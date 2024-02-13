# Instruction code heavily inspired by Andreas Köpf
# source: https://github.com/andreaskoepf/epfl-megatron/tree/local_changes/
"""Processing data for instruction tuning.
Example:
python instruct/preprocess_instruct_data.py --input=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1.jsonl \
    --output_prefix=/pure-mlo-scratch/alhernan/data/medmc/medmc-v1 \
    --tokenizer_type=SentencePieceTokenizer \
    --vocab_file=/pure-mlo-scratch/llama/tokenizer.model \
    --chunk_size=32 --workers=32 \
    --vocab_extra_ids_list "[bib_ref],[/bib_ref],[fig_ref],[/fig_ref],[bib],[/bib],[fig],[/fig],[table],[/table],[formula],[/formula],<|im_start|>,<|im_end|>" \
    --question_key=input \
    --answer_key=output \
    --system_key=instruction
"""

import sys
import json
import time
import itertools
from pathlib import Path
from typing import Optional
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

import torch

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from megatron.tokenizer import build_tokenizer
from megatron.tokenizer.tokenizer import AbstractTokenizer
from megatron.data.indexed_dataset import make_builder
from megatron.data.instruction_dataset import Role


class Encoder(object):
    tokenizer: Optional[AbstractTokenizer] = None

    def __init__(self, args: Namespace):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, line: str) -> tuple[int, list[int], list[int]]:
        # get data
        assert Encoder.tokenizer is not None
        data = json.loads(line)
        # tokenize and get roles
        tokens = []
        roles = []
        weights = []
        for i in data["messages"]:
            if "name" in i:
                text = format_message(i["content"], i["role"], i["name"])
            else:
                text = format_message(i["content"], i["role"])
            token = Encoder.tokenizer.tokenize(text)
            tokens += token

            if i["role"] == "system":
                roles += [Role.system.value]*len(token)
                if self.args.weight_key:
                    weights += [float(i[self.args.weight_key])]*len(token)
                else:
                    weights += [self.args.system_weight]*len(token)
            elif i["role"] == "user":
                roles += [Role.prompter.value]*len(token)
                if self.args.weight_key:
                    weights += [float(i[self.args.weight_key])]*len(token)
                else:
                    weights += [self.args.prompter_weight]*len(token)
            elif i["role"] == "assistant":
                roles += [Role.assistant.value]*len(token)
                if self.args.weight_key:
                    weights += [float(i[self.args.weight_key])]*len(token)
                else:
                    weights += [self.args.assistant_weight]*len(token)
            else:
                raise ValueError(f"Unknown role {i['role']}")

        return len(line), tokens, roles, weights

    @property
    def special_tokens(self) -> dict:
        return self.tokenizer._special_tokens


class DatasetWriter:
    def __init__(self, prefix: str, vocab_size: int, dataset_impl: str = "mmap",
                 feature: str = "text"):
        self.vocab_size = vocab_size
        self.dataset_impl = dataset_impl
        self.bin_fname = f"{prefix}-{feature}.bin"
        self.idx_fname = f"{prefix}-{feature}.idx"
        self.builder = None

    def add_item(self, tokens: list[int]):
        self.builder.add_item(torch.IntTensor(tokens))

    def __enter__(self):
        self.builder = make_builder(self.bin_fname, impl=self.dataset_impl,
                                    vocab_size=self.vocab_size)
        return self

    def __exit__(self, *_):
        self.builder.finalize(self.idx_fname)
        self.builder = None


def format_message(message: str, role: str, name: str = None) -> str:
    if role == "system" and name:
        return f"<|im_start|>{role} name={name}\n{message}<|im_end|>"
    else:
        return f"<|im_start|>{role}\n{message}<|im_end|>"


def get_args():
    parser = ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, nargs="+",
                       help='Path(s) to input JSONL file(s)')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 'FalconTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab_file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge_file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_prefix', type=Path, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk_size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log_interval', type=int, default=100,
                       help='Interval between progress updates')
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--vocab_extra_ids_list', type=str, default=None,
                       help='comma separated list of special vocab ids to add to the tokenizer')
    group.add_argument("--no_new_tokens", action="store_false", dest="new_tokens",
                       help=("Whether to add special tokens (e.g. CLS, MASK, etc) "
                             "in the sentencepiece tokenizer or not"))

    group.add_argument('--weight_key', help='key to extract loss weight (optional)')
    group.add_argument("--system_weight", type=float, default=0.0, help="")
    group.add_argument("--prompter_weight", type=float, default=0.0, help="")
    group.add_argument("--assistant_weight", type=float, default=1.0, help="")

    group.add_argument("--do_packing", action="store_true", help="")
    group.add_argument("--max_packing_size", type=int, default=0, help="")

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    return args


def ffd_packing(data, max_bin_size, eos_token=[]):
    bins = []
    bins_results = []
    sorted_data = sorted(data, key=lambda x: x["size"], reverse=True)
    for i in range(len(sorted_data)):
        add_new = True
        for j in range(len(bins)):
            if bins[j] + (sorted_data[i]["size"] + len(eos_token)) <= max_bin_size:
                bins[j] += (sorted_data[i]["size"] + len(eos_token))
                bins_results[j].append(sorted_data[i])
                add_new = False
                break
        if add_new:
            if sorted_data[i]["size"] > max_bin_size:
                print(f"Warning: seq_len {sorted_data[i]['size']} > max_packing_seq_len {max_bin_size}")
            bins.append(sorted_data[i]["size"])
            bins_results.append([sorted_data[i]])

    tokens_list = []
    roles_list = []
    weights_list = []
    for bin_res in bins_results:
        tokens = []
        roles = []
        weights = []
        for x in bin_res:
            if tokens:
                tokens.extend(eos_token)
                # No loss for eos token
                roles.extend([Role.system.value]*len(eos_token))
                weights.extend([0.0]*len(eos_token))
            tokens.extend(x["tokens"])
            roles.extend(x["roles"])
            weights.extend(x["weights"])
        assert len(tokens) == len(roles) == len(weights)
        tokens_list.append(tokens)
        roles_list.append(roles)
        weights_list.append(weights)
    print(f"Packing done!\noriginal size: {len(data)}\nnew size: {len(tokens_list)}")
    return tokens_list, roles_list, weights_list


def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    vocab_size = build_tokenizer(args).vocab_size
    special_tokens = build_tokenizer(args)._special_tokens
    fs = map(open, args.input)
    processed_data = []
    with Pool(args.workers, initializer=encoder.initializer) as pool, \
            DatasetWriter(args.output_prefix, vocab_size, args.dataset_impl,
                          "text") as token_writer, \
            DatasetWriter(args.output_prefix, 16, args.dataset_impl,
                          "role") as role_writer, \
            DatasetWriter(args.output_prefix, None, args.dataset_impl,
                          "weight") as weigth_writer:

        f = itertools.chain(*fs)
        docs = pool.imap(encoder.encode, f, args.chunk_size)
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (size, tokens, roles, weights) in enumerate(docs, start=1):
            total_bytes_processed += size
            token_writer.add_item(tokens)
            role_writer.add_item(roles)
            weigth_writer.add_item(weights)
            assert len(tokens) == len(roles) == len(weights)
            processed_data.append(
                {"tokens": tokens, "roles": roles, "weights": weights, "size": len(tokens)}
            )
            if i % args.log_interval == 0:
                elapsed = time.time() - proc_start
                mbs = total_bytes_processed/1024/1024/elapsed
                print(f"Processed {i} documents ({i/elapsed} docs/s, {mbs} MB/s).")
        print("Done! Now finalizing.")

    for f in fs:
        f.close()

    if args.do_packing:
        assert args.max_packing_size > 0
        packed_tokens, packed_roles, packed_weights = ffd_packing(
            processed_data, args.max_packing_size, eos_token=[special_tokens["</s>"]]
        )

        with DatasetWriter(
            f"{args.output_prefix}_packed_{args.max_packing_size}",
            vocab_size, args.dataset_impl, "text"
        ) as packed_token_writer:
            for i in packed_tokens:
                packed_token_writer.add_item(i)

        with DatasetWriter(
            f"{args.output_prefix}_packed_{args.max_packing_size}",
            16, args.dataset_impl, "role"
        ) as packed_role_writer:
            for i in packed_roles:
                packed_role_writer.add_item(i)

        with DatasetWriter(
            f"{args.output_prefix}_packed_{args.max_packing_size}",
            None, args.dataset_impl, "weight"
        ) as packed_weight_writer:
            for i in packed_weights:
                packed_weight_writer.add_item(i)


if __name__ == '__main__':
    main()
