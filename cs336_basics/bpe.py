from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
import numpy as np

from tqdm import tqdm
import regex as re

from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab_size: int,
        eos: bytes = b"<|endoftext|>",
        num_chunk_processes: int = 4,
        pat: str = PAT,
        special_tokens: list[str] | None = None,
    ):
        self.vocab_size = vocab_size
        self.eos = eos
        self.num_chunk_processes = num_chunk_processes
        self.pat = pat
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens = list(set(self.special_tokens + [eos.decode("utf-8")]))

        self.vocab = self._init_vocab()
        self.merges = []

    def _init_vocab(self) -> set[bytes]:
        vocab = set(bytes([b]) for b in range(256))
        for special_token in self.special_tokens:
            vocab.add(special_token.encode())

        return vocab

    @staticmethod
    def _read_chunk(f: BinaryIO, boundaries: tuple[int, int]) -> str:
        start, end = boundaries
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")

    # def read_chunks(self, input_path: str) -> list[str]:
    #     with open(input_path, "rb") as f:
    #         boundaries = find_chunk_boundaries(
    #             file=f, desired_num_chunks=self.num_chunk_processes, split_special_token=self.eos
    #         )
    #         chunks = [self._read_chunk(f, boundary) for boundary in zip(boundaries[:-1], boundaries[1:])]
    #
    #     return chunks

    def _split_chunk_to_docs(self, chunk: str) -> list[str]:
        delim = "|".join(re.escape(s) for s in self.special_tokens)

        return re.split(delim, chunk)

    def pretokenize_doc(self, doc: str):
        counter = Counter()
        for match in re.finditer(self.pat, doc):
            counter[match.group()] += 1

        return counter

    def _pretokenize_with_chunk_boundaries(self, input_path: str, boundaries: tuple[int, int]) -> Counter:
        counter = Counter()

        with open(input_path, "rb") as f:
            chunk = self._read_chunk(f, boundaries)
        for doc in tqdm(self._split_chunk_to_docs(chunk)):
            counter += self.pretokenize_doc(doc)

        return counter

    def pretokenize(self, input_path: str | os.PathLike) -> Counter:
        counter = Counter()

        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                file=f, desired_num_chunks=self.num_chunk_processes, split_special_token=self.eos
            )

        boundary_edges = list(zip(boundaries[:-1], boundaries[1:]))
        worker_args = [(input_path, edges) for edges in boundary_edges]

        with Pool(self.num_chunk_processes) as p:
            chunk_counters = p.starmap(self._pretokenize_with_chunk_boundaries, worker_args)

        for chunk_counter in chunk_counters:
            counter += chunk_counter

        return counter

    # TODO: Handle special tokens
    @staticmethod
    def _get_token_pair_indices(pretoken: str) -> dict[tuple[bytes, bytes], list[int]]:
        """For each token pair in a pretoken, list its starting indices"""
        byte_pair_to_indices = defaultdict(list)
        pretoken_bytes = pretoken.encode()
        for idx in range(len(pretoken_bytes) - 1):
            byte_pair_to_indices[(pretoken_bytes[idx : idx + 1], pretoken_bytes[idx + 1 : idx + 2])].append(idx)

        return byte_pair_to_indices

    @staticmethod
    def merge_token_pair_in_token_pairs_to_indices(
        token_pairs_to_indices: dict[tuple[bytes, bytes], dict[str, list[int]]], merge_pair: tuple[bytes, bytes]
    ) -> None:
        """Update the mapping of token pairs to indices by merging a token pair and updating the index list of overlapping pairs"""
        merge_pair_idxs_per_pretoken = token_pairs_to_indices[merge_pair]

        all_new_merges = defaultdict(lambda: defaultdict(list))
        for pair, pair_idxs_per_pretoken in list(token_pairs_to_indices.items()):
            if pair == merge_pair:
                continue

            # token pair overlaps with merged token pair to its left
            if pair[0] == merge_pair[1] or pair[1] == merge_pair[0]:
                for pretoken, pair_idxs in list(pair_idxs_per_pretoken.items()):
                    # merge_pair_idxs =

                    merge_pair_idxs = merge_pair_idxs_per_pretoken.get(pretoken)
                    if merge_pair_idxs is None:
                        continue

                    # token pair overlaps with merged token pair to its left
                    if pair[1] == merge_pair[0]:
                        new_merges = [idx for idx in pair_idxs if idx + len(pair[0]) in merge_pair_idxs]
                        if len(new_merges) > 0:
                            pair_idxs[:] = [idx for idx in pair_idxs if idx not in new_merges]
                            all_new_merges[(pair[0], b"".join(merge_pair))][pretoken] += new_merges

                    # token pair overlaps with merged token pair to its right
                    if pair[0] == merge_pair[1]:
                        new_merges = [idx for idx in merge_pair_idxs if idx + len(merge_pair[0]) in pair_idxs]
                        if len(new_merges) > 0:
                            pair_idxs[:] = [idx for idx in pair_idxs if idx - len(merge_pair[0]) not in new_merges]
                            all_new_merges[(b"".join(merge_pair), pair[1])][pretoken] += new_merges

                    if len(pair_idxs) == 0:
                        del token_pairs_to_indices[pair][pretoken]

        token_pairs_to_indices.update(all_new_merges)
        del token_pairs_to_indices[merge_pair]

    def tokenize(self, pretokenize_counter: Counter[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab = self._init_vocab()
        merges = []

        # create pretoken -> (token_tuple -> list of indices)
        token_pair_idxs = defaultdict(dict)
        for pretoken in pretokenize_counter:
            token_indices = self._get_token_pair_indices(pretoken)
            for token_pair, idxs in token_indices.items():
                token_pair_idxs[token_pair][pretoken] = idxs

        for _ in tqdm(range(self.vocab_size - len(vocab))):
            # determine most common token pair
            merge_pair = max(
                token_pair_idxs.items(),
                key=lambda item: (
                    sum(pretokenize_counter[pretoken] * len(idxs) for pretoken, idxs in item[1].items()),
                    item[0],
                ),
            )[0]

            # merge
            vocab.add(b"".join(merge_pair))
            merges.append(merge_pair)

            # update pretoken_token_dict
            self.merge_token_pair_in_token_pairs_to_indices(token_pair_idxs, merge_pair)

        self.vocab = vocab
        self.merges = merges

        return dict(enumerate(vocab)), merges


if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=1000)

    # input_path = "data/TinyStories-valid.txt"
    input_path = "../tests/fixtures/tinystories_sample_5M.txt"
    # "tinystories_sample_5M.txt"

    counter = tokenizer.pretokenize(input_path=input_path)
    print(counter.most_common(20))

    vocab, merges = tokenizer.tokenize(counter)
    # print(f"{vocab=}")
    print(f"{merges=}")

    def safe_decode(byte_data: bytes, encoding: str = "utf-8") -> str:
        try:
            return byte_data.decode(encoding)
        except UnicodeDecodeError:
            return ""

    print([safe_decode(v) for v in vocab.values()])


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str] | None = None,
    eos: bytes = b"<|endoftext|>",
    num_chunk_processes: int = 16,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    tokenizer = Tokenizer(vocab_size, special_tokens=special_tokens, num_chunk_processes=num_chunk_processes, eos=eos)
    counter = tokenizer.pretokenize(input_path=input_path)

    vocab, merges = tokenizer.tokenize(counter)

    return vocab, merges
