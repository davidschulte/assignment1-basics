from cs336_basics.pretokenization_example import find_chunk_boundaries
import json
import os
import pathlib

from tqdm import tqdm
import regex as re

from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO, Self

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

    def _init_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for idx, special_token in enumerate(self.special_tokens):
            vocab[idx + 256] = special_token.encode()

        return vocab

    @staticmethod
    def _read_chunk(f: BinaryIO, boundaries: tuple[int, int]) -> str:
        start, end = boundaries
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")

    def _split_chunk_to_segments(self, chunk: str) -> list[str]:
        delim = "|".join(re.escape(s) for s in self.special_tokens)

        return [seg for seg in re.split(f"({delim})", chunk) if len(seg) > 0]

    def pretokenize_doc(self, doc: str):
        counter = Counter()
        for match in re.finditer(self.pat, doc):
            counter[match.group()] += 1

        return counter

    def _pretokenize_with_chunk_boundaries(self, input_path: str, boundaries: tuple[int, int]) -> Counter:
        counter = Counter()

        with open(input_path, "rb") as f:
            chunk = self._read_chunk(f, boundaries)
        for seg in tqdm(self._split_chunk_to_segments(chunk)):
            if seg not in self.special_tokens:
                counter += self.pretokenize_doc(seg)

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

    @staticmethod
    def get_byte_pairs(text: str) -> list[tuple[bytes, bytes]]:
        text_bytes = text.encode()
        return [(bytes(text_bytes[i : i + 1]), bytes(text_bytes[i + 1 : i + 2])) for i in range(len(text_bytes) - 1)]

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

    @staticmethod
    def merge(tokens: list[bytes], merge_pair: tuple[bytes, bytes]) -> list[bytes]:
        new_tokens = []
        idx = 0
        while idx < len(tokens) - 1:
            if (tokens[idx], tokens[idx + 1]) == merge_pair:
                new_tokens.append(b"".join(merge_pair))
                idx += 2
            else:
                new_tokens.append(tokens[idx])
                idx += 1

        if idx == len(tokens) - 1:
            new_tokens.append(tokens[idx])

        return new_tokens

    @staticmethod
    def count_token_pairs(tokens: list[bytes]) -> Counter[tuple[bytes, bytes]]:
        return Counter(zip(tokens[:-1], tokens[1:]))

    def train_vocab_and_merges(self, pretoken_counter: Counter[str]) -> Self:
        vocab = self._init_vocab()
        merges = []
        """
        Steps:
        1. Initialize dicts:
        pretoken -> pretoken_count √
        pretoken -> tokens
        token_pair -> (pretokens in which its included, num how often)

        2. Counter number of token pair occurrences

        3. Find the maximum

        4. Merge
        4.1. Find all pairs that have to be checked: overlaps from both sides or the merge pair itself
        4.2. Find all their pretokens where these pairs are both included: Either overlaps left AND right or merge pair itself
        4.3. For all overlap pairs and merge pair, remove the counter for all the selected pretokens
        4.4. For all of the pretokens, merge tokens with merge pair
        4.5. For all of the pretokens, update the counts of overlap tokens

        """
        pretoken_to_tokens = {pretoken: [bytes([b]) for b in pretoken.encode()] for pretoken in pretoken_counter}
        token_pairs_to_pretoken_and_count = defaultdict(dict)

        for pretoken, tokens in pretoken_to_tokens.items():
            token_pair_counter = self.count_token_pairs(tokens)
            for token_pair, token_pair_count in token_pair_counter.items():
                token_pairs_to_pretoken_and_count[token_pair][pretoken] = token_pair_count

        for idx in tqdm(range(len(vocab), self.vocab_size)):
            # determine most common token pair
            merge_pair = max(
                token_pairs_to_pretoken_and_count.items(),
                key=lambda item: (
                    sum(
                        token_pair_count * pretoken_counter[pretoken] for pretoken, token_pair_count in item[1].items()
                    ),
                    item[0],
                ),
            )[0]

            # merge
            vocab[idx] = b"".join(merge_pair)
            merges.append(merge_pair)

            pretokens_to_merge = set()
            left_overlap_token_pairs = set(
                token_pair for token_pair in token_pairs_to_pretoken_and_count if token_pair[1] == merge_pair[0]
            )
            right_overlap_token_pairs = set(
                token_pair for token_pair in token_pairs_to_pretoken_and_count if token_pair[0] == merge_pair[1]
            )
            pretokens_to_merge = (
                set(
                    pretoken
                    for pair in left_overlap_token_pairs
                    for pretoken in list(token_pairs_to_pretoken_and_count[pair].keys())
                )
                & set(
                    pretoken
                    for pair in right_overlap_token_pairs
                    for pretoken in list(token_pairs_to_pretoken_and_count[pair].keys())
                )
            ) | set(token_pairs_to_pretoken_and_count[merge_pair].keys())

            for token_pair in left_overlap_token_pairs | right_overlap_token_pairs | set([merge_pair]):
                for pretoken in pretokens_to_merge & set(token_pairs_to_pretoken_and_count[token_pair].keys()):
                    token_pairs_to_pretoken_and_count[token_pair].pop(pretoken, None)

            for pretoken in pretokens_to_merge:
                tokens = pretoken_to_tokens[pretoken]
                updated_tokens = self.merge(tokens, merge_pair)

                if updated_tokens != tokens:
                    tokens = updated_tokens
                    pretoken_to_tokens[pretoken] = tokens

                token_pair_counter = self.count_token_pairs(tokens)
                for token_pair, token_pair_count in token_pair_counter.items():
                    token_pairs_to_pretoken_and_count[token_pair][pretoken] = token_pair_count

        self.vocab = vocab
        self.merges = merges

        return self

    def fit(self, input_path: str | os.PathLike):
        pretoken_counter = self.pretokenize(input_path)
        self.train_vocab_and_merges(pretoken_counter)

    def save(self, output_dir: str | os.PathLike):
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab_size": len(self.vocab),
            "eos": self.eos.decode(errors="surrogateescape"),
            "num_chunk_processes": self.num_chunk_processes,
            "pat": self.pat,
            "special_tokens": self.special_tokens,
        }

        (output_dir / "config.json").write_text(json.dumps(config))
        (output_dir / "vocab.json").write_text(
            json.dumps({idx: v.decode(errors="surrogateescape") for idx, v in self.vocab.items()})
        )
        (output_dir / "merges.json").write_text(
            json.dumps([[token.decode(errors="surrogateescape") for token in pair] for pair in self.merges])
        )

    @classmethod
    def load(cls, dir: str | os.PathLike) -> Self:
        dir = pathlib.Path(dir)
        config = json.loads((dir / "config.json").read_text())
        config["eos"] = config["eos"].encode(errors="surrogateescape")

        vocab = json.loads((dir / "vocab.json").read_text())
        vocab = {int(idx): v.encode(errors="surrogateescape") for idx, v in vocab.items()}

        merges = json.loads((dir / "merges.json").read_text())
        merges = [tuple(token.encode(errors="surrogateescape") for token in pair) for pair in merges]

        tokenizer = Tokenizer(**config)
        tokenizer.vocab = vocab
        tokenizer.merges = merges

        return tokenizer


if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=1000)

    # input_path = "data/TinyStories-valid.txt"
    input_path = "../tests/fixtures/tinystories_sample_5M.txt"
    # "tinystories_sample_5M.txt"

    counter = tokenizer.pretokenize(input_path=input_path)
    print(counter.most_common(20))

    vocab, merges = tokenizer.train_vocab_and_merges(counter)
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
) -> Tokenizer:
    tokenizer = Tokenizer(vocab_size, special_tokens=special_tokens, num_chunk_processes=num_chunk_processes, eos=eos)
    tokenizer.fit(input_path)

    return tokenizer
