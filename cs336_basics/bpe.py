from cs336_basics.pretokenization_example import find_chunk_boundaries

from tqdm import tqdm
import regex as re

from collections import Counter
from typing import BinaryIO

# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class PreTokenizer:
    def __init__(self, eos: bytes = b"<|endoftext|>", num_chunk_processes: int = 4, pat: str = PAT):
        self.eos = eos
        self.num_chunk_processes = num_chunk_processes
        self.pat = pat

    @staticmethod
    def _read_chunk(f: BinaryIO, boundaries: tuple[int, int]) -> str:
        start, end = boundaries
        f.seek(start)
        return f.read(end - start).decode("utf-8", errors="ignore")

    def read_chunks(self, input_path: str) -> list[str]:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                file=f, desired_num_chunks=self.num_chunk_processes, split_special_token=self.eos
            )
            chunks = [self._read_chunk(f, boundary) for boundary in zip(boundaries[:-1], boundaries[1:])]

        return chunks

    def _split_chunk_to_docs(self, chunk: str) -> list[str]:
        return chunk.split(self.eos.decode("utf-8"))

    def pretokenize_doc(self, doc: str):
        counter = Counter()
        for match in re.finditer(self.pat, doc):
            counter[match.group()] += 1

        return counter

    def pretokenize(self, input_path: str) -> Counter:
        counter = Counter()

        for chunk in self.read_chunks(input_path=input_path):
            for doc in tqdm(self._split_chunk_to_docs(chunk)):
                counter += self.pretokenize_doc(doc)

        return counter


if __name__ == "__main__":
    pretokenizer = PreTokenizer()

    input_path = "data/TinyStories-valid.txt"

    counter = pretokenizer.pretokenize(input_path=input_path)

    print(counter.most_common(3))


# def train_bpe(
#     input_path: str,
#     vocab_size: int,
#     special_tokens=list[str],
#     eos: bytes = b"<|endoftext|>",
#     num_chunk_processes: int = 4,
# ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
