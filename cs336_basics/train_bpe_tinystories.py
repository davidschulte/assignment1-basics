import json

from bpe import train_bpe

DATASET_PATH = "data/TinyStories-train.txt"
VOCAB_SIZE = 10_000

if __name__ == "__main__":
    vocab, merges = train_bpe(input_path=DATASET_PATH, vocab_size=VOCAB_SIZE)

    with open("data/tokenizer/vocab.json", "w") as f:
        json.dump(vocab, f)

    with open("data/tokenizer/merges.json", "w") as f:
        json.dump(merges, f)
