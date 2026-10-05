from bpe import train_bpe

DATASET_PATH = "data/TinyStories-train.txt"
VOCAB_SIZE = 10_000
OUTPUT_DIR = "data/tokenizers/tinystories"

if __name__ == "__main__":
    tokenizer = train_bpe(input_path=DATASET_PATH, vocab_size=VOCAB_SIZE)

    tokenizer.save(OUTPUT_DIR)
