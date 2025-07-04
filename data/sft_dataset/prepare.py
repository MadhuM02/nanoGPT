
import os
import pickle
import numpy as np
from datasets import load_dataset
import tiktoken

def prepare_ag_news():
    """Downloads and prepares the AG News dataset."""
    # --- Download and process dataset ---
    print("Downloading AG News dataset...")
    dataset = load_dataset("ag_news")

    # --- Format and save to text files ---
    data_dir = os.path.dirname(__file__)
    train_txt_path = os.path.join(data_dir, 'train.txt')
    val_txt_path = os.path.join(data_dir, 'val.txt')

    # Class names for AG News
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    def format_and_save(split, path):
        with open(path, 'w', encoding='utf-8') as f:
            for example in dataset[split]:
                # Combine title and description for more context
                text = example['text']
                label = class_names[example['label']]
                # Format as "[CLASS]: text"
                f.write(f"[{label}]: {text}\n")

    print("Formatting and saving train/validation splits...")
    format_and_save('train', train_txt_path)
    format_and_save('test', val_txt_path) # Using test split as validation

    # --- Tokenization ---
    print("Tokenizing data...")
    enc = tiktoken.get_encoding("gpt2")

    def tokenize_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = f.read()
        return enc.encode_ordinary(data)

    train_ids = tokenize_file(train_txt_path)
    val_ids = tokenize_file(val_txt_path)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # --- Save tokenized data ---
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    # --- Save metadata ---
    meta = {
        'vocab_size': enc.n_vocab,
        'itos': {i: enc.decode([i]) for i in range(enc.n_vocab)},
        'stoi': {v: k for k, v in {i: enc.decode([i]) for i in range(enc.n_vocab)}.items()},
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset preparation complete!")

if __name__ == '__main__':
    prepare_ag_news()
