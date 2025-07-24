import os
import json
import pickle
import numpy as np
from datasets import load_dataset, concatenate_datasets
import tiktoken
from tqdm import tqdm

def prepare_large_sft_dataset():
    """Downloads and prepares a large, diverse SFT dataset combining multiple sources."""
    print("🚀 Preparing large SFT dataset...")
    
    data_dir = os.path.dirname(__file__)
    train_jsonl_path = os.path.join(data_dir, 'train.jsonl')
    val_jsonl_path = os.path.join(data_dir, 'val.jsonl')
    
    # Initialize lists to store all examples
    all_train_examples = []
    all_val_examples = []
    
    # Dataset 1: AG News (4-class news classification)
    print("📰 Loading AG News dataset...")
    ag_news = load_dataset("ag_news")
    ag_class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    
    for split, examples_list in [('train', all_train_examples), ('test', all_val_examples)]:
        for example in tqdm(ag_news[split], desc=f"Processing AG News {split}"):
            text = example['text']
            label = ag_class_names[example['label']]
            # Create prompt-completion format
            prompt = f"Classify this news article:\n{text}\nCategory:"
            completion = f" {label}"
            formatted_example = {"prompt": prompt, "completion": completion}
            examples_list.append(formatted_example)
    
    # Dataset 2: IMDB Movie Reviews (sentiment classification)
    print("🎬 Loading IMDB dataset...")
    imdb = load_dataset("imdb")
    sentiment_labels = ['negative', 'positive']
    
    # Use a subset to balance dataset sizes
    max_imdb_samples = 10000  # Limit to prevent IMDB from dominating
    
    for split, examples_list in [('train', all_train_examples), ('test', all_val_examples)]:
        count = 0
        for example in tqdm(imdb[split], desc=f"Processing IMDB {split}"):
            if count >= max_imdb_samples:
                break
            text = example['text']
            label = sentiment_labels[example['label']]
            # Create prompt-completion format
            prompt = f"Analyze the sentiment of this movie review:\n{text}\nSentiment:"
            completion = f" {label}"
            formatted_example = {"prompt": prompt, "completion": completion}
            examples_list.append(formatted_example)
            count += 1
    
    # Dataset 3: Yelp Reviews (5-star rating prediction)
    print("⭐ Loading Yelp Reviews dataset...")
    try:
        yelp = load_dataset("yelp_review_full")
        max_yelp_samples = 8000  # Limit samples
        
        for split, examples_list in [('train', all_train_examples), ('test', all_val_examples)]:
            count = 0
            for example in tqdm(yelp[split], desc=f"Processing Yelp {split}"):
                if count >= max_yelp_samples:
                    break
                text = example['text']
                stars = example['label'] + 1  # Convert 0-4 to 1-5 stars
                # Create prompt-completion format
                prompt = f"Rate this business review from 1-5 stars:\n{text}\nRating:"
                completion = f" {stars} stars"
                formatted_example = {"prompt": prompt, "completion": completion}
                examples_list.append(formatted_example)
                count += 1
    except Exception as e:
        print(f"Warning: Could not load Yelp dataset: {e}")
    
    # Dataset 4: Financial PhraseBank (financial sentiment)
    print("💰 Loading Financial PhraseBank dataset...")
    try:
        financial = load_dataset("financial_phrasebank", "sentences_allagree")
        financial_labels = ['negative', 'neutral', 'positive']
        
        # Split into train/val (80/20)
        total_examples = len(financial['train'])
        train_size = int(0.8 * total_examples)
        
        for i, example in enumerate(tqdm(financial['train'], desc="Processing Financial data")):
            text = example['sentence']
            label = financial_labels[example['label']]
            # Create prompt-completion format
            prompt = f"Determine the financial sentiment:\n{text}\nSentiment:"
            completion = f" {label}"
            formatted_example = {"prompt": prompt, "completion": completion}
            
            if i < train_size:
                all_train_examples.append(formatted_example)
            else:
                all_val_examples.append(formatted_example)
    except Exception as e:
        print(f"Warning: Could not load Financial PhraseBank: {e}")
    
    # Dataset 5: Emotion Detection
    print("😊 Loading Emotion dataset...")
    try:
        emotion = load_dataset("emotion")
        emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        max_emotion_samples = 5000  # Limit samples
        
        for split, examples_list in [('train', all_train_examples), ('validation', all_val_examples)]:
            count = 0
            for example in tqdm(emotion[split], desc=f"Processing Emotion {split}"):
                if count >= max_emotion_samples:
                    break
                text = example['text']
                label = emotion_labels[example['label']]
                # Create prompt-completion format
                prompt = f"Identify the emotion in this text:\n{text}\nEmotion:"
                completion = f" {label}"
                formatted_example = {"prompt": prompt, "completion": completion}
                examples_list.append(formatted_example)
                count += 1
    except Exception as e:
        print(f"Warning: Could not load Emotion dataset: {e}")
    
    # Dataset 6: SMS Spam Classification
    print("📱 Loading SMS Spam dataset...")
    try:
        sms_spam = load_dataset("sms_spam")
        spam_labels = ['ham', 'spam']
        
        # Split into train/val (80/20)
        total_examples = len(sms_spam['train'])
        train_size = int(0.8 * total_examples)
        
        for i, example in enumerate(tqdm(sms_spam['train'], desc="Processing SMS data")):
            text = example['sms']
            label = spam_labels[example['label']]
            # Create prompt-completion format
            prompt = f"Classify this SMS message:\n{text}\nType:"
            completion = f" {label}"
            formatted_example = {"prompt": prompt, "completion": completion}
            
            if i < train_size:
                all_train_examples.append(formatted_example)
            else:
                all_val_examples.append(formatted_example)
    except Exception as e:
        print(f"Warning: Could not load SMS Spam dataset: {e}")
    
    # Shuffle the combined datasets
    import random
    random.seed(42)
    random.shuffle(all_train_examples)
    random.shuffle(all_val_examples)
    
    print(f"📊 Dataset Statistics:")
    print(f"  Training examples: {len(all_train_examples):,}")
    print(f"  Validation examples: {len(all_val_examples):,}")
    print(f"  Total examples: {len(all_train_examples) + len(all_val_examples):,}")
    
    # Write to JSONL files
    print("💾 Writing JSONL files...")
    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for example in tqdm(all_train_examples, desc="Writing train JSONL"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    with open(val_jsonl_path, 'w', encoding='utf-8') as f:
        for example in tqdm(all_val_examples, desc="Writing val JSONL"):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    
    # --- Tokenization ---
    print("🔤 Tokenizing data...")
    enc = tiktoken.get_encoding("gpt2")
    
    def tokenize_jsonl_file(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        tokenized_examples = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                example = json.loads(line.strip())
                # Combine prompt and completion
                full_text = example['prompt'] + example['completion']
                tokens = enc.encode_ordinary(full_text)
                if len(tokens) > 500:  # Leave room for padding/special tokens
                    # Truncate long sequences
                    tokens = tokens[:500]
                tokenized_examples.extend(tokens)
                tokenized_examples.append(enc.encode_ordinary('\n\n')[0])  # Add separator
        
        return tokenized_examples
    
    print("Tokenizing training data...")
    train_ids = tokenize_jsonl_file(train_jsonl_path)
    print("Tokenizing validation data...")
    val_ids = tokenize_jsonl_file(val_jsonl_path)
    
    print(f"📈 Tokenization Results:")
    print(f"  Train tokens: {len(train_ids):,}")
    print(f"  Val tokens: {len(val_ids):,}")
    print(f"  Total tokens: {len(train_ids) + len(val_ids):,}")
    
    # --- Save tokenized data ---
    print("💿 Saving binary files...")
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))
    
    # --- Save metadata ---
    meta = {
        'vocab_size': enc.n_vocab,
        'itos': {i: enc.decode([i]) for i in range(enc.n_vocab)},
        'stoi': {v: k for k, v in {i: enc.decode([i]) for i in range(enc.n_vocab)}.items()},
        'dataset_info': {
            'total_examples': len(all_train_examples) + len(all_val_examples),
            'train_examples': len(all_train_examples),
            'val_examples': len(all_val_examples),
            'train_tokens': len(train_ids),
            'val_tokens': len(val_ids),
            'datasets_included': [
                'AG News (news classification)',
                'IMDB (movie sentiment)',
                'Yelp Reviews (rating prediction)', 
                'Financial PhraseBank (financial sentiment)',
                'Emotion (emotion detection)',
                'SMS Spam (spam classification)'
            ]
        }
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("✅ Large SFT dataset preparation complete!")
    print(f"📁 Files created:")
    print(f"  - train.bin ({len(train_ids):,} tokens)")
    print(f"  - val.bin ({len(val_ids):,} tokens)")
    print(f"  - meta.pkl (metadata)")
    print(f"  - train.jsonl ({len(all_train_examples):,} examples)")
    print(f"  - val.jsonl ({len(all_val_examples):,} examples)")
    
    # Estimate training time
    tokens_per_iter = 8 * 256  # batch_size * block_size from config
    estimated_iters = len(train_ids) // tokens_per_iter
    print(f"🕐 Estimated training iterations needed: {estimated_iters:,}")
    print(f"💡 Consider increasing max_iters in config to: {min(estimated_iters, 5000)}")

if __name__ == '__main__':
    prepare_large_sft_dataset()
