"""
Prepare an extra large, diverse SFT dataset for nanoGPT
This combines multiple datasets across different domains and tasks
"""

import os
import pickle
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import argparse

# we now encode with tiktoken
import tiktoken

def get_tokenizer():
    """Get the tokenizer"""
    enc = tiktoken.get_encoding("gpt2")
    return enc

def prepare_conversation_dataset(dataset_name, dataset_config=None, text_field='text', response_field=None):
    """Prepare a dataset for conversation/instruction following"""
    print(f"Loading {dataset_name}...")
    
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the training split
    if 'train' in dataset:
        train_data = dataset['train']
    else:
        train_data = dataset
    
    conversations = []
    
    for item in tqdm(train_data, desc=f"Processing {dataset_name}"):
        if response_field and response_field in item:
            # This is a Q&A format
            conversation = f"Human: {item[text_field]}\n\nAssistant: {item[response_field]}"
        else:
            # This is a text completion format
            conversation = f"Complete the following: {item[text_field]}"
        
        conversations.append(conversation)
    
    return conversations

def prepare_classification_dataset(dataset_name, dataset_config=None, text_field='text', label_field='label', class_names=None):
    """Prepare a dataset for classification tasks"""
    print(f"Loading {dataset_name}...")
    
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the training split
    if 'train' in dataset:
        train_data = dataset['train']
    else:
        train_data = dataset
    
    conversations = []
    
    for item in tqdm(train_data, desc=f"Processing {dataset_name}"):
        text = item[text_field]
        label = item[label_field]
        
        # Convert label to text
        if class_names and isinstance(label, int):
            label_text = class_names[label]
        else:
            label_text = str(label)
        
        conversation = f"Classify this text: {text}\n\nCategory: {label_text}"
        conversations.append(conversation)
    
    return conversations

def prepare_extra_large_dataset():
    """Prepare an extra large, diverse SFT dataset"""
    
    conversations = []
    
    # 1. News Classification (AG News)
    print("=== Processing AG News ===")
    ag_news = prepare_classification_dataset(
        "ag_news",
        text_field='text',
        label_field='label',
        class_names=['World', 'Sports', 'Business', 'Technology']
    )
    conversations.extend(ag_news)
    
    # 2. Movie Reviews (IMDB)
    print("=== Processing IMDB Reviews ===")
    imdb = prepare_classification_dataset(
        "imdb",
        text_field='text',
        label_field='label',
        class_names=['Negative', 'Positive']
    )
    conversations.extend(imdb)
    
    # 3. Yelp Reviews
    print("=== Processing Yelp Reviews ===")
    yelp = prepare_classification_dataset(
        "yelp_review_full",
        text_field='text',
        label_field='label',
        class_names=['1 star', '2 stars', '3 stars', '4 stars', '5 stars']
    )
    conversations.extend(yelp)
    
    # 4. Financial News (Financial PhraseBank)
    print("=== Processing Financial PhraseBank ===")
    try:
        financial = prepare_classification_dataset(
            "financial_phrasebank",
            dataset_config="sentences_allagree",
            text_field='sentence',
            label_field='label',
            class_names=['negative', 'neutral', 'positive']
        )
        conversations.extend(financial)
    except Exception as e:
        print(f"Warning: Could not load financial_phrasebank: {e}")
    
    # 5. Emotion Classification
    print("=== Processing Emotion Dataset ===")
    emotion = prepare_classification_dataset(
        "emotion",
        text_field='text',
        label_field='label',
        class_names=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    )
    conversations.extend(emotion)
    
    # 6. SMS Spam Detection
    print("=== Processing SMS Spam ===")
    sms_spam = prepare_classification_dataset(
        "sms_spam",
        text_field='sms',
        label_field='label',
        class_names=['ham', 'spam']
    )
    conversations.extend(sms_spam)
    
    # 7. Question Answering (SQuAD)
    print("=== Processing SQuAD ===")
    try:
        squad = load_dataset("squad")
        squad_conversations = []
        for item in tqdm(squad['train'], desc="Processing SQuAD"):
            question = item['question']
            context = item['context']
            answer = item['answers']['text'][0] if item['answers']['text'] else "No answer"
            
            conversation = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
            squad_conversations.append(conversation)
        
        conversations.extend(squad_conversations)
    except Exception as e:
        print(f"Warning: Could not load SQuAD: {e}")
    
    # 8. Common Sense Question Answering
    print("=== Processing CommonsenseQA ===")
    try:
        commonsense = load_dataset("commonsense_qa")
        cs_conversations = []
        for item in tqdm(commonsense['train'], desc="Processing CommonsenseQA"):
            question = item['question']
            choices = item['choices']
            answer_key = item['answerKey']
            
            # Format choices
            choice_text = ""
            for i, choice in enumerate(choices['text']):
                choice_text += f"{choices['label'][i]}: {choice}\n"
            
            conversation = f"Question: {question}\n\nChoices:\n{choice_text}\nAnswer: {answer_key}"
            cs_conversations.append(conversation)
        
        conversations.extend(cs_conversations)
    except Exception as e:
        print(f"Warning: Could not load CommonsenseQA: {e}")
    
    # 9. Natural Language Inference
    print("=== Processing SNLI ===")
    try:
        snli = load_dataset("snli")
        snli_conversations = []
        for item in tqdm(snli['train'], desc="Processing SNLI"):
            if item['label'] != -1:  # Skip unlabeled examples
                premise = item['premise']
                hypothesis = item['hypothesis']
                label = ['entailment', 'neutral', 'contradiction'][item['label']]
                
                conversation = f"Premise: {premise}\n\nHypothesis: {hypothesis}\n\nRelation: {label}"
                snli_conversations.append(conversation)
        
        conversations.extend(snli_conversations)
    except Exception as e:
        print(f"Warning: Could not load SNLI: {e}")
    
    # 10. Summarization
    print("=== Processing CNN/DailyMail ===")
    try:
        cnn_dm = load_dataset("cnn_dailymail", "3.0.0")
        sum_conversations = []
        for i, item in enumerate(tqdm(cnn_dm['train'], desc="Processing CNN/DailyMail")):
            if i >= 50000:  # Limit to first 50k for size
                break
            article = item['article']
            summary = item['highlights']
            
            conversation = f"Article: {article}\n\nSummary: {summary}"
            sum_conversations.append(conversation)
        
        conversations.extend(sum_conversations)
    except Exception as e:
        print(f"Warning: Could not load CNN/DailyMail: {e}")
    
    print(f"\nTotal conversations collected: {len(conversations)}")
    
    # Shuffle the conversations
    np.random.shuffle(conversations)
    
    # Split into train/val
    split_idx = int(0.8 * len(conversations))
    train_conversations = conversations[:split_idx]
    val_conversations = conversations[split_idx:]
    
    print(f"Training conversations: {len(train_conversations)}")
    print(f"Validation conversations: {len(val_conversations)}")
    
    return train_conversations, val_conversations

def save_dataset(conversations, filename):
    """Save conversations to a text file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for conversation in conversations:
            f.write(conversation + '\n\n')  # Double newline to separate conversations

def tokenize_and_save(conversations, filename):
    """Tokenize conversations and save as binary file"""
    enc = get_tokenizer()
    
    all_tokens = []
    for conversation in tqdm(conversations, desc=f"Tokenizing {filename}"):
        tokens = enc.encode(conversation)
        all_tokens.extend(tokens)
        all_tokens.append(enc.encode('\n')[0])  # Add newline token between conversations
    
    # Convert to numpy array
    tokens_array = np.array(all_tokens, dtype=np.uint16)
    
    # Save as binary file
    tokens_array.tofile(filename)
    
    print(f"Saved {len(all_tokens)} tokens to {filename}")
    return len(all_tokens)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/sft_dataset_xl', help='Directory to save the dataset')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Prepare the dataset
    train_conversations, val_conversations = prepare_extra_large_dataset()
    
    # Save text files
    print("Saving text files...")
    save_dataset(train_conversations, os.path.join(args.data_dir, 'train.txt'))
    save_dataset(val_conversations, os.path.join(args.data_dir, 'val.txt'))
    
    # Tokenize and save binary files
    print("Tokenizing and saving binary files...")
    train_tokens = tokenize_and_save(train_conversations, os.path.join(args.data_dir, 'train.bin'))
    val_tokens = tokenize_and_save(val_conversations, os.path.join(args.data_dir, 'val.bin'))
    
    # Create metadata
    enc = get_tokenizer()
    meta = {
        'vocab_size': enc.n_vocab,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'train_conversations': len(train_conversations),
        'val_conversations': len(val_conversations)
    }
    
    with open(os.path.join(args.data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Dataset preparation complete!")
    print(f"Train: {len(train_conversations)} conversations, {train_tokens} tokens")
    print(f"Val: {len(val_conversations)} conversations, {val_tokens} tokens")
    print(f"Vocab size: {enc.n_vocab}")
