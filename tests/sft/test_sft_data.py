"""
Test script to verify SFT data loading works correctly
"""
import os
import sys
sys.path.append('.')

import torch
import numpy as np
import tiktoken

# Set up basic config
dataset = 'sft_dataset'
batch_size = 2
block_size = 128
device = 'cpu'
master_process = True

# Load the data loading functions from train_sft.py
data_dir = os.path.join('data', dataset)

# Load examples
train_examples = []
val_examples = []

def load_examples():
    """Load all examples from text files"""
    global train_examples, val_examples
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Load training examples
    train_path = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_examples = [line.strip() for line in f if line.strip()]
    
    # Load validation examples  
    val_path = os.path.join(data_dir, 'val.txt')
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            val_examples = [line.strip() for line in f if line.strip()]
    
    print(f"📚 Loaded {len(train_examples)} training examples, {len(val_examples)} validation examples")

def get_batch(split):
    """Get a batch of complete examples for SFT training"""
    enc = tiktoken.get_encoding("gpt2")
    
    # Choose examples based on split
    examples = train_examples if split == 'train' else val_examples
    
    if len(examples) == 0:
        print("No examples found!")
        return None, None
    
    # Sample random examples
    sampled_examples = np.random.choice(examples, size=batch_size, replace=True)
    
    x_list = []
    y_list = []
    
    for example in sampled_examples:
        # Tokenize the example
        tokens = enc.encode_ordinary(example)
        
        # Truncate or pad to block_size
        if len(tokens) >= block_size:
            # Truncate to block_size
            tokens = tokens[:block_size]
            x_tokens = tokens[:-1] + [enc.eot_token]  # Add end token
            y_tokens = tokens[1:] + [enc.eot_token]   # Shifted for next token prediction
        else:
            # Pad with end tokens
            pad_length = block_size - len(tokens)
            x_tokens = tokens + [enc.eot_token] * pad_length
            y_tokens = tokens[1:] + [enc.eot_token] * (pad_length + 1)
            y_tokens = y_tokens[:block_size]  # Ensure exactly block_size
        
        # Ensure we have exactly block_size tokens
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        
        # Pad if needed (should not happen with above logic, but safety check)
        while len(x_tokens) < block_size:
            x_tokens.append(enc.eot_token)
        while len(y_tokens) < block_size:
            y_tokens.append(enc.eot_token)
            
        x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
        y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    return x, y

if __name__ == '__main__':
    print("Testing SFT data loading...")
    
    # Load examples
    load_examples()
    
    if len(train_examples) == 0:
        print("No training examples found!")
        sys.exit(1)
    
    print(f"First training example: {train_examples[0][:100]}...")
    
    # Test batch generation
    print("\nTesting batch generation...")
    x, y = get_batch('train')
    
    if x is not None:
        print(f"Batch shape: x={x.shape}, y={y.shape}")
        print(f"First sequence (x): {x[0][:20]}")
        print(f"First sequence (y): {y[0][:20]}")
        
        # Decode to verify
        enc = tiktoken.get_encoding("gpt2")
        decoded = enc.decode(x[0][:50].tolist())
        print(f"Decoded first 50 tokens: {decoded}")
        
        print("✅ SFT data loading test passed!")
    else:
        print("❌ Failed to generate batch")
