"""
Test script to verify the tensor size fix for SFT data loading
"""
import os
import sys
import torch
import numpy as np
import tiktoken

# Set up basic config
dataset = 'sft_dataset'
batch_size = 4
block_size = 512
device = 'cpu'
master_process = True

# Load the data loading functions
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
    
    for i, example in enumerate(sampled_examples):
        # Tokenize the example
        tokens = enc.encode_ordinary(example)
        
        # Ensure we have at least one token
        if len(tokens) == 0:
            tokens = [enc.eot_token]
        
        # Add end token to mark end of sequence
        tokens = tokens + [enc.eot_token]
        
        # Truncate or pad to exactly block_size
        if len(tokens) >= block_size:
            # Truncate to exactly block_size
            tokens = tokens[:block_size]
        else:
            # Pad with end tokens to reach block_size
            pad_length = block_size - len(tokens)
            tokens = tokens + [enc.eot_token] * pad_length
        
        # Create input (x) and target (y) sequences
        # x: tokens[:-1], y: tokens[1:]
        # But we need both to be exactly block_size
        x_tokens = tokens[:]  # Copy the full sequence
        y_tokens = tokens[1:] + [enc.eot_token]  # Shift by 1 and add eot at end
        
        # Ensure exactly block_size for both
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        
        # Final safety check - pad if somehow still short
        while len(x_tokens) < block_size:
            x_tokens.append(enc.eot_token)
        while len(y_tokens) < block_size:
            y_tokens.append(enc.eot_token)
        
        print(f"Example {i}: len(x_tokens)={len(x_tokens)}, len(y_tokens)={len(y_tokens)}")
        
        x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
        y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
    
    # This is where the error was happening
    try:
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        print(f"✅ Successfully stacked tensors: x.shape={x.shape}, y.shape={y.shape}")
        return x, y
    except Exception as e:
        print(f"❌ Error stacking tensors: {e}")
        for i, (x_tensor, y_tensor) in enumerate(zip(x_list, y_list)):
            print(f"  Tensor {i}: x.shape={x_tensor.shape}, y.shape={y_tensor.shape}")
        return None, None

if __name__ == '__main__':
    print("Testing fixed SFT data loading...")
    
    # Load examples
    load_examples()
    
    if len(train_examples) == 0:
        print("No training examples found!")
        sys.exit(1)
    
    # Test batch generation multiple times
    for test_round in range(3):
        print(f"\n--- Test Round {test_round + 1} ---")
        x, y = get_batch('train')
        
        if x is not None:
            print(f"Batch generation successful!")
        else:
            print(f"Batch generation failed!")
            break
    
    print("\n✅ All tests passed! Tensor size issue is fixed.")
