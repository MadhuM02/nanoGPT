#!/usr/bin/env python3
"""
Test script to verify SFT loss masking is working correctly
"""

import os
import sys
import torch
import tiktoken
import numpy as np

# Add the nanoGPT directory to path
sys.path.append('/mnt/workspace/nanoGPT')

def test_sft_loss_masking():
    """Test that SFT loss masking correctly ignores instruction tokens"""
    
    print("🧪 Testing SFT Loss Masking")
    print("=" * 50)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Create test examples with clear instruction/response boundaries
    test_examples = [
        "Translate to French: Hello world -> Bonjour le monde",
        "Sentiment: This movie is amazing -> Positive", 
        "Category: The cat sat on the mat -> Animals",
        "Summary: The quick brown fox jumps over the lazy dog -> A fox jumps over a dog"
    ]
    
    print("📝 Test Examples:")
    for i, example in enumerate(test_examples):
        print(f"  {i+1}. {example}")
    
    # Simulate the get_batch logic for SFT
    block_size = 64
    batch_size = len(test_examples)
    
    x_list = []
    y_list = []
    instruction_masks = []
    
    response_markers = ["->", ":", "Answer:", "Response:", "Category:", "Sentiment:", "Summary:"]
    
    for example in test_examples:
        # Find instruction/response boundary
        instruction_end = -1
        for marker in response_markers:
            pos = example.find(marker)
            if pos != -1:
                instruction_end = pos + len(marker)
                break
        
        # Tokenize
        tokens = enc.encode_ordinary(example)
        tokens = tokens + [enc.eot_token]  # Add end token
        
        # Find token boundary for instruction/response split
        instruction_token_end = -1
        if instruction_end > 0:
            instruction_text = example[:instruction_end]
            instruction_tokens = enc.encode_ordinary(instruction_text)
            instruction_token_end = len(instruction_tokens)
        
        # Create instruction mask (1 for instruction, 0 for response)
        instruction_mask = [1] * len(tokens)
        if instruction_token_end > 0 and instruction_token_end < len(tokens):
            for i in range(instruction_token_end, len(tokens)):
                instruction_mask[i] = 0
        
        # Pad/truncate to block_size
        if len(tokens) >= block_size:
            tokens = tokens[:block_size]
            instruction_mask = instruction_mask[:block_size]
        else:
            pad_length = block_size - len(tokens)
            tokens = tokens + [enc.eot_token] * pad_length
            instruction_mask = instruction_mask + [0] * pad_length
        
        # Create x, y sequences
        x_tokens = tokens[:]
        y_tokens = tokens[1:] + [enc.eot_token]
        
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        instruction_mask = instruction_mask[:block_size]
        
        x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
        y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
        instruction_masks.append(torch.tensor(instruction_mask, dtype=torch.float32))
    
    # Stack into batches
    X = torch.stack(x_list)
    Y = torch.stack(y_list)
    instruction_mask_tensor = torch.stack(instruction_masks)
    
    # Create loss mask (SFT: compute loss only on response tokens)
    loss_mask = 1.0 - instruction_mask_tensor  # 0 for instruction, 1 for response
    Y_masked = Y.clone()
    Y_masked[loss_mask == 0] = -1  # Mask out instruction tokens
    
    print(f"\n📊 Batch Analysis:")
    print(f"  Batch shape: {X.shape}")
    print(f"  Block size: {block_size}")
    
    # Analyze each example
    for i in range(batch_size):
        example = test_examples[i]
        x_tokens = X[i].numpy()
        y_original = Y[i].numpy()
        y_masked = Y_masked[i].numpy()
        instr_mask = instruction_mask_tensor[i].numpy()
        loss_mask_sample = loss_mask[i].numpy()
        
        # Count tokens
        total_tokens = len(x_tokens)
        instruction_tokens = int(instr_mask.sum())
        response_tokens = int(loss_mask_sample.sum())
        masked_out = (y_masked == -1).sum()
        
        print(f"\n  Example {i+1}: {example}")
        print(f"    Original text: {repr(example)}")
        
        # Decode and show the split
        try:
            # Find where instruction ends
            instruction_end_idx = -1
            for j, mask_val in enumerate(instr_mask):
                if mask_val == 0:  # First response token
                    instruction_end_idx = j
                    break
            
            if instruction_end_idx > 0:
                instr_tokens = x_tokens[:instruction_end_idx]
                resp_tokens = x_tokens[instruction_end_idx:]
                
                # Remove padding tokens for decoding
                instr_tokens = [t for t in instr_tokens if t != enc.eot_token]
                resp_tokens = [t for t in resp_tokens if t != enc.eot_token]
                
                instr_text = enc.decode(instr_tokens) if instr_tokens else ""
                resp_text = enc.decode(resp_tokens) if resp_tokens else ""
                
                print(f"    Instruction part: {repr(instr_text)}")
                print(f"    Response part: {repr(resp_text)}")
            
        except Exception as e:
            print(f"    Decode error: {e}")
        
        print(f"    Token counts:")
        print(f"      Total: {total_tokens}")
        print(f"      Instruction: {instruction_tokens} (masked out from loss)")
        print(f"      Response: {response_tokens} (used for loss)")
        print(f"      Masked in Y: {masked_out}")
        print(f"    Loss computation: {'✅ Response only' if response_tokens > 0 else '❌ No response tokens'}")
        
        # Show which tokens are masked
        print(f"    Y targets (first 20): {y_original[:20].tolist()}")
        print(f"    Y masked (first 20): {y_masked[:20].tolist()}")
        print(f"    Instruction mask:     {instr_mask[:20].tolist()}")
        print(f"    Loss mask:            {loss_mask_sample[:20].tolist()}")
    
    # Test loss computation difference
    print(f"\n🔬 Loss Computation Test:")
    
    # Create dummy logits
    vocab_size = enc.n_vocab
    dummy_logits = torch.randn(batch_size, block_size, vocab_size)
    
    # Compute standard loss (all tokens)
    loss_standard = torch.nn.functional.cross_entropy(
        dummy_logits.view(-1, vocab_size), 
        Y.view(-1), 
        ignore_index=-1
    )
    
    # Compute SFT loss (response tokens only)
    loss_sft = torch.nn.functional.cross_entropy(
        dummy_logits.view(-1, vocab_size), 
        Y_masked.view(-1), 
        ignore_index=-1
    )
    
    print(f"  Standard loss (all tokens): {loss_standard.item():.4f}")
    print(f"  SFT loss (response only): {loss_sft.item():.4f}")
    print(f"  Difference: {abs(loss_standard.item() - loss_sft.item()):.4f}")
    
    # Count how many tokens are actually used for loss
    standard_count = (Y.view(-1) != -1).sum().item()
    sft_count = (Y_masked.view(-1) != -1).sum().item()
    
    print(f"  Tokens used in standard loss: {standard_count}")
    print(f"  Tokens used in SFT loss: {sft_count}")
    print(f"  Reduction: {(standard_count - sft_count) / standard_count * 100:.1f}%")
    
    # Verify masking is working
    if sft_count < standard_count:
        print("  ✅ SFT loss masking is working - fewer tokens used for loss")
    else:
        print("  ❌ SFT loss masking is NOT working - same number of tokens")
    
    print("\n" + "=" * 50)
    print("🎯 SFT Loss Masking Test Complete")
    
    return {
        'batch_shape': X.shape,
        'standard_loss': loss_standard.item(),
        'sft_loss': loss_sft.item(),
        'tokens_standard': standard_count,
        'tokens_sft': sft_count,
        'masking_working': sft_count < standard_count
    }

if __name__ == "__main__":
    result = test_sft_loss_masking()
    print(f"\n📋 Summary:")
    print(f"  Batch shape: {result['batch_shape']}")
    print(f"  Standard loss: {result['standard_loss']:.4f}")
    print(f"  SFT loss: {result['sft_loss']:.4f}")
    print(f"  Tokens (standard): {result['tokens_standard']}")
    print(f"  Tokens (SFT): {result['tokens_sft']}")
    print(f"  Masking working: {'✅ Yes' if result['masking_working'] else '❌ No'}")
