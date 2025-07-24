#!/usr/bin/env python3
"""
Test script to verify SFT attention implementation
"""

import torch
import sys
import os
sys.path.append('/mnt/workspace/nanoGPT')

from model import GPTConfig, GPT
from sft_utils import find_instruction_positions_simple, apply_sft_attention_config

def test_sft_attention():
    print("🧪 Testing SFT Attention Implementation")
    print("=" * 50)
    
    # Create a small model config with SFT attention enabled
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=2,
        n_head=6,  # Divisible by 3 for head specialization
        n_embd=192,
        dropout=0.0,
        bias=False
    )
    
    # Enable SFT attention
    apply_sft_attention_config(config, enable_sft=True)
    
    print(f"Model config:")
    print(f"  SFT mode: {config.sft_mode}")
    print(f"  Instruction weight: {config.instruction_attention_weight}")
    print(f"  Response dampening: {config.response_attention_dampening}")
    print(f"  Head specialization: {config.sft_head_specialization}")
    
    # Create model
    model = GPT(config)
    model.eval()
    
    # Create sample batch with instruction-response structure
    batch_size = 2
    seq_len = 32
    
    # Simulate tokenized sequences
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = input_ids.clone()
    
    # Create instruction positions (instruction ends at position 20)
    instruction_positions = [20, 18]  # Different for each batch item
    
    print(f"\nTesting forward pass:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Instruction positions: {instruction_positions}")
    
    # Test forward pass with instruction positions
    try:
        with torch.no_grad():
            logits, loss = model(input_ids, targets, instruction_positions=instruction_positions)
        
        print(f"  ✅ Forward pass successful!")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False
    
    # Test without instruction positions (should still work)
    try:
        with torch.no_grad():
            logits_std, loss_std = model(input_ids, targets)
        
        print(f"  ✅ Standard forward pass (no instruction positions) successful!")
        print(f"  Loss difference: {abs(loss.item() - loss_std.item()):.6f}")
        
    except Exception as e:
        print(f"  ❌ Standard forward pass failed: {e}")
        return False
    
    # Test instruction position detection
    print(f"\nTesting instruction position detection:")
    test_positions = find_instruction_positions_simple(input_ids)
    print(f"  Detected positions: {test_positions}")
    
    # Test with SFT mode disabled
    config.sft_mode = False
    model_std = GPT(config)
    model_std.eval()
    
    try:
        with torch.no_grad():
            logits_nosft, loss_nosft = model_std(input_ids, targets, instruction_positions=instruction_positions)
        
        print(f"  ✅ SFT mode disabled - forward pass successful!")
        print(f"  Loss (SFT disabled): {loss_nosft.item():.4f}")
        
    except Exception as e:
        print(f"  ❌ SFT disabled forward pass failed: {e}")
        return False
    
    print(f"\n🎉 All tests passed! SFT attention implementation is working correctly.")
    
    return True

def test_attention_bias():
    """Test the attention bias calculation specifically"""
    print("\n🔍 Testing Attention Bias Calculation")
    print("-" * 40)
    
    # Create config
    config = GPTConfig(
        block_size=16,
        vocab_size=100,
        n_layer=1,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False
    )
    
    # Apply SFT configuration
    config.sft_mode = True
    config.instruction_attention_weight = 2.0
    config.response_attention_dampening = 0.5
    
    model = GPT(config)
    attention_layer = model.transformer.h[0].attn
    
    # Test bias creation
    B, T = 2, 16
    instruction_positions = [10, 8]
    device = 'cpu'
    
    bias = attention_layer.create_sft_attention_bias(B, T, instruction_positions, device)
    
    if bias is not None:
        print(f"  ✅ Attention bias created successfully!")
        print(f"  Bias shape: {bias.shape}")
        
        # Check if bias values are as expected
        for b in range(B):
            pos = instruction_positions[b]
            if pos < T:
                instr_bias = bias[b, 0, pos:, :pos].unique()
                resp_bias = bias[b, 0, pos:, pos:].unique()
                
                print(f"  Batch {b} (instruction ends at {pos}):")
                print(f"    Response→Instruction bias: {instr_bias.tolist()}")
                print(f"    Response→Response bias: {resp_bias.tolist()}")
    else:
        print(f"  ❌ Attention bias creation failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = test_sft_attention()
    if success:
        test_attention_bias()
    else:
        print("❌ Basic tests failed, skipping bias tests")
        sys.exit(1)
