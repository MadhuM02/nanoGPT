#!/usr/bin/env python3
"""
Test to verify SFT attention behaves differently from standard attention
"""

import torch
from model import GPTConfig, GPT

def test_attention_differences():
    """Test that SFT attention produces different outputs than standard attention"""
    
    print("🧪 Testing SFT vs Standard Attention Differences")
    print("=" * 60)
    
    # Create two identical models, one with SFT attention, one without
    base_config = {
        'block_size': 128,
        'vocab_size': 1000,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'dropout': 0.0  # Disable dropout for reproducible results
    }
    
    # Standard model
    standard_config = GPTConfig(**base_config, sft_mode=False)
    standard_model = GPT(standard_config)
    
    # SFT model
    sft_config = GPTConfig(**base_config, 
                          sft_mode=True,
                          instruction_attention_weight=1.5,
                          response_attention_dampening=0.8)
    sft_model = GPT(sft_config)
    
    # Copy weights to make models identical except for attention logic
    sft_model.load_state_dict(standard_model.state_dict(), strict=False)
    
    # Test input
    batch_size, seq_len = 2, 64
    x = torch.randint(0, 1000, (batch_size, seq_len))
    instruction_positions = [32, 40]  # Instruction ends at position 32 and 40
    
    print(f"Input shape: {x.shape}")
    print(f"Instruction positions: {instruction_positions}")
    
    # Forward pass through both models
    with torch.no_grad():
        # Standard model (no instruction positions)
        standard_logits, _ = standard_model(x)
        
        # SFT model (with instruction positions)
        sft_logits, _ = sft_model(x, instruction_positions=instruction_positions)
    
    print(f"Standard logits shape: {standard_logits.shape}")
    print(f"SFT logits shape: {sft_logits.shape}")
    
    # Compare outputs
    logits_diff = torch.abs(standard_logits - sft_logits).mean().item()
    max_diff = torch.abs(standard_logits - sft_logits).max().item()
    
    print(f"\n📊 Results:")
    print(f"  Mean absolute difference: {logits_diff:.6f}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    
    if logits_diff > 1e-6:
        print(f"✅ SUCCESS: SFT attention produces different outputs!")
        print(f"   The attention mechanism is working as expected.")
    else:
        print(f"⚠️  WARNING: No significant difference detected.")
        print(f"   SFT attention may not be properly implemented.")
    
    # Test without instruction positions for SFT model (should be similar to standard)
    with torch.no_grad():
        sft_no_positions_logits, _ = sft_model(x, instruction_positions=None)
    
    no_pos_diff = torch.abs(standard_logits - sft_no_positions_logits).mean().item()
    print(f"\n🔍 Control Test (SFT without instruction positions):")
    print(f"  Difference from standard: {no_pos_diff:.6f}")
    
    if no_pos_diff < 1e-6:
        print(f"✅ GOOD: SFT model behaves like standard when no positions provided")
    else:
        print(f"⚠️  Unexpected difference when no instruction positions provided")
    
    return logits_diff > 1e-6

def test_attention_bias_application():
    """Test that attention bias is correctly applied"""
    
    print(f"\n🎯 Testing Attention Bias Application")
    print("=" * 60)
    
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=1,
        n_head=2,
        n_embd=64,
        sft_mode=True,
        instruction_attention_weight=2.0,  # Strong bias for testing
        response_attention_dampening=0.5
    )
    
    model = GPT(config)
    
    # Small test case
    batch_size, seq_len = 1, 16
    x = torch.randint(0, 100, (batch_size, seq_len))
    instruction_positions = [8]  # Instruction ends at position 8
    
    print(f"Sequence length: {seq_len}")
    print(f"Instruction end: {instruction_positions[0]}")
    print(f"Instruction weight: {config.instruction_attention_weight}")
    print(f"Response dampening: {config.response_attention_dampening}")
    
    with torch.no_grad():
        # Test with SFT bias
        sft_logits, _ = model(x, instruction_positions=instruction_positions)
        
        # Test without SFT bias (set sft_mode to False temporarily)
        model.transformer.h[0].attn.sft_mode = False
        standard_logits, _ = model(x, instruction_positions=instruction_positions)
        model.transformer.h[0].attn.sft_mode = True  # Reset
    
    bias_effect = torch.abs(sft_logits - standard_logits).mean().item()
    print(f"\nAttention bias effect: {bias_effect:.6f}")
    
    if bias_effect > 1e-5:
        print(f"✅ SUCCESS: Attention bias is being applied!")
    else:
        print(f"❌ ISSUE: Attention bias may not be working properly.")
    
    return bias_effect > 1e-5

if __name__ == "__main__":
    try:
        # Test 1: Basic attention differences
        test1_passed = test_attention_differences()
        
        # Test 2: Attention bias application
        test2_passed = test_attention_bias_application()
        
        print(f"\n🏁 Final Results:")
        print(f"=" * 60)
        print(f"✅ SFT Attention Differences: {'PASS' if test1_passed else 'FAIL'}")
        print(f"✅ Attention Bias Application: {'PASS' if test2_passed else 'FAIL'}")
        
        if test1_passed and test2_passed:
            print(f"\n🎉 All tests passed! SFT attention is working correctly.")
        else:
            print(f"\n⚠️  Some tests failed. Check the implementation.")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
