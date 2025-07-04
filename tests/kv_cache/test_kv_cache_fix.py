#!/usr/bin/env python3
"""
Test script to verify the KV cache fix
"""
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_kv_cache_fix():
    """Test that the KV cache fix works"""
    print("Testing KV cache fix...")
    
    try:
        # Import required modules
        from model import GPTConfig, GPT
        from kv_cache import add_kv_cache_support
        
        print("✓ Imports successful")
        
        # Create a small test model
        config = GPTConfig(
            block_size=64,
            vocab_size=100,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
            bias=True
        )
        
        print("✓ GPTConfig created")
        print(f"  Config type: {type(config)}")
        print(f"  Has vocab_size: {hasattr(config, 'vocab_size')}")
        print(f"  vocab_size value: {config.vocab_size}")
        
        # Create model
        model = GPT(config)
        print("✓ GPT model created")
        
        # Add KV cache support
        model = add_kv_cache_support(model)
        print("✓ KV cache support added")
        
        # Test forward_with_cache
        x = torch.randint(0, 100, (1, 10))
        logits, past_key_values = model.forward_with_cache(x, use_cache=True)
        print("✓ forward_with_cache successful")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Past key values type: {type(past_key_values)}")
        
        # Test incremental generation
        next_token = torch.randint(0, 100, (1, 1))
        logits2, past_key_values2 = model.forward_with_cache(
            next_token, past_key_values, use_cache=True, start_pos=10
        )
        print("✓ Incremental generation successful")
        print(f"  Next logits shape: {logits2.shape}")
        
        print("\n🎉 All tests passed! KV cache fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_kv_cache_fix()
    sys.exit(0 if success else 1)
