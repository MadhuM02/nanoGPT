#!/usr/bin/env python3
"""
Minimal test for KV cache fix
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== KV Cache Fix Test ===")

try:
    # Test dataclass import
    from dataclasses import dataclass
    print("✓ dataclass import OK")
    
    # Test model imports
    from model import GPTConfig, GPT
    print("✓ Model imports OK")
    
    # Test config creation
    config = GPTConfig(vocab_size=100, n_layer=2, n_head=2, n_embd=32)
    print("✓ Config creation OK")
    print(f"  Config type: {type(config)}")
    print(f"  Has vocab_size: {hasattr(config, 'vocab_size')}")
    print(f"  vocab_size = {config.vocab_size}")
    
    # Test model creation
    import torch
    model = GPT(config)
    print("✓ Model creation OK")
    
    # Test KV cache import
    from kv_cache import add_kv_cache_support
    print("✓ KV cache import OK")
    
    # Test adding KV cache support
    model = add_kv_cache_support(model)
    print("✓ KV cache support added OK")
    
    print("\nTesting forward_with_cache...")
    x = torch.randint(0, 100, (1, 5))
    logits, past_kv = model.forward_with_cache(x, use_cache=True)
    print("✓ forward_with_cache SUCCESS!")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Past KV type: {type(past_kv)}")
    
    print("\n🎉 ALL TESTS PASSED! KV cache fix is working correctly!")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
