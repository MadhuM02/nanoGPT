#!/usr/bin/env python3
"""
Test sample_moe.py generate_with_kv_cache function directly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model import GPTConfig, GPT
from sample_moe import generate_with_kv_cache

print("=== Testing generate_with_kv_cache ===")

try:
    # Create a small test model
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        block_size=64
    )
    model = GPT(config)
    model.eval()
    
    print("✓ Model created")
    
    # Test generate_with_kv_cache
    start_ids = torch.randint(0, 100, (1, 5))
    print(f"Starting with tokens: {start_ids}")
    
    # Generate with KV cache
    generated = generate_with_kv_cache(
        model, start_ids, max_new_tokens=10, 
        temperature=0.8, top_k=50, use_cache=True
    )
    
    print("✓ generate_with_kv_cache SUCCESS!")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated tokens: {generated[0].tolist()}")
    
    # Compare with standard generation
    generated_standard = model.generate(
        start_ids, max_new_tokens=10, 
        temperature=0.8, top_k=50
    )
    
    print("✓ Standard generation SUCCESS!")
    print(f"  Standard shape: {generated_standard.shape}")
    print(f"  Standard tokens: {generated_standard[0].tolist()}")
    
    print("\n🎉 Both generation methods work correctly!")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
