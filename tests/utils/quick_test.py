#!/usr/bin/env python3
"""
Simple test to verify the gradient checkpointing fix
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing gradient checkpointing fix...")

# Test 1: Import the model
try:
    from model import GPT, GPTConfig
    print("✅ Model imports successful")
except Exception as e:
    print(f"❌ Model import failed: {e}")
    exit(1)

# Test 2: Create a minimal model
try:
    config = GPTConfig(
        n_layer=2,
        n_head=2, 
        n_embd=64,
        block_size=32,
        vocab_size=100,
        num_experts=2,
        top_k_experts=1
    )
    model = GPT(config)
    print("✅ Model creation successful")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    exit(1)

# Test 3: Test gradient checkpointing methods
try:
    model.gradient_checkpointing_enable()
    print("✅ gradient_checkpointing_enable() works")
except Exception as e:
    print(f"❌ gradient_checkpointing_enable() failed: {e}")
    exit(1)

try:
    model.gradient_checkpointing_disable()
    print("✅ gradient_checkpointing_disable() works")
except Exception as e:
    print(f"❌ gradient_checkpointing_disable() failed: {e}")
    exit(1)

print("\n🎉 SUCCESS! The AttributeError has been fixed.")
print("You can now run training with gradient checkpointing:")
print("  python train_moe_advanced.py --config minimal --monitor")
