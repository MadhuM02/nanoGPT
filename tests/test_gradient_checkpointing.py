#!/usr/bin/env python3
"""
Test gradient checkpointing functionality
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model import GPT, GPTConfig

def test_gradient_checkpointing():
    """Test that gradient checkpointing works correctly"""
    print("Testing gradient checkpointing...")
    
    # Create a small model for testing
    config = GPTConfig(
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=64,
        dropout=0.0,
        bias=False,
        vocab_size=1000,
        num_experts=2,
        top_k_experts=1,
    )
    
    model = GPT(config)
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test enabling gradient checkpointing
    print("\nTesting gradient_checkpointing_enable()...")
    try:
        model.gradient_checkpointing_enable()
        print("✅ gradient_checkpointing_enable() works")
    except Exception as e:
        print(f"❌ gradient_checkpointing_enable() failed: {e}")
        return False
    
    # Test disabling gradient checkpointing  
    print("\nTesting gradient_checkpointing_disable()...")
    try:
        model.gradient_checkpointing_disable()
        print("✅ gradient_checkpointing_disable() works")
    except Exception as e:
        print(f"❌ gradient_checkpointing_disable() failed: {e}")
        return False
    
    # Test forward pass with gradient checkpointing enabled
    print("\nTesting forward pass with gradient checkpointing...")
    model.gradient_checkpointing_enable()
    model.train()  # Make sure we're in training mode
    
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    try:
        # Test data
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Forward pass
        logits, loss = model(x, y)
        
        # Backward pass
        loss.backward()
        
        print(f"✅ Forward/backward pass with gradient checkpointing successful")
        print(f"   Loss: {loss.item():.4f}")
        print(f"   Logits shape: {logits.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Gradient Checkpointing Test ===\n")
    
    try:
        success = test_gradient_checkpointing()
        if success:
            print("\n🎉 All gradient checkpointing tests passed!")
            print("The AttributeError should now be fixed.")
        else:
            print("\n❌ Gradient checkpointing tests failed.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
