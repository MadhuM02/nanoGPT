#!/usr/bin/env python3
"""
Test script for the MoE implementation with new parameters
"""

import torch
from model import GPTConfig, GPT

def test_moe():
    # Create a small config for testing with new parameters
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        num_experts=4,
        top_k_experts=2,
        dropout=0.1,
        bias=False,
        expert_dropout=0.1,
        capacity_factor=1.25,
        expert_activation='swish',
        load_balancing_loss_coef=0.01
    )
    
    print(f"Config created successfully:")
    print(f"  num_experts: {config.num_experts}")
    print(f"  top_k_experts: {config.top_k_experts}")
    print(f"  expert_dropout: {config.expert_dropout}")
    print(f"  capacity_factor: {config.capacity_factor}")
    print(f"  expert_activation: {config.expert_activation}")
    
    # Create model
    model = GPT(config)
    model.eval()
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test input
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass without targets
    with torch.no_grad():
        logits, loss = model(idx)
    
    print(f"Input shape: {idx.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Test with targets for training mode
    model.train()
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits, loss = model(idx, targets)
    
    print(f"Training loss: {loss.item():.4f}")
    
    # Test generation
    print("Testing generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(idx[:1, :10], max_new_tokens=5)
        print(f"Generation successful! Generated shape: {generated.shape}")
    
    print("✅ MoE test passed!")

def test_different_activations():
    """Test different expert activation functions"""
    activations = ['gelu', 'swish', 'relu']
    
    for activation in activations:
        print(f"\nTesting {activation} activation...")
        config = GPTConfig(
            n_layer=1,
            n_head=2,
            n_embd=64,
            block_size=32,
            vocab_size=100,
            num_experts=2,
            expert_activation=activation
        )
        
        model = GPT(config)
        x = torch.randint(0, 100, (1, 16))
        
        with torch.no_grad():
            logits, _ = model(x)
            print(f"  {activation} activation works! Output shape: {logits.shape}")

if __name__ == "__main__":
    print("Testing MoE configuration...")
    
    try:
        test_moe()
        print("\n✅ Basic MoE test passed!")
        
        test_different_activations()
        print("\n✅ All activation tests passed!")
        
        print("\n🎉 All tests passed! The MoE configuration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
