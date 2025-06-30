#!/usr/bin/env python3
"""
Quick test of memory-optimized configurations
"""
import torch
from model import GPT, GPTConfig
from v100_config import get_minimal_config, get_debug_config

def test_minimal_config():
    print("Testing minimal configuration...")
    config = get_minimal_config()
    
    print("Minimal config parameters:")
    key_params = ['n_layer', 'n_embd', 'num_experts', 'batch_size', 'block_size']
    for k in key_params:
        print(f"  {k}: {config[k]}")
    
    # Create model
    model_config = GPTConfig(
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size'],
        dropout=config['dropout'],
        bias=config['bias'],
        vocab_size=50304,
        num_experts=config['num_experts'],
        top_k_experts=config['top_k_experts'],
        load_balancing_loss_coef=config['load_balancing_loss_coef'],
        expert_dropout=config['expert_dropout'],
        capacity_factor=config['capacity_factor'],
        expert_activation=config['expert_activation']
    )
    
    model = GPT(model_config)
    print(f"Model created with {model.get_num_params() / 1e6:.1f}M parameters")
    
    if torch.cuda.is_available():
        model = model.cuda()
        
        # Test training step
        batch_size = config['batch_size']
        seq_len = config['block_size']
        x = torch.randint(0, 50304, (batch_size, seq_len), device='cuda')
        y = torch.randint(0, 50304, (batch_size, seq_len), device='cuda')
        
        # Forward and backward
        logits, loss = model(x, y)
        loss.backward()
        
        memory_used = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory used: {memory_used:.2f} GB")
        print("✅ Minimal config test PASSED")
        
        return True
    else:
        print("No CUDA available, but model creation successful")
        return True

if __name__ == "__main__":
    try:
        test_minimal_config()
        print("\n🎉 Ready for training with minimal config!")
        print("Run: python train_moe_advanced.py --config minimal --monitor")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Try debug config instead:")
        print("Run: python train_moe_advanced.py --config debug --monitor")
