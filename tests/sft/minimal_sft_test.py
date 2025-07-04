#!/usr/bin/env python3
"""
Minimal test for SFT attention
"""

try:
    print("Importing modules...")
    import torch
    from model import GPTConfig, GPT
    
    print("Creating config...")
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        sft_mode=True,
        instruction_attention_weight=1.5,
        response_attention_dampening=0.8
    )
    
    print("Creating model...")
    model = GPT(config)
    
    print("Testing forward pass...")
    x = torch.randint(0, 1000, (2, 64))  # batch_size=2, seq_len=64
    instruction_positions = [32, 40]  # Mock instruction end positions
    
    with torch.no_grad():
        logits, loss = model(x, instruction_positions=instruction_positions)
    
    print(f"✅ Success! Logits shape: {logits.shape}")
    print(f"✅ Loss: {loss}")
    print(f"✅ SFT attention is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
