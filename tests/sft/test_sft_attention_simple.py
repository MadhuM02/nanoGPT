#!/usr/bin/env python3
"""
Simple test for SFT attention implementation
"""
import torch
from model import GPTConfig, GPT

def test_sft_attention():
    """Test SFT attention with a small model"""
    print("Testing SFT Attention Implementation...")
    
    # Create a small test configuration
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=4,  # Use 4 heads (divisible by 3 for specialization)
        n_embd=128,  # 128 is divisible by 4
        dropout=0.0,
        bias=True,
        # SFT-specific settings
        sft_mode=True,
        instruction_attention_weight=1.5,
        response_attention_dampening=0.8,
        sft_head_specialization=True
    )
    
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}")
    print(f"Head dimension: {config.n_embd // config.n_head}")
    print(f"SFT mode: {config.sft_mode}")
    
    # Create model
    model = GPT(config)
    model.eval()
    
    # Create test input
    batch_size = 2
    seq_len = 32
    instruction_end = 20  # First 20 tokens are instruction, last 12 are response
    
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    instruction_positions = [instruction_end, instruction_end]  # Same for both samples
    
    print(f"Input shape: {x.shape}")
    print(f"Instruction positions: {instruction_positions}")
    
    # Test forward pass without SFT attention
    print("\n1. Testing standard attention...")
    model.transformer.h[0].attn.sft_mode = False
    try:
        with torch.no_grad():
            logits_standard = model(x)
        print(f"✅ Standard attention output shape: {logits_standard.shape}")
    except Exception as e:
        print(f"❌ Standard attention failed: {e}")
        return False
    
    # Test forward pass with SFT attention
    print("\n2. Testing SFT attention...")
    model.transformer.h[0].attn.sft_mode = True
    try:
        with torch.no_grad():
            logits_sft = model(x, instruction_positions=instruction_positions)
        print(f"✅ SFT attention output shape: {logits_sft.shape}")
    except Exception as e:
        print(f"❌ SFT attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare outputs
    diff = torch.abs(logits_standard - logits_sft).mean()
    print(f"\n3. Comparing outputs...")
    print(f"Mean absolute difference: {diff:.6f}")
    
    if diff > 0:
        print("✅ SFT attention produces different outputs (expected)")
    else:
        print("⚠️  SFT and standard attention produce identical outputs")
    
    print("\n✅ SFT attention test completed successfully!")
    return True

if __name__ == "__main__":
    test_sft_attention()
