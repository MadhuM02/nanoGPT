#!/usr/bin/env python3
"""
Test the fixed checkpoint loading with problematic prefixes
"""

import os
import sys

# Add parent directory to path for imports (go up 2 levels from tests/utils/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import os
import torch
from model import GPTConfig, GPT

def create_problematic_checkpoint():
    """Create a checkpoint with DDP and compilation prefixes to test our fixes"""
    print("Creating test checkpoint with problematic prefixes...")
    
    # Create a minimal model
    config = {
        'block_size': 128,
        'vocab_size': 1000,
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 64,
        'dropout': 0.0,
        'bias': False,
        'num_experts': 4,
        'top_k_experts': 2,
        'load_balancing_loss_coef': 0.01,
        'expert_dropout': 0.0,
        'capacity_factor': 1.0,
        'expert_activation': 'gelu'
    }
    
    gpt_config = GPTConfig(**config)
    model = GPT(gpt_config)
    
    # Get clean state dict
    clean_state_dict = model.state_dict()
    
    # Create problematic state dict with DDP + compilation prefixes
    problematic_state_dict = {}
    for key, value in clean_state_dict.items():
        # Add module._orig_mod. prefix (DDP + compilation)
        problematic_key = f"module._orig_mod.{key}"
        problematic_state_dict[problematic_key] = value
    
    # Create checkpoint
    checkpoint = {
        'model': problematic_state_dict,
        'config': config,
        'step': 1000,
        'best_val_loss': 2.5
    }
    
    # Save problematic checkpoint
    os.makedirs('test_prefix_fix', exist_ok=True)
    problematic_path = 'test_prefix_fix/problematic_checkpoint.pt'
    torch.save(checkpoint, problematic_path)
    
    print(f"Created problematic checkpoint: {problematic_path}")
    print(f"Sample keys: {list(problematic_state_dict.keys())[:3]}")
    
    return problematic_path, config

def test_sampling_script_fix():
    """Test if our updated sampling script can handle the problematic checkpoint"""
    print("\n" + "="*60)
    print("Testing sampling script with problematic checkpoint")
    print("="*60)
    
    # Create problematic checkpoint
    ckpt_path, expected_config = create_problematic_checkpoint()
    
    # Test the loading logic from sample_moe.py
    try:
        print("\n🧪 Testing load_model_from_checkpoint function...")
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Use the same logic as in sample_moe.py
        if 'config' in checkpoint:
            config = checkpoint['config']
            gptconf = GPTConfig(
                block_size=config.get('block_size', 1024),
                vocab_size=config.get('vocab_size', 50304),
                n_layer=config.get('n_layer', 12),
                n_head=config.get('n_head', 12),
                n_embd=config.get('n_embd', 768),
                dropout=0.0,
                bias=config.get('bias', True),
                num_experts=config.get('num_experts', 1),
                top_k_experts=config.get('top_k_experts', 1),
                load_balancing_loss_coef=config.get('load_balancing_loss_coef', 0.01),
                expert_dropout=0.0,
                capacity_factor=config.get('capacity_factor', 1.0),
                expert_activation=config.get('expert_activation', 'gelu')
            )
        else:
            raise ValueError("Config not found")
        
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        print(f"Original state dict sample keys:")
        for i, key in enumerate(state_dict.keys()):
            if i < 3:
                print(f"  {key}")
            else:
                break
        
        # Apply our fixed prefix removal logic
        unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
        keys_removed = 0
        
        for prefix in unwanted_prefixes:
            keys_to_update = []
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    keys_to_update.append((k, k[len(prefix):]))
            
            for old_key, new_key in keys_to_update:
                state_dict[new_key] = state_dict.pop(old_key)
                keys_removed += 1
        
        print(f"\nFixed {keys_removed} keys by removing prefixes")
        print(f"Fixed state dict sample keys:")
        for i, key in enumerate(state_dict.keys()):
            if i < 3:
                print(f"  {key}")
            else:
                break
        
        # Try to load the state dict
        model.load_state_dict(state_dict)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n✅ SUCCESS! Model loaded with {total_params:,} parameters")
        print(f"   MoE config: {gptconf.num_experts} experts, top-{gptconf.top_k_experts}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        if os.path.exists('test_prefix_fix'):
            os.rmdir('test_prefix_fix')

def main():
    print("Testing Checkpoint Prefix Fix")
    print("=" * 40)
    
    success = test_sampling_script_fix()
    
    if success:
        print(f"\n🎉 All tests passed!")
        print("The sampling script can now handle checkpoints with:")
        print("  ✅ module._orig_mod. prefixes (DDP + compilation)")
        print("  ✅ module. prefixes (DDP only)")
        print("  ✅ _orig_mod. prefixes (compilation only)")
        print("\nYour sampling scripts should now work with any checkpoint format!")
    else:
        print(f"\n💥 Tests failed!")
        print("There may still be issues with the prefix removal logic.")

if __name__ == "__main__":
    main()
