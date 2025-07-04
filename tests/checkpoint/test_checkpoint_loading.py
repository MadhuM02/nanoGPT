#!/usr/bin/env python3
"""
Test checkpoint loading functionality
"""

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from model import GPTConfig, GPT
from train_moe_advanced import load_checkpoint, save_checkpoint

def create_test_checkpoint_with_prefixes():
    """Create a test checkpoint with problematic prefixes"""
    print("Creating test checkpoint with prefixes...")
    
    # Create a small MoE model
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        num_experts=4,
        top_k_experts=2,
        load_balancing_loss_coef=0.01,
        expert_dropout=0.0,
        capacity_factor=1.0,
        expert_activation='gelu'
    )
    
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get clean state dict
    clean_state_dict = model.state_dict()
    
    # Create problematic state dict with module._orig_mod. prefix
    problematic_state_dict = {}
    for key, value in clean_state_dict.items():
        problematic_key = f"module._orig_mod.{key}"
        problematic_state_dict[problematic_key] = value
    
    # Save checkpoint with problematic prefixes
    problematic_checkpoint = {
        'model': problematic_state_dict,
        'optimizer': optimizer.state_dict(),
        'config': {
            'block_size': 128,
            'vocab_size': 1000,
            'n_layer': 2,
            'n_head': 4,
            'n_embd': 128,
            'dropout': 0.0,
            'bias': False,
            'num_experts': 4,
            'top_k_experts': 2,
            'load_balancing_loss_coef': 0.01,
            'expert_dropout': 0.0,
            'capacity_factor': 1.0,
            'expert_activation': 'gelu',
            'out_dir': 'test_checkpoint'
        },
        'step': 1000,
        'best_val_loss': 2.5
    }
    
    os.makedirs('test_checkpoint', exist_ok=True)
    checkpoint_path = 'test_checkpoint/problematic_checkpoint.pt'
    torch.save(problematic_checkpoint, checkpoint_path)
    
    print(f"Created checkpoint: {checkpoint_path}")
    print(f"Sample problematic keys: {list(problematic_state_dict.keys())[:3]}")
    
    return checkpoint_path, model, optimizer

def test_checkpoint_loading():
    """Test the enhanced checkpoint loading"""
    print("Testing Enhanced Checkpoint Loading")
    print("=" * 50)
    
    # Create test checkpoint with prefixes
    checkpoint_path, original_model, original_optimizer = create_test_checkpoint_with_prefixes()
    
    # Create a new model (same architecture)
    config = GPTConfig(
        block_size=128,
        vocab_size=1000,
        n_layer=2,
        n_head=4,
        n_embd=128,
        dropout=0.0,
        bias=False,
        num_experts=4,
        top_k_experts=2,
        load_balancing_loss_coef=0.01,
        expert_dropout=0.0,
        capacity_factor=1.0,
        expert_activation='gelu'
    )
    
    new_model = GPT(config)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    
    print(f"\nOriginal model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"New model parameters: {sum(p.numel() for p in new_model.parameters()):,}")
    
    # Test loading
    try:
        print(f"\n🧪 Testing checkpoint loading from {checkpoint_path}")
        loaded_config, loaded_step, loaded_best_val_loss = load_checkpoint(
            checkpoint_path, new_model, new_optimizer, device='cpu'
        )
        
        print(f"✅ Successfully loaded checkpoint!")
        print(f"  Loaded step: {loaded_step}")
        print(f"  Loaded best_val_loss: {loaded_best_val_loss}")
        print(f"  Config keys: {list(loaded_config.keys()) if loaded_config else 'None'}")
        
        # Verify parameters match
        original_params = {name: param.clone() for name, param in original_model.named_parameters()}
        loaded_params = {name: param.clone() for name, param in new_model.named_parameters()}
        
        params_match = True
        for name in original_params:
            if not torch.allclose(original_params[name], loaded_params[name], atol=1e-6):
                print(f"⚠️ Parameter mismatch: {name}")
                params_match = False
        
        if params_match:
            print("✅ All parameters match perfectly!")
        else:
            print("⚠️ Some parameters don't match")
            
    except Exception as e:
        print(f"❌ Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test saving with new function
    try:
        print(f"\n🧪 Testing checkpoint saving...")
        save_checkpoint_path = 'test_checkpoint/test_save.pt'
        save_checkpoint(new_model, new_optimizer, loaded_config, loaded_step + 100, 
                       loaded_best_val_loss - 0.1, save_checkpoint_path, is_best=True)
        
        print("✅ Checkpoint saving successful!")
        
    except Exception as e:
        print(f"❌ Checkpoint saving failed: {e}")
        return False
    
    # Cleanup
    try:
        import shutil
        shutil.rmtree('test_checkpoint')
        print("\n🧹 Cleaned up test files")
    except:
        pass
    
    return True

if __name__ == "__main__":
    success = test_checkpoint_loading()
    if success:
        print("\n🎉 All checkpoint loading tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
