"""
Simple import test for SFT training script to verify refactoring.
"""

import os
import sys

def test_sft_imports():
    """Test that SFT training script imports work correctly."""
    print("🧪 Testing SFT training script imports...")
    
    # Add the proper path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    try:
        # Test model import
        from model import GPTConfig, GPT
        print("✓ Model imports work")
        
        # Test checkpoint utils import  
        from training.utils.checkpoint_utils import (
            load_checkpoint_and_create_model,
            save_checkpoint,
            get_model_info
        )
        print("✓ Checkpoint utils imports work")
        
        # Test configurator import path
        configurator_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'training', 'utils', 'configurator.py'
        )
        configurator_exists = os.path.exists(configurator_path)
        print(f"✓ Configurator path exists: {configurator_exists}")
        
        # Test basic model creation
        model_args = {
            'n_layer': 2,
            'n_head': 2, 
            'n_embd': 64,
            'block_size': 128,
            'bias': False,
            'vocab_size': 1000,
            'dropout': 0.0,
            'num_experts': 4,
            'top_k_experts': 2,
            'load_balancing_loss_coef': 0.01,
            'expert_dropout': 0.0,
            'capacity_factor': 1.0,
            'expert_activation': 'gelu'
        }
        
        model, updated_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(
            init_from='scratch',
            resume_from='',
            model_args=model_args,
            meta_vocab_size=1000,
            device='cpu',
            master_process=True,
            block_size=128
        )
        
        print(f"✓ Model created successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test model info
        info = get_model_info(model)
        print(f"✓ Model info: {info['total_parameters']:,} params, {info['model_size_mb']:.1f}MB")
        
        print("✅ All SFT import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_sft_imports()
    sys.exit(0 if success else 1)
