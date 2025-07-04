"""
Test for checkpoint utilities functionality.
"""

import os
import sys
import tempfile
import torch

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from model import GPTConfig, GPT
from training.utils.checkpoint_utils import (
    load_checkpoint_and_create_model,
    save_checkpoint,
    save_best_checkpoint,
    load_optimizer_state,
    get_model_info,
    find_latest_checkpoint,
    _detect_vocab_size_from_weights,
    _clean_state_dict_keys,
    _extract_iteration_number
)


def test_checkpoint_utilities():
    """Test checkpoint utility functions."""
    print("🧪 Testing checkpoint utilities...")
    
    # Create a simple model for testing
    model_args = {
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 64,
        'block_size': 128,
        'bias': False,
        'vocab_size': 1000,
        'dropout': 0.0
    }
    
    print("1. Testing model creation from scratch...")
    model, updated_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(
        init_from='scratch',
        resume_from='',
        model_args=model_args.copy(),
        meta_vocab_size=1000,
        device='cpu',
        master_process=True
    )
    
    assert model is not None, "Model should be created"
    assert iter_num == 0, "Iteration should be 0 for scratch"
    assert checkpoint is None, "No checkpoint for scratch"
    print("✓ Model creation from scratch works")
    
    print("2. Testing model info extraction...")
    info = get_model_info(model)
    assert 'total_parameters' in info, "Should have parameter count"
    assert 'model_size_mb' in info, "Should have model size"
    print(f"✓ Model info: {info['total_parameters']:,} params, {info['model_size_mb']:.1f}MB")
    
    print("3. Testing checkpoint saving...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a simple optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Test saving checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            model_args=model_args,
            iter_num=100,
            best_val_loss=2.5,
            config={'test': True},
            out_dir=tmp_dir,
            checkpoint_name='test_ckpt.pt',
            master_process=True
        )
        
        # Check if file was created
        checkpoint_path = os.path.join(tmp_dir, 'test_ckpt.pt')
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"
        print("✓ Checkpoint saving works")
        
        print("4. Testing checkpoint loading...")
        # Test loading the checkpoint
        loaded_model, loaded_args, loaded_iter, loaded_loss, loaded_checkpoint = load_checkpoint_and_create_model(
            init_from='resume',
            resume_from=checkpoint_path,
            model_args=model_args.copy(),
            meta_vocab_size=None,
            device='cpu',
            master_process=True
        )
        
        assert loaded_model is not None, "Loaded model should exist"
        assert loaded_iter == 100, f"Should load iter_num 100, got {loaded_iter}"
        assert loaded_loss == 2.5, f"Should load best_val_loss 2.5, got {loaded_loss}"
        print("✓ Checkpoint loading works")
        
        print("5. Testing optimizer state loading...")
        new_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4)
        optimizer_loaded = load_optimizer_state(loaded_checkpoint, new_optimizer, master_process=True)
        assert optimizer_loaded, "Optimizer state should load successfully"
        print("✓ Optimizer state loading works")
        
        print("6. Testing find_latest_checkpoint...")
        latest = find_latest_checkpoint(tmp_dir)
        assert latest is not None, "Should find the checkpoint"
        assert latest.endswith('test_ckpt.pt'), "Should find our test checkpoint"
        print("✓ Find latest checkpoint works")
        
        print("7. Testing best model saving...")
        save_best_checkpoint(
            model=loaded_model,
            optimizer=new_optimizer,
            model_args=loaded_args,
            iter_num=150,
            val_loss=2.3,
            config={'test': True},
            out_dir=tmp_dir,
            master_process=True
        )
        
        best_path = os.path.join(tmp_dir, 'best_model.pt')
        assert os.path.exists(best_path), "Best model file should exist"
        print("✓ Best model saving works")

    print("8. Testing utility functions...")
    
    # Test vocab size detection
    test_state_dict = {
        'transformer.wte.weight': torch.randn(5000, 64),
        'other.weight': torch.randn(10, 10)
    }
    vocab_size = _detect_vocab_size_from_weights(test_state_dict, master_process=False)
    assert vocab_size == 5000, f"Should detect vocab_size 5000, got {vocab_size}"
    
    # Test state dict cleaning
    dirty_state_dict = {
        'module.layer1.weight': torch.randn(10, 10),
        '_orig_mod.layer2.weight': torch.randn(5, 5),
        'clean.weight': torch.randn(3, 3)
    }
    clean_state_dict = _clean_state_dict_keys(dirty_state_dict.copy(), master_process=False)
    expected_keys = {'layer1.weight', 'layer2.weight', 'clean.weight'}
    assert set(clean_state_dict.keys()) == expected_keys, "Should clean unwanted prefixes"
    
    # Test iteration extraction
    test_checkpoint = {'iter_num': 42}
    iter_num = _extract_iteration_number(test_checkpoint)
    assert iter_num == 42, f"Should extract iter_num 42, got {iter_num}"
    
    test_checkpoint = {'step': 99}
    iter_num = _extract_iteration_number(test_checkpoint)
    assert iter_num == 99, f"Should extract step 99, got {iter_num}"
    
    test_checkpoint = {}
    iter_num = _extract_iteration_number(test_checkpoint)
    assert iter_num == 0, f"Should default to 0, got {iter_num}"
    
    print("✓ All utility functions work")
    
    print("✅ All checkpoint utility tests passed!")


def test_model_loading_formats():
    """Test loading different model formats."""
    print("\n🧪 Testing different model loading formats...")
    
    model_args = {
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 64,
        'block_size': 128,
        'bias': False,
        'vocab_size': 1000,
        'dropout': 0.0,
        'num_experts': 4,
        'top_k_experts': 2
    }
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create and save model with different checkpoint formats
        model, _, _, _, _ = load_checkpoint_and_create_model(
            init_from='scratch',
            resume_from='',
            model_args=model_args.copy(),
            device='cpu',
            master_process=True
        )
        
        # Test format 1: model_args in checkpoint
        checkpoint1 = {
            'model': model.state_dict(),
            'model_args': model_args,
            'iter_num': 50,
            'best_val_loss': 3.0
        }
        path1 = os.path.join(tmp_dir, 'format1.pt')
        torch.save(checkpoint1, path1)
        
        # Test format 2: config in checkpoint  
        checkpoint2 = {
            'model': model.state_dict(),
            'config': model_args,
            'step': 75,
            'best_val_loss': 2.8
        }
        path2 = os.path.join(tmp_dir, 'format2.pt')
        torch.save(checkpoint2, path2)
        
        # Test loading format 1
        loaded1, _, iter1, loss1, _ = load_checkpoint_and_create_model(
            init_from='resume',
            resume_from=path1,
            model_args={'n_layer': 1},  # This should be overridden
            device='cpu',
            master_process=True
        )
        assert iter1 == 50, f"Should load iter 50, got {iter1}"
        assert loss1 == 3.0, f"Should load loss 3.0, got {loss1}"
        
        # Test loading format 2  
        loaded2, _, iter2, loss2, _ = load_checkpoint_and_create_model(
            init_from='resume',
            resume_from=path2,
            model_args={'n_layer': 1},  # This should be overridden
            device='cpu',
            master_process=True
        )
        assert iter2 == 75, f"Should load step 75, got {iter2}"
        assert loss2 == 2.8, f"Should load loss 2.8, got {loss2}"
        
        print("✓ Different checkpoint formats work")
        
        # Test state dict with unwanted prefixes
        dirty_state_dict = {}
        for key, tensor in model.state_dict().items():
            dirty_state_dict[f'module._orig_mod.{key}'] = tensor
            
        checkpoint3 = {
            'model': dirty_state_dict,
            'model_args': model_args,
            'iter_num': 25
        }
        path3 = os.path.join(tmp_dir, 'dirty.pt')
        torch.save(checkpoint3, path3)
        
        # Should still load successfully after cleaning
        loaded3, _, iter3, _, _ = load_checkpoint_and_create_model(
            init_from='resume',
            resume_from=path3,
            model_args=model_args.copy(),
            device='cpu',
            master_process=True
        )
        assert iter3 == 25, f"Should load iter 25, got {iter3}"
        print("✓ Dirty state dict cleaning works")
    
    print("✅ All model loading format tests passed!")


if __name__ == '__main__':
    try:
        test_checkpoint_utilities()
        test_model_loading_formats()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
