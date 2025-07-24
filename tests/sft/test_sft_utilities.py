"""
Test for refactored SFT training script utilities integration.
"""

import os
import sys
import tempfile

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def test_sft_utilities_integration():
    """Test that the SFT utilities work correctly."""
    print("🧪 Testing SFT utilities integration...")
    
    try:
        from training.utils.sft_utils import create_sft_utilities
        
        # Create temporary data directory for testing
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test data files
            train_file = os.path.join(tmp_dir, 'train.txt')
            val_file = os.path.join(tmp_dir, 'val.txt')
            
            with open(train_file, 'w') as f:
                f.write("Question: What is 2+2? Answer: 4\n")
                f.write("Category: Math Summary: Simple addition\n")
                f.write("Sentiment: This is great! Output: Positive\n")
            
            with open(val_file, 'w') as f:
                f.write("Question: What is 3+3? Answer: 6\n")
                f.write("Category: Science Summary: Chemistry basics\n")
            
            # Test creating utilities
            data_manager, loss_analyzer, training_utils = create_sft_utilities(tmp_dir)
            print("✓ SFT utilities created successfully")
            
            # Test loading examples
            data_manager.load_examples(master_process=True)
            print(f"✓ Loaded {len(data_manager.train_examples)} train examples, {len(data_manager.val_examples)} val examples")
            
            # Test batch creation
            batch_data = data_manager.get_batch('train', batch_size=2, block_size=64, device='cpu', device_type='cpu')
            assert len(batch_data) == 3, "Should return X, Y, loss_mask for SFT data"
            X, Y, loss_mask = batch_data
            print(f"✓ Created batch: X shape {X.shape}, Y shape {Y.shape}, loss_mask shape {loss_mask.shape}")
            
            # Test loss analyzer
            import torch
            import torch.nn.functional as F
            
            # Create dummy logits
            vocab_size = 50257  # GPT-2 vocab size
            logits = torch.randn(2, 64, vocab_size)
            
            per_sample_losses = loss_analyzer.compute_per_sample_loss(logits, Y)
            print(f"✓ Computed per-sample losses: {per_sample_losses.shape}")
            
            # Test learning rate scheduler
            lr = training_utils.get_cosine_lr_with_warmup(
                it=50, learning_rate=1e-4, warmup_iters=100, 
                lr_decay_iters=1000, min_lr=1e-6
            )
            print(f"✓ Learning rate at step 50: {lr:.6f}")
            
            # Test loss spike detection
            loss_history = [2.5, 2.4, 2.3, 2.2, 2.1]
            is_spike = training_utils.detect_loss_spike(loss_history, 5.0, spike_threshold=2.0)
            assert is_spike, "Should detect loss spike"
            print("✓ Loss spike detection works")
            
            print("✅ All SFT utilities integration tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sft_training_imports():
    """Test that SFT training script imports work."""
    print("\n🧪 Testing SFT training script imports...")
    
    try:
        # Test basic imports
        from training.utils.sft_utils import create_sft_utilities
        from training.utils.checkpoint_utils import load_checkpoint_and_create_model
        
        print("✓ All SFT training imports work")
        
        # Test that we can create utilities without errors
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_manager, loss_analyzer, training_utils = create_sft_utilities(tmp_dir)
            print("✓ Utilities creation works")
        
        print("✅ SFT training imports test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success1 = test_sft_utilities_integration()
    success2 = test_sft_training_imports()
    
    if success1 and success2:
        print("\n🎉 All SFT refactoring tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
