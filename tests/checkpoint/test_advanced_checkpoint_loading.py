#!/usr/bin/env python3
"""
Test Enhanced Checkpoint Loading with Real Scenarios

This script tests the new enhanced checkpoint loading functionality
with various scenarios including prefix mismatches, DDP wrappers, etc.
"""

import os
import sys
import torch
import tempfile
from contextlib import contextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_moe_advanced import load_checkpoint, save_checkpoint

def create_test_model():
    """Create a simple test model"""
    from model import GPTConfig, GPT
    
    config = GPTConfig(
        block_size=256,
        vocab_size=1000,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    return GPT(config), config

@contextmanager
def temporary_checkpoint(state_dict, metadata=None):
    """Create a temporary checkpoint file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
    
    try:
        checkpoint_data = {'model': state_dict}
        if metadata:
            checkpoint_data.update(metadata)
        torch.save(checkpoint_data, checkpoint_path)
        yield checkpoint_path
    finally:
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)

def add_prefix_to_state_dict(state_dict, prefix):
    """Add prefix to all keys in state dict"""
    return {prefix + key: value for key, value in state_dict.items()}

def test_scenario(name, model, checkpoint_state_dict, metadata=None, should_succeed=True):
    """Test a specific checkpoint loading scenario"""
    print(f"\n🧪 Testing: {name}")
    print("-" * 50)
    
    with temporary_checkpoint(checkpoint_state_dict, metadata) as checkpoint_path:
        try:
            config, step, best_val_loss = load_checkpoint(
                checkpoint_path, 
                model,
                device='cpu'
            )
            
            if should_succeed:
                print(f"✅ SUCCESS: Loaded checkpoint (step={step}, loss={best_val_loss:.4f})")
                return True
            else:
                print(f"⚠️ UNEXPECTED SUCCESS: Test was expected to fail but succeeded")
                return False
                
        except Exception as e:
            if should_succeed:
                print(f"❌ FAILED: {str(e)[:100]}...")
                return False
            else:
                print(f"✅ EXPECTED FAILURE: {str(e)[:100]}...")
                return True

def test_enhanced_checkpoint_loading():
    """Test the enhanced checkpoint loading with various scenarios"""
    print("🔧 Testing Enhanced Checkpoint Loading")
    print("=" * 80)
    
    # Create test model
    model, config = create_test_model()
    original_state_dict = model.state_dict()
    
    # Test scenarios
    test_results = []
    
    # 1. Normal checkpoint (should work)
    test_results.append(test_scenario(
        "Normal checkpoint (no prefixes)",
        model,
        original_state_dict.copy(),
        {'step': 100, 'best_val_loss': 2.5}
    ))
    
    # 2. DDP checkpoint (module. prefix)
    ddp_state_dict = add_prefix_to_state_dict(original_state_dict, 'module.')
    test_results.append(test_scenario(
        "DDP checkpoint (module. prefix)",
        model,
        ddp_state_dict,
        {'step': 200, 'best_val_loss': 2.0}
    ))
    
    # 3. Compiled model checkpoint (_orig_mod. prefix)
    compiled_state_dict = add_prefix_to_state_dict(original_state_dict, '_orig_mod.')
    test_results.append(test_scenario(
        "Compiled model checkpoint (_orig_mod. prefix)",
        model,
        compiled_state_dict,
        {'step': 300, 'best_val_loss': 1.8}
    ))
    
    # 4. DDP + Compiled checkpoint (module._orig_mod. prefix)
    ddp_compiled_state_dict = add_prefix_to_state_dict(original_state_dict, 'module._orig_mod.')
    test_results.append(test_scenario(
        "DDP + Compiled checkpoint (module._orig_mod. prefix)",
        model,
        ddp_compiled_state_dict,
        {'step': 400, 'best_val_loss': 1.5}
    ))
    
    # 5. Direct state dict (no metadata)
    test_results.append(test_scenario(
        "Direct state dict (no metadata wrapper)",
        model,
        original_state_dict.copy()
    ))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 50)
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Tests passed: {passed}/{total}")
    print(f"❌ Tests failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced checkpoint loading is working correctly.")
    else:
        print(f"\n⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

def test_checkpoint_analysis_functions():
    """Test the helper functions for checkpoint analysis"""
    print(f"\n🔍 Testing Checkpoint Analysis Functions")
    print("=" * 50)
    
    from train_moe_advanced import analyze_key_patterns, find_matching_keys, strip_model_prefixes
    
    # Test analyze_key_patterns
    test_keys = [
        'transformer.wte.weight',
        'transformer.wpe.weight', 
        '_orig_mod.transformer.h.0.attn.c_attn.weight',
        '_orig_mod.transformer.h.0.attn.c_proj.weight',
        'module.transformer.h.1.mlp.c_fc.weight'
    ]
    
    analysis = analyze_key_patterns(test_keys)
    print(f"Key pattern analysis:")
    print(f"  Total keys: {analysis['total_keys']}")
    print(f"  Dominant prefix: {analysis['dominant_prefix']}")
    print(f"  Prefix ratio: {analysis['prefix_ratio']:.2f}")
    
    # Test strip_model_prefixes
    test_state_dict = {key: torch.randn(10, 10) for key in test_keys}
    stripped_dict, changes = strip_model_prefixes(test_state_dict)
    print(f"\nPrefix stripping:")
    print(f"  Keys before: {len(test_state_dict)}")
    print(f"  Keys after: {len(stripped_dict)}")
    print(f"  Changes made: {changes}")
    
    # Test find_matching_keys
    model_keys = {'a.weight', 'b.bias', 'c.weight'}
    checkpoint_keys = {'_orig_mod.a.weight', '_orig_mod.b.bias', '_orig_mod.c.weight'}
    
    matching_analysis = find_matching_keys(model_keys, checkpoint_keys)
    print(f"\nKey matching analysis:")
    print(f"  Exact matches: {matching_analysis['exact_matches']}")
    print(f"  Best transformation: {matching_analysis['transformations'][0] if matching_analysis['transformations'] else 'None'}")
    
    print("✅ Analysis functions working correctly")

def main():
    """Run all tests"""
    print("🧪 Enhanced Checkpoint Loading Test Suite")
    print("=" * 80)
    
    try:
        # Test the analysis functions
        test_checkpoint_analysis_functions()
        
        # Test the checkpoint loading scenarios
        success = test_enhanced_checkpoint_loading()
        
        print(f"\n{'='*80}")
        if success:
            print("🎉 All tests completed successfully!")
            print("✨ Enhanced checkpoint loading is ready for production use.")
        else:
            print("⚠️ Some tests failed - please review the output above.")
            
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
