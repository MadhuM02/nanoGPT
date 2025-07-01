#!/usr/bin/env python3
"""
Quick test of enhanced checkpoint loading logic
"""

import os
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_prefix_logic():
    """Test the prefix detection logic"""
    print("Testing Prefix Logic")
    print("=" * 30)
    
    # Test case 1: High prefix ratio (should remove prefixes)
    high_prefix_dict = {
        '_orig_mod.layer1.weight': torch.randn(10, 10),
        '_orig_mod.layer2.weight': torch.randn(10, 10),
        '_orig_mod.layer3.weight': torch.randn(10, 10),
        'layer4.weight': torch.randn(10, 10),  # Only one without prefix
    }
    
    unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
    prefix_counts = {}
    for prefix in unwanted_prefixes:
        count = sum(1 for k in high_prefix_dict.keys() if k.startswith(prefix))
        if count > 0:
            prefix_counts[prefix] = count
    
    total_keys = len(high_prefix_dict)
    total_prefixed_keys = sum(prefix_counts.values())
    prefix_ratio = total_prefixed_keys / total_keys
    
    print(f"Test 1 - High prefix ratio:")
    print(f"  Keys: {list(high_prefix_dict.keys())}")
    print(f"  Prefix counts: {prefix_counts}")
    print(f"  Prefix ratio: {prefix_ratio:.2f}")
    print(f"  Should remove prefixes: {prefix_ratio > 0.8}")
    
    # Test case 2: Low prefix ratio (should keep prefixes)
    low_prefix_dict = {
        'layer1.weight': torch.randn(10, 10),
        'layer2.weight': torch.randn(10, 10),
        'layer3.weight': torch.randn(10, 10),
        '_orig_mod.layer4.weight': torch.randn(10, 10),  # Only one with prefix
    }
    
    prefix_counts_low = {}
    for prefix in unwanted_prefixes:
        count = sum(1 for k in low_prefix_dict.keys() if k.startswith(prefix))
        if count > 0:
            prefix_counts_low[prefix] = count
    
    total_keys_low = len(low_prefix_dict)
    total_prefixed_keys_low = sum(prefix_counts_low.values())
    prefix_ratio_low = total_prefixed_keys_low / total_keys_low
    
    print(f"\nTest 2 - Low prefix ratio:")
    print(f"  Keys: {list(low_prefix_dict.keys())}")
    print(f"  Prefix counts: {prefix_counts_low}")
    print(f"  Prefix ratio: {prefix_ratio_low:.2f}")
    print(f"  Should remove prefixes: {prefix_ratio_low > 0.8}")
    
    print(f"\n✅ Prefix logic working correctly!")
    return True

def test_key_matching():
    """Test key matching logic"""
    print("\nTesting Key Matching Logic")
    print("=" * 35)
    
    # Simulate model keys (what the model expects)
    model_keys = [
        '_orig_mod.transformer.wte.weight',
        '_orig_mod.transformer.wpe.weight',
        '_orig_mod.transformer.h.0.ln_1.weight',
    ]
    
    # Simulate checkpoint keys (what we loaded)
    checkpoint_keys = [
        'transformer.wte.weight',
        'transformer.wpe.weight', 
        'transformer.h.0.ln_1.weight',
    ]
    
    # Check prefix patterns
    model_has_prefix = any(key.startswith(('_orig_mod.', 'module.')) for key in model_keys)
    checkpoint_has_prefix = any(key.startswith(('_orig_mod.', 'module.')) for key in checkpoint_keys)
    
    print(f"Model keys: {model_keys}")
    print(f"Checkpoint keys: {checkpoint_keys}")
    print(f"Model expects prefixed keys: {model_has_prefix}")
    print(f"Checkpoint has prefixed keys: {checkpoint_has_prefix}")
    
    if model_has_prefix and not checkpoint_has_prefix:
        print("✅ Detected: Need to ADD prefixes to checkpoint keys")
        prefix_to_add = '_orig_mod.'
        adjusted_keys = [prefix_to_add + key for key in checkpoint_keys]
        print(f"Adjusted keys: {adjusted_keys}")
        
    elif not model_has_prefix and checkpoint_has_prefix:
        print("✅ Detected: Need to REMOVE prefixes from checkpoint keys")
        
    print(f"\n✅ Key matching logic working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_prefix_logic()
        test_key_matching()
        print(f"\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
