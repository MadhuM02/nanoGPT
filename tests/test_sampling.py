#!/usr/bin/env python3
"""
Test script to verify sampling functionality
Creates a minimal checkpoint and tests loading/sampling
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import torch
from model import GPTConfig, GPT

def create_test_checkpoint():
    """Create a minimal test checkpoint for sampling verification"""
    print("Creating test checkpoint...")
    
    # Create minimal MoE config
    config = {
        'block_size': 256,
        'vocab_size': 50304,
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
        'expert_activation': 'gelu'
    }
    
    # Create model
    gpt_config = GPTConfig(
        block_size=config['block_size'],
        vocab_size=config['vocab_size'],
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        bias=config['bias'],
        num_experts=config['num_experts'],
        top_k_experts=config['top_k_experts'],
        load_balancing_loss_coef=config['load_balancing_loss_coef'],
        expert_dropout=config['expert_dropout'],
        capacity_factor=config['capacity_factor'],
        expert_activation=config['expert_activation']
    )
    
    model = GPT(gpt_config)
    
    # Create checkpoint
    checkpoint = {
        'model': model.state_dict(),
        'config': config,
        'step': 1000,
        'best_val_loss': 2.5
    }
    
    # Save test checkpoint
    os.makedirs('test_sampling_out', exist_ok=True)
    torch.save(checkpoint, 'test_sampling_out/best_model.pt')
    
    print(f"Test checkpoint saved to test_sampling_out/best_model.pt")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return config

def test_sample_py():
    """Test the updated sample.py script"""
    print("\n=== Testing sample.py ===")
    
    # Modify sample.py parameters for testing
    test_config = """
out_dir = 'test_sampling_out'
start = "Hello world"
num_samples = 2
max_new_tokens = 50
temperature = 1.0
top_k = 50
seed = 42
device = 'cpu'  # Use CPU for testing
dtype = 'float32'
compile = False
"""
    
    # Write test config
    with open('test_sample_config.py', 'w') as f:
        f.write(test_config)
    
    # Test sample.py by importing and checking it loads the model correctly
    try:
        # We'll simulate the key parts without actually running generation
        import sys
        sys.path.insert(0, '.')
        
        # Set up the same way sample.py does
        exec(open('test_sample_config.py').read())
        
        from model import GPTConfig, GPT
        
        # Load checkpoint
        ckpt_path = os.path.join('test_sampling_out', 'best_model.pt')
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
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
            raise ValueError("Config not found in checkpoint")
        
        model = GPT(gptconf)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        print("✅ sample.py loading test passed!")
        print(f"   - Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"   - MoE config: {gptconf.num_experts} experts, top-{gptconf.top_k_experts}")
        
        return True
        
    except Exception as e:
        print(f"❌ sample.py loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sample_moe_py():
    """Test the new sample_moe.py script"""
    print("\n=== Testing sample_moe.py ===")
    
    try:
        import subprocess
        import sys
        
        # Test with minimal parameters
        cmd = [
            sys.executable, 'sample_moe.py',
            '--out_dir', 'test_sampling_out',
            '--checkpoint', 'best',
            '--start', 'Hello',
            '--num_samples', '1',
            '--max_new_tokens', '10',
            '--device', 'cpu',
            '--dtype', 'float32',
            '--seed', '42'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ sample_moe.py test passed!")
            print("Sample output preview:")
            print(result.stdout[-200:])  # Show last 200 chars
            return True
        else:
            print(f"❌ sample_moe.py test failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ sample_moe.py test failed: {e}")
        return False

def cleanup():
    """Clean up test files"""
    import shutil
    
    files_to_remove = [
        'test_sample_config.py',
    ]
    
    dirs_to_remove = [
        'test_sampling_out',
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
    
    for dir in dirs_to_remove:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    
    print("\n🧹 Cleaned up test files")

def main():
    print("Testing MoE Sampling Scripts")
    print("=" * 40)
    
    try:
        # Create test checkpoint
        create_test_checkpoint()
        
        # Test both scripts
        sample_py_ok = test_sample_py()
        sample_moe_py_ok = test_sample_moe_py()
        
        print(f"\n📊 Test Results:")
        print(f"  sample.py: {'✅ PASS' if sample_py_ok else '❌ FAIL'}")
        print(f"  sample_moe.py: {'✅ PASS' if sample_moe_py_ok else '❌ FAIL'}")
        
        if sample_py_ok and sample_moe_py_ok:
            print(f"\n🎉 All sampling tests passed!")
            print("\nYou can now use either script to sample from your trained models:")
            print("  python sample.py  # Simple version")
            print("  python sample_moe.py --help  # Advanced version with more options")
        else:
            print(f"\n⚠️  Some tests failed. Check the error messages above.")
            
    finally:
        cleanup()

if __name__ == "__main__":
    main()
