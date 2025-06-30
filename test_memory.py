"""
Memory testing script to find optimal configuration for your V100 setup
"""
import torch
import gc
from model import GPT, GPTConfig
from v100_config import get_debug_config, get_minimal_config, get_memory_efficient_config, get_v100_config

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return 0

def test_config_memory(config_name, config, max_attempts=3):
    """Test if a configuration fits in memory"""
    print(f"\nTesting {config_name} configuration...")
    print(f"  Model: {config['n_layer']}L, {config['n_embd']}D, {config['num_experts']}E")
    print(f"  Training: batch={config['batch_size']}, seq={config['block_size']}")
    
    for attempt in range(max_attempts):
        try:
            clear_memory()
            
            # Create model config
            model_config = GPTConfig(
                n_layer=config['n_layer'],
                n_head=config['n_head'],
                n_embd=config['n_embd'],
                block_size=config['block_size'],
                dropout=config['dropout'],
                bias=config['bias'],
                vocab_size=50304,
                num_experts=config['num_experts'],
                top_k_experts=config['top_k_experts'],
                load_balancing_loss_coef=config['load_balancing_loss_coef'],
                expert_dropout=config['expert_dropout'],
                capacity_factor=config['capacity_factor'],
                expert_activation=config['expert_activation']
            )
            
            # Create model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = GPT(model_config).to(device)
            
            memory_after_model = get_memory_usage()
            print(f"  Memory after model creation: {memory_after_model:.2f} GB")
            
            # Test forward pass
            batch_size = config['batch_size']
            seq_len = config['block_size']
            x = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, model_config.vocab_size, (batch_size, seq_len), device=device)
            
            # Forward pass
            logits, loss = model(x, y)
            memory_after_forward = get_memory_usage()
            print(f"  Memory after forward pass: {memory_after_forward:.2f} GB")
            
            # Backward pass
            loss.backward()
            memory_after_backward = get_memory_usage()
            print(f"  Memory after backward pass: {memory_after_backward:.2f} GB")
            
            print(f"  ✅ {config_name} configuration PASSED (peak: {memory_after_backward:.2f} GB)")
            
            # Cleanup
            del model, logits, loss, x, y
            clear_memory()
            
            return True, memory_after_backward
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ {config_name} configuration FAILED - OOM on attempt {attempt + 1}")
                clear_memory()
                if attempt < max_attempts - 1:
                    print(f"  Retrying...")
                continue
            else:
                print(f"  ❌ {config_name} configuration FAILED - {e}")
                return False, 0
        except Exception as e:
            print(f"  ❌ {config_name} configuration FAILED - {e}")
            return False, 0
    
    return False, 0

def find_optimal_config():
    """Find the largest configuration that fits in memory"""
    if not torch.cuda.is_available():
        print("CUDA not available, using debug config")
        return 'debug'
    
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("Testing configurations from smallest to largest...")
    
    configs = [
        ('debug', get_debug_config()),
        ('minimal', get_minimal_config()),
        ('memory', get_memory_efficient_config()),
        ('v100', get_v100_config()),
    ]
    
    best_config = 'debug'
    best_memory = 0
    
    for config_name, config in configs:
        success, peak_memory = test_config_memory(config_name, config)
        if success:
            best_config = config_name
            best_memory = peak_memory
        else:
            break  # Stop at first failure
    
    print(f"\n🎯 RECOMMENDATION: Use '{best_config}' configuration")
    print(f"   Peak memory usage: {best_memory:.2f} GB")
    print(f"   Command: python train_moe_advanced.py --config {best_config}")
    
    return best_config

def quick_memory_check():
    """Quick check of available memory vs model requirements"""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total memory: {total_memory:.1f} GB")
    
    # Rough memory estimates for different configs
    estimates = {
        'debug': 0.5,
        'minimal': 2.5,
        'memory': 6.0,
        'v100': 12.0,
    }
    
    print("\nEstimated memory requirements:")
    for config, memory in estimates.items():
        status = "✅" if memory < total_memory * 0.8 else "❌"
        print(f"  {config:8}: ~{memory:4.1f} GB {status}")
    
    recommended = [name for name, mem in estimates.items() if mem < total_memory * 0.8]
    if recommended:
        print(f"\nRecommended configs: {', '.join(recommended)}")
    else:
        print("\nWarning: Even debug config might be tight!")

if __name__ == "__main__":
    print("=== Memory Configuration Tester ===\n")
    
    # Quick check first
    quick_memory_check()
    print()
    
    # Ask user if they want full test
    response = input("Run full memory test? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        find_optimal_config()
    else:
        print("\nFor OOM issues, try:")
        print("  python train_moe_advanced.py --config minimal")
        print("  python train_moe_advanced.py --config debug")
