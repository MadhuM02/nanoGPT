"""
Advanced training configurations optimized for 4x V100 16GB setup
"""

# V100 optimized configuration
import torch

def get_v100_config():
    """Get memory-optimized config for 4x V100 16GB GPUs"""
    config = {
        # Model architecture optimized for V100 memory constraints
        'n_layer': 12,  # Reduced from 16 to save memory
        'n_head': 12,   # Reduced from 16 
        'n_embd': 768,  # Reduced from 1024 to save significant memory
        'block_size': 512,  # Reduced context length for memory efficiency
        'dropout': 0.1,
        'bias': False,
        
        # MoE configuration - more conservative for memory
        'num_experts': 8,   # Reduced from 16 to save memory
        'top_k_experts': 2,  # Keep top-2 routing
        'load_balancing_loss_coef': 0.01,
        'expert_dropout': 0.15,  # Slightly higher for regularization
        'capacity_factor': 1.0,   # Reduced from 1.25
        'expert_activation': 'gelu',  # GELU is more memory efficient than swish
        
        # Training hyperparameters optimized for memory
        'batch_size': 4,  # Reduced from 8 per GPU
        'gradient_accumulation_steps': 32,  # Increased to maintain effective batch size
        'learning_rate': 3e-4,  # Slightly lower for MoE stability
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # Memory optimization
        'dtype': 'float16',  # Use FP16 for memory efficiency
        'compile': True,     # PyTorch 2.0 compilation
        
        # V100 specific optimizations
        'max_iters': 500000,
        'warmup_iters': 4000,
        'lr_decay_iters': 500000,
        'min_lr': 3e-5,
        
        # Evaluation and logging
        'eval_interval': 1000,
        'log_interval': 10,
        'eval_iters': 100,
        'checkpoint_interval': 5000,  # Save checkpoint every 5000 steps
        
        # Checkpointing
        'always_save_checkpoint': True,
        'out_dir': 'checkpoints/out-moe-v100',
        
        # Distributed training
        'backend': 'nccl',
    }
    return config

def get_memory_efficient_config():
    """Ultra memory-efficient configuration for longer sequences"""
    config = get_v100_config()
    config.update({
        'n_layer': 8,    # Even fewer layers
        'n_embd': 768,   # Smaller embedding
        'block_size': 256,  # Much shorter sequences to save memory
        'batch_size': 2,     # Very small batch size
        'gradient_accumulation_steps': 64,  # High accumulation to maintain effective batch
        'num_experts': 4,    # Minimal experts
        'expert_dropout': 0.3,  # High expert dropout
        'use_gradient_checkpointing': True,
        'dtype': 'float16',  # Force FP16
    })
    return config

def get_performance_config():
    """Performance-optimized configuration with memory constraints"""
    config = get_v100_config()
    config.update({
        'n_layer': 10,   # Moderate depth
        'n_embd': 640,   # Moderate embedding size
        'block_size': 384,   # Shorter sequences for speed and memory
        'batch_size': 6,    # Moderate batch size
        'gradient_accumulation_steps': 16,
        'num_experts': 12,   # Moderate number of experts
        'expert_dropout': 0.1,  # Lower dropout for performance
        'compile': True,
        'dtype': 'float16',  # Stick with FP16 for V100
    })
    return config

def get_minimal_config():
    """Minimal configuration for very tight memory constraints"""
    config = {
        # Very small model for testing/debugging
        'n_layer': 6,
        'n_head': 8,
        'n_embd': 384,
        'block_size': 128,
        'dropout': 0.1,
        'bias': False,
        
        # Minimal MoE
        'num_experts': 4,
        'top_k_experts': 1,  # Top-1 for memory savings
        'load_balancing_loss_coef': 0.01,
        'expert_dropout': 0.2,
        'capacity_factor': 0.8,
        'expert_activation': 'gelu',
        
        # Very conservative training params
        'batch_size': 1,
        'gradient_accumulation_steps': 128,  # Very high accumulation
        'learning_rate': 2e-4,
        'weight_decay': 0.05,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 0.5,
        
        # Memory optimizations
        'dtype': 'float16',
        'compile': False,  # Disable compilation to save memory
        'use_gradient_checkpointing': True,
        
        # Reduced training params
        'max_iters': 100000,
        'warmup_iters': 1000,
        'lr_decay_iters': 100000,
        'min_lr': 2e-5,
        
        # More frequent evaluation to monitor progress
        'eval_interval': 500,
        'log_interval': 50,
        'eval_iters': 20,
        
        'always_save_checkpoint': True,
        'out_dir': 'checkpoints/out-moe-minimal',
        'backend': 'nccl',
    }
    return config

def get_debug_config():
    """Debug configuration for development and testing"""
    config = {
        # Tiny model for quick testing
        'n_layer': 2,
        'n_head': 4,
        'n_embd': 128,
        'block_size': 64,
        'dropout': 0.0,
        'bias': False,
        
        # Simple MoE for testing
        'num_experts': 2,
        'top_k_experts': 1,
        'load_balancing_loss_coef': 0.01,
        'expert_dropout': 0.0,
        'capacity_factor': 1.0,
        'expert_activation': 'gelu',
        
        # Fast iteration for debugging
        'batch_size': 2,
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'grad_clip': 1.0,
        
        'dtype': 'float32',  # Use FP32 for debugging
        'compile': False,
        
        'max_iters': 1000,
        'warmup_iters': 100,
        'lr_decay_iters': 1000,
        'min_lr': 1e-4,
        
        'eval_interval': 100,
        'log_interval': 10,
        'eval_iters': 5,
        
        'always_save_checkpoint': False,
        'out_dir': 'checkpoints/out-debug',
        'backend': 'nccl',
    }
    return config

def auto_select_config(available_memory_gb):
    """Auto-select configuration based on available GPU memory"""
    if available_memory_gb >= 14:
        print("Using V100 config (sufficient memory)")
        return get_v100_config()
    elif available_memory_gb >= 10:
        print("Using memory-efficient config (moderate memory)")
        return get_memory_efficient_config()
    elif available_memory_gb >= 6:
        print("Using minimal config (low memory)")
        return get_minimal_config()
    else:
        print("Using debug config (very low memory)")
        return get_debug_config()

# Additional optimization techniques
OPTIMIZATION_TECHNIQUES = {
    'expert_parallelism': {
        'description': 'Distribute experts across GPUs',
        'memory_saving': 'High',
        'complexity': 'High',
        'implementation': 'ExpertParallelMoE'
    },
    
    'switch_transformer': {
        'description': 'Use Switch Transformer with capacity limits',
        'memory_saving': 'Medium',
        'complexity': 'Medium', 
        'implementation': 'SwitchTransformer'
    },
    
    'gradient_checkpointing': {
        'description': 'Trade compute for memory',
        'memory_saving': 'High',
        'complexity': 'Low',
        'implementation': 'GradientCheckpointingMoE'
    },
    
    'flash_attention': {
        'description': 'Memory-efficient attention',
        'memory_saving': 'Medium',
        'complexity': 'Low',
        'implementation': 'FlashAttentionMoE'
    },
    
    'mixed_precision': {
        'description': 'FP16/BF16 training',
        'memory_saving': 'Medium',
        'complexity': 'Low',
        'implementation': 'Built-in PyTorch'
    },
    
    'tensor_parallelism': {
        'description': 'Split large tensors across GPUs',
        'memory_saving': 'High',
        'complexity': 'Very High',
        'implementation': 'Custom'
    },
    
    'dynamic_batching': {
        'description': 'Adaptive batch sizes based on sequence length',
        'memory_saving': 'Medium',
        'complexity': 'Medium',
        'implementation': 'Custom'
    },
    
    'expert_caching': {
        'description': 'Cache frequently used expert outputs',
        'memory_saving': 'Low',
        'complexity': 'Medium',
        'implementation': 'Custom'
    }
}

def print_optimization_recommendations():
    """Print recommendations for V100 setup"""
    print("=== V100 4x16GB Optimization Recommendations ===\n")
    
    print("1. MEMORY OPTIMIZATIONS:")
    print("   - Use FP16/BF16 mixed precision training")
    print("   - Enable gradient checkpointing for deeper models")
    print("   - Use expert parallelism to distribute MoE across GPUs")
    print("   - Implement dynamic batching for variable sequence lengths")
    
    print("\n2. COMPUTE OPTIMIZATIONS:")
    print("   - Use PyTorch 2.0 compilation with torch.compile()")
    print("   - Enable Flash Attention for memory-efficient attention")
    print("   - Use Switch Transformer with capacity limits")
    print("   - Optimize expert routing with better load balancing")
    
    print("\n3. DISTRIBUTED TRAINING:")
    print("   - Use expert parallelism across 4 GPUs")
    print("   - Implement pipeline parallelism for very large models")
    print("   - Use NCCL backend for optimal GPU communication")
    print("   - Consider ZeRO optimizer states sharding")
    
    print("\n4. HYPERPARAMETER TUNING:")
    print("   - Start with 16 experts, top-k=2")
    print("   - Use lower learning rates (3e-4) for MoE stability")
    print("   - Increase warmup steps (4000) for better convergence")
    print("   - Use expert dropout (0.1) for regularization")
    
    print("\n5. MONITORING:")
    print("   - Track expert utilization and load balance")
    print("   - Monitor GPU memory usage across devices")
    print("   - Watch for expert collapse (unused experts)")
    print("   - Log routing entropy for diversity metrics")

def print_memory_recommendations():
    """Print memory optimization recommendations"""
    print("=== Memory Optimization Recommendations ===\n")
    
    print("1. IMMEDIATE MEMORY SAVERS:")
    print("   - Use smaller batch size (1-4 per GPU)")
    print("   - Reduce sequence length (128-512 tokens)")
    print("   - Use fewer experts (4-8 instead of 16+)")
    print("   - Enable gradient checkpointing")
    print("   - Use FP16 mixed precision")
    
    print("\n2. MODEL SIZE OPTIMIZATIONS:")
    print("   - Reduce embedding dimension (384-768 instead of 1024+)")
    print("   - Use fewer layers (6-12 instead of 16+)")
    print("   - Use fewer attention heads (8-12 instead of 16+)")
    print("   - Switch to top-1 expert routing")
    
    print("\n3. TRAINING OPTIMIZATIONS:")
    print("   - Increase gradient accumulation steps")
    print("   - Use expert dropout (0.2-0.3)")
    print("   - Disable model compilation temporarily")
    print("   - Clear cache between iterations")
    
    print("\n4. CONFIGURATION HIERARCHY:")
    print("   - debug: Tiny model for testing (2GB)")
    print("   - minimal: Small model for tight constraints (6GB)")
    print("   - memory: Conservative model (10GB)")
    print("   - v100: Standard model (14GB+)")

if __name__ == "__main__":
    print_optimization_recommendations()
    print_memory_recommendations()
    
    print("\n=== Available Configurations ===")
    configs = {
        'debug': get_debug_config(),
        'minimal': get_minimal_config(),
        'memory': get_memory_efficient_config(),
        'v100': get_v100_config(),
        'performance': get_performance_config()
    }
    
    for name, config in configs.items():
        params = ((config['n_layer'] * config['n_head'] * config['n_embd'] * 
                  (3 * config['n_embd'] + 4 * config['n_embd'] * config['num_experts'])) // 1e6)
        print(f"{name:12}: {config['n_layer']:2}L {config['n_embd']:4}D {config['num_experts']:2}E "
              f"batch={config['batch_size']} seq={config['block_size']} (~{params}M params)")
    
    print(f"\nFor OOM issues, try:")
    print(f"  python train_moe_advanced.py --config minimal")
    print(f"  python train_moe_advanced.py --config debug")
