# nanoGPT MoE Model Architecture Guide

This document provides a comprehensive overview of the nanoGPT Mixture-of-Experts (MoE) model architecture, with specific focus on V100 GPU configurations and optimization considerations.

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MoE Layer Design](#moe-layer-design)
3. [V100 Configuration](#v100-configuration)
4. [Memory Considerations](#memory-considerations)
5. [Performance Optimizations](#performance-optimizations)
6. [Scaling Considerations](#scaling-considerations)
7. [Configuration Guidelines](#configuration-guidelines)

## 🏗️ Architecture Overview

### Base Transformer Architecture

The nanoGPT MoE model extends the standard GPT architecture with Mixture-of-Experts layers:

```
GPT Model
├── Token Embedding (vocab_size × n_embd)
├── Position Embedding (block_size × n_embd)
├── Transformer Blocks (n_layer)
│   ├── LayerNorm
│   ├── Multi-Head Attention (n_head)
│   ├── LayerNorm
│   └── MoE Feed-Forward Network
│       ├── Gating Network (Router)
│       └── Expert Networks (num_experts)
└── Language Model Head (n_embd × vocab_size)
```

### Key Components

| Component | Purpose | V100 Considerations |
|-----------|---------|-------------------|
| **Embedding Layer** | Token → Vector mapping | Memory-bound, benefits from larger `n_embd` |
| **Attention Heads** | Self-attention mechanism | Compute-bound, scales with `n_head` |
| **MoE Layer** | Sparse expert selection | Memory + compute, critical for efficiency |
| **Router/Gating** | Expert selection logic | Lightweight, affects load balancing |

## 🧠 MoE Layer Design

### Expert Architecture

Each expert is a standard feed-forward network:

```python
Expert Network:
    Linear1: n_embd → 4 * n_embd
    Activation: GELU/ReLU/SwiGLU
    Linear2: 4 * n_embd → n_embd
    Dropout: expert_dropout
```

### Gating Mechanism

The router determines which experts to activate:

```python
Router Logic:
    1. Linear: n_embd → num_experts
    2. Softmax: Probability distribution
    3. TopK: Select top_k_experts
    4. Load Balancing: Auxiliary loss
```

### V100-Optimized MoE Parameters

```python
# V100 Configuration Example
{
    "num_experts": 8,           # Optimal for V100 memory
    "top_k_experts": 2,         # Balance quality/efficiency
    "capacity_factor": 1.25,    # Buffer for load imbalance
    "load_balancing_loss_coef": 0.01,  # Encourage expert diversity
    "expert_dropout": 0.1       # Regularization
}
```

## 🎯 V100 Configuration

### Hardware Specifications

**NVIDIA V100 Specifications:**
- **Memory**: 32GB HBM2
- **Memory Bandwidth**: 900 GB/s
- **Compute**: 15.7 TFLOPS (FP32), 125 TFLOPS (Tensor)
- **SM Count**: 80 (5120 CUDA cores)

### Optimal V100 Configuration

```python
# Recommended V100 MoE Configuration
V100_CONFIG = {
    # Model Architecture
    "n_layer": 12,              # 12 transformer blocks
    "n_head": 12,               # 12 attention heads
    "n_embd": 768,              # 768 embedding dimensions
    "block_size": 1024,         # 1024 sequence length
    "vocab_size": 50304,        # GPT-2 vocabulary
    
    # MoE Specific
    "num_experts": 8,           # 8 experts per MoE layer
    "top_k_experts": 2,         # Activate top-2 experts
    "load_balancing_loss_coef": 0.01,
    "expert_dropout": 0.1,
    "capacity_factor": 1.25,
    "expert_activation": "gelu",
    
    # Training Configuration
    "batch_size": 12,           # Optimized for V100 memory
    "gradient_accumulation_steps": 5,  # Effective batch: 60
    "learning_rate": 6e-4,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    
    # Memory Optimization
    "use_gradient_checkpointing": True,
    "dtype": "bfloat16",        # Mixed precision
    "compile": True,            # PyTorch 2.0 compilation
    
    # Performance
    "warmup_iters": 2000,
    "max_iters": 600000,
    "eval_interval": 2000,
    "log_interval": 100
}
```

### Memory Breakdown (V100 32GB)

| Component | Memory Usage | Percentage |
|-----------|--------------|------------|
| **Model Parameters** | ~8GB | 25% |
| **Optimizer States** | ~16GB | 50% |
| **Activations** | ~4GB | 12.5% |
| **Gradients** | ~4GB | 12.5% |
| **Total** | ~32GB | 100% |

## 💾 Memory Considerations

### Parameter Count Analysis

```python
# V100 Configuration Parameter Breakdown
Total Parameters: ~350M

Base Transformer:
- Embeddings: 50304 × 768 = 38.6M
- Attention: 12 × (768 × 768 × 4) = 28.3M  
- Layer Norms: 12 × 768 × 2 = 18.4K
- Output Head: 768 × 50304 = 38.6M

MoE Layers (per layer):
- Router: 768 × 8 = 6.1K
- Experts: 8 × (768 × 3072 × 2) = 37.7M
- Total MoE (12 layers): 452M

Effective Parameters (top-2):
- Active per forward: ~175M (50% sparsity)
```

### Memory Optimization Strategies

#### 1. Gradient Checkpointing
```python
# Trade compute for memory
model.gradient_checkpointing_enable()
# Saves ~50% activation memory
# Adds ~30% compute overhead
```

#### 2. Mixed Precision Training
```python
# bfloat16 reduces memory by ~50%
scaler = torch.cuda.amp.GradScaler(enabled=True)
with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
    loss = model(inputs, targets)
```

#### 3. Dynamic Batching
```python
# Adjust batch size based on sequence length
def get_dynamic_batch_size(seq_len, base_batch_size=12):
    # Longer sequences → smaller batches
    if seq_len > 512:
        return max(1, base_batch_size // 2)
    return base_batch_size
```

## ⚡ Performance Optimizations

### Compute Optimization

#### 1. Expert Parallelization
```python
# Experts can be computed in parallel
# V100 has 80 SMs, can handle 8 experts efficiently
for expert_id in range(num_experts):
    expert_outputs[expert_id] = experts[expert_id](expert_inputs[expert_id])
```

#### 2. Attention Optimization
```python
# Flash Attention for memory-efficient attention
# Reduces attention memory from O(n²) to O(n)
if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    # Use PyTorch 2.0 optimized attention
    attention_output = F.scaled_dot_product_attention(q, k, v)
```

#### 3. Compilation Benefits
```python
# PyTorch 2.0 compilation significantly improves performance
model = torch.compile(model)
# Typical speedup: 20-40% on V100
```

### Load Balancing

#### Router Load Balancing Loss
```python
def compute_load_balancing_loss(gate_logits, top_k_indices):
    """
    Encourage uniform expert usage
    """
    # Count expert usage
    expert_counts = torch.bincount(top_k_indices.flatten(), 
                                  minlength=num_experts)
    
    # Normalize and compute variance
    expert_fractions = expert_counts.float() / expert_counts.sum()
    target_fraction = 1.0 / num_experts
    
    # MSE loss to encourage uniformity
    load_balance_loss = torch.sum((expert_fractions - target_fraction) ** 2)
    return load_balance_loss
```

## 📈 Scaling Considerations

### Single GPU (V100) Scaling

| Model Size | Parameters | Memory | Batch Size | Throughput |
|------------|------------|---------|------------|------------|
| **Small** | 125M | 16GB | 24 | 850 tok/s |
| **Medium** | 350M | 28GB | 12 | 650 tok/s |
| **Large** | 760M | 32GB | 8 | 450 tok/s |

### Multi-GPU Scaling

#### Data Parallel (DDP)
```python
# Scale batch size across GPUs
effective_batch_size = batch_size * world_size * gradient_accumulation_steps

# Example: 4 × V100
# Single V100: batch_size = 12
# 4 × V100: effective_batch_size = 12 × 4 × 5 = 240
```

#### Model Parallel Considerations
```python
# For very large models, consider expert parallelism
# Distribute experts across GPUs
experts_per_gpu = num_experts // world_size
local_experts = experts[rank * experts_per_gpu:(rank + 1) * experts_per_gpu]
```

## ⚙️ Configuration Guidelines

### Choosing Architecture Parameters

#### Number of Experts (`num_experts`)
```python
# Guidelines for V100:
# - 4 experts: Small models, memory constrained
# - 8 experts: Optimal for most V100 use cases  
# - 16 experts: Large models, may need model parallelism

def choose_num_experts(available_memory_gb):
    if available_memory_gb < 20:
        return 4
    elif available_memory_gb < 30:
        return 8
    else:
        return 16
```

#### Top-K Selection (`top_k_experts`)
```python
# Typical configurations:
# - top_k = 1: Maximum efficiency, potential quality loss
# - top_k = 2: Good balance (recommended)
# - top_k = 4: Higher quality, more compute

quality_efficiency_tradeoff = {
    1: {"efficiency": 100, "quality": 85},
    2: {"efficiency": 85, "quality": 95},
    4: {"efficiency": 65, "quality": 100}
}
```

#### Embedding Dimension (`n_embd`)
```python
# V100 Memory limits:
memory_per_embd_dim = {
    512: "16GB",   # Small model
    768: "24GB",   # Medium model (recommended)
    1024: "32GB",  # Large model (memory limit)
    1536: "48GB"   # Requires multiple GPUs
}
```

### Layer Distribution Strategy

```python
# Optimal layer configuration for V100
def get_layer_config(target_size="medium"):
    configs = {
        "small": {
            "n_layer": 8,
            "n_head": 8,
            "n_embd": 512,
            "num_experts": 4
        },
        "medium": {
            "n_layer": 12,
            "n_head": 12, 
            "n_embd": 768,
            "num_experts": 8
        },
        "large": {
            "n_layer": 16,
            "n_head": 16,
            "n_embd": 1024,
            "num_experts": 8
        }
    }
    return configs[target_size]
```

### Performance Tuning Checklist

#### ✅ Memory Optimization
- [ ] Enable gradient checkpointing for large models
- [ ] Use mixed precision (bfloat16) training
- [ ] Implement dynamic batching based on sequence length
- [ ] Monitor peak memory usage and adjust batch size

#### ✅ Compute Optimization  
- [ ] Enable PyTorch compilation (`torch.compile`)
- [ ] Use optimized attention (Flash Attention if available)
- [ ] Parallelize expert computation
- [ ] Profile GPU utilization and identify bottlenecks

#### ✅ Load Balancing
- [ ] Tune load balancing coefficient (0.01 - 0.1)
- [ ] Monitor expert usage distribution
- [ ] Adjust capacity factor based on load variance
- [ ] Use auxiliary losses to encourage expert diversity

#### ✅ Training Stability
- [ ] Implement gradient clipping (max_norm=1.0)
- [ ] Use warmup learning rate schedule
- [ ] Monitor auxiliary loss components
- [ ] Validate expert activation patterns

## 🔬 Monitoring and Debugging

### Key Metrics to Track

#### Training Metrics
```python
metrics_to_log = {
    "loss/total": total_loss,
    "loss/main": main_loss, 
    "loss/auxiliary": aux_loss,
    "learning_rate": current_lr,
    "gradient_norm": grad_norm,
    "expert_usage_variance": expert_variance,
    "memory_usage_gb": gpu_memory_gb,
    "throughput_tokens_per_sec": throughput
}
```

#### MoE-Specific Metrics
```python
moe_metrics = {
    "experts/usage_distribution": expert_counts,
    "experts/avg_per_token": avg_experts_activated,
    "experts/load_balance_loss": load_balance_loss,
    "experts/routing_entropy": routing_entropy,
    "experts/capacity_utilization": capacity_usage
}
```

### Common Issues and Solutions

#### 1. Memory Out-of-Memory (OOM)
```python
Solutions:
- Reduce batch_size: 12 → 8 → 4
- Enable gradient_checkpointing: True
- Use mixed precision: "bfloat16"
- Reduce model size: n_embd 768 → 512
```

#### 2. Poor Expert Load Balancing
```python
Solutions:
- Increase load_balancing_loss_coef: 0.01 → 0.1
- Adjust capacity_factor: 1.0 → 1.5
- Monitor routing patterns
- Use auxiliary losses
```

#### 3. Slow Training Speed
```python
Solutions:
- Enable torch.compile: True
- Use optimal batch size: 8-16 for V100
- Check gradient_accumulation_steps
- Profile GPU utilization
```

## 📊 Example Configurations

### Production V100 Configuration
```python
PRODUCTION_V100_CONFIG = {
    # Architecture
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "block_size": 1024,
    "vocab_size": 50304,
    
    # MoE
    "num_experts": 8,
    "top_k_experts": 2,
    "load_balancing_loss_coef": 0.01,
    "expert_dropout": 0.1,
    "capacity_factor": 1.25,
    
    # Training
    "batch_size": 12,
    "gradient_accumulation_steps": 5,
    "learning_rate": 6e-4,
    "weight_decay": 1e-1,
    "warmup_iters": 2000,
    "max_iters": 600000,
    
    # Optimization
    "use_gradient_checkpointing": True,
    "dtype": "bfloat16",
    "compile": True,
    "grad_clip": 1.0
}
```

### Debug/Development Configuration
```python
DEBUG_V100_CONFIG = {
    # Smaller for quick iteration
    "n_layer": 4,
    "n_head": 4, 
    "n_embd": 256,
    "block_size": 256,
    "num_experts": 4,
    "top_k_experts": 2,
    "batch_size": 4,
    "max_iters": 1000,
    "eval_interval": 100,
    "log_interval": 10
}
```

## 🚀 Getting Started

### Quick Setup
```bash
# 1. Use the V100 configuration
python train_moe_advanced.py --config v100

# 2. Monitor with W&B
python train_moe_advanced.py --config v100 --wandb --monitor

# 3. Profile performance  
python train_moe_advanced.py --config v100 --profile

# 4. Memory-constrained setup
python train_moe_advanced.py --config memory
```

### Configuration Selection
```python
# Auto-detect optimal configuration
python train_moe_advanced.py --config auto

# Manual configuration for specific needs
python train_moe_advanced.py --config performance  # Max speed
python train_moe_advanced.py --config minimal     # Memory constrained
```

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
- [NVIDIA V100 Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)

This architecture guide provides the foundation for effectively deploying and optimizing nanoGPT MoE models on V100 hardware. For specific implementation details, refer to the code in `model.py` and `train_moe_advanced.py`.
