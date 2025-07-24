# Training Directory Structure

This directory contains all training scripts organized by training type and functionality.

## 📁 Directory Structure

### `/standard/` - Standard GPT Training
- `train.py` - Main GPT training script (original nanoGPT)

### `/moe/` - Mixture-of-Experts Training
- `train_moe_advanced.py` - Advanced MoE training with optimizations
- `moe_monitor.py` - MoE training monitoring utilities
- `moe_optimizations.py` - MoE performance optimizations

### `/sft/` - Supervised Fine-Tuning
- `train_sft.py` - SFT training script with loss masking
- `sft_attention.py` - SFT-specific attention implementations
- `sft_utils.py` - SFT utility functions and configs
- `sft_position_encoding.py` - Position encoding for SFT
- `test_sft_distributed.sh` - Distributed SFT testing script

### `/utils/` - Training Utilities
- `configurator.py` - Configuration management utility
- `bench.py` - Benchmarking and performance testing
- `v100_config.py` - V100 GPU-specific configurations

## 🚀 How to Run Training

### Standard GPT Training
```bash
# From project root
python training/standard/train.py

# With custom config
python training/standard/train.py config/train_gpt2.py
```

### MoE Training
```bash
# Basic MoE training
python training/moe/train_moe_advanced.py

# With monitoring
python training/moe/train_moe_advanced.py --wandb_log=True

# Monitor during training
python training/moe/moe_monitor.py
```

### SFT Training
```bash
# Single GPU SFT
python training/sft/train_sft.py config/sft_config.py

# Distributed SFT
bash training/sft/test_sft_distributed.sh

# With custom SFT config
python training/sft/train_sft.py config/sft_stable_config.py
```

### Benchmarking
```bash
# Performance benchmarking
python training/utils/bench.py

# V100-specific setup
python training/utils/v100_config.py
```

## 📝 Configuration Files

Training configurations are still located in `/config/`:
- `config/train_gpt2.py` - Standard GPT-2 training
- `config/train_shakespeare_char.py` - Character-level training
- `config/sft_config.py` - SFT training configuration
- `config/sft_stable_config.py` - Stable SFT configuration
- `config/sft_distributed_config.py` - Distributed SFT configuration

## 🎯 Training Types

### 1. Standard Training (`/standard/`)
- **Purpose**: Train standard transformer models
- **Use Cases**: Language modeling, pretraining
- **Features**: Standard attention, causal masking

### 2. MoE Training (`/moe/`)
- **Purpose**: Train Mixture-of-Experts models
- **Use Cases**: Large-scale models, efficiency improvements
- **Features**: Expert routing, load balancing, sparsity

### 3. SFT Training (`/sft/`)
- **Purpose**: Supervised fine-tuning on instruction data
- **Use Cases**: Chat models, instruction following
- **Features**: Loss masking, attention modifications, response-only training

## 🔧 Training Features

### Standard Training Features
- ✅ Gradient accumulation
- ✅ Learning rate scheduling
- ✅ Gradient clipping
- ✅ Checkpointing
- ✅ Wandb logging

### MoE Training Features
- ✅ Expert routing with top-k selection
- ✅ Load balancing loss
- ✅ Expert dropout
- ✅ Capacity factor control
- ✅ Expert monitoring

### SFT Training Features
- ✅ Instruction-response loss masking
- ✅ SFT-specific attention patterns
- ✅ Per-sample loss computation
- ✅ Detailed logging and debugging
- ✅ Distributed training support

## 🚀 Quick Start Examples

### Train a small GPT model
```bash
python training/standard/train.py \
    --dataset=shakespeare_char \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=5000
```

### Train MoE model
```bash
python training/moe/train_moe_advanced.py \
    --num_experts=8 \
    --top_k_experts=2 \
    --dataset=openwebtext \
    --max_iters=10000
```

### Fine-tune with SFT
```bash
python training/sft/train_sft.py \
    --init_from=resume \
    --resume_from=out/best_model.pt \
    --dataset=sft_dataset \
    --max_iters=2000
```

## 🛠️ Utilities and Monitoring

### Configuration Management
```bash
# Use configurator for easy config override
python training/standard/train.py config/custom.py --learning_rate=1e-4
```

### Performance Monitoring
```bash
# Benchmark training performance
python training/utils/bench.py

# Monitor MoE expert usage
python training/moe/moe_monitor.py --checkpoint_dir=out-moe/
```

## 📊 Output Directories

Training outputs are organized by type:
- `out/` - Standard training outputs
- `out-moe/` - MoE training outputs  
- `out-sft/` - SFT training outputs
- `out-sft-large/` - Large SFT training outputs

## 🔄 Migration from Old Structure

If you have scripts that reference the old paths:

**Old:**
```bash
python train.py
python train_moe_advanced.py
python train_sft.py
```

**New:**
```bash
python training/standard/train.py
python training/moe/train_moe_advanced.py
python training/sft/train_sft.py
```

## 🎯 Training Goals

This organization provides:
- **Clear Separation**: Different training types in separate folders
- **Easy Discovery**: Related scripts grouped together
- **Scalability**: Easy to add new training types
- **Maintainability**: Logical organization for development
