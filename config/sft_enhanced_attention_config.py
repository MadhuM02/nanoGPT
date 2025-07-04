# Enhanced SFT configuration with instruction-aware attention
# Optimized for supervised fine-tuning with proper attention patterns

# Model configuration
sft_mode = True  # Enable SFT-specific attention patterns
instruction_attention_weight = 1.5  # Boost attention from response to instruction tokens
response_attention_dampening = 0.8  # Reduce attention within response tokens
sft_head_specialization = True  # Enable attention head specialization (requires n_head % 3 == 0)

# Distributed training
init_from = 'resume'  # 'resume' to continue training, 'gpt2*' for fine-tuning from GPT-2
out_dir = 'checkpoints/out-sft-enhanced'
eval_interval = 200
log_interval = 10
eval_iters = 50
eval_only = False

# Data
dataset = 'sft_dataset'
gradient_accumulation_steps = 16  # Increase for stable distributed training
batch_size = 4  # Per-device batch size
block_size = 512  # Sequence length

# Model  
n_layer = 12
n_head = 12  # Must be divisible by 3 for head specialization
n_embd = 768
dropout = 0.1  # Slightly higher dropout for SFT
bias = False

# AdamW optimizer with conservative settings for SFT
learning_rate = 3e-5  # Lower LR for fine-tuning stability
max_iters = 5000
weight_decay = 0.01
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5  # Aggressive gradient clipping for stability

# Learning rate decay
decay_lr = True
warmup_iters = 300  # Longer warmup for SFT
lr_decay_iters = 5000
min_lr = 3e-6  # 10% of max learning rate

# DDP settings
backend = 'nccl'

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Memory optimization
gradient_checkpointing = True

# Enhanced attention logging
log_attention_stats = True  # Log attention pattern statistics during training

# MoE configuration (optional)
num_experts = 1  # Set to > 1 to enable MoE
top_k_experts = 1
load_balancing_loss_coef = 0.01
expert_dropout = 0.0
capacity_factor = 1.0
expert_activation = 'gelu'

# SFT-specific loss configuration
instruction_loss_weight = 0.0  # Don't compute loss on instruction tokens
response_loss_weight = 1.0  # Full loss weight on response tokens
