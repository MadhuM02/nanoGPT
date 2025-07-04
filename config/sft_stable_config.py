# Successful SFT Configuration Summary
# Based on working single-GPU training with stable memory usage and good performance

import time

# --- Dataset settings ---
dataset = 'sft_dataset'  # Large multi-task dataset
gradient_accumulation_steps = 4  # Working well for single GPU
batch_size = 1  # Optimal for memory usage
block_size = 512  # Matches pretrained model

# --- Model settings ---
init_from = 'resume'
out_dir = 'checkpoints/out-sft-stable'
resume_from = 'checkpoints/out-moe-v100/best_model.pt'

# --- Training settings (PROVEN WORKING) ---
eval_interval = 100  # Good balance
log_interval = 50    # Provides good monitoring
eval_iters = 200     # Sufficient for evaluation
eval_only = False
always_save_checkpoint = True
wandb_log = True
wandb_project = 'nanoGPT-SFT-Stable'
wandb_run_name = 'stable-sft-run-' + str(time.time())

# --- Optimizer settings (WORKING WELL) ---
learning_rate = 5e-6  # Showing good loss reduction
max_iters = 15000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 500
lr_decay_iters = 12000
min_lr = 5e-7

# --- Hardware settings (STABLE) ---
compile = False  # Critical for stability
dtype = 'float16'  # Working efficiently
backend = 'nccl'

# --- For Distributed Training (when ready) ---
# Use these settings for multi-GPU:
# gradient_accumulation_steps = 8  # Increase for distributed
# batch_size = 1  # Keep minimal per GPU
# eval_iters = 50  # Reduce for distributed evaluation
