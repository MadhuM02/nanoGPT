# Memory-optimized SFT configuration for distributed training
import time

# --- Dataset settings ---
dataset = 'sft_dataset'  # Large multi-task dataset
gradient_accumulation_steps = 8  # Higher accumulation, lower per-step memory
batch_size = 1  # Minimal batch size per GPU
block_size = 256  # Reduced context length to save memory

# --- Model settings ---
# Initialize from a pretrained model
init_from = 'resume'  # Load weights from a checkpoint
out_dir = 'out-sft-distributed'   # Output directory for distributed SFT
# Path to the pretrained checkpoint
resume_from = 'out-moe-v100/best_model.pt'  # Use the best pretrained model

# --- Training settings ---
eval_interval = 50   # Evaluate more frequently but with fewer examples
log_interval = 1     # Log every step to monitor memory
eval_iters = 50      # Fewer evaluation iterations to save memory
eval_only = False
always_save_checkpoint = True
wandb_log = False    # Disable wandb to save memory
wandb_project = 'nanoGPT-SFT-Distributed'
wandb_run_name = 'distributed-sft-run-' + str(time.time())

# --- Optimizer settings (memory efficient) ---
learning_rate = 5e-6  # Lower learning rate for stable fine-tuning
max_iters = 100       # Shorter test run
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 10     # Short warmup
lr_decay_iters = 80   # Most of the run
min_lr = 5e-7

# --- Hardware compatibility (memory optimized) ---
compile = False      # Disable compilation to save memory
dtype = 'float16'    # Use float16 for memory efficiency
backend = 'nccl'     # Use NCCL backend for multi-GPU communication

# Additional memory optimizations
use_gradient_checkpointing = True  # Trade compute for memory
