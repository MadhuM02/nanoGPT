# Ultra memory-optimized SFT configuration for distributed training
# Designed to work even on heavily constrained memory scenarios
import time

# --- Dataset settings ---
dataset = 'sft_dataset'  # Large multi-task dataset
gradient_accumulation_steps = 8  # High accumulation, minimal per-step memory (must be divisible by number of GPUs)
batch_size = 1  # Minimal batch size per GPU
block_size = 128  # Very small context length to save memory

# --- Model settings ---
# Initialize from a pretrained model
init_from = 'resume'  # Load weights from a checkpoint
out_dir = 'checkpoints/out-sft-ultra-small'   # Output directory for ultra small SFT
# Path to the pretrained checkpoint
resume_from = 'checkpoints/out-moe-v100/best_model.pt'  # Use the best pretrained model

# --- Training settings ---
eval_interval = 100   # Evaluate less frequently
log_interval = 1     # Log every step to monitor memory
eval_iters = 20      # Very few evaluation iterations to save memory
eval_only = False
always_save_checkpoint = True
wandb_log = False    # Disable wandb to save memory
wandb_project = 'nanoGPT-SFT-UltraSmall'
wandb_run_name = 'ultra-small-sft-run-' + str(time.time())

# --- Optimizer settings (memory efficient) ---
learning_rate = 1e-6  # Very low learning rate for stable fine-tuning
max_iters = 50       # Very short test run
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 5     # Very short warmup
lr_decay_iters = 40   # Most of the run
min_lr = 1e-7

# --- Hardware compatibility (ultra memory optimized) ---
compile = False      # Disable compilation to save memory and avoid DDP issues
dtype = 'float16'    # Use float16 for memory efficiency
backend = 'nccl'     # Use NCCL backend for multi-GPU communication

# Additional memory optimizations
use_gradient_checkpointing = True  # Trade compute for memory

# Additional ultra-conservative settings
max_seq_len = 128    # Match block_size
vocab_size = 50257   # GPT-2 vocab size, ensure this matches the model
