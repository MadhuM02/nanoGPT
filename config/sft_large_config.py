# Large SFT configuration - optimized for diverse multi-task dataset
import time

# --- Dataset settings ---
dataset = 'sft_dataset'  # Large multi-task dataset
gradient_accumulation_steps = 4  # Increase accumulation to maintain effective batch size
batch_size = 1  # Reduce batch size per GPU to fit in memory
block_size = 512  # Match the pretrained model's block size

# --- Model settings ---
# Initialize from a pretrained model
init_from = 'resume'  # Load weights from a checkpoint
out_dir = 'checkpoints/out-sft-large'   # Output directory for large SFT checkpoints
# Path to the pretrained checkpoint
resume_from = 'checkpoints/out-moe-v100/best_model.pt'  # Use the best pretrained model

# --- Training settings ---
eval_interval = 100  # Evaluate less frequently for efficiency with longer training
log_interval = 50     # Log more frequently to monitor progress
eval_iters = 200      # More evaluation iterations for better estimates
eval_only = False
always_save_checkpoint = True
wandb_log = True
wandb_project = 'nanoGPT-SFT-Large'
wandb_run_name = 'large-sft-run-' + str(time.time())

# --- Optimizer settings (stabilized for loss explosion prevention) ---
learning_rate = 1e-6  # Reduced from 5e-6 to prevent loss spikes
max_iters = 5000     # Much longer training - multiple epochs
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 0.5       # Reduced from 1.0 for more aggressive gradient clipping
decay_lr = True
warmup_iters = 1000   # Increased from 500 for more stable warmup
lr_decay_iters = 10000 # Start decay earlier to avoid instabilities
min_lr = 1e-7         # Reduced proportionally

# --- Hardware compatibility ---
compile = False  # Keep disabled for stability in multi-GPU
dtype = 'float16'  # Use float16 for memory efficiency on V100s
backend = 'nccl'  # Use NCCL backend for multi-GPU communication

# --- Performance optimizations ---
# Consider these settings for faster training:
# dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
# compile = True  # Enable if no Triton errors
