# SFT configuration
import time

# --- Dataset settings ---
dataset = 'sft_dataset'  # Replace with your SFT dataset name
gradient_accumulation_steps = 1
batch_size = 8
block_size = 256

# --- Model settings ---
# Initialize from a pretrained model
init_from = 'resume'  # Load weights from a checkpoint
out_dir = 'checkpoints/out-sft'   # Output directory for SFT checkpoints
# Path to the pretrained checkpoint
resume_from = 'checkpoints/out-moe-v100/best_model.pt'  # Use the best pretrained model
# --- Training settings ---
eval_interval = 250
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
wandb_log = True
wandb_project = 'nanoGPT-SFT'
wandb_run_name = 'sft-run-' + str(time.time())

# --- Optimizer settings ---
learning_rate = 1e-5
max_iters = 1000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 50
lr_decay_iters = 1000
min_lr = 1e-6

# --- Hardware compatibility ---
compile = False  # Disable compilation to avoid Triton/PTX errors
dtype = 'float32'  # Use float32 for stable fine-tuning
