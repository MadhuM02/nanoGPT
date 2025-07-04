# Configuration for SFT training with extra large dataset
# This config is optimized for the extra large multi-task dataset

# wandb logging
wandb_log = True
wandb_project = 'nanoGPT-SFT-XL'
wandb_run_name = 'sft-extra-large'

# data
dataset = 'sft_dataset_xl'
gradient_accumulation_steps = 16  # Increased for stability with larger dataset
batch_size = 2  # Smaller batch size to fit in memory
block_size = 512  # Keep same context length

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # No dropout for fine-tuning
bias = False

# MoE configuration
num_experts = 8
top_k_experts = 2
load_balancing_loss_coef = 0.01
expert_dropout = 0.0  # No dropout for fine-tuning
capacity_factor = 1.0
expert_activation = 'gelu'

# optimization
learning_rate = 1e-4  # Lower learning rate for fine-tuning
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
dtype = 'bfloat16'
compile = True

# learning rate schedule
decay_lr = True
warmup_iters = 1000  # Shorter warmup for fine-tuning
lr_decay_iters = 50000  # Longer decay for extra large dataset
min_lr = 1e-5

# evaluation
eval_interval = 500  # More frequent evaluation
log_interval = 10
eval_iters = 100
eval_only = False

# checkpointing
out_dir = 'out-sft-xl'
checkpoint_interval = 2000
always_save_checkpoint = True
init_from = 'resume'  # Resume from existing checkpoint

# training
max_iters = 50000  # More iterations for extra large dataset
backend = 'nccl'

# system
device = 'cuda'
device_type = 'cuda'
