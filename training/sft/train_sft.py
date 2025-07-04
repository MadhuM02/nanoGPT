"""
SFT (Supervised Fine-Tuning) training script for MoE models
Based on train_moe_advanced.py but with configurator.py support
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model import GPTConfig, GPT
from training.utils.checkpoint_utils import (
    load_checkpoint_and_create_model, 
    save_checkpoint, 
    save_best_checkpoint,
    load_optimizer_state,
    get_model_info
)
from training.utils.sft_utils import create_sft_utilities

# Set CUDA memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'checkpoints/out-sft'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
resume_from = 'checkpoints/out-moe-v100/best_model.pt'  # path to checkpoint to resume from
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'nanoGPT-SFT'
wandb_run_name = 'sft-' + str(int(time.time()))
# data
dataset = 'sft_dataset'
gradient_accumulation_steps = 5 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512  # Match the pretrained model's block size
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# MoE specific
num_experts = 8
top_k_experts = 2
load_balancing_loss_coef = 0.01
expert_dropout = 0.0
capacity_factor = 1.0
expert_activation = 'gelu'
# adamw optimizer
learning_rate = 1e-5  # Lower learning rate for fine-tuning
max_iters = 2000      # Fewer iterations for fine-tuning
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 2000 # should be ~= max_iters per Chinchilla
min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(os.path.dirname(__file__), '..', 'utils', 'configurator.py')).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    try:
        # Initialize process group with timeout
        import datetime
        init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        print(f"✅ DDP initialized: rank {ddp_rank}/{ddp_world_size} on device {device}")
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        if master_process:
            print(f"DDP: gradient_accumulation_steps={gradient_accumulation_steps}, ddp_world_size={ddp_world_size}")
        
        # Auto-adjust gradient_accumulation_steps if not divisible by ddp_world_size
        if gradient_accumulation_steps % ddp_world_size != 0:
            original_steps = gradient_accumulation_steps
            # Round up to the nearest multiple of ddp_world_size
            gradient_accumulation_steps = ((gradient_accumulation_steps + ddp_world_size - 1) // ddp_world_size) * ddp_world_size
            if master_process:
                print(f"⚠️  Auto-adjusted gradient_accumulation_steps from {original_steps} to {gradient_accumulation_steps} to be divisible by {ddp_world_size}")
        
        gradient_accumulation_steps //= ddp_world_size
        if master_process:
            print(f"Adjusted gradient_accumulation_steps to {gradient_accumulation_steps} for DDP")
    except Exception as e:
        print(f"❌ Failed to initialize distributed training: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        print("Falling back to single GPU training")
        ddp = False
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
# For single GPU training, ensure we're not accidentally triggering DDP
if not ddp and 'RANK' in os.environ:
    # Clear distributed environment variables to avoid conflicts
    for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        if key in os.environ:
            del os.environ[key]
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# SFT data loader - reads complete examples
data_dir = os.path.join('data', dataset)

# Initialize SFT utilities
data_manager, loss_analyzer, training_utils = create_sft_utilities(data_dir)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
loss_history = []  # Track recent losses for spike detection
loss_spike_threshold = 2.0  # If loss increases by this factor, it's a spike

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Load SFT examples from text files
data_manager.load_examples(master_process)

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  num_experts=num_experts, top_k_experts=top_k_experts,
                  load_balancing_loss_coef=load_balancing_loss_coef,
                  expert_dropout=expert_dropout, capacity_factor=capacity_factor,
                  expert_activation=expert_activation) # start with model_args from command line

# Load model using checkpoint utility
model, model_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(
    init_from=init_from,
    resume_from=resume_from,
    model_args=model_args,
    meta_vocab_size=meta_vocab_size,
    device=device,
    master_process=master_process,
    block_size=block_size
)

# Log model information
if master_process:
    model_info = get_model_info(model)
    print(f"✓ Model loaded: {model_info['total_parameters']:,} total params, "
          f"{model_info['trainable_parameters']:,} trainable, "
          f"{model_info['model_size_mb']:.1f}MB")

# Ensure model is on the correct device and dtype before optimizer creation
model.to(device)
if ptdtype != torch.float32:
    model = model.to(dtype=ptdtype)
    if master_process:
        print(f"✓ Model converted to {ptdtype}")

# Force CUDA memory cleanup after model loading and dtype conversion
if device_type == 'cuda':
    torch.cuda.empty_cache()
    if master_process:
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"Memory after model loading: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# Enable gradient checkpointing if specified in config
use_gradient_checkpointing = config.get('use_gradient_checkpointing', False)
if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    if master_process:
        print("✓ Gradient checkpointing enabled")

# initialize a GradScaler. If enabled=False scaler is a no-op
# Only enable for true float16, disabled for float32 and other dtypes
use_scaler = (dtype == 'float16' and ptdtype == torch.float16)
use_scaler = False
try:
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
except Exception as e:
    if master_process:
        print(f"Warning: Could not initialize GradScaler, using fallback. Error: {e}")
    use_scaler = False
    scaler = torch.cuda.amp.GradScaler(enabled=False)
if master_process:
    print(f"GradScaler enabled: {use_scaler} (dtype: {dtype}, ptdtype: {ptdtype})")

# optimizer - create after model is in final dtype
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state from checkpoint if available
if init_from == 'resume' and checkpoint is not None:
    optimizer_loaded = load_optimizer_state(checkpoint, optimizer, master_process, skip_on_dtype_mismatch=True)
    if not optimizer_loaded:
        iter_num = 0  # Reset iteration if optimizer couldn't be loaded

checkpoint = None # free up memory

# compile the model
if compile:
    if master_process:
        print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    try:
        # Ensure model is in the right state before DDP wrapping
        if master_process:
            print(f"Wrapping model in DDP on device {ddp_local_rank}...")
        # Use broadcast_buffers=False and find_unused_parameters=False for better performance
        model = DDP(model, device_ids=[ddp_local_rank], 
                   broadcast_buffers=False, find_unused_parameters=True)
        if master_process:
            print(f"✅ Model wrapped in DDP with device {ddp_local_rank}")
        
        # Force memory cleanup after DDP wrapping
        if device_type == 'cuda':
            torch.cuda.empty_cache()
            if master_process:
                allocated = torch.cuda.memory_allocated(device) / 1024**3
                reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"Memory after DDP setup: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"❌ Could not wrap model in DDP: {e}")
        print("Falling back to single GPU training")
        ddp = False
        master_process = True  # Reset master process for fallback

# Create estimate loss function using utilities
estimate_loss = training_utils.create_estimate_loss_fn(
    model, data_manager, eval_iters, batch_size, block_size, device, device_type, ctx
)

# learning rate decay scheduler (cosine with warmup)  
def get_lr(it):
    return training_utils.get_cosine_lr_with_warmup(
        it, learning_rate, warmup_iters, lr_decay_iters, min_lr
    )

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
batch_data = data_manager.get_batch('train', batch_size, block_size, device, device_type)  # fetch the very first batch
if len(batch_data) == 3:
    X, Y, loss_mask = batch_data  # SFT data with loss mask
    using_sft_masking = True
else:
    X, Y = batch_data  # Binary data fallback
    loss_mask = None
    using_sft_masking = False

if master_process and using_sft_masking:
    print("✅ Using SFT loss masking - only computing loss on response tokens")
elif master_process:
    print("⚠️  Using standard loss computation - computing loss on all tokens")

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                # Save regular checkpoint
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    model_args=model_args,
                    iter_num=iter_num,
                    best_val_loss=best_val_loss,
                    config=config,
                    out_dir=out_dir,
                    checkpoint_name='ckpt.pt',
                    master_process=master_process
                )
                
                # Save best model checkpoint (without optimizer for deployment)
                if losses['val'] < best_val_loss:
                    save_best_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        model_args=model_args,
                        iter_num=iter_num,
                        val_loss=losses['val'],
                        config=config,
                        out_dir=out_dir,
                        master_process=master_process
                    )
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of model.no_sync() it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            
            # Detailed logging for loss explosion debugging
            if master_process and (iter_num % log_interval == 0 or loss.item() > 10.0):
                # Compute per-sample losses for detailed analysis
                with torch.no_grad():
                    per_sample_losses = loss_analyzer.compute_per_sample_loss(logits, Y)
                    
                    # Log samples if loss is high or at regular intervals
                    if loss.item() > 10.0 or iter_num % (log_interval * 10) == 0:
                        print(f"\n🚨 DETAILED SAMPLE ANALYSIS - Loss: {loss.item():.4f}")
                        
                        # Show loss masking statistics
                        if using_sft_masking and loss_mask is not None:
                            total_tokens = Y.numel()
                            masked_tokens = (Y == -1).sum().item()
                            response_tokens = total_tokens - masked_tokens
                            mask_ratio = masked_tokens / total_tokens * 100
                            print(f"   📊 Loss masking: {masked_tokens}/{total_tokens} tokens masked ({mask_ratio:.1f}%)")
                            print(f"   📊 Computing loss on {response_tokens} response tokens only")
                        
                        loss_analyzer.log_training_samples(X, Y, per_sample_losses, iter_num, 
                                                         sample_limit=2, master_process=master_process)
                        
                        # Additional loss analysis
                        max_loss_idx = torch.argmax(per_sample_losses)
                        min_loss_idx = torch.argmin(per_sample_losses)
                        print(f"\n   Loss statistics:")
                        print(f"     Mean per-sample loss: {per_sample_losses.mean().item():.4f}")
                        print(f"     Max per-sample loss: {per_sample_losses.max().item():.4f} (sample {max_loss_idx})")
                        print(f"     Min per-sample loss: {per_sample_losses.min().item():.4f} (sample {min_loss_idx})")
                        print(f"     Loss std: {per_sample_losses.std().item():.4f}")
                        
                        # Check for NaN or inf in logits
                        nan_count = torch.isnan(logits).sum().item()
                        inf_count = torch.isinf(logits).sum().item()
                        if nan_count > 0 or inf_count > 0:
                            print(f"     ⚠️  CRITICAL: NaN count: {nan_count}, Inf count: {inf_count}")
            
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        batch_data = data_manager.get_batch('train', batch_size, block_size, device, device_type)
        if len(batch_data) == 3:
            X, Y, loss_mask = batch_data  # SFT data with loss mask
        else:
            X, Y = batch_data  # Binary data fallback
            loss_mask = None
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        # Clean up intermediate tensors to prevent memory accumulation
        del logits
    # clip the gradient
    if grad_clip != 0.0:
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        
        # Loss spike detection and recovery
        if training_utils.detect_loss_spike(loss_history, lossf, loss_spike_threshold):
            print(f"⚠️  Loss spike detected! Current: {lossf:.4f}, Recent avg: {sum(loss_history[-10:]) / min(len(loss_history), 10):.4f}")
            print(f"🔧 Reducing learning rate temporarily...")
            # Temporarily reduce learning rate
            training_utils.apply_spike_recovery(optimizer, recovery_factor=0.1)
            # Force a CUDA cache cleanup to help with potential memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Track loss history
        loss_history.append(lossf)
        if len(loss_history) > 50:  # Keep only recent history
            loss_history.pop(0)
        
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Memory monitoring (only on master process to avoid distributed issues)
        if torch.cuda.is_available() and master_process:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            current_lr = optimizer.param_groups[0]['lr']
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, "
                  f"mem {memory_allocated:.2f}/{memory_reserved:.2f}GB, lr {current_lr:.2e}")
        elif master_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, lr {current_lr:.2e}")
        
        # Less frequent cleanup to avoid interfering with NCCL communications
        if iter_num % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
