"""
Simport os
import time
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Set CUDA memory allocation configuration to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'd Fine-Tuning) training script for MoE models
Based on train_moe_advanced.py but with configurator.py support
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-sft'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'resume' # 'scratch' or 'resume' or 'gpt2*'
resume_from = 'out-moe-v100/best_model.pt'  # path to checkpoint to resume from
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
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# Debug: Print key config values
print(f"DEBUG: gradient_accumulation_steps = {gradient_accumulation_steps}")
print(f"DEBUG: batch_size = {batch_size}")
print(f"DEBUG: block_size = {block_size}")
print(f"DEBUG: config file used: {config.get('config', 'default')}")
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    try:
        init_process_group(backend=backend)
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

# Load examples once at startup
train_examples = []
val_examples = []

# Initialize tokenizer once globally to avoid memory leaks
import tiktoken
enc = tiktoken.get_encoding("gpt2")

def load_examples():
    """Load all examples from text files"""
    global train_examples, val_examples
    
    # Load training examples
    train_path = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding='utf-8') as f:
            train_examples = [line.strip() for line in f if line.strip()]
    
    # Load validation examples  
    val_path = os.path.join(data_dir, 'val.txt')
    if os.path.exists(val_path):
        with open(val_path, 'r', encoding='utf-8') as f:
            val_examples = [line.strip() for line in f if line.strip()]
    
    # Only master process should print to avoid spam in distributed training
    if master_process:
        print(f"📚 Loaded {len(train_examples)} training examples, {len(val_examples)} validation examples")

def get_batch(split):
    """Get a batch of complete examples for SFT training"""
    # Use global tokenizer to avoid repeated imports/creations
    global enc
    
    # Choose examples based on split
    examples = train_examples if split == 'train' else val_examples
    
    if len(examples) == 0:
        # Fallback to binary data if text examples not available
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    else:
        # Sample random examples
        sampled_examples = np.random.choice(examples, size=batch_size, replace=True)
        
        x_list = []
        y_list = []
        
        for example in sampled_examples:
            # Tokenize the example
            tokens = enc.encode_ordinary(example)
            
            # Ensure we have at least one token
            if len(tokens) == 0:
                tokens = [enc.eot_token]
            
            # Add end token to mark end of sequence
            tokens = tokens + [enc.eot_token]
            
            # Truncate or pad to exactly block_size
            if len(tokens) >= block_size:
                # Truncate to exactly block_size
                tokens = tokens[:block_size]
            else:
                # Pad with end tokens to reach block_size
                pad_length = block_size - len(tokens)
                tokens = tokens + [enc.eot_token] * pad_length
            
            # Create input (x) and target (y) sequences
            # x: tokens[:-1], y: tokens[1:]
            # But we need both to be exactly block_size
            x_tokens = tokens[:]  # Copy the full sequence
            y_tokens = tokens[1:] + [enc.eot_token]  # Shift by 1 and add eot at end
            
            # Ensure exactly block_size for both
            x_tokens = x_tokens[:block_size]
            y_tokens = y_tokens[:block_size]
            
            # Final safety check - pad if somehow still short
            while len(x_tokens) < block_size:
                x_tokens.append(enc.eot_token)
            while len(y_tokens) < block_size:
                y_tokens.append(enc.eot_token)
            
            x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
            y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        # Clean up intermediate lists to free memory
        del x_list, y_list
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

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
load_examples()

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,
                  num_experts=num_experts, top_k_experts=top_k_experts,
                  load_balancing_loss_coef=load_balancing_loss_coef,
                  expert_dropout=expert_dropout, capacity_factor=capacity_factor,
                  expert_activation=expert_activation) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    if master_process:
        print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        if master_process:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {resume_from}")
    # resume training from a checkpoint
    ckpt_path = resume_from
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_args' in checkpoint:
        checkpoint_model_args = checkpoint['model_args']
    elif 'config' in checkpoint:
        checkpoint_model_args = checkpoint['config']
    else:
        raise ValueError("Cannot find model configuration in checkpoint")
    
    if master_process:
        print(f"Checkpoint model args: {checkpoint_model_args}")
    
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # MoE specific parameters
    for k in ['num_experts', 'top_k_experts', 'expert_activation']:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    
    # Ensure vocab_size is set - detect from the actual model weights
    state_dict = checkpoint['model']
    
    # Check embedding layer size to infer vocab_size
    for key in state_dict.keys():
        if key.endswith('transformer.wte.weight') or key.endswith('lm_head.weight'):
            checkpoint_vocab_size = state_dict[key].shape[0]
            model_args['vocab_size'] = checkpoint_vocab_size
            if master_process:
                print(f"Detected vocab_size from checkpoint weights: {checkpoint_vocab_size}")
            break
    
    # Fallback if we couldn't detect from weights
    if 'vocab_size' not in model_args or model_args['vocab_size'] is None:
        if 'vocab_size' in checkpoint_model_args and checkpoint_model_args['vocab_size'] is not None:
            model_args['vocab_size'] = checkpoint_model_args['vocab_size']
            if master_process:
                print(f"Using checkpoint vocab_size: {model_args['vocab_size']}")
        elif meta_vocab_size is not None:
            model_args['vocab_size'] = meta_vocab_size
            if master_process:
                print(f"Using dataset vocab_size: {meta_vocab_size}")
        else:
            model_args['vocab_size'] = 50304  # Default GPT-2 vocab size
            if master_process:
                print(f"Using default vocab_size: 50304")
    
    if master_process:
        print(f"Final model args: {model_args}")
    
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    
    # Now load the state dict
    state_dict = checkpoint['model']
    # fix the keys of the state dict :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefixes = ['module._orig_mod.', '_orig_mod.', 'module.']
    for prefix in unwanted_prefixes:
        keys_to_fix = [k for k in state_dict.keys() if k.startswith(prefix)]
        if keys_to_fix:
            if master_process:
                print(f"Removing {len(keys_to_fix)} keys with prefix '{prefix}'")
            for k in keys_to_fix:
                new_key = k[len(prefix):]
                state_dict[new_key] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    
    # Handle different checkpoint formats for iteration tracking
    if 'iter_num' in checkpoint:
        iter_num = checkpoint['iter_num']
    elif 'step' in checkpoint:
        iter_num = checkpoint['step']
    else:
        iter_num = 0
        
    best_val_loss = checkpoint.get('best_val_loss', 1e9)
elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

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
if init_from == 'resume' and 'optimizer' in checkpoint:
    # Skip optimizer loading to avoid dtype mismatches - start with fresh optimizer
    if master_process:
        print("⚠️  Skipping optimizer state loading to avoid dtype mismatches")
        print("Starting with fresh optimizer state for fine-tuning")
    iter_num = 0
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
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
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

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            # Clean up intermediate tensors
            del X, Y, logits, loss
        out[split] = losses.mean()
        # Force cleanup after each split (only on master process to avoid sync issues)
        if torch.cuda.is_available() and master_process:
            torch.cuda.empty_cache()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
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
        # Ensure all processes are synchronized before evaluation
        if ddp:
            try:
                torch.distributed.barrier()
            except Exception as e:
                if master_process:
                    print(f"Warning: Distributed barrier failed: {e}")
                    print("Continuing without synchronization...")
        
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
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                # Clean up checkpoint dict to free memory (only on master process)
                del checkpoint
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        
        # Clean up intermediate tensors to prevent memory accumulation
        del logits
        if micro_step == gradient_accumulation_steps - 1:
            # Force cleanup after the last micro step
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # Memory monitoring (only on master process to avoid distributed issues)
        if torch.cuda.is_available() and master_process:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, "
                  f"mem {memory_allocated:.2f}/{memory_reserved:.2f}GB")
        elif master_process:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Force cleanup every 10 steps to prevent memory accumulation (all processes)
        if iter_num % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
