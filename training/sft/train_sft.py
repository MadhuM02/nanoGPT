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
    """Get a batch of complete examples for SFT training with instruction-response awareness"""
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
        instruction_masks = []  # Track instruction vs response regions
        
        for example in sampled_examples:
            # Detect instruction-response boundary markers
            # Common patterns: "Category:", "Sentiment:", "Summary:", etc.
            response_markers = ["Category:", "Sentiment:", "Summary:", "Answer:", "Translation:", "Output:"]
            
            # Find the first response marker
            instruction_end = -1
            for marker in response_markers:
                pos = example.find(marker)
                if pos != -1:
                    instruction_end = pos + len(marker)
                    break
            
            # Tokenize the example
            tokens = enc.encode_ordinary(example)
            
            # Find token boundary for instruction/response split
            instruction_token_end = -1
            if instruction_end > 0:
                # Tokenize up to the instruction end to find token boundary
                instruction_text = example[:instruction_end]
                instruction_tokens = enc.encode_ordinary(instruction_text)
                instruction_token_end = len(instruction_tokens)
            
            # Ensure we have at least one token
            if len(tokens) == 0:
                tokens = [enc.eot_token]
            
            # Add end token to mark end of sequence
            tokens = tokens + [enc.eot_token]
            
            # Create instruction mask (1 for instruction, 0 for response)
            instruction_mask = [1] * len(tokens)
            if instruction_token_end > 0 and instruction_token_end < len(tokens):
                # Mark response tokens
                for i in range(instruction_token_end, len(tokens)):
                    instruction_mask[i] = 0
            
            # Truncate or pad to exactly block_size
            if len(tokens) >= block_size:
                # Truncate to exactly block_size
                tokens = tokens[:block_size]
                instruction_mask = instruction_mask[:block_size]
            else:
                # Pad with end tokens to reach block_size
                pad_length = block_size - len(tokens)
                tokens = tokens + [enc.eot_token] * pad_length
                instruction_mask = instruction_mask + [0] * pad_length  # Padding is response-like
            
            # Create input (x) and target (y) sequences
            # x: tokens[:-1], y: tokens[1:]
            # But we need both to be exactly block_size
            x_tokens = tokens[:]  # Copy the full sequence
            y_tokens = tokens[1:] + [enc.eot_token]  # Shift by 1 and add eot at end
            
            # Ensure exactly block_size for both
            x_tokens = x_tokens[:block_size]
            y_tokens = y_tokens[:block_size]
            instruction_mask = instruction_mask[:block_size]
            
            # Final safety check - pad if somehow still short
            while len(x_tokens) < block_size:
                x_tokens.append(enc.eot_token)
            while len(y_tokens) < block_size:
                y_tokens.append(enc.eot_token)
            while len(instruction_mask) < block_size:
                instruction_mask.append(0)
            
            x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
            y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
            instruction_masks.append(torch.tensor(instruction_mask, dtype=torch.float32))
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        instruction_mask_tensor = torch.stack(instruction_masks)
        
        # Create SFT loss mask: 0 for instruction tokens (no loss), 1 for response tokens (compute loss)
        # instruction_mask: 1 for instruction, 0 for response
        # loss_mask: 0 for instruction, 1 for response (inverted)
        loss_mask = 1.0 - instruction_mask_tensor
        
        # Apply loss mask to targets: set instruction tokens to -1 (ignored in cross_entropy)
        y_masked = y.clone()
        y_masked[loss_mask == 0] = -1  # Mask out instruction tokens
        
        # Clean up intermediate lists to free memory
        del x_list, y_list, instruction_masks
    
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y_masked = y_masked.pin_memory().to(device, non_blocking=True)
        if len(examples) > 0:  # Only return loss_mask for SFT data
            loss_mask = loss_mask.pin_memory().to(device, non_blocking=True)
            return x, y_masked, loss_mask
        else:
            return x, y_masked  # For binary data fallback
    else:
        x, y_masked = x.to(device), y_masked.to(device)
        if len(examples) > 0:
            loss_mask = loss_mask.to(device)
            return x, y_masked, loss_mask
        else:
            return x, y_masked

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

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            batch_data = get_batch(split)
            if len(batch_data) == 3:
                X, Y, loss_mask = batch_data  # SFT data with loss mask
            else:
                X, Y = batch_data  # Binary data fallback
                loss_mask = None
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            # Clean up intermediate tensors
            del X, Y, logits, loss
            if loss_mask is not None:
                del loss_mask
        out[split] = losses.mean()
        # Clean up tensors
        del losses
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

# Utility functions for detailed loss analysis and debugging
def log_training_samples(X, Y, loss_per_sample=None, step=None, sample_limit=3):
    """
    Log detailed information about training samples to debug loss explosion
    
    Args:
        X: Input tensor (batch_size, seq_len)
        Y: Target tensor (batch_size, seq_len)
        loss_per_sample: Optional per-sample losses
        step: Current training step
        sample_limit: Maximum number of samples to log (to avoid spam)
    """
    global enc
    
    if not master_process:  # Only log on master process
        return
        
    batch_size, seq_len = X.shape
    
    print(f"\n📋 Sample Logging (Step {step}):")
    print(f"   Batch shape: {X.shape}, Device: {X.device}")
    print(f"   X range: [{X.min().item()}, {X.max().item()}]")
    print(f"   Y range: [{Y.min().item()}, {Y.max().item()}]")
    
    # Log first few samples in detail
    for i in range(min(sample_limit, batch_size)):
        x_sample = X[i].cpu().numpy()
        y_sample = Y[i].cpu().numpy()
        
        # Check for problematic values
        x_invalid = np.sum((x_sample < 0) | (x_sample >= enc.n_vocab))
        y_invalid = np.sum((y_sample < 0) | (y_sample >= enc.n_vocab))
        
        print(f"\n   Sample {i}:")
        print(f"     X tokens: {x_sample[:10].tolist()}...{x_sample[-10:].tolist()}")
        print(f"     Y tokens: {y_sample[:10].tolist()}...{y_sample[-10:].tolist()}")
        print(f"     Invalid X tokens: {x_invalid}, Invalid Y tokens: {y_invalid}")
        
        if loss_per_sample is not None:
            print(f"     Sample loss: {loss_per_sample[i].item():.4f}")
        
        # Try to decode samples (handle potential encoding errors)
        try:
            x_text = enc.decode(x_sample.tolist()[:50])  # First 50 tokens
            y_text = enc.decode(y_sample.tolist()[:50])
            print(f"     X text: {repr(x_text)}")
            print(f"     Y text: {repr(y_text)}")
        except Exception as e:
            print(f"     Decode error: {e}")
        
        # Check for suspicious patterns
        if len(np.unique(x_sample)) < 5:
            print(f"     ⚠️  WARNING: X sample has very low diversity ({len(np.unique(x_sample))} unique tokens)")
        if len(np.unique(y_sample)) < 5:
            print(f"     ⚠️  WARNING: Y sample has very low diversity ({len(np.unique(y_sample))} unique tokens)")
        
        # Check for extremely repetitive patterns
        if seq_len > 20:
            # Check if first 10 tokens repeat
            first_10 = x_sample[:10]
            repeats = 0
            for j in range(10, seq_len - 10, 10):
                if np.array_equal(x_sample[j:j+10], first_10):
                    repeats += 1
            if repeats > seq_len // 20:  # More than 5% repetition
                print(f"     ⚠️  WARNING: High repetition detected in X ({repeats} repeating blocks)")

def compute_per_sample_loss(logits, targets):
    """
    Compute loss for each sample in the batch individually
    
    Args:
        logits: Model output (batch_size, seq_len, vocab_size)
        targets: Target tokens (batch_size, seq_len)
    
    Returns:
        per_sample_losses: Tensor of shape (batch_size,)
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for loss computation
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = targets.view(-1)  # (batch_size * seq_len,)
    
    # Compute loss for each token
    token_losses = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1, reduction='none')
    
    # Reshape back to (batch_size, seq_len)
    token_losses = token_losses.view(batch_size, seq_len)
    
    # Sum losses for each sample (ignoring -1 tokens)
    per_sample_losses = []
    for i in range(batch_size):
        mask = targets[i] != -1
        if mask.sum() > 0:
            sample_loss = token_losses[i][mask].mean()
        else:
            sample_loss = torch.tensor(0.0, device=token_losses.device)
        per_sample_losses.append(sample_loss)
    
    return torch.stack(per_sample_losses)

# training loop
batch_data = get_batch('train')  # fetch the very first batch
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
            
            # Detailed logging for loss explosion debugging
            if master_process and (iter_num % log_interval == 0 or loss.item() > 10.0):
                # Compute per-sample losses for detailed analysis
                with torch.no_grad():
                    per_sample_losses = compute_per_sample_loss(logits, Y)
                    
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
                        
                        log_training_samples(X, Y, per_sample_losses, iter_num, sample_limit=2)
                        
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
        batch_data = get_batch('train')
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
        if len(loss_history) > 0:
            recent_avg_loss = sum(loss_history[-10:]) / min(len(loss_history), 10)
            if lossf > recent_avg_loss * loss_spike_threshold and len(loss_history) >= 5:
                print(f"⚠️  Loss spike detected! Current: {lossf:.4f}, Recent avg: {recent_avg_loss:.4f}")
                print(f"🔧 Reducing learning rate temporarily...")
                # Temporarily reduce learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
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
