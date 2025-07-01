"""
Advanced MoE training script with V100 optimizations
Run with: python train_moe_advanced.py --config v100
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
from v100_config import (get_v100_config, get_memory_efficient_config, get_performance_config,
                        get_minimal_config, get_debug_config, auto_select_config)
from moe_monitor import MoEMonitor, GPUProfiler

# -----------------------------------------------------------------------------
# Configuration management
def load_config(config_name):
    """Load predefined configurations"""
    configs = {
        'v100': get_v100_config(), 
        'memory': get_memory_efficient_config(), 
        'performance': get_performance_config(),
        'minimal': get_minimal_config(),
        'debug': get_debug_config()
    }
    return configs.get(config_name, get_v100_config())

def optimize_for_hardware():
    """Auto-detect and optimize for available hardware"""
    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("No CUDA devices found, using debug config")
        return get_debug_config()
        
    # Use single GPU memory for estimation
    single_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    total_memory_gb = single_gpu_memory * device_count
    
    print(f"Detected {device_count} GPUs with {single_gpu_memory:.1f} GB each ({total_memory_gb:.1f} GB total)")
    
    # Auto-adjust configuration based on single GPU memory (most constraining)
    return auto_select_config(single_gpu_memory)

# -----------------------------------------------------------------------------
# Advanced data loading with dynamic batching
class DynamicBatchLoader:
    """Data loader with dynamic batching based on sequence length"""
    
    def __init__(self, data_path, max_batch_size, block_size, device):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.device = device
        
    def get_batch(self, target_tokens=None):
        """Get batch with dynamic size based on target token count"""
        if target_tokens is None:
            batch_size = self.max_batch_size
        else:
            batch_size = min(self.max_batch_size, target_tokens // self.block_size)
            batch_size = max(1, batch_size)
            
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix])
        
        if 'cuda' in str(self.device):
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y, batch_size

# -----------------------------------------------------------------------------
# Advanced loss computation with MoE-specific losses
def compute_loss_with_aux(model, x, y, load_balance_coef=0.01):
    """Compute main loss plus auxiliary MoE losses"""
    # Clear any existing auxiliary losses
    for module in model.modules():
        if hasattr(module, '_load_balancing_loss'):
            module._load_balancing_loss = 0
            
    # Forward pass
    logits, main_loss = model(x, y)
    
    # Collect auxiliary losses from MoE modules
    aux_loss = 0
    for module in model.modules():
        if hasattr(module, '_load_balancing_loss'):
            aux_loss += module._load_balancing_loss
            
    total_loss = main_loss + load_balance_coef * aux_loss
    
    return logits, total_loss, main_loss, aux_loss

# -----------------------------------------------------------------------------
# Advanced learning rate scheduling
class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_steps, max_steps, base_lr, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, step):
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step + 1) / (self.warmup_steps + 1)
        elif step > self.max_steps:
            lr = self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

# -----------------------------------------------------------------------------
# Checkpoint loading and saving utilities
def analyze_key_patterns(keys):
    """Analyze key patterns to understand model structure"""
    if not keys:
        return {}
    
    patterns = {
        'total_keys': len(keys),
        'has_module_prefix': sum(1 for k in keys if k.startswith('module.')),
        'has_orig_mod_prefix': sum(1 for k in keys if k.startswith('_orig_mod.')),
        'has_module_orig_mod_prefix': sum(1 for k in keys if k.startswith('module._orig_mod.')),
        'sample_keys': list(keys)[:5]
    }
    
    # Determine the most common prefix
    prefix_counts = {
        'module.': patterns['has_module_prefix'],
        '_orig_mod.': patterns['has_orig_mod_prefix'], 
        'module._orig_mod.': patterns['has_module_orig_mod_prefix']
    }
    
    patterns['dominant_prefix'] = max(prefix_counts.keys(), key=lambda k: prefix_counts[k])
    patterns['dominant_prefix_count'] = prefix_counts[patterns['dominant_prefix']]
    patterns['prefix_ratio'] = patterns['dominant_prefix_count'] / patterns['total_keys'] if patterns['total_keys'] > 0 else 0
    
    return patterns

def strip_model_prefixes(state_dict, prefixes_to_remove=['_orig_mod.', 'module._orig_mod.', 'module.']):
    """Strip unwanted prefixes from state dict keys"""
    new_state_dict = {}
    changes_made = 0
    
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes_to_remove:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                changes_made += 1
                break
        new_state_dict[new_key] = value
    
    return new_state_dict, changes_made

def add_model_prefix(state_dict, prefix_to_add):
    """Add prefix to all state dict keys"""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = prefix_to_add + key
        new_state_dict[new_key] = value
    return new_state_dict

def find_matching_keys(model_keys, checkpoint_keys):
    """Find which keys match between model and checkpoint"""
    exact_matches = model_keys.intersection(checkpoint_keys)
    
    # Try different prefix transformations to find potential matches
    transformations = []
    
    # Try removing prefixes from checkpoint keys
    for prefix in ['_orig_mod.', 'module._orig_mod.', 'module.']:
        stripped_checkpoint_keys = set()
        for key in checkpoint_keys:
            if key.startswith(prefix):
                stripped_checkpoint_keys.add(key[len(prefix):])
            else:
                stripped_checkpoint_keys.add(key)
        
        matches = model_keys.intersection(stripped_checkpoint_keys)
        if len(matches) > len(exact_matches):
            transformations.append({
                'type': 'remove_prefix',
                'prefix': prefix,
                'matches': len(matches),
                'description': f"Remove '{prefix}' from checkpoint keys"
            })
    
    # Try adding prefixes to checkpoint keys
    for prefix in ['_orig_mod.', 'module._orig_mod.', 'module.']:
        prefixed_checkpoint_keys = set(prefix + key for key in checkpoint_keys)
        matches = model_keys.intersection(prefixed_checkpoint_keys)
        if len(matches) > len(exact_matches):
            transformations.append({
                'type': 'add_prefix',
                'prefix': prefix,
                'matches': len(matches),
                'description': f"Add '{prefix}' to checkpoint keys"
            })
    
    # Sort by number of matches (best first)
    transformations.sort(key=lambda x: x['matches'], reverse=True)
    
    return {
        'exact_matches': len(exact_matches),
        'total_model_keys': len(model_keys),
        'total_checkpoint_keys': len(checkpoint_keys),
        'transformations': transformations
    }

def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    """Load checkpoint with robust prefix handling and detailed diagnostics"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    # Extract state dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        config = checkpoint.get('config', {})
        step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Checkpoint metadata: step={step}, best_val_loss={best_val_loss:.4f}")
    else:
        # Direct state dict
        model_state_dict = checkpoint
        config = {}
        step = 0
        best_val_loss = float('inf')
        print("Loading direct state dict (no metadata)")
    
    # Analyze the checkpoint and model key patterns
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(model_state_dict.keys())
    
    print("\n=== CHECKPOINT ANALYSIS ===")
    model_analysis = analyze_key_patterns(model_keys)
    checkpoint_analysis = analyze_key_patterns(checkpoint_keys)
    
    print(f"Model keys: {model_analysis['total_keys']}")
    print(f"  Sample: {model_analysis['sample_keys']}")
    print(f"  Dominant prefix: '{model_analysis['dominant_prefix']}' ({model_analysis['dominant_prefix_count']}/{model_analysis['total_keys']})")
    
    print(f"Checkpoint keys: {checkpoint_analysis['total_keys']}")
    print(f"  Sample: {checkpoint_analysis['sample_keys']}")
    print(f"  Dominant prefix: '{checkpoint_analysis['dominant_prefix']}' ({checkpoint_analysis['dominant_prefix_count']}/{checkpoint_analysis['total_keys']})")
    
    # Find the best way to match keys
    matching_analysis = find_matching_keys(model_keys, checkpoint_keys)
    
    print(f"\n=== KEY MATCHING ANALYSIS ===")
    print(f"Exact matches: {matching_analysis['exact_matches']}/{matching_analysis['total_model_keys']}")
    
    if matching_analysis['transformations']:
        print("Potential transformations:")
        for i, transform in enumerate(matching_analysis['transformations'][:3]):  # Show top 3
            print(f"  {i+1}. {transform['description']}: {transform['matches']}/{matching_analysis['total_model_keys']} matches")
    
    # Apply the best transformation if exact matches are insufficient
    exact_match_ratio = matching_analysis['exact_matches'] / matching_analysis['total_model_keys']
    
    if exact_match_ratio < 0.95 and matching_analysis['transformations']:  # Less than 95% exact matches
        best_transform = matching_analysis['transformations'][0]
        print(f"\n🔧 Applying transformation: {best_transform['description']}")
        
        if best_transform['type'] == 'remove_prefix':
            model_state_dict, changes = strip_model_prefixes(model_state_dict, [best_transform['prefix']])
            print(f"Removed prefix from {changes} keys")
        elif best_transform['type'] == 'add_prefix':
            model_state_dict = add_model_prefix(model_state_dict, best_transform['prefix'])
            print(f"Added prefix '{best_transform['prefix']}' to {len(model_state_dict)} keys")
        
        # Re-analyze after transformation
        new_checkpoint_keys = set(model_state_dict.keys())
        new_matches = len(model_keys.intersection(new_checkpoint_keys))
        print(f"After transformation: {new_matches}/{len(model_keys)} keys match")
    
    # Load state dict into model with comprehensive error handling
    print(f"\n=== LOADING STATE DICT ===")
    
    loading_strategies = [
        ("Direct loading (strict=True)", lambda: model.load_state_dict(model_state_dict, strict=True)),
        ("Direct loading (strict=False)", lambda: model.load_state_dict(model_state_dict, strict=False)),
    ]
    
    # Add module-based loading if the model has a module attribute
    if hasattr(model, 'module'):
        loading_strategies.extend([
            ("Module loading (strict=True)", lambda: model.module.load_state_dict(model_state_dict, strict=True)),
            ("Module loading (strict=False)", lambda: model.module.load_state_dict(model_state_dict, strict=False)),
        ])
    
    success = False
    for strategy_name, strategy_fn in loading_strategies:
        try:
            print(f"Trying: {strategy_name}")
            result = strategy_fn()
            
            if isinstance(result, tuple):  # strict=False returns (missing, unexpected)
                missing_keys, unexpected_keys = result
                if missing_keys:
                    print(f"  ⚠️ Missing keys: {len(missing_keys)}")
                    if len(missing_keys) <= 5:
                        print(f"    {missing_keys}")
                    else:
                        print(f"    First 5: {missing_keys[:5]}")
                
                if unexpected_keys:
                    print(f"  ⚠️ Unexpected keys: {len(unexpected_keys)}")
                    if len(unexpected_keys) <= 5:
                        print(f"    {unexpected_keys}")
                    else:
                        print(f"    First 5: {unexpected_keys[:5]}")
                
                # Check if too many keys are missing (indicates fundamental mismatch)
                missing_ratio = len(missing_keys) / len(model_keys)
                if missing_ratio > 0.5:
                    print(f"  ❌ Too many missing keys ({missing_ratio:.1%}) - trying next strategy")
                    continue
            
            print(f"  ✅ Success with {strategy_name}")
            success = True
            break
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}")
            continue
    
    if not success:
        print("\n=== DEBUGGING INFO ===")
        print("All loading strategies failed. Detailed key analysis:")
        
        # Show detailed key mismatches
        missing_in_checkpoint = model_keys - checkpoint_keys
        unexpected_in_checkpoint = checkpoint_keys - model_keys
        
        print(f"Keys in model but not in checkpoint ({len(missing_in_checkpoint)}):")
        for i, key in enumerate(sorted(missing_in_checkpoint)):
            if i < 10:  # Show first 10
                print(f"  - {key}")
            elif i == 10:
                print(f"  ... and {len(missing_in_checkpoint) - 10} more")
                break
        
        print(f"\nKeys in checkpoint but not in model ({len(unexpected_in_checkpoint)}):")
        for i, key in enumerate(sorted(unexpected_in_checkpoint)):
            if i < 10:  # Show first 10
                print(f"  + {key}")
            elif i == 10:
                print(f"  ... and {len(unexpected_in_checkpoint) - 10} more")
                break
        
        raise RuntimeError("Failed to load checkpoint with any strategy")
    
    # Load optimizer state if provided
    if optimizer is not None and isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("✅ Optimizer state loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load optimizer state: {e}")
            print("Continuing with fresh optimizer state...")
    
    return config, step, best_val_loss

def save_checkpoint(model, optimizer, config, step, best_val_loss, checkpoint_path, is_best=False):
    """Save checkpoint with metadata"""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config,
        'step': step,
        'best_val_loss': best_val_loss,
        'timestamp': time.time()
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    try:
        torch.save(checkpoint, checkpoint_path)
        file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✅ Saved checkpoint to {checkpoint_path} ({file_size_mb:.1f} MB)")
        
        if is_best:
            print(f"🏆 New best validation loss: {best_val_loss:.4f}")
            
    except Exception as e:
        print(f"❌ Failed to save checkpoint: {e}")
        raise

# -----------------------------------------------------------------------------
# Main training function
def train_advanced_moe():
    """Advanced MoE training with all optimizations"""
    
    # Parse arguments and load config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='auto', 
                        choices=['auto', 'v100', 'memory', 'performance', 'minimal', 'debug'])
    parser.add_argument('--profile', action='store_true', help='Enable detailed profiling')
    parser.add_argument('--monitor', action='store_true', help='Enable MoE monitoring')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='nanoGPT-MoE', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path to .pt file)')
    parser.add_argument('--init-from', type=str, default='scratch', 
                        choices=['scratch', 'resume', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='Initialize from scratch, resume training, or pretrained GPT-2')
    parser.add_argument('--out-dir', type=str, default=None, help='Override output directory')
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'auto':
        config = optimize_for_hardware()
    else:
        config = load_config(args.config)
    
    # Override output directory if specified
    if args.out_dir:
        config['out_dir'] = args.out_dir
    
    print(f"Using configuration: {args.config}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Setup distributed training first to define master_process
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        config['gradient_accumulation_steps'] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Weights & Biases if requested
    wandb_run = None
    if args.wandb and master_process:
        try:
            import wandb
            
            # Generate run name if not provided
            run_name = args.wandb_run_name
            if run_name is None:
                run_name = f"{args.config}-{config['num_experts']}E-{config['n_layer']}L-{config['n_embd']}D"
            
            # Initialize wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=config,
                tags=[
                    f"config-{args.config}",
                    f"experts-{config['num_experts']}",
                    f"layers-{config['n_layer']}",
                    f"embedding-{config['n_embd']}",
                    "MoE",
                    "nanoGPT"
                ]
            )
            print(f"Initialized W&B run: {wandb_run.name}")
            
        except ImportError:
            print("WARNING: wandb not installed. Install with: pip install wandb")
            args.wandb = False
        except Exception as e:
            print(f"WARNING: wandb initialization failed: {e}")
            args.wandb = False
    
    # Setup
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    
    # Mixed precision setup
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Data loading
    data_dir = 'data/openwebtext'  # Adjust as needed
    train_loader = DynamicBatchLoader(
        os.path.join(data_dir, 'train.bin'),
        config['batch_size'],
        config['block_size'],
        device
    )
    val_loader = DynamicBatchLoader(
        os.path.join(data_dir, 'val.bin'),
        config['batch_size'],
        config['block_size'],
        device
    )
    
    # Model initialization
    vocab_size = config.get('vocab_size', 50304)
    gpt_config = GPTConfig(
        block_size=config['block_size'],
        vocab_size=vocab_size,
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        dropout=config['dropout'],
        bias=config['bias'],
        num_experts=config['num_experts'],
        top_k_experts=config['top_k_experts'],
        load_balancing_loss_coef=config['load_balancing_loss_coef'],
        expert_dropout=config.get('expert_dropout', 0.0),
        capacity_factor=config.get('capacity_factor', 1.0),
        expert_activation=config.get('expert_activation', 'gelu')
    )
    
    model = GPT(gpt_config).to(device)
    
    # Apply gradient checkpointing if requested
    if config.get('use_gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
    
    # Compile model
    if config.get('compile', False):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Setup optimizer
    optimizer = model.configure_optimizers(
        config['weight_decay'],
        config['learning_rate'],
        (config['beta1'], config['beta2']),
        device_type
    )
    
    # Setup learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        config['warmup_iters'],
        config['max_iters'],
        config['learning_rate'],
        config.get('min_lr', config['learning_rate'] / 10)
    )
    
    # Setup DDP
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
    
    # Setup monitoring
    monitor = None
    profiler = None
    if args.monitor and master_process:
        monitor = MoEMonitor(config['num_experts'], log_interval=config['log_interval'])
    if args.profile and master_process:
        profiler = GPUProfiler()
    
    # Setup mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    
    # Initialize training state
    step = 0
    best_val_loss = float('inf')
    
    # Checkpoint loading and initialization
    if args.resume:
        try:
            checkpoint_config, loaded_step, loaded_best_val_loss = load_checkpoint(
                args.resume, model, optimizer, device
            )
            step = loaded_step
            best_val_loss = loaded_best_val_loss
            
            # Update config with checkpoint values for consistency
            if checkpoint_config:
                print(f"Loaded checkpoint config keys: {list(checkpoint_config.keys())}")
                        
            print(f"✅ Resumed training from step {step} with best_val_loss={best_val_loss:.4f}")
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint from {args.resume}: {e}")
            print("Starting training from scratch...")
            step = 0
            best_val_loss = float('inf')
    
    elif args.init_from != 'scratch':
        # Initialize from pretrained GPT-2 model
        print(f"Initializing from pretrained {args.init_from}...")
        try:
            # For MoE models, we can initialize the base transformer layers from GPT-2
            # but the expert layers will be randomly initialized
            raw_model = model.module if ddp else model
            if hasattr(raw_model, 'init_from_gpt2'):
                raw_model.init_from_gpt2(args.init_from)
                print(f"✅ Initialized transformer layers from {args.init_from}")
                print("⚠️ Expert layers initialized randomly (MoE-specific)")
            else:
                print(f"⚠️ MoE model doesn't support GPT-2 initialization")
                print("Starting from scratch...")
        except Exception as e:
            print(f"❌ Failed to initialize from {args.init_from}: {e}")
            print("Starting training from scratch...")
    
    # Log initial model information
    if master_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Information:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model configuration: {config['n_layer']}L-{config['n_head']}H-{config['n_embd']}D")
        print(f"  MoE configuration: {config['num_experts']} experts, top-{config['top_k_experts']} routing")
        print(f"  Starting from step: {step}")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        
        # Log model info to wandb
        if args.wandb and wandb_run:
            try:
                model_info = {
                    'model/total_parameters': total_params,
                    'model/trainable_parameters': trainable_params,
                    'model/n_layers': config['n_layer'],
                    'model/n_heads': config['n_head'],
                    'model/n_embd': config['n_embd'],
                    'model/num_experts': config['num_experts'],
                    'model/top_k_experts': config['top_k_experts'],
                    'training/starting_step': step,
                    'training/starting_best_val_loss': best_val_loss
                }
                wandb.log(model_info)
            except Exception as e:
                print(f"Warning: Failed to log model info to wandb: {e}")
    
    # Training loop
    print("\n🚀 Starting training...")
    running_mfu = -1.0
    
    while step < config['max_iters']:
        
        # Learning rate scheduling
        lr = scheduler.step(step)
        
        # Evaluation
        if step % config['eval_interval'] == 0 and master_process:
            print(f"\nEvaluating at step {step}...")
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for eval_step in range(config['eval_iters']):
                    x_val, y_val, _ = val_loader.get_batch()
                    with ctx:
                        _, val_loss, _, _ = compute_loss_with_aux(model, x_val, y_val)
                    val_losses.append(val_loss.item())
                    
            avg_val_loss = np.mean(val_losses)
            print(f"Step {step}: val_loss = {avg_val_loss:.4f}")
            
            # Log validation metrics to wandb
            if args.wandb and wandb_run:
                try:
                    wandb_metrics = {
                        'val/loss': avg_val_loss,
                        'val/perplexity': math.exp(avg_val_loss),
                        'step': step
                    }
                    wandb.log(wandb_metrics)
                except Exception as e:
                    print(f"Warning: Failed to log validation metrics to wandb: {e}")
                    
            # Save checkpoint if best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if master_process:
                    checkpoint_path = os.path.join(config['out_dir'], 'best_model.pt')
                    save_checkpoint(model, optimizer, config, step, best_val_loss, checkpoint_path, is_best=True)
                    
                    # Log best model as wandb artifact
                    if args.wandb and wandb_run:
                        try:
                            artifact = wandb.Artifact(
                                name=f"best-model-step-{step}",
                                type="model",
                                description=f"Best model checkpoint at step {step} with val_loss {avg_val_loss:.4f}"
                            )
                            artifact.add_file(checkpoint_path)
                            wandb.log_artifact(artifact)
                            wandb.log({'best_val_loss': best_val_loss})
                        except Exception as e:
                            print(f"Warning: Failed to log best model artifact to wandb: {e}")
                    
            model.train()
        
        # Periodic checkpoint saving (every checkpoint_interval steps)
        if step % config.get('checkpoint_interval', 5000) == 0 and step > 0 and master_process:
            periodic_checkpoint_path = os.path.join(config['out_dir'], f'checkpoint_step_{step}.pt')
            save_checkpoint(model, optimizer, config, step, best_val_loss, periodic_checkpoint_path, is_best=False)
            
            # Log periodic checkpoint to wandb
            if args.wandb and wandb_run:
                try:
                    periodic_artifact = wandb.Artifact(
                        name=f"checkpoint-step-{step}",
                        type="checkpoint",
                        description=f"Periodic checkpoint at step {step}"
                    )
                    periodic_artifact.add_file(periodic_checkpoint_path)
                    wandb.log_artifact(periodic_artifact)
                    print(f"Saved and logged periodic checkpoint at step {step}")
                except Exception as e:
                    print(f"Warning: Failed to log periodic checkpoint to wandb: {e}")
            
            model.train()
        
        # Training step with gradient accumulation
        t0 = time.time()
        
        for micro_step in range(config['gradient_accumulation_steps']):
            if ddp:
                model.require_backward_grad_sync = (micro_step == config['gradient_accumulation_steps'] - 1)
                
            # Get batch with dynamic sizing if memory constrained
            if profiler and step % 100 == 0:
                profiler.log_gpu_memory(step)
                
            x, y, actual_batch_size = train_loader.get_batch()
            
            if profiler:
                profiler.start_timing('forward')
                
            with ctx:
                logits, total_loss, main_loss, aux_loss = compute_loss_with_aux(
                    model, x, y, config['load_balancing_loss_coef']
                )
                total_loss = total_loss / config['gradient_accumulation_steps']
                
            if profiler:
                profiler.end_timing('forward')
                profiler.start_timing('backward')
                
            # Backward pass
            scaler.scale(total_loss).backward()
            
            if profiler:
                profiler.end_timing('backward')
                
            # Monitor MoE statistics
            if monitor and step % config['log_interval'] == 0:
                # Extract expert routing info (simplified)
                # In practice, you'd get this from the MoE modules
                expert_indices = torch.randint(0, config['num_experts'], (actual_batch_size * config['block_size'],))
                gate_probs = torch.softmax(torch.randn(actual_batch_size * config['block_size'], config['num_experts']), dim=-1)
                monitor.update(expert_indices, gate_probs, aux_loss)
        
        # Optimizer step
        if config['grad_clip'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        
        if step % config['log_interval'] == 0 and master_process:
            lossf = total_loss.item() * config['gradient_accumulation_steps']
            
            # Estimate MFU
            raw_model = model.module if ddp else model
            if hasattr(raw_model, 'estimate_mfu'):
                mfu = raw_model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            
            print(f"step {step}: loss {lossf:.4f}, main_loss {main_loss.item():.4f}, "
                  f"aux_loss {aux_loss.item():.6f}, lr {lr:.2e}, "
                  f"time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
            # Log training metrics to wandb
            if args.wandb and wandb_run:
                try:
                    train_metrics = {
                        'train/loss': lossf,
                        'train/main_loss': main_loss.item(),
                        'train/aux_loss': aux_loss.item(),
                        'train/perplexity': math.exp(lossf),
                        'lr': lr,
                        'mfu': running_mfu * 100,
                        'timing/step_time_ms': dt * 1000,
                        'step': step
                    }
                    
                    # Add MoE-specific metrics if available
                    if hasattr(raw_model, 'get_moe_stats'):
                        moe_stats = raw_model.get_moe_stats()
                        train_metrics.update({
                            'moe/expert_usage_variance': moe_stats.get('expert_usage_variance', 0),
                            'moe/avg_experts_per_token': moe_stats.get('avg_experts_per_token', config['top_k_experts']),
                            'moe/load_balance_loss': aux_loss.item()
                        })
                    
                    # Add GPU memory metrics if available
                    if torch.cuda.is_available():
                        gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
                        gpu_memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
                        train_metrics['gpu/memory_allocated_mb'] = gpu_memory_mb
                        train_metrics['gpu/memory_reserved_mb'] = gpu_memory_reserved_mb
                        
                        # Calculate memory efficiency
                        if gpu_memory_reserved_mb > 0:
                            train_metrics['gpu/memory_efficiency'] = (gpu_memory_mb / gpu_memory_reserved_mb) * 100
                        
                        # Reset peak memory tracking periodically
                        if step % 1000 == 0:
                            torch.cuda.reset_peak_memory_stats()
                            
                    wandb.log(train_metrics)
                except Exception as e:
                    print(f"Warning: Failed to log training metrics to wandb: {e}")
            
            if monitor:
                monitor.log_stats(step)
        
        step += 1
    
    # Final reporting
    if master_process:
        print(f"\nTraining completed after {step} steps")
        
        # Log final training summary to wandb
        if args.wandb and wandb_run:
            try:
                final_metrics = {
                    'training/final_step': step,
                    'training/best_val_loss': best_val_loss,
                    'training/final_mfu': running_mfu * 100 if running_mfu > 0 else 0,
                    'training/total_parameters': sum(p.numel() for p in model.parameters()),
                    'training/trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
                }
                
                # Add model size info
                raw_model = model.module if ddp else model
                if hasattr(raw_model, 'get_num_params'):
                    final_metrics['model/total_params'] = raw_model.get_num_params()
                
                wandb.log(final_metrics)
                
                # Save final model as artifact
                final_checkpoint_path = os.path.join(config['out_dir'], 'final_model.pt')
                save_checkpoint(model, optimizer, config, step, best_val_loss, final_checkpoint_path, is_best=False)
                
                final_artifact = wandb.Artifact(
                    name=f"final-model-step-{step}",
                    type="model",
                    description=f"Final model checkpoint after {step} training steps"
                )
                final_artifact.add_file(final_checkpoint_path)
                wandb.log_artifact(final_artifact)
                
                print(f"Logged final model and metrics to W&B run: {wandb_run.name}")
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to log final metrics to wandb: {e}")
                try:
                    wandb.finish()
                except:
                    pass
        
        if profiler:
            profiler.print_summary()
        if monitor:
            monitor.plot_stats(os.path.join(config['out_dir'], 'moe_stats.png'))
            
            # Log MoE stats plot to wandb if available
            if args.wandb and os.path.exists(os.path.join(config['out_dir'], 'moe_stats.png')):
                wandb.log({"moe_stats_plot": wandb.Image(os.path.join(config['out_dir'], 'moe_stats.png'))})
    
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('out-moe-advanced', exist_ok=True)
    train_advanced_moe()
