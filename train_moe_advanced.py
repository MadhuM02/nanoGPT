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
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'auto':
        config = optimize_for_hardware()
    else:
        config = load_config(args.config)
    
    print(f"Using configuration: {args.config}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Setup distributed training
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
    
    # Training loop
    print("Starting training...")
    step = 0
    best_val_loss = float('inf')
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
            
            # Save checkpoint if best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if master_process:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'config': config,
                        'step': step,
                        'best_val_loss': best_val_loss
                    }
                    os.makedirs(config['out_dir'], exist_ok=True)
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'best_model.pt'))
                    
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
            
            if monitor:
                monitor.log_stats(step)
        
        step += 1
    
    # Final reporting
    if master_process:
        print(f"\nTraining completed after {step} steps")
        if profiler:
            profiler.print_summary()
        if monitor:
            monitor.plot_stats(os.path.join(config['out_dir'], 'moe_stats.png'))
    
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs('out-moe-advanced', exist_ok=True)
    train_advanced_moe()
