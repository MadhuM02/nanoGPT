#!/usr/bin/env python3
"""
Quick memory test for distributed SFT training
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Set memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def check_memory():
    """Check current CUDA memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return allocated, reserved
    return 0, 0

def test_model_loading():
    """Test loading the model and checking memory"""
    print("Testing model loading and memory usage...")
    
    # Initialize distributed if running with torchrun
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        is_master = dist.get_rank() == 0
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        is_master = True
    
    if is_master:
        print(f"Using device: {device}")
    
    # Check initial memory
    if is_master:
        print("1. Initial memory:")
        check_memory()
    
    # Load model
    try:
        from model import GPTConfig, GPT
        
        # Create a smaller model for testing
        config = GPTConfig(
            block_size=256,  # Reduced from 512
            vocab_size=50304,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=False,
            num_experts=8,
            top_k_experts=2,
            load_balancing_loss_coef=0.01,
            expert_dropout=0.0,
            capacity_factor=1.0,
            expert_activation='gelu'
        )
        
        model = GPT(config).to(device)
        
        if is_master:
            print("2. After model creation:")
            check_memory()
        
        # Wrap in DDP if distributed
        if 'RANK' in os.environ:
            model = DDP(model, device_ids=[local_rank])
            if is_master:
                print("3. After DDP wrapping:")
                check_memory()
        
        # Test forward pass
        x = torch.randint(0, 50304, (1, 256), device=device)
        y = torch.randint(0, 50304, (1, 256), device=device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits, loss = model(x, y)
        
        if is_master:
            print("4. After forward pass:")
            check_memory()
        
        # Test backward pass
        loss.backward()
        
        if is_master:
            print("5. After backward pass:")
            allocated, reserved = check_memory()
            
            if reserved > 14.0:  # If using more than 14GB reserved
                print("⚠️  High memory usage detected!")
            else:
                print("✅ Memory usage looks reasonable")
        
        # Cleanup
        del x, y, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if is_master:
            print("6. After cleanup:")
            check_memory()
        
        print("✅ Memory test completed successfully")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False
    
    finally:
        if 'RANK' in os.environ:
            dist.destroy_process_group()
    
    return True

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)
