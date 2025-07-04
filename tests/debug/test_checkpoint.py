#!/usr/bin/env python3
"""
Test checkpoint loading
"""

import torch
import os
from model import GPTConfig, GPT

# Test loading the pretrained checkpoint
ckpt_path = 'out-moe-v100/best_model.pt'

try:
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print("✓ Checkpoint loaded successfully")
    
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model_args' in checkpoint:
        print(f"Model args: {checkpoint['model_args']}")
    
    if 'iter_num' in checkpoint:
        print(f"Iteration number: {checkpoint['iter_num']}")
    
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']}")
        
    # Try to create model from checkpoint
    model_args = checkpoint['model_args']
    print(f"Creating model with args: {model_args}")
    
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    print("✓ Model created successfully")
    
    # Try to load state dict
    state_dict = checkpoint['model']
    print(f"State dict has {len(state_dict)} keys")
    
    # Check for common prefix issues
    sample_keys = list(state_dict.keys())[:5]
    print(f"Sample keys: {sample_keys}")
    
    # Clean up unwanted prefixes
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    print("✓ State dict loaded successfully")
    
    print("✓ All checkpoint operations successful")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
