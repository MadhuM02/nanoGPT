"""
Checkpoint utility functions for nanoGPT training scripts.
Provides reusable checkpoint loading, saving, and management functionality.
"""

import os
import sys
import torch
import pickle
from typing import Dict, Any, Optional, Tuple

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from model import GPTConfig, GPT


def load_checkpoint_and_create_model(
    init_from: str,
    resume_from: str,
    model_args: Dict[str, Any],
    meta_vocab_size: Optional[int] = None,
    device: str = 'cuda',
    master_process: bool = True,
    block_size: Optional[int] = None
) -> Tuple[GPT, Dict[str, Any], int, float, Optional[Dict]]:
    """
    Load a checkpoint and create a model based on init_from parameter.
    
    Args:
        init_from: One of 'scratch', 'resume', or 'gpt2*'
        resume_from: Path to checkpoint file (used when init_from='resume')
        model_args: Dictionary of model arguments
        meta_vocab_size: Vocab size from dataset metadata
        device: Device to load checkpoint on
        master_process: Whether this is the master process (for logging)
        block_size: Optional block size for model surgery
        
    Returns:
        Tuple of (model, updated_model_args, iter_num, best_val_loss, checkpoint_dict)
    """
    iter_num = 0
    best_val_loss = 1e9
    checkpoint_dict = None
    
    if init_from == 'scratch':
        # Initialize a new model from scratch
        if master_process:
            print("Initializing a new model from scratch")
        
        # Determine the vocab size for from-scratch training
        if meta_vocab_size is None:
            if master_process:
                print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
    elif init_from == 'resume':
        if master_process:
            print(f"Resuming training from {resume_from}")
        
        # Load checkpoint
        checkpoint_dict = torch.load(resume_from, map_location=device)
        
        # Handle different checkpoint formats for model configuration
        if 'model_args' in checkpoint_dict:
            checkpoint_model_args = checkpoint_dict['model_args']
        elif 'config' in checkpoint_dict:
            checkpoint_model_args = checkpoint_dict['config']
        else:
            raise ValueError("Cannot find model configuration in checkpoint")
        
        if master_process:
            print(f"Checkpoint model args: {checkpoint_model_args}")
        
        # Force these config attributes to be equal otherwise we can't resume training
        # The rest of the attributes (e.g. dropout) can stay as desired from command line
        core_model_keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
        for k in core_model_keys:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        
        # MoE specific parameters
        moe_keys = ['num_experts', 'top_k_experts', 'expert_activation']
        for k in moe_keys:
            if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]
        
        # Ensure vocab_size is set - detect from the actual model weights
        state_dict = checkpoint_dict['model']
        vocab_size_detected = _detect_vocab_size_from_weights(state_dict, master_process)
        
        if vocab_size_detected is not None:
            model_args['vocab_size'] = vocab_size_detected
        elif 'vocab_size' not in model_args or model_args['vocab_size'] is None:
            # Fallback logic for vocab_size
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
        
        # Create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Clean and load the state dict
        cleaned_state_dict = _clean_state_dict_keys(state_dict, master_process)
        model.load_state_dict(cleaned_state_dict)
        
        # Handle different checkpoint formats for iteration tracking
        iter_num = _extract_iteration_number(checkpoint_dict)
        best_val_loss = checkpoint_dict.get('best_val_loss', 1e9)
        
    elif init_from.startswith('gpt2'):
        if master_process:
            print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        
        # Initialize from OpenAI GPT-2 weights
        override_args = {k: v for k, v in model_args.items() if k in ['dropout']}
        model = GPT.from_pretrained(init_from, override_args)
        
        # Read off the created config params for checkpointing
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    
    else:
        raise ValueError(f"Unknown init_from value: {init_from}")
    
    # Perform model surgery if block_size is smaller than model's block_size
    if block_size is not None and block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size  # Update for checkpoint saving
        if master_process:
            print(f"Cropped model block_size to {block_size}")
    
    return model, model_args, iter_num, best_val_loss, checkpoint_dict


def _detect_vocab_size_from_weights(state_dict: Dict[str, torch.Tensor], master_process: bool = True) -> Optional[int]:
    """
    Detect vocabulary size from model weights by examining embedding layers.
    
    Args:
        state_dict: Model state dictionary
        master_process: Whether this is the master process (for logging)
        
    Returns:
        Detected vocab_size or None if not found
    """
    # Check embedding layer size to infer vocab_size
    embedding_keys = ['transformer.wte.weight', 'lm_head.weight']
    
    for key in state_dict.keys():
        if any(key.endswith(embed_key) for embed_key in embedding_keys):
            vocab_size = state_dict[key].shape[0]
            if master_process:
                print(f"Detected vocab_size from checkpoint weights ({key}): {vocab_size}")
            return vocab_size
    
    return None


def _clean_state_dict_keys(state_dict: Dict[str, torch.Tensor], master_process: bool = True) -> Dict[str, torch.Tensor]:
    """
    Clean state dict keys by removing unwanted prefixes that sometimes appear in checkpoints.
    
    Args:
        state_dict: Original model state dictionary
        master_process: Whether this is the master process (for logging)
        
    Returns:
        Cleaned state dictionary
    """
    # Fix the keys of the state dict
    # Sometimes checkpoints get these prefixes, need to debug more
    unwanted_prefixes = ['module._orig_mod.', '_orig_mod.', 'module.']
    
    for prefix in unwanted_prefixes:
        keys_to_fix = [k for k in state_dict.keys() if k.startswith(prefix)]
        if keys_to_fix:
            if master_process:
                print(f"Removing {len(keys_to_fix)} keys with prefix '{prefix}'")
            for k in keys_to_fix:
                new_key = k[len(prefix):]
                state_dict[new_key] = state_dict.pop(k)
    
    return state_dict


def _extract_iteration_number(checkpoint_dict: Dict[str, Any]) -> int:
    """
    Extract iteration number from checkpoint with different possible formats.
    
    Args:
        checkpoint_dict: Checkpoint dictionary
        
    Returns:
        Iteration number (0 if not found)
    """
    if 'iter_num' in checkpoint_dict:
        return checkpoint_dict['iter_num']
    elif 'step' in checkpoint_dict:
        return checkpoint_dict['step']
    else:
        return 0


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    model_args: Dict[str, Any],
    iter_num: int,
    best_val_loss: float,
    config: Dict[str, Any],
    out_dir: str,
    checkpoint_name: str = 'ckpt.pt',
    master_process: bool = True
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save (will be unwrapped if DDP)
        optimizer: Optimizer to save
        model_args: Model configuration arguments
        iter_num: Current iteration number
        best_val_loss: Best validation loss achieved
        config: Training configuration
        out_dir: Output directory
        checkpoint_name: Name of checkpoint file
        master_process: Whether this is the master process
    """
    if not master_process:
        return
    
    # Unwrap DDP if necessary
    raw_model = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    
    os.makedirs(out_dir, exist_ok=True)
    checkpoint_path = os.path.join(out_dir, checkpoint_name)
    
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    # Clean up checkpoint dict to free memory
    del checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_best_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer, 
    model_args: Dict[str, Any],
    iter_num: int,
    val_loss: float,
    config: Dict[str, Any],
    out_dir: str,
    master_process: bool = True
) -> None:
    """
    Save the best model checkpoint (without optimizer state for deployment).
    
    Args:
        model: Model to save
        optimizer: Optimizer (for consistency, not saved in best model)
        model_args: Model configuration arguments
        iter_num: Current iteration number
        val_loss: Current validation loss
        config: Training configuration
        out_dir: Output directory
        master_process: Whether this is the master process
    """
    if not master_process:
        return
    
    # Unwrap DDP if necessary
    raw_model = model.module if hasattr(model, 'module') else model
    
    best_checkpoint = {
        'model': raw_model.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': val_loss,
        'config': config,
    }
    
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, 'best_model.pt')
    
    print(f"Saving best model to {best_path}")
    torch.save(best_checkpoint, best_path)
    
    # Clean up
    del best_checkpoint
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_latest_checkpoint(out_dir: str, checkpoint_patterns: list = None) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        out_dir: Directory to search
        checkpoint_patterns: List of checkpoint filename patterns to look for
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if checkpoint_patterns is None:
        checkpoint_patterns = ['ckpt.pt', 'best_model.pt', 'checkpoint.pt']
    
    if not os.path.exists(out_dir):
        return None
    
    for pattern in checkpoint_patterns:
        checkpoint_path = os.path.join(out_dir, pattern)
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    
    # Look for any .pt files as fallback
    pt_files = [f for f in os.listdir(out_dir) if f.endswith('.pt')]
    if pt_files:
        # Sort by modification time and return the newest
        pt_files.sort(key=lambda f: os.path.getmtime(os.path.join(out_dir, f)), reverse=True)
        return os.path.join(out_dir, pt_files[0])
    
    return None


def load_optimizer_state(checkpoint_dict: Dict[str, Any], optimizer: torch.optim.Optimizer, 
                        master_process: bool = True, skip_on_dtype_mismatch: bool = True) -> bool:
    """
    Load optimizer state from checkpoint with error handling.
    
    Args:
        checkpoint_dict: Checkpoint dictionary
        optimizer: Optimizer to load state into
        master_process: Whether this is the master process (for logging)
        skip_on_dtype_mismatch: Whether to skip loading on dtype mismatches
        
    Returns:
        True if optimizer state was loaded successfully, False otherwise
    """
    if 'optimizer' not in checkpoint_dict:
        if master_process:
            print("No optimizer state found in checkpoint")
        return False
    
    try:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if master_process:
            print("✓ Optimizer state loaded successfully")
        return True
    except Exception as e:
        if master_process:
            if skip_on_dtype_mismatch and ('dtype' in str(e).lower() or 'type' in str(e).lower()):
                print(f"⚠️  Skipping optimizer state loading due to dtype mismatch: {e}")
                print("Starting with fresh optimizer state for fine-tuning")
            else:
                print(f"❌ Failed to load optimizer state: {e}")
        return False


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get information about a model for logging and debugging.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    if hasattr(model, 'config'):
        config_dict = model.config.__dict__ if hasattr(model.config, '__dict__') else {}
        info.update({f'config_{k}': v for k, v in config_dict.items()})
    
    return info
