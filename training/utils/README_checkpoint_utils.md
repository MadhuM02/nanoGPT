# Checkpoint Utilities

This module provides comprehensive checkpoint management utilities for nanoGPT training scripts. It centralizes checkpoint loading, saving, and management functionality to reduce code duplication and improve maintainability.

## Features

- **Unified checkpoint loading**: Handles different checkpoint formats and initialization methods
- **Robust error handling**: Gracefully handles missing files, format mismatches, and dtype issues
- **Memory management**: Automatic cleanup and memory optimization
- **Flexible model initialization**: Support for scratch, resume, and GPT-2 pretrained initialization
- **Comprehensive validation**: Vocab size detection, state dict cleaning, and format validation

## Main Functions

### `load_checkpoint_and_create_model()`

Loads a checkpoint and creates a model based on the initialization method.

```python
model, model_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(
    init_from='resume',  # 'scratch', 'resume', or 'gpt2*'
    resume_from='checkpoints/out/ckpt.pt',
    model_args=model_args,
    meta_vocab_size=50304,
    device='cuda',
    master_process=True,
    block_size=1024
)
```

**Parameters:**
- `init_from`: Initialization method ('scratch', 'resume', or 'gpt2*')
- `resume_from`: Path to checkpoint file (used when init_from='resume')
- `model_args`: Dictionary of model configuration arguments
- `meta_vocab_size`: Vocabulary size from dataset metadata (optional)
- `device`: Device to load checkpoint on
- `master_process`: Whether this is the master process (for logging)
- `block_size`: Optional block size for model surgery

**Returns:**
- `model`: Loaded/created GPT model
- `updated_model_args`: Updated model arguments (may be modified by checkpoint)
- `iter_num`: Training iteration number
- `best_val_loss`: Best validation loss achieved
- `checkpoint_dict`: Loaded checkpoint dictionary (None for scratch/gpt2)

### `save_checkpoint()`

Saves a complete training checkpoint including model and optimizer state.

```python
save_checkpoint(
    model=model,
    optimizer=optimizer,
    model_args=model_args,
    iter_num=iter_num,
    best_val_loss=best_val_loss,
    config=config,
    out_dir='checkpoints/out',
    checkpoint_name='ckpt.pt',
    master_process=True
)
```

### `save_best_checkpoint()`

Saves a deployment-ready checkpoint with only the model state (no optimizer).

```python
save_best_checkpoint(
    model=model,
    optimizer=optimizer,  # Not saved, just for API consistency
    model_args=model_args,
    iter_num=iter_num,
    val_loss=val_loss,
    config=config,
    out_dir='checkpoints/out',
    master_process=True
)
```

### `load_optimizer_state()`

Loads optimizer state with error handling for dtype mismatches.

```python
success = load_optimizer_state(
    checkpoint_dict=checkpoint,
    optimizer=optimizer,
    master_process=True,
    skip_on_dtype_mismatch=True
)
```

### `get_model_info()`

Extracts detailed information about a model for logging and debugging.

```python
info = get_model_info(model)
print(f"Model: {info['total_parameters']:,} params, {info['model_size_mb']:.1f}MB")
```

## Utility Functions

### `find_latest_checkpoint()`

Finds the most recent checkpoint in a directory.

```python
checkpoint_path = find_latest_checkpoint(
    out_dir='checkpoints/out',
    checkpoint_patterns=['ckpt.pt', 'best_model.pt']
)
```

### Internal Utilities

- `_detect_vocab_size_from_weights()`: Detects vocabulary size from model weights
- `_clean_state_dict_keys()`: Removes unwanted prefixes from state dict keys
- `_extract_iteration_number()`: Extracts iteration number from different checkpoint formats

## Usage Examples

### Basic Training Script Integration

```python
from training.utils.checkpoint_utils import (
    load_checkpoint_and_create_model,
    save_checkpoint,
    load_optimizer_state
)

# Load/create model
model, model_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(
    init_from=init_from,
    resume_from=resume_from,
    model_args=model_args,
    meta_vocab_size=meta_vocab_size,
    device=device,
    master_process=master_process
)

# Create optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# Load optimizer state if resuming
if init_from == 'resume' and checkpoint is not None:
    load_optimizer_state(checkpoint, optimizer, master_process)

# Training loop...

# Save checkpoint
if should_save:
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        model_args=model_args,
        iter_num=iter_num,
        best_val_loss=best_val_loss,
        config=config,
        out_dir=out_dir,
        master_process=master_process
    )
```

### Checkpoint Format Compatibility

The utilities handle multiple checkpoint formats:

1. **Standard format**: `{'model': state_dict, 'model_args': args, 'iter_num': num, ...}`
2. **Config format**: `{'model': state_dict, 'config': args, 'step': num, ...}`
3. **Dirty state dict**: Automatically cleans prefixes like `module.`, `_orig_mod.`, etc.

### Error Handling

The utilities provide robust error handling:

- **Missing files**: Graceful handling with informative error messages
- **Dtype mismatches**: Optional skipping of optimizer loading on dtype conflicts
- **Format variations**: Automatic detection and handling of different checkpoint formats
- **Memory management**: Automatic cleanup to prevent memory leaks

## Integration with Training Scripts

All major training scripts now use these utilities:

- `training/sft/train_sft.py`: SFT training with MoE support
- `training/standard/train.py`: Standard GPT training
- `training/moe/train_moe_advanced.py`: Advanced MoE training

This provides consistent checkpoint handling across all training workflows.

## Testing

Run the comprehensive test suite:

```bash
python tests/utils/test_checkpoint_utils.py
```

The tests cover:
- Model creation from scratch
- Checkpoint saving and loading
- Different checkpoint formats
- State dict cleaning
- Optimizer state handling
- Error scenarios

## Migration Guide

To migrate existing training scripts:

1. Import the utilities:
   ```python
   from training.utils.checkpoint_utils import (
       load_checkpoint_and_create_model,
       save_checkpoint,
       load_optimizer_state
   )
   ```

2. Replace manual checkpoint loading with:
   ```python
   model, model_args, iter_num, best_val_loss, checkpoint = load_checkpoint_and_create_model(...)
   ```

3. Replace manual checkpoint saving with:
   ```python
   save_checkpoint(...)
   ```

4. Use `load_optimizer_state()` for robust optimizer loading

This refactoring significantly reduces code duplication and provides more robust checkpoint handling across all training scripts.
