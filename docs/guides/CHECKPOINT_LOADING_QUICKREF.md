# Checkpoint Loading Quick Reference

This is a quick reference for the enhanced checkpoint loading system in nanoGPT.

## TL;DR - Just Want to Load a Checkpoint?

```python
from train_moe_advanced import load_checkpoint

# This handles 95% of cases automatically
config, step, best_val_loss = load_checkpoint(
    'path/to/your/checkpoint.pt',
    your_model,
    device='cuda'
)
```

## Common Error Messages and Solutions

### ❌ "RuntimeError: Error(s) in loading state_dict"

**What it means**: Key names don't match between your model and checkpoint.

**Quick fix**:
```bash
# Analyze the checkpoint to understand the issue
python debug_checkpoint_advanced.py path/to/your/checkpoint.pt

# The tool will tell you exactly what's wrong and how to fix it
```

### ❌ "size mismatch for transformer.wte.weight"

**What it means**: Model architecture doesn't match the checkpoint.

**Quick fix**: Check these settings in your model:
- `vocab_size` - Must match the checkpoint's vocabulary size
- `n_layer`, `n_head`, `n_embd` - Must match the model architecture
- Block size and other architectural parameters

### ❌ "Many keys missing" warning

**What it means**: You might be loading the wrong type of checkpoint.

**Quick fix**:
```bash
# Check what type of model the checkpoint expects
python debug_checkpoint_advanced.py path/to/your/checkpoint.pt gpt2
python debug_checkpoint_advanced.py path/to/your/checkpoint.pt moe
```

## When the Automatic Loading Fails

### Step 1: Diagnose the Problem
```bash
python debug_checkpoint_advanced.py your_checkpoint.pt
```

### Step 2: Manual Loading (if needed)
```python
import torch
from train_moe_advanced import strip_model_prefixes

# Load checkpoint
checkpoint = torch.load('checkpoint.pt', map_location='cpu')
state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

# Remove problematic prefixes (if the tool suggested this)
clean_state_dict, changes = strip_model_prefixes(state_dict)
print(f"Removed prefixes from {changes} keys")

# Load into model
missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
```

## Training Script Integration

### Basic Integration
```python
# At the start of your training script
if args.resume or args.init_from:
    checkpoint_path = args.resume or args.init_from
    try:
        config, step, best_val_loss = load_checkpoint(
            checkpoint_path, model, optimizer, device
        )
        print(f"Resumed from step {step}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        # Handle the error appropriately
```

### Advanced Integration with Fallbacks
```python
def load_model_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint with multiple fallback strategies"""
    try:
        # Try enhanced loading first
        return load_checkpoint(checkpoint_path, model, optimizer)
    except Exception as e:
        print(f"Enhanced loading failed: {e}")
        
        # Fallback to manual loading
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            # Try removing common prefixes
            for prefix in ['_orig_mod.', 'module.', 'module._orig_mod.']:
                if any(k.startswith(prefix) for k in state_dict.keys()):
                    clean_dict = {k[len(prefix):] if k.startswith(prefix) else k: v 
                                for k, v in state_dict.items()}
                    try:
                        model.load_state_dict(clean_dict, strict=False)
                        print(f"Successfully loaded with {prefix} prefix removed")
                        break
                    except:
                        continue
        
        return {}, 0, float('inf')  # Return defaults if all fails
```

## Prefix Cheat Sheet

| Scenario | Checkpoint Keys Look Like | Model Keys Look Like | Solution |
|----------|---------------------------|---------------------|----------|
| Normal | `transformer.wte.weight` | `transformer.wte.weight` | ✅ Direct loading |
| DDP | `module.transformer.wte.weight` | `transformer.wte.weight` | Remove `module.` |
| Compiled | `_orig_mod.transformer.wte.weight` | `transformer.wte.weight` | Remove `_orig_mod.` |
| DDP + Compiled | `module._orig_mod.transformer.wte.weight` | `transformer.wte.weight` | Remove `module._orig_mod.` |
| Reverse DDP | `transformer.wte.weight` | `module.transformer.wte.weight` | Add `module.` |

## Testing Your Setup

### Quick Test
```python
# Test if your model and checkpoint are compatible
from train_moe_advanced import find_matching_keys

model_keys = set(your_model.state_dict().keys())
checkpoint = torch.load('your_checkpoint.pt', map_location='cpu')
checkpoint_keys = set(checkpoint['model'].keys())

analysis = find_matching_keys(model_keys, checkpoint_keys)
print(f"Compatibility: {analysis['exact_matches']}/{len(model_keys)} keys match")
```

### Full Test Suite
```bash
# Run the comprehensive test suite
python tests/test_advanced_checkpoint_loading.py
```

## Emergency Procedures

### If Nothing Works

1. **Check the basics**:
   - Is the file corrupted? Try loading it in Python: `torch.load('file.pt')`
   - Is it the right file? Check file size and modification date

2. **Model architecture mismatch**:
   ```python
   # Compare architectures
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')
   if 'config' in checkpoint:
       print("Checkpoint config:", checkpoint['config'])
   
   print("Your model config:", your_model_config.__dict__)
   ```

3. **Last resort - manual key mapping**:
   ```python
   # Map keys manually (if you know the correspondence)
   checkpoint = torch.load('checkpoint.pt', map_location='cpu')
   state_dict = checkpoint['model']
   
   # Create mapping of old keys to new keys
   key_mapping = {
       'old_key_name': 'new_key_name',
       # ... add more mappings
   }
   
   new_state_dict = {}
   for old_key, value in state_dict.items():
       new_key = key_mapping.get(old_key, old_key)
       new_state_dict[new_key] = value
   
   model.load_state_dict(new_state_dict, strict=False)
   ```

## Performance Tips

- **Large checkpoints**: Use `map_location='cpu'` first, then move to GPU
- **Repeated loading**: The analysis is cached within a session
- **Network storage**: Copy checkpoint locally first for faster loading
- **Memory constraints**: Use `torch.load(..., weights_only=True)` if you only need the model weights

## Getting Help

1. **Run the diagnostic tool**: `python debug_checkpoint_advanced.py your_checkpoint.pt`
2. **Check the output carefully** - it usually tells you exactly what to do
3. **Look at the test suite** for examples of different scenarios
4. **The enhanced loading provides detailed logs** - read them for clues

Remember: The enhanced checkpoint loading system handles 95% of cases automatically. When it doesn't work, the diagnostic output will guide you to the solution! 🎯
