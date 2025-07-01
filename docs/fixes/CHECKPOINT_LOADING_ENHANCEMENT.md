# ✅ Enhanced Checkpoint Loading for Advanced MoE Training

The `train_moe_advanced.py` script now includes robust checkpoint loading functionality with support for resuming training and handling various checkpoint formats.

## 🔄 Checkpoint Loading Features

### 1. **Resume Training**
```bash
# Resume from a specific checkpoint
python train_moe_advanced.py --config v100 --resume path/to/checkpoint.pt

# Resume from best model in output directory
python train_moe_advanced.py --config v100 --resume out-moe-v100/best_model.pt
```

### 2. **Robust Prefix Handling**
The checkpoint loader automatically handles problematic prefixes from:
- **DDP (Distributed Data Parallel)**: `module.` prefix
- **Model Compilation**: `_orig_mod.` prefix  
- **Combined DDP + Compilation**: `module._orig_mod.` prefix

### 3. **Graceful Error Handling**
- Non-strict loading as fallback
- Detailed error messages
- Continues training from scratch if checkpoint loading fails

## 🔧 Checkpoint Loading Process

### Step 1: Load and Validate
```python
checkpoint = torch.load(checkpoint_path, map_location=device)
```

### Step 2: Extract Components
- Model state dict
- Optimizer state dict
- Training configuration
- Training step and best validation loss

### Step 3: Handle Prefixes
```python
unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
# Automatically strip problematic prefixes
```

### Step 4: Load with Fallback
- Try strict loading first
- Fall back to non-strict if needed
- Report missing/unexpected keys

## 📊 Checkpoint Information

When loading a checkpoint, the script displays:
```
Loading checkpoint from out-moe-v100/best_model.pt
Checkpoint contains: step=15000, best_val_loss=2.3456
No problematic prefixes found
✅ Model state loaded successfully (strict=True)
✅ Optimizer state loaded successfully
✅ Resumed training from step 15000 with best_val_loss=2.3456
```

## 🔗 Integration with Weights & Biases

Checkpoint metadata is automatically logged to W&B:
- Model parameters and configuration
- Starting step and validation loss
- Training resumption events

## 📁 Checkpoint File Structure

Saved checkpoints contain:
```python
{
    'model': model.state_dict(),           # Model parameters
    'optimizer': optimizer.state_dict(),   # Optimizer state
    'config': config,                      # Training configuration
    'step': step,                          # Current training step
    'best_val_loss': best_val_loss,       # Best validation loss
    'timestamp': time.time()               # Save timestamp
}
```

## 🚀 Usage Examples

### Basic Resume
```bash
# Train for some steps
python train_moe_advanced.py --config v100

# Resume from best checkpoint
python train_moe_advanced.py --config v100 --resume out-moe-v100/best_model.pt
```

### Resume with Different Configuration
```bash
# Resume but override output directory
python train_moe_advanced.py --config v100 --resume old_run/best_model.pt --out-dir new_run
```

### Resume with Monitoring
```bash
# Resume with full monitoring and W&B logging
python train_moe_advanced.py --config v100 --resume checkpoint.pt --wandb --monitor --profile
```

## ⚠️ Important Notes

### 1. **Configuration Compatibility**
- Checkpoint config is loaded but current config takes precedence
- Model architecture must match (layers, embedding size, experts)
- Only training state (step, best_val_loss) is restored from checkpoint

### 2. **Optimizer State**
- Optimizer state is restored when available
- Training continues with correct learning rate schedule
- Adam momentum and variance are preserved

### 3. **DDP Compatibility**
- Works with both single-GPU and multi-GPU training
- Checkpoint prefixes are automatically handled
- DDP setup doesn't affect checkpoint loading

## 🔍 Troubleshooting

### Common Issues

**1. Architecture Mismatch**
```
RuntimeError: Error(s) in loading state_dict for GPT:
        size mismatch for transformer.h.0.mlp.experts.0.c_fc.weight
```
**Solution**: Ensure model architecture matches checkpoint

**2. Missing Keys**
```
⚠️ Missing keys: ['transformer.h.0.mlp.experts.0.c_fc.weight', ...]
```
**Solution**: Usually safe to ignore for MoE models with different expert counts

**3. Unexpected Keys**
```
⚠️ Unexpected keys: ['_orig_mod.transformer.wte.weight', ...]
```
**Solution**: Automatically handled by prefix removal

## 📈 Best Practices

1. **Regular Checkpointing**: Save checkpoints every 1000-5000 steps
2. **Keep Best Model**: Always maintain `best_model.pt` for easy resumption
3. **Monitor Loading**: Check logs for successful checkpoint loading
4. **Test Resume**: Verify resumed training continues smoothly
5. **Backup Checkpoints**: Keep multiple checkpoint versions

## 🎯 Next Steps

The enhanced checkpoint loading enables:
- **Robust Training**: Recover from interruptions
- **Experimentation**: Try different configurations from same checkpoint
- **Model Sharing**: Easily distribute trained models
- **Fine-tuning**: Start from pre-trained checkpoints

This checkpoint system ensures reliable and flexible training workflows for large MoE models! 🚀
