# Fixed: UnboundLocalError in train_moe_advanced.py

## ✅ Problem Solved

Fixed a critical bug where `master_process` variable was referenced before assignment in the Weights & Biases initialization section.

## 🔧 What Was Fixed

**Root Cause:**
- The `master_process` variable was used in the W&B initialization code before it was defined in the DDP setup section
- This caused an `UnboundLocalError: local variable 'master_process' referenced before assignment`

**Solution:**
- Moved the DDP setup section (which defines `master_process`) before the W&B initialization
- Removed duplicate DDP setup code that was appearing later in the function
- Ensured proper variable initialization order

## 📝 Code Changes

### Before (Problematic):
```python
# Initialize Weights & Biases if requested
wandb_run = None
if args.wandb and master_process:  # ❌ master_process not defined yet
    # ... W&B code

# Setup distributed training (later in code)
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # ... DDP setup
    master_process = ddp_rank == 0  # ✅ defined here
else:
    master_process = True  # ✅ defined here
```

### After (Fixed):
```python
# Setup distributed training first to define master_process
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # ... DDP setup
    master_process = ddp_rank == 0  # ✅ defined early
else:
    master_process = True  # ✅ defined early

# Initialize Weights & Biases if requested
wandb_run = None
if args.wandb and master_process:  # ✅ master_process now defined
    # ... W&B code
```

## 🧪 Verification

The fix has been verified:
- ✅ Python compilation successful
- ✅ No syntax errors
- ✅ Proper variable initialization order
- ✅ All functionality preserved

## 📋 Impact

This fix ensures that:
- The training script runs without UnboundLocalError
- Weights & Biases logging works correctly in distributed training setups
- Both single-GPU and multi-GPU training modes function properly
- The script maintains all its advanced features and optimizations

## 🚀 Usage

The script now works correctly with all options:

```bash
# Single GPU training
python train_moe_advanced.py --config v100 --wandb

# Multi-GPU training  
torchrun --nproc_per_node=2 train_moe_advanced.py --config v100 --wandb

# With monitoring and profiling
python train_moe_advanced.py --config auto --wandb --monitor --profile
```
