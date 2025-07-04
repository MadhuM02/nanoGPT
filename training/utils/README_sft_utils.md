# SFT (Supervised Fine-Tuning) Utilities

This module provides comprehensive utilities for SFT training in nanoGPT, including data management, loss analysis, and training utilities. The utilities have been extracted from the main training script to improve modularity and reusability.

## Overview

The SFT utilities are organized into three main classes:

1. **`SFTDataManager`**: Handles dataset loading and batch processing with instruction-response awareness
2. **`SFTLossAnalyzer`**: Provides detailed loss analysis and debugging capabilities  
3. **`SFTTrainingUtils`**: Contains general training utilities like learning rate scheduling and spike detection

## Features

### SFTDataManager

- **Intelligent instruction-response parsing**: Automatically detects response markers like "Answer:", "Category:", etc.
- **Loss masking**: Only computes loss on response tokens, not instruction tokens
- **Flexible data loading**: Supports both text files and binary data fallback
- **Memory efficient**: Proper cleanup and memory management
- **Device optimization**: Supports CPU/GPU with proper tensor pinning

### SFTLossAnalyzer

- **Per-sample loss computation**: Analyze individual sample contributions to loss
- **Detailed sample logging**: Debug training with comprehensive sample analysis
- **Token-level analysis**: Check for invalid tokens, low diversity, and repetitive patterns
- **Text decoding**: Convert tokens back to readable text for debugging

### SFTTrainingUtils

- **Cosine learning rate with warmup**: Standard learning rate scheduling
- **Loss spike detection**: Automatically detect and handle training instabilities
- **Spike recovery**: Apply learning rate reduction for stability
- **Flexible configuration**: Easy to customize thresholds and parameters

## Quick Start

```python
from training.utils.sft_utils import create_sft_utilities

# Create all utilities
data_manager, loss_analyzer, training_utils = create_sft_utilities('data/sft_dataset')

# Load data
data_manager.load_examples(master_process=True)

# Get a training batch
X, Y, loss_mask = data_manager.get_batch('train', batch_size=8, block_size=512, device='cuda', device_type='cuda')

# Analyze losses during training
per_sample_losses = loss_analyzer.compute_per_sample_loss(logits, Y)
loss_analyzer.log_training_samples(X, Y, per_sample_losses, step=100, master_process=True)

# Use learning rate scheduling
lr = training_utils.get_cosine_lr_with_warmup(iter_num, base_lr, warmup_iters, decay_iters, min_lr)

# Detect and handle loss spikes
if training_utils.detect_loss_spike(loss_history, current_loss):
    training_utils.apply_spike_recovery(optimizer)
```

## Detailed Usage

### Data Management

```python
from training.utils.sft_utils import SFTDataManager

# Initialize data manager
data_manager = SFTDataManager('data/sft_dataset', tokenizer_name='gpt2')

# Load examples from train.txt and val.txt
data_manager.load_examples(master_process=True)

# Get a batch with instruction-response masking
batch_data = data_manager.get_batch(
    split='train',
    batch_size=12,
    block_size=512,
    device='cuda',
    device_type='cuda'
)

X, Y, loss_mask = batch_data
# X: input tokens, Y: target tokens (with -1 for masked instruction tokens), loss_mask: mask tensor
```

### Loss Analysis

```python
from training.utils.sft_utils import SFTLossAnalyzer

# Initialize loss analyzer
loss_analyzer = SFTLossAnalyzer(tokenizer_name='gpt2')

# Compute per-sample losses
per_sample_losses = loss_analyzer.compute_per_sample_loss(logits, targets)

# Log detailed sample information for debugging
loss_analyzer.log_training_samples(
    X=input_tokens,
    Y=target_tokens, 
    loss_per_sample=per_sample_losses,
    step=iteration_num,
    sample_limit=3,
    master_process=True
)
```

### Training Utilities

```python
from training.utils.sft_utils import SFTTrainingUtils

training_utils = SFTTrainingUtils()

# Learning rate scheduling
current_lr = training_utils.get_cosine_lr_with_warmup(
    it=current_iteration,
    learning_rate=base_learning_rate,
    warmup_iters=100,
    lr_decay_iters=2000,
    min_lr=1e-6
)

# Loss spike detection
loss_history = [2.5, 2.3, 2.1, 2.0, 5.8]  # Recent losses
is_spike = training_utils.detect_loss_spike(
    loss_history=loss_history,
    current_loss=5.8,
    spike_threshold=2.0,
    min_history=5
)

if is_spike:
    # Apply recovery by reducing learning rate
    training_utils.apply_spike_recovery(optimizer, recovery_factor=0.1)
```

## Data Format

### Text Files

The SFT utilities expect data in simple text format:

**train.txt / val.txt:**
```
Question: What is the capital of France? Answer: Paris
Category: Geography Summary: European capitals
Sentiment: This is amazing! Output: Positive
Translation: Hello Answer: Bonjour
```

### Response Markers

The system automatically detects instruction-response boundaries using these markers:
- `Category:`
- `Sentiment:`  
- `Summary:`
- `Answer:`
- `Translation:`
- `Output:`

Only tokens after these markers are used for loss computation.

### Binary Data Fallback

If text files are not available, the system falls back to binary data format:
- `train.bin` / `val.bin`: Memory-mapped numpy arrays of token IDs

## Integration with Training Scripts

The utilities integrate seamlessly with training scripts:

```python
# In your training script
from training.utils.sft_utils import create_sft_utilities

# Replace manual data loading
data_manager, loss_analyzer, training_utils = create_sft_utilities(data_dir)
data_manager.load_examples(master_process)

# Replace manual batch creation  
def get_batch(split):
    return data_manager.get_batch(split, batch_size, block_size, device, device_type)

# Replace manual loss analysis
def analyze_loss(logits, targets, step):
    per_sample_losses = loss_analyzer.compute_per_sample_loss(logits, targets)
    if step % 100 == 0:
        loss_analyzer.log_training_samples(X, Y, per_sample_losses, step, master_process=master_process)

# Replace manual learning rate scheduling
def get_lr(it):
    return training_utils.get_cosine_lr_with_warmup(it, learning_rate, warmup_iters, lr_decay_iters, min_lr)
```

## Advanced Features

### Custom Response Markers

You can extend the response marker detection by modifying the `_process_example` method:

```python
class CustomSFTDataManager(SFTDataManager):
    def _process_example(self, example, block_size):
        # Add custom response markers
        response_markers = ["Category:", "Answer:", "Output:", "Response:", "Result:"]
        # ... rest of processing
```

### Custom Loss Analysis

Extend the loss analyzer for domain-specific analysis:

```python
class CustomSFTLossAnalyzer(SFTLossAnalyzer):
    def analyze_domain_specific_patterns(self, X, Y):
        # Custom analysis logic
        pass
```

## Performance Considerations

- **Memory Management**: All utilities include proper tensor cleanup and memory management
- **Device Optimization**: Supports tensor pinning for efficient GPU transfers
- **Batch Processing**: Efficient vectorized operations for batch processing
- **Memory Mapping**: Uses memory-mapped files for large binary datasets

## Testing

Run the comprehensive test suite:

```bash
python tests/sft/test_sft_utilities.py
```

The tests cover:
- Data loading and batch creation
- Instruction-response masking
- Loss analysis functionality
- Learning rate scheduling
- Loss spike detection
- Integration with training workflows

## Migration from Manual Implementation

To migrate from manual SFT implementation:

1. **Replace data loading**:
   ```python
   # Old
   train_examples = load_text_files()
   
   # New  
   data_manager = SFTDataManager(data_dir)
   data_manager.load_examples()
   ```

2. **Replace batch creation**:
   ```python
   # Old
   def get_batch(split): # manual implementation
   
   # New
   batch_data = data_manager.get_batch(split, batch_size, block_size, device, device_type)
   ```

3. **Replace loss analysis**:
   ```python
   # Old
   def log_samples(): # manual logging
   
   # New
   loss_analyzer.log_training_samples(X, Y, losses, step, master_process=master_process)
   ```

4. **Replace utilities**:
   ```python
   # Old
   def get_lr(): # manual scheduler
   
   # New
   lr = training_utils.get_cosine_lr_with_warmup(...)
   ```

This refactoring significantly reduces code duplication and provides more robust, tested utilities for SFT training.
