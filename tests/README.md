# nanoGPT Test Suite

This directory contains all test scripts for nanoGPT, organized by functionality including MoE, SFT, KV Cache, and more.

## 📁 Directory Structure

### `/sft/` - Supervised Fine-Tuning Tests
- `test_sft_attention.py` - Tests SFT-specific attention mechanisms
- `test_sft_attention_detailed.py` - Detailed SFT attention analysis
- `test_sft_attention_simple.py` - Simplified SFT attention tests
- `test_sft_data.py` - SFT data loading and preprocessing tests
- `test_sft_loss_masking.py` - Tests loss masking for instruction vs response tokens
- `minimal_sft_test.py` - Minimal SFT training test

### `/kv_cache/` - KV Cache Tests
- `test_kv_cache_fix.py` - Tests KV cache implementation fixes
- `test_kv_quality.py` - Quality tests for KV cache
- `minimal_kv_test.py` - Minimal KV cache functionality test
- `debug_kv_step.py` - Step-by-step KV cache debugging
- `test_generate_kv.py` - Tests text generation with KV cache

### `/memory/` - Memory Management Tests
- `test_memory.py` - Memory usage and leak tests
- `test_tensor_fix.py` - Tensor memory management tests

### `/moe/` - Mixture-of-Experts Tests
- `test_moe.py` - MoE layer functionality tests

### `/sampling/` - Text Generation and Sampling Tests
- `test_sampling.py` - Text generation sampling tests
- `test_prob_sampling.py` - Probability sampling algorithm tests

### `/wandb/` - Weights & Biases Integration Tests
- `test_wandb_logging.py` - W&B logging functionality tests
- `demo_wandb.py` - W&B integration demo and examples

### `/distributed/` - Distributed Training Tests
- `test_distributed_minimal.sh` - Minimal distributed training test
- `test_distributed_nccl.sh` - NCCL distributed training test
- `test_minimal_distributed.sh` - Basic distributed setup test
- `test_memory_configs.sh` - Memory configuration tests for distributed training

### `/debug/` - Debug and Checkpoint Tests
- `debug_checkpoint.py` - Checkpoint loading/saving debug utilities
- `debug_checkpoint_advanced.py` - Advanced checkpoint debugging
- `test_checkpoint.py` - Checkpoint functionality tests

### `/analysis/` - Analysis and Profiling Scripts
- `analyze_sft_attention.py` - SFT attention pattern analysis
- `analyze_sft_attention_new.py` - Updated SFT attention analysis
- `analyze_training_loss.py` - Training loss pattern analysis

### `/utils/` - Test Utilities and Setup Scripts
- `run_tests.py` - Main test runner script
- `organize_tests.py` - Test organization utilities
- `quick_test.py` - Quick functionality tests
- `update_test_imports.py` - Test import management
- `test_minimal.py` - Minimal functionality tests
- `test_prefix_fix.py` - Prefix handling tests
- `test_v100_setup.py` - V100 GPU setup tests

## 🚀 How to Run All Tests

### Option 1: Use the Test Runner (Recommended)
```bash
cd tests
python utils/run_tests.py
```

### Option 2: Use pytest for Category-based Testing
```bash
cd tests

# Run all tests in all categories
python -m pytest . -v

# Run tests by category
python -m pytest sft/ -v          # SFT tests only
python -m pytest memory/ -v       # Memory tests only
python -m pytest kv_cache/ -v     # KV cache tests only
python -m pytest moe/ -v          # MoE tests only
python -m pytest sampling/ -v     # Sampling tests only
python -m pytest wandb/ -v        # W&B tests only
```

### Option 3: Run Individual Tests
```bash
cd tests

# SFT tests
python sft/test_sft_attention.py
python sft/test_sft_loss_masking.py

# Memory tests  
python memory/test_memory.py

# KV cache tests
python kv_cache/test_kv_cache_fix.py

# MoE tests
python moe/test_moe.py

# Sampling tests
python sampling/test_sampling.py

# W&B tests
python wandb/test_wandb_logging.py
```

### Option 4: Quick Tests Only
```bash
cd tests
python utils/quick_test.py        # Fast smoke tests
```

## 📋 Available Tests

### Core Functionality Tests
- **`test_moe.py`** - Tests basic MoE model creation and forward pass
- **`test_gradient_checkpointing.py`** - Tests gradient checkpointing functionality
- **`test_minimal.py`** - Tests minimal configuration loading

### Memory and Performance Tests
- **`test_memory.py`** - Tests memory usage across different configurations
- **`test_v100_setup.py`** - Tests V100-specific configurations

### Sampling Tests
- **`test_sampling.py`** - Tests both sample.py and sample_moe.py scripts
- **`test_prefix_fix.py`** - Tests checkpoint prefix removal functionality

### Integration Tests
- **`test_wandb_logging.py`** - Tests Weights & Biases logging functionality

### Quick Tests
- **`quick_test.py`** - Quick smoke test for basic functionality

## 🧪 Test Categories

### Unit Tests
- Model creation and configuration
- MoE layer functionality
- Gradient checkpointing

### Integration Tests
- Sampling script functionality
- wandb logging integration
- Memory optimization features

### System Tests
- Full model training (minimal)
- Checkpoint loading/saving
- Cross-platform compatibility

## 📊 Expected Outputs

### ✅ Successful Test Run
```
nanoGPT MoE Test Suite
========================================
Found 8 test files:
  test_gradient_checkpointing.py
  test_memory.py
  test_minimal.py
  test_moe.py
  test_prefix_fix.py
  test_sampling.py
  test_v100_setup.py
  test_wandb_logging.py

============================================================
Running test_moe.py
============================================================
✅ test_moe.py PASSED (2.1s)

[... other tests ...]

============================================================
TEST SUMMARY
============================================================
Tests run: 8
Passed: 8
Failed: 0
Total time: 45.2s

🎉 All tests passed!
```

### ❌ Failed Test Example
```
❌ test_moe.py FAILED (1.5s)
Return code: 1
STDERR:
RuntimeError: CUDA out of memory...
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Tests automatically fall back to CPU
   - Reduce batch sizes in test configurations
   - Use `test_minimal.py` for very constrained environments

2. **Import Errors**
   - Tests automatically add parent directory to Python path
   - Ensure you're running from the `tests/` directory
   - Check that all required dependencies are installed

3. **Model Loading Errors**
   - Tests create temporary checkpoints automatically
   - Cleanup happens automatically after each test
   - Manual cleanup: `rm -rf test_*_out/`

### Environment Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- tiktoken (for sampling tests)
- wandb (for wandb integration tests)

### Running Specific Test Categories

```bash
# Test only core functionality
python test_moe.py
python test_gradient_checkpointing.py

# Test only sampling
python test_sampling.py
python test_prefix_fix.py

# Test only memory/performance
python test_memory.py
python test_v100_setup.py

# Test only integrations
python test_wandb_logging.py
```

## 📝 Adding New Tests

1. Create a new file: `test_your_feature.py`
2. Add the standard header:
   ```python
   #!/usr/bin/env python3
   """
   Test script for your feature
   """
   
   import os
   import sys
   
   # Add parent directory to path for imports
   sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
   
   from model import GPTConfig, GPT
   # ... other imports
   ```

3. Follow the pattern of existing tests
4. Include cleanup in your test
5. Return proper exit codes (0 for success, 1 for failure)

## 🎯 Test Goals

These tests ensure:
- ✅ MoE models can be created and used correctly
- ✅ Memory optimizations work on constrained hardware
- ✅ Sampling scripts work with all checkpoint formats
- ✅ Gradient checkpointing reduces memory usage
- ✅ wandb integration logs metrics correctly
- ✅ All configurations are valid and loadable
- ✅ Cross-platform compatibility

## 📈 Continuous Testing

Run tests regularly during development:
```bash
# Quick smoke test
python quick_test.py

# Full test suite
python run_tests.py

# Memory-specific tests before training
python test_memory.py
```

This ensures that changes don't break existing functionality and that the codebase remains robust across different environments and use cases.
