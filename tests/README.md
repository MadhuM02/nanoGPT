# nanoGPT MoE Test Suite

This directory contains all test scripts for the nanoGPT Mixture-of-Experts implementation.

## 🚀 Quick Start

Run all tests:
```bash
cd tests
python run_tests.py
```

Run individual tests:
```bash
cd tests
python test_moe.py           # Test basic MoE functionality
python test_sampling.py      # Test sampling scripts
python test_memory.py        # Test memory configurations
python test_wandb_logging.py # Test wandb integration
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
