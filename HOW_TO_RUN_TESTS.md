# How to Run All Tests - Quick Reference

## ✅ **Recommended: Use the Test Runner**
```bash
cd tests
python utils/run_tests.py
```

This will:
- Discover all test files in all categories
- Run them sequentially with proper error handling  
- Provide a comprehensive summary
- Show which tests passed/failed

## 🏃‍♂️ **Quick Options**

### Run by Category
```bash
cd tests

# All tests
python -m pytest . -v

# By category
python -m pytest sft/ -v
python -m pytest memory/ -v  
python -m pytest kv_cache/ -v
python -m pytest moe/ -v
```

### Individual Tests
```bash
cd tests

# Key functionality tests
python sft/test_sft_attention.py
python memory/test_memory.py
python moe/test_moe.py
python kv_cache/test_kv_cache_fix.py
```

### Quick Smoke Test
```bash
cd tests
python utils/quick_test.py
```

## 📊 **Expected Output**
```
nanoGPT Test Suite
========================================
Found 24 test files:
  sft/test_sft_attention.py
  sft/test_sft_loss_masking.py
  memory/test_memory.py
  kv_cache/test_kv_cache_fix.py
  moe/test_moe.py
  [... more tests ...]

============================================================
Running sft/test_sft_attention.py
============================================================
✅ sft/test_sft_attention.py PASSED (2.1s)

[... test results ...]

============================================================
TEST SUMMARY
============================================================
Tests run: 24
Passed: 22
Failed: 2
Total time: 45.2s

🎉 All tests passed!
```

## 🐛 **If Tests Fail**

1. **Import Errors**: Tests should auto-fix paths, but if issues persist:
   ```bash
   cd tests
   python fix_imports.py
   ```

2. **CUDA Memory**: Tests auto-fallback to CPU if GPU unavailable

3. **Missing Dependencies**: Install with:
   ```bash
   pip install torch tiktoken wandb numpy
   ```

## 📂 **Test Organization**
All tests are organized by functionality:
- `sft/` - Supervised Fine-Tuning
- `kv_cache/` - KV Cache functionality  
- `memory/` - Memory management
- `moe/` - Mixture-of-Experts
- `sampling/` - Text generation
- `wandb/` - W&B integration
- `distributed/` - Multi-GPU training
- `debug/` - Debug utilities
- `analysis/` - Performance analysis
- `utils/` - Test management tools
