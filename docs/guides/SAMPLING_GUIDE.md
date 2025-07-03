# Model Sampling Guide

This guide explains how to sample (generate text) from your trained nanoGPT MoE models using the provided sampling scripts.

## 📁 Available Sampling Scripts

### 1. `sample.py` - Simple Sampling Script
- **Purpose**: Quick and simple text generation
- **Best for**: Basic sampling with minimal configuration
- **Features**: Works with both MoE and standard models

### 2. `sample_moe.py` - Advanced MoE Sampling Script  
- **Purpose**: Advanced sampling with extensive configuration options
- **Best for**: Detailed control over generation parameters
- **Features**: 
  - Comprehensive checkpoint handling with automatic prefix fixing
  - **NEW**: KV cache support for 2-10x faster generation
  - **NEW**: Performance profiling and batch generation
  - **NEW**: Robust error handling and fallback strategies
  - Flexible MoE parameter detection

## 🚀 Quick Start

### Basic Sampling with `sample.py`

1. **Default sampling** (loads from `out-moe-v100/best_model.pt`):
```bash
python sample.py
```

2. **Custom output directory**:
```bash
# Edit sample.py and change the out_dir variable
out_dir = 'your-output-directory'
```

### Advanced Sampling with `sample_moe.py`

1. **Basic usage**:
```bash
python sample_moe.py --out_dir out-moe-v100
```

2. **Custom prompt**:
```bash
python sample_moe.py --out_dir out-moe-v100 \
    --start "Once upon a time" \
    --num_samples 5 \
    --max_new_tokens 200
```

3. **Load specific checkpoint**:
```bash
python sample_moe.py --out_dir out-moe-v100 \
    --checkpoint final \
    --temperature 0.8
```

4. **🚀 NEW: Fast generation with KV cache**:
```bash
# Enable KV cache for 2-10x faster generation
python sample_moe.py --use_kv_cache \
    --max_new_tokens 500 \
    --profile
```

5. **🚀 NEW: Batch generation with performance monitoring**:
```bash
# Generate multiple samples efficiently
python sample_moe.py --use_kv_cache \
    --batch_size 4 \
    --num_samples 8 \
    --profile
```

## 📋 Complete Usage Examples

### Example 1: Creative Writing
```bash
python sample_moe.py \
    --out_dir out-moe-v100 \
    --start "In a world where artificial intelligence" \
    --num_samples 3 \
    --max_new_tokens 500 \
    --temperature 0.9 \
    --top_k 50 \
    --seed 42
```

### Example 2: Code Generation
```bash
python sample_moe.py \
    --out_dir out-moe-v100 \
    --start "def fibonacci(n):" \
    --num_samples 2 \
    --max_new_tokens 200 \
    --temperature 0.3 \
    --top_k 20
```

### Example 3: Dialog Generation
```bash
python sample_moe.py \
    --out_dir out-moe-v100 \
    --start "Human: Hello, how are you?\nAI:" \
    --num_samples 5 \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_k 40
```

### Example 4: Using a Prompt File
```bash
# Create a prompt file
echo "Write a short story about a robot learning to paint:" > my_prompt.txt

# Use it for generation
python sample_moe.py \
    --out_dir out-moe-v100 \
    --start "FILE:my_prompt.txt" \
    --num_samples 1 \
    --max_new_tokens 800
```

## ⚙️ Parameter Reference

### `sample_moe.py` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--out_dir` | `out-moe-v100` | Directory containing model checkpoints |
| `--checkpoint` | `best` | Which checkpoint to load (`best`, `final`, `latest`) |
| `--start` | `Hello, I am a language model,` | Starting prompt for generation |
| `--num_samples` | `5` | Number of samples to generate |
| `--max_new_tokens` | `300` | Maximum new tokens per sample |
| `--temperature` | `0.8` | Sampling temperature (creativity) |
| `--top_k` | `200` | Top-k sampling (vocabulary restriction) |
| `--seed` | `1337` | Random seed for reproducibility |
| `--device` | `cuda` | Device to use (`cuda`, `cpu`, `cuda:0`, etc.) |
| `--dtype` | `auto` | Model precision (`float32`, `float16`, `bfloat16`, `auto`) |
| `--compile` | `False` | Enable PyTorch 2.0 compilation |

### Temperature Guide
- **0.1-0.3**: Very focused, deterministic (good for code, facts)
- **0.4-0.7**: Balanced creativity and coherence  
- **0.8-1.0**: Creative, diverse outputs
- **1.0+**: Very creative, potentially incoherent

### Top-k Guide
- **1-20**: Very restrictive vocabulary (focused outputs)
- **50-100**: Moderate vocabulary restriction
- **200-500**: Light vocabulary restriction  
- **No limit**: Full vocabulary available

## 📂 Checkpoint Types

### Automatic Checkpoint Discovery
The scripts automatically find checkpoints in this order:

1. **`best`**: `best_model.pt` → `ckpt.pt`
2. **`final`**: `final_model.pt` → `ckpt.pt`  
3. **`latest`**: Most recently modified `.pt` file

### Checkpoint Formats Supported
- ✅ New format with `config` dict (from `train_moe_advanced.py`)
- ✅ Old format with `model_args` dict
- ✅ Compiled model artifacts (automatically cleaned)
- ✅ Both MoE and standard GPT models

## 🛠️ Troubleshooting

### Common Issues

1. **"No checkpoint found"**:
   ```bash
   # Check what's in your output directory
   ls -la out-moe-v100/
   
   # Specify exact checkpoint file
   python sample_moe.py --out_dir out-moe-v100 --checkpoint your_file.pt
   ```

2. **CUDA out of memory**:
   ```bash
   # Use CPU instead
   python sample_moe.py --device cpu
   
   # Or use smaller precision
   python sample_moe.py --dtype float16
   ```

3. **Import errors**:
   ```bash
   # Make sure you're in the right directory
   cd /path/to/nanoGPT
   
   # Check if model.py exists
   ls model.py
   ```

4. **Checkpoint format errors**:
   - The scripts handle both old and new checkpoint formats
   - If you get errors, the checkpoint might be corrupted
   - Try a different checkpoint file

### Debug Mode
```bash
# For detailed error information
python -u sample_moe.py --out_dir out-moe-v100 --num_samples 1
```

## 🎯 Tips for Better Generation

### 1. **Choose the Right Temperature**
- Low temperature (0.1-0.4) for factual, consistent text
- Medium temperature (0.5-0.8) for balanced creativity
- High temperature (0.9-1.2) for creative, diverse outputs

### 2. **Optimize Top-k**
- Small top-k (10-50) for focused, coherent text
- Medium top-k (100-200) for balanced diversity
- Large top-k (500+) for maximum vocabulary

### 3. **Craft Good Prompts**
- Be specific about the desired output format
- Include context and style indicators
- Use examples in your prompt when helpful

### 4. **Seed for Reproducibility**
- Use the same seed to reproduce exact outputs
- Change seed to get different variations of the same prompt

### 5. **Batch Generation**
- Generate multiple samples at once for variety
- Compare outputs to find the best generation

## 📊 Model Information

When loading a model, the scripts display:
- Total parameter count
- MoE configuration (if applicable)
- Model architecture details
- Checkpoint metadata

Example output:
```
Loading checkpoint from out-moe-v100/best_model.pt
Model config: n_layer=12, n_head=12, n_embd=768
MoE config: num_experts=8, top_k=2
Model loaded successfully. Total parameters: 124,439,808
```

## 🔄 Integration with Training

### Using Different Configs
The sampling scripts work with models trained using any configuration:

```bash
# Sample from memory-efficient model
python sample_moe.py --out_dir out-moe-memory

# Sample from performance-optimized model  
python sample_moe.py --out_dir out-moe-performance

# Sample from debug model
python sample_moe.py --out_dir out-moe-debug
```

### Monitoring During Training
You can sample from checkpoints while training is still running:
```bash
# Sample from the latest checkpoint
python sample_moe.py --out_dir out-moe-v100 --checkpoint latest
```

## 🎉 Ready to Generate!

Both sampling scripts are now ready to use with your trained MoE models. Start with the simple examples above and experiment with different parameters to find what works best for your use case!

## 🚀 NEW: KV Cache Performance Enhancement

### What is KV Cache?
KV Cache is a powerful optimization that dramatically speeds up text generation by avoiding redundant computations in the attention mechanism.

### Performance Benefits
- **2-10x faster generation** for long sequences (>100 tokens)
- **Linear scaling** instead of quadratic with sequence length  
- **Memory efficient** caching of attention keys and values

### When to Use KV Cache
✅ **Recommended for:**
- Long sequence generation (>100 tokens)
- Interactive applications (chatbots, assistants)
- Batch inference with multiple sequences
- Production deployments requiring low latency

⚠️ **Consider carefully for:**
- Very short sequences (<50 tokens) - overhead may outweigh benefits
- Memory-constrained environments

### Example Performance Comparison
```bash
# Standard generation (baseline)
python sample_moe.py --max_new_tokens 500 --profile
# Output: ~45 tokens/second

# With KV cache (2-10x faster)
python sample_moe.py --use_kv_cache --max_new_tokens 500 --profile  
# Output: ~120-450 tokens/second (depending on sequence length)
```

### KV Cache Usage Examples
```bash
# Basic KV cache usage
python sample_moe.py --use_kv_cache

# Long sequence generation with KV cache
python sample_moe.py --use_kv_cache --max_new_tokens 1000

# Batch generation with KV cache
python sample_moe.py --use_kv_cache --batch_size 4 --num_samples 8

# Full performance optimization
python sample_moe.py --use_kv_cache --compile --dtype bfloat16 --profile
```

### Performance Monitoring
Use `--profile` to see detailed performance metrics:
```
📊 Performance metrics:
  Tokens generated: 500
  Time taken: 2.34s  
  Tokens/second: 213.7
  Cache strategy: KV cache
  GPU memory: 1247.3 MB
```

### For more details, see: [KV Cache Guide](KV_CACHE_GUIDE.md)

**Quick commands to get started:**
```bash
# Simple sampling
python sample.py

# Advanced sampling with custom prompt
python sample_moe.py --start "Your custom prompt here" --num_samples 3

# Creative writing
python sample_moe.py --start "Once upon a time" --temperature 0.9 --max_new_tokens 500
```
