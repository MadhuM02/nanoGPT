# Sampling Directory

This directory contains text generation and sampling scripts.

## 📁 Contents

- `sample.py` - Standard text generation from GPT models
- `sample_moe.py` - Text generation from MoE models with expert routing

## 🚀 Usage

### Standard Sampling
```bash
# Generate text from standard model
python sampling/sample.py \
    --init_from=gpt2-medium \
    --start="Hello, I'm a language model," \
    --num_samples=5 \
    --max_new_tokens=100

# Sample from custom checkpoint
python sampling/sample.py \
    --init_from=resume \
    --out_dir=checkpoints/out/ \
    --start="Once upon a time" \
    --temperature=0.8
```

### MoE Sampling
```bash
# Generate from MoE model
python sampling/sample_moe.py \
    --init_from=resume \
    --out_dir=checkpoints/out-moe/ \
    --start="The future of AI is" \
    --num_samples=3 \
    --max_new_tokens=200

# With specific sampling parameters
python sampling/sample_moe.py \
    --checkpoint=checkpoints/out-moe/best_model.pt \
    --start="Explain quantum computing:" \
    --temperature=0.7 \
    --top_k=50
```

## 🎛️ Parameters

### Common Parameters
- `--init_from`: Model initialization ('gpt2', 'gpt2-medium', 'resume')
- `--out_dir`: Output directory for resume mode
- `--start`: Starting text prompt
- `--num_samples`: Number of samples to generate
- `--max_new_tokens`: Maximum tokens to generate
- `--temperature`: Sampling temperature (0.0 = deterministic)
- `--top_k`: Top-k sampling parameter
- `--seed`: Random seed for reproducibility

### MoE-Specific Parameters
- `--checkpoint`: Specific checkpoint file to load
- `--show_expert_usage`: Display expert routing information
- `--expert_analysis`: Detailed expert usage analysis

## 📋 Examples

### Creative Writing
```bash
python sampling/sample.py \
    --init_from=gpt2-large \
    --start="In a world where magic and technology coexist," \
    --temperature=0.9 \
    --num_samples=3 \
    --max_new_tokens=500
```

### Code Generation
```bash
python sampling/sample.py \
    --init_from=resume \
    --out_dir=checkpoints/out-code/ \
    --start="def fibonacci(n):" \
    --temperature=0.2 \
    --max_new_tokens=200
```

### Instruction Following (SFT Models)
```bash
python sampling/sample.py \
    --init_from=resume \
    --out_dir=checkpoints/out-sft/ \
    --start="Question: What is the capital of France? Answer:" \
    --temperature=0.1 \
    --max_new_tokens=50
```

## 🔧 Advanced Usage

### Batch Generation
```bash
# Generate multiple samples with different temperatures
for temp in 0.5 0.7 0.9; do
    python sampling/sample.py \
        --start="The meaning of life is" \
        --temperature=$temp \
        --num_samples=1 \
        --max_new_tokens=100
done
```

### Expert Analysis (MoE)
```bash
# Analyze expert usage patterns
python sampling/sample_moe.py \
    --checkpoint=out-moe/best_model.pt \
    --start="Artificial intelligence will" \
    --expert_analysis=True \
    --max_new_tokens=100
```

## 🎯 Use Cases

### Standard Sampling (`sample.py`)
- **Text completion**: Continue existing text
- **Creative writing**: Generate stories, poems
- **Code generation**: Complete code snippets
- **Q&A**: Answer questions with fine-tuned models

### MoE Sampling (`sample_moe.py`)
- **Expert analysis**: Understand routing patterns
- **Specialized generation**: Leverage expert specialization
- **Performance testing**: Compare MoE vs standard models
- **Research**: Study expert behavior

## 🚀 Quick Examples

```bash
# Simple text generation
python sampling/sample.py --start="AI is"

# Creative story
python sampling/sample.py \
    --start="Once upon a time in a distant galaxy," \
    --temperature=0.8 \
    --max_new_tokens=300

# Technical explanation
python sampling/sample.py \
    --start="Machine learning is" \
    --temperature=0.3 \
    --max_new_tokens=150

# MoE model with expert tracking
python sampling/sample_moe.py \
    --start="The future of computing" \
    --show_expert_usage=True
```
