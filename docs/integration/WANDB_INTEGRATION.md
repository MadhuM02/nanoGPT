# Weights & Biases Integration Guide

This guide explains how to use the comprehensive Weights & Biases (wandb) integration in the nanoGPT MoE training script.

## Prerequisites

Install wandb if you haven't already:
```bash
pip install wandb
```

Login to your wandb account:
```bash
wandb login
```

## Basic Usage

### Enable wandb Logging

Add the `--wandb` flag when running training:

```bash
python train_moe_advanced.py --config v100 --wandb
```

### Custom Project and Run Names

Specify custom project and run names:

```bash
python train_moe_advanced.py --config v100 --wandb \
    --wandb-project "my-nanogpt-experiments" \
    --wandb-run-name "v100-8experts-12layers"
```

If you don't specify a run name, one will be auto-generated based on your configuration:
- Format: `{config}-{num_experts}E-{n_layer}L-{n_embd}D`
- Example: `v100-8E-12L-768D`

## Logged Metrics

### Training Metrics (logged every `log_interval` steps)

- **Losses:**
  - `train/loss`: Total training loss
  - `train/main_loss`: Main language modeling loss
  - `train/aux_loss`: Auxiliary load balancing loss
  - `train/perplexity`: Training perplexity (exp(loss))

- **Learning:**
  - `lr`: Current learning rate
  - `mfu`: Model FLOPS Utilization percentage
  - `timing/step_time_ms`: Time per training step in milliseconds

- **MoE-Specific:**
  - `moe/expert_usage_variance`: Variance in expert usage (load balancing)
  - `moe/avg_experts_per_token`: Average number of experts used per token
  - `moe/load_balance_loss`: Load balancing auxiliary loss

- **Hardware:**
  - `gpu/memory_allocated_mb`: GPU memory allocated in MB
  - `gpu/memory_reserved_mb`: GPU memory reserved in MB

### Validation Metrics (logged every `eval_interval` steps)

- `val/loss`: Validation loss
- `val/perplexity`: Validation perplexity
- `best_val_loss`: Best validation loss achieved so far

### Final Training Summary

- `training/final_step`: Total training steps completed
- `training/best_val_loss`: Best validation loss achieved
- `training/final_mfu`: Final MFU percentage
- `training/total_parameters`: Total model parameters
- `training/trainable_parameters`: Trainable model parameters
- `model/total_params`: Model parameter count

## Artifacts (Model Checkpoints)

The integration logs training metrics to WandB. Model checkpoints are saved locally but artifact logging is disabled to reduce storage costs and upload time.

### Best Model Checkpoints
- **Type:** `model`
- **Name:** `best-model-step-{step}`
- **When:** Saved whenever validation loss improves
- **Description:** Includes validation loss in description

### Periodic Checkpoints
- **Type:** `checkpoint`
- **Name:** `checkpoint-step-{step}`
- **When:** Saved every `checkpoint_interval` steps (default: 5000)
- **Description:** Regular training checkpoints for recovery

### Final Model
- **Type:** `model`
- **Name:** `final-model-step-{step}`
- **When:** Saved at the end of training
- **Description:** Final model state after all training

## Configuration Integration

The entire training configuration is logged to wandb, including:
- Model architecture parameters
- MoE configuration
- Training hyperparameters
- Memory optimization settings

## Tags

Auto-generated tags help organize runs:
- `config-{config_name}`: Configuration used (e.g., `config-v100`)
- `experts-{num_experts}`: Number of experts (e.g., `experts-8`)
- `layers-{n_layer}`: Number of transformer layers
- `embedding-{n_embd}`: Embedding dimension
- `MoE`: Indicates this is a Mixture-of-Experts run
- `nanoGPT`: Indicates this is a nanoGPT experiment

## Example Training Commands

### Basic V100 Training with wandb
```bash
python train_moe_advanced.py --config v100 --wandb
```

### Memory-Constrained Training
```bash
python train_moe_advanced.py --config memory --wandb \
    --wandb-project "memory-experiments" \
    --wandb-run-name "ultra-memory-efficient"
```

### Performance Optimization
```bash
python train_moe_advanced.py --config performance --wandb \
    --wandb-project "performance-tests" \
    --wandb-run-name "speed-optimized"
```

### Debug/Testing
```bash
python train_moe_advanced.py --config debug --wandb \
    --wandb-project "debug-runs" \
    --wandb-run-name "quick-test"
```

## Viewing Results

1. **Web Dashboard:** Go to [wandb.ai](https://wandb.ai) and navigate to your project
2. **Real-time Monitoring:** Watch metrics update in real-time during training
3. **Compare Runs:** Use wandb's comparison tools to analyze different configurations
4. **Download Artifacts:** Download model checkpoints directly from the web interface

## Advanced Features

### Custom Metrics

You can extend the logging by modifying the training script to include additional metrics:

```python
# Add custom metrics to the wandb_metrics dictionary
if args.wandb and wandb_run:
    train_metrics.update({
        'custom/my_metric': my_metric_value,
        'custom/another_metric': another_value
    })
    wandb.log(train_metrics)
```

### Hyperparameter Sweeps

Use wandb sweeps for hyperparameter optimization:

```yaml
# sweep.yaml
program: train_moe_advanced.py
method: bayes
metric:
  name: val/loss
  goal: minimize
parameters:
  learning_rate:
    min: 1e-5
    max: 1e-3
  num_experts:
    values: [4, 8, 16]
  expert_dropout:
    min: 0.05
    max: 0.3
```

Then run:
```bash
wandb sweep sweep.yaml
wandb agent YOUR_SWEEP_ID
```

## Troubleshooting

### wandb Not Installed
If you see a warning about wandb not being installed:
```bash
pip install wandb
```

### Login Issues
If wandb fails to initialize, make sure you're logged in:
```bash
wandb login
```

### Memory Issues with Artifacts
If logging large checkpoints causes memory issues, you can disable artifact logging by modifying the script or reducing checkpoint frequency.

### Offline Mode
For environments without internet access:
```bash
export WANDB_MODE=offline
```

## Benefits of wandb Integration

1. **Experiment Tracking:** Keep track of all your training runs
2. **Metric Visualization:** Real-time plots and analysis
3. **Model Versioning:** Automatic checkpoint management
4. **Collaboration:** Share results with your team
5. **Reproducibility:** Complete configuration and code tracking
6. **Comparison:** Easy comparison between different runs
7. **Resource Monitoring:** Track GPU usage and training efficiency

This integration makes it easy to monitor your MoE training experiments, compare different configurations, and maintain a complete record of your research.
