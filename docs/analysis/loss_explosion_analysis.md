# SFT Training Analysis: Loss Explosion Causes and Solutions

## Observed Loss Pattern
From your training log:
- Iterations 477-497: Loss around 39-40 (stable)
- Iterations 498-507: Loss jumps to 41-43 (explosion)
- Iterations 508+: Loss drops to 34-35 (recovery and improvement)

## Potential Causes of Loss Explosion

### 1. Learning Rate Issues
**Most Likely Cause**: The learning rate schedule might be causing spikes.

Current config:
- learning_rate = 5e-6
- warmup_iters = 500 (you're past this)
- lr_decay_iters = 12000
- Using cosine decay

**Problem**: Around iteration 500, you might be hitting a point where the cosine decay creates an unfavorable learning rate.

### 2. Gradient Accumulation and Clipping
Current settings:
- gradient_accumulation_steps = 4
- grad_clip = 1.0

**Problem**: If some batches have very different loss magnitudes, gradient accumulation can amplify instabilities.

### 3. Data Quality Issues
**Problem**: The SFT dataset might have some examples that are:
- Much longer/shorter than typical
- Contain unusual tokens or patterns
- Have formatting inconsistencies

### 4. MoE Load Balancing
Your model has 8 experts with top-2 routing. Load balancing loss could be causing instabilities.

### 5. Float16 Precision Issues
Using float16 can sometimes cause numerical instabilities, especially with large gradients.

## Recommended Solutions

### Immediate Fixes for Current Config:

1. **Lower Learning Rate**
```python
learning_rate = 1e-6  # Reduce from 5e-6
```

2. **Increase Gradient Clipping**
```python
grad_clip = 0.5  # Reduce from 1.0 for more aggressive clipping
```

3. **Adjust Warmup/Decay**
```python
warmup_iters = 1000  # Longer warmup
lr_decay_iters = 10000  # Start decay earlier
```

4. **Add Learning Rate Monitoring**
The current config should log LR, but let's make it more visible.

### Advanced Solutions:

1. **Use Linear Decay Instead of Cosine**
2. **Add Learning Rate Restarts**
3. **Implement Loss Spike Detection and Recovery**
4. **Use Mixed Precision with Loss Scaling**

## Quick Config Update
