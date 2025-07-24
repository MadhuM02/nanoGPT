#!/bin/bash

# Test distributed training with NCCL optimizations
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL optimizations for better stability
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800  # 30 minute timeout
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "=== Distributed Training Test with NCCL Optimizations ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NCCL_TIMEOUT: $NCCL_TIMEOUT seconds"
echo ""

# Check GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo ""
echo "Running distributed training with memory-optimized settings..."

# Use memory-efficient settings directly in command line
torchrun --standalone --nproc_per_node=2 train_sft.py \
    batch_size=1 \
    block_size=256 \
    gradient_accumulation_steps=8 \
    max_iters=10 \
    eval_interval=5 \
    eval_iters=5 \
    log_interval=1 \
    compile=False \
    out_dir=out-sft-distributed-test \
    learning_rate=1e-6 \
    warmup_iters=2

echo ""
echo "Test completed with exit code: $?"
