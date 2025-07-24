#!/bin/bash

# Simple distributed test to debug the issue
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Minimal Distributed Training Debug Test ==="
echo ""

# Check GPU availability
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "Running distributed training for 10 seconds to see actual errors..."

# Run with shorter timeout and better error capture
timeout 15 torchrun --standalone --nproc_per_node=2 --nnodes=1 --node_rank=0 train_sft.py \
    config=config/sft_ultra_small_config.py \
    max_iters=2 \
    eval_only=False \
    eval_interval=1 \
    log_interval=1 \
    compile=False

echo ""
echo "Exit code: $?"
