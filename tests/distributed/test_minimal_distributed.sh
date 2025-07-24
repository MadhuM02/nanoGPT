#!/bin/bash

# Minimal distributed test
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_TIMEOUT=300  # 5 minute timeout

echo "=== Minimal Distributed Test ==="

# Very quick test - just 2 iterations
timeout 120 torchrun --standalone --nproc_per_node=2 train_sft.py \
    batch_size=1 \
    block_size=128 \
    gradient_accumulation_steps=2 \
    max_iters=2 \
    eval_interval=1 \
    eval_iters=1 \
    log_interval=1 \
    compile=False \
    out_dir=out-sft-minimal-test

echo "Exit code: $?"
