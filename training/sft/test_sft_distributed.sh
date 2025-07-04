#!/bin/bash
# Test script to verify SFT training works in both single GPU and distributed modes

echo "🧪 Testing SFT Training Modes"
echo "=============================="

# Test 1: Single GPU mode
echo "📝 Test 1: Single GPU mode"
echo "Running: python train_sft.py config=config/sft_distributed_config.py max_iters=5 eval_only=False eval_interval=5"

python train_sft.py config=config/sft_distributed_config.py max_iters=5 eval_only=False eval_interval=5

if [ $? -eq 0 ]; then
    echo "✅ Single GPU test PASSED"
else
    echo "❌ Single GPU test FAILED"
    exit 1
fi

echo ""
echo "📝 Test 2: Distributed mode (if available)"

# Check if we have multiple GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Available GPUs: $GPU_COUNT"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Running: torchrun --nproc_per_node=2 train_sft.py config=config/sft_distributed_config.py max_iters=5 eval_only=False eval_interval=5"
    
    # Set environment variables for memory optimization
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_LAUNCH_BLOCKING=1
    
    torchrun --nproc_per_node=2 train_sft.py config=config/sft_distributed_config.py max_iters=5 eval_only=False eval_interval=5
    
    if [ $? -eq 0 ]; then
        echo "✅ Distributed test PASSED"
    else
        echo "❌ Distributed test FAILED"
        exit 1
    fi
else
    echo "⚠️  Skipping distributed test (only 1 GPU available)"
fi

echo ""
echo "🎉 All tests completed successfully!"
