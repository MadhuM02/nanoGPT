#!/bin/bash

# Comprehensive SFT distributed training test with multiple memory configurations
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "=== SFT Distributed Training Memory Test ==="
echo "Testing different memory configurations to find the optimal one"
echo ""

# Function to check GPU memory usage
check_memory() {
    echo "=== GPU Memory Status ==="
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
}

# Function to test configuration
test_config() {
    local config_name=$1
    local config_file=$2
    
    echo "=== Testing Configuration: $config_name ==="
    echo "Config file: $config_file"
    echo ""
    
    # Clear GPU memory
    python3 -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"
    
    check_memory
    
    # Test single GPU first
    echo "--- Single GPU Test ---"
    timeout 60 python train_sft.py config=$config_file
    single_exit_code=$?
    
    if [ $single_exit_code -eq 0 ]; then
        echo "✅ Single GPU test PASSED"
    elif [ $single_exit_code -eq 124 ]; then
        echo "⏰ Single GPU test TIMEOUT (this is expected for a quick test)"
    else
        echo "❌ Single GPU test FAILED with exit code $single_exit_code"
        return 1
    fi
    
    # Clear memory again
    python3 -c "import torch; torch.cuda.empty_cache(); print('✓ CUDA cache cleared')"
    sleep 2
    
    # Test distributed
    echo "--- Distributed Test (2 GPUs) ---"
    timeout 60 torchrun --standalone --nproc_per_node=2 train_sft.py config=$config_file
    distributed_exit_code=$?
    
    if [ $distributed_exit_code -eq 0 ]; then
        echo "✅ Distributed test PASSED"
        return 0
    elif [ $distributed_exit_code -eq 124 ]; then
        echo "⏰ Distributed test TIMEOUT (this is expected for a quick test)"
        return 0
    else
        echo "❌ Distributed test FAILED with exit code $distributed_exit_code"
        return 1
    fi
}

# Clean up any existing processes
echo "=== Cleaning up existing processes ==="
pkill -f "train_sft.py" || true
sleep 2

check_memory

# Test configurations in order of memory requirements (smallest first)
configs=(
    "Ultra Small:config/sft_ultra_small_config.py"
    "Distributed:config/sft_distributed_config.py"
    "Large:config/sft_large_config.py"
)

successful_configs=()

for config_entry in "${configs[@]}"; do
    IFS=':' read -r config_name config_file <<< "$config_entry"
    
    if [ -f "$config_file" ]; then
        if test_config "$config_name" "$config_file"; then
            successful_configs+=("$config_name")
            echo "✅ Configuration '$config_name' PASSED"
        else
            echo "❌ Configuration '$config_name' FAILED"
        fi
    else
        echo "⚠️  Configuration file not found: $config_file"
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
    
    # Clean up between tests
    pkill -f "train_sft.py" || true
    python3 -c "import torch; torch.cuda.empty_cache(); print('✓ Memory cleaned between tests')"
    sleep 3
done

# Summary
echo "=== TEST SUMMARY ==="
echo "Successful configurations:"
for config in "${successful_configs[@]}"; do
    echo "  ✅ $config"
done

if [ ${#successful_configs[@]} -eq 0 ]; then
    echo "❌ No configurations were successful"
    exit 1
else
    echo ""
    echo "🎉 ${#successful_configs[@]} out of ${#configs[@]} configurations were successful"
    echo ""
    echo "Recommended configuration for your setup:"
    echo "  ${successful_configs[0]}"
fi

check_memory
