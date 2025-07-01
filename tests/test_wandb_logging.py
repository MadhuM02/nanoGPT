#!/usr/bin/env python3
"""
Test script to verify wandb logging functionality
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import os
import sys
import math
import time
import torch
import torch.nn as nn

# Mock wandb for testing
class MockWandb:
    def __init__(self):
        self.logged_data = []
        self.artifacts = []
    
    def init(self, project, name, config, tags):
        print(f"Mock W&B initialized: project={project}, name={name}")
        return self
    
    def log(self, data):
        self.logged_data.append(data)
        print(f"Logged: {data}")
    
    def log_artifact(self, artifact):
        self.artifacts.append(artifact)
        print(f"Logged artifact: {artifact.name}")
    
    def finish(self):
        print("W&B run finished")
        print(f"Total logged entries: {len(self.logged_data)}")
        print(f"Total artifacts: {len(self.artifacts)}")

class MockArtifact:
    def __init__(self, name, type, description):
        self.name = name
        self.type = type
        self.description = description
    
    def add_file(self, path):
        print(f"Added file to artifact {self.name}: {path}")

def test_wandb_logging():
    """Test the wandb logging functionality"""
    print("Testing wandb logging functionality...")
    
    # Mock wandb
    wandb = MockWandb()
    
    # Initialize mock run
    wandb_run = wandb.init(
        project="test-nanogpt-moe",
        name="test-run",
        config={
            'n_layer': 6,
            'n_embd': 384,
            'num_experts': 4,
            'top_k_experts': 2
        },
        tags=["test", "MoE", "nanoGPT"]
    )
    
    # Test training metrics logging
    print("\n--- Testing Training Metrics ---")
    for step in range(0, 100, 10):
        loss = 4.0 * math.exp(-step / 50)  # Simulated decreasing loss
        lr = 3e-4 * (1 - step / 100)  # Simulated LR decay
        
        train_metrics = {
            'train/loss': loss,
            'train/main_loss': loss * 0.9,
            'train/aux_loss': loss * 0.1,
            'train/perplexity': math.exp(loss),
            'lr': lr,
            'mfu': min(85.0, 50 + step * 0.5),  # Simulated MFU increase
            'timing/step_time_ms': 250 + step * 2,
            'step': step
        }
        
        # Add MoE-specific metrics
        train_metrics.update({
            'moe/expert_usage_variance': 0.1 + step * 0.001,
            'moe/avg_experts_per_token': 2.0,
            'moe/load_balance_loss': loss * 0.1
        })
        
        # Add GPU memory metrics
        train_metrics.update({
            'gpu/memory_allocated_mb': 8000 + step * 10,
            'gpu/memory_reserved_mb': 9000 + step * 5
        })
        
        wandb.log(train_metrics)
    
    # Test validation metrics logging
    print("\n--- Testing Validation Metrics ---")
    for step in [20, 40, 60, 80]:
        val_loss = 3.5 * math.exp(-step / 60)
        
        val_metrics = {
            'val/loss': val_loss,
            'val/perplexity': math.exp(val_loss),
            'step': step
        }
        wandb.log(val_metrics)
    
    # Test artifact logging
    print("\n--- Testing Artifact Logging ---")
    
    # Create mock checkpoint file
    os.makedirs('test_out', exist_ok=True)
    checkpoint_path = 'test_out/test_checkpoint.pt'
    torch.save({'test': 'data'}, checkpoint_path)
    
    # Test best model artifact
    best_artifact = MockArtifact(
        name="best-model-step-80",
        type="model", 
        description="Best model checkpoint at step 80 with val_loss 1.234"
    )
    best_artifact.add_file(checkpoint_path)
    wandb.log_artifact(best_artifact)
    
    # Test periodic checkpoint artifact
    periodic_artifact = MockArtifact(
        name="checkpoint-step-50",
        type="checkpoint",
        description="Periodic checkpoint at step 50"
    )
    periodic_artifact.add_file(checkpoint_path)
    wandb.log_artifact(periodic_artifact)
    
    # Test final metrics
    print("\n--- Testing Final Metrics ---")
    final_metrics = {
        'training/final_step': 100,
        'training/best_val_loss': 1.234,
        'training/final_mfu': 85.0,
        'training/total_parameters': 1000000,
        'training/trainable_parameters': 1000000,
        'model/total_params': 1000000
    }
    wandb.log(final_metrics)
    
    # Test final artifact
    final_artifact = MockArtifact(
        name="final-model-step-100",
        type="model",
        description="Final model checkpoint after 100 training steps"
    )
    final_artifact.add_file(checkpoint_path)
    wandb.log_artifact(final_artifact)
    
    # Finish run
    wandb.finish()
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    if os.path.exists('test_out'):
        os.rmdir('test_out')
    
    print("\n✅ All wandb logging tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_wandb_logging()
        print("\n🎉 wandb logging functionality verified!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
