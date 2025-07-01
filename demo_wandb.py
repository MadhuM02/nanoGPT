#!/usr/bin/env python3
"""
Quick example demonstrating wandb integration with nanoGPT MoE training

This script runs a short training session with minimal configuration
to demonstrate the wandb logging functionality.
"""

import subprocess
import sys
import os

def check_wandb():
    """Check if wandb is installed and user is logged in"""
    try:
        import wandb
        print("✅ wandb is installed")
        
        # Check if logged in by trying to get user info
        try:
            api = wandb.Api()
            user = api.default_entity
            print(f"✅ Logged in as: {user}")
            return True
        except Exception:
            print("❌ Not logged in to wandb. Please run: wandb login")
            return False
            
    except ImportError:
        print("❌ wandb not installed. Please run: pip install wandb")
        return False

def run_quick_training():
    """Run a quick training session with wandb logging"""
    print("🚀 Starting quick MoE training with wandb logging...")
    
    # Use minimal config for quick testing
    cmd = [
        "python", "train_moe_advanced.py",
        "--config", "debug",  # Minimal config for quick training
        "--wandb",
        "--wandb-project", "nanogpt-moe-demo",
        "--wandb-run-name", "quick-demo-run"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print("Note: This will run for a few iterations to demonstrate logging")
    print("Press Ctrl+C to stop early if needed\n")
    
    try:
        # Run the training script
        result = subprocess.run(cmd, cwd="/mnt/workspace/nanoGPT", capture_output=False)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            print("🌐 Check your wandb dashboard to see the logged metrics and artifacts")
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running training: {e}")

def main():
    print("nanoGPT MoE wandb Integration Demo")
    print("=" * 40)
    
    # Check prerequisites
    if not check_wandb():
        print("\nPlease install wandb and login before running this demo:")
        print("  pip install wandb")
        print("  wandb login")
        return 1
    
    # Check if training script exists
    train_script = "/mnt/workspace/nanoGPT/train_moe_advanced.py"
    if not os.path.exists(train_script):
        print(f"❌ Training script not found: {train_script}")
        return 1
    
    print("\n📊 This demo will:")
    print("  • Run a short MoE training session")
    print("  • Log metrics to wandb in real-time")
    print("  • Save model checkpoints as artifacts")
    print("  • Demonstrate all wandb integration features")
    
    # Ask for confirmation
    response = input("\nWould you like to continue? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Demo cancelled.")
        return 0
    
    # Run the demo
    run_quick_training()
    
    print("\n🎉 Demo complete!")
    print("Visit https://wandb.ai to see your results")
    return 0

if __name__ == "__main__":
    sys.exit(main())
