#!/usr/bin/env python3
"""
Checkpoint debugging utility for nanoGPT MoE models
Helps diagnose and fix checkpoint loading issues
"""

import os
import sys
import torch
import argparse
from collections import defaultdict

def analyze_checkpoint(ckpt_path):
    """Analyze checkpoint structure and identify potential issues"""
    print(f"🔍 Analyzing checkpoint: {ckpt_path}")
    print("=" * 60)
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        print(f"📋 Checkpoint top-level keys: {list(checkpoint.keys())}")
        
        # Analyze model state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            print(f"\n📊 Model state dict analysis:")
            print(f"   Total parameters: {len(state_dict)} entries")
            
            # Analyze key prefixes
            prefixes = defaultdict(int)
            for key in state_dict.keys():
                parts = key.split('.')
                if len(parts) > 1:
                    prefixes[parts[0]] += 1
                if len(parts) > 2:
                    prefixes[f"{parts[0]}.{parts[1]}"] += 1
                if len(parts) > 3:
                    prefixes[f"{parts[0]}.{parts[1]}.{parts[2]}"] += 1
            
            print(f"\n🏷️  Key prefix analysis:")
            for prefix, count in sorted(prefixes.items()):
                if count > 1:  # Only show prefixes with multiple occurrences
                    print(f"   {prefix}: {count} keys")
            
            # Check for common problematic prefixes
            problematic_prefixes = ['module.', '_orig_mod.', 'module._orig_mod.']
            found_prefixes = []
            
            for prefix in problematic_prefixes:
                if any(key.startswith(prefix) for key in state_dict.keys()):
                    found_prefixes.append(prefix)
            
            if found_prefixes:
                print(f"\n⚠️  Found problematic prefixes: {found_prefixes}")
                print("   These indicate the model was saved with DDP/compilation wrappers")
            else:
                print(f"\n✅ No problematic prefixes found")
            
            # Sample a few keys
            sample_keys = list(state_dict.keys())[:10]
            print(f"\n📝 Sample keys (first 10):")
            for key in sample_keys:
                print(f"   {key}")
            
            if len(state_dict) > 10:
                print(f"   ... and {len(state_dict) - 10} more")
        
        # Analyze config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"\n⚙️  Configuration:")
            important_keys = ['n_layer', 'n_head', 'n_embd', 'num_experts', 'top_k_experts', 'vocab_size', 'block_size']
            for key in important_keys:
                if key in config:
                    print(f"   {key}: {config[key]}")
            
            # Check for MoE
            if config.get('num_experts', 1) > 1:
                print(f"   🧠 MoE Model: {config['num_experts']} experts, top-{config.get('top_k_experts', 1)}")
            else:
                print(f"   🧠 Standard GPT Model")
        
        elif 'model_args' in checkpoint:
            print(f"\n⚙️  Legacy model_args format detected")
            model_args = checkpoint['model_args']
            print(f"   Keys: {list(model_args.keys())}")
        
        # Analyze other metadata
        other_keys = [k for k in checkpoint.keys() if k not in ['model', 'config', 'model_args']]
        if other_keys:
            print(f"\n📎 Other metadata:")
            for key in other_keys:
                value = checkpoint[key]
                if isinstance(value, (int, float, str)):
                    print(f"   {key}: {value}")
                else:
                    print(f"   {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return False

def fix_checkpoint(ckpt_path, output_path=None):
    """Fix checkpoint by removing problematic prefixes"""
    if output_path is None:
        base, ext = os.path.splitext(ckpt_path)
        output_path = f"{base}_fixed{ext}"
    
    print(f"\n🔧 Fixing checkpoint...")
    print(f"   Input: {ckpt_path}")
    print(f"   Output: {output_path}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
            
            # Remove problematic prefixes
            unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
            keys_fixed = 0
            
            for prefix in unwanted_prefixes:
                keys_to_update = []
                for k in list(state_dict.keys()):
                    if k.startswith(prefix):
                        keys_to_update.append((k, k[len(prefix):]))
                
                for old_key, new_key in keys_to_update:
                    state_dict[new_key] = state_dict.pop(old_key)
                    keys_fixed += 1
            
            print(f"   Fixed {keys_fixed} parameter keys")
            
            # Save fixed checkpoint
            torch.save(checkpoint, output_path)
            print(f"✅ Fixed checkpoint saved to: {output_path}")
            
            return output_path
        else:
            print("❌ No 'model' key found in checkpoint")
            return None
            
    except Exception as e:
        print(f"❌ Error fixing checkpoint: {e}")
        return None

def test_loading(ckpt_path):
    """Test loading the checkpoint with the sampling script logic"""
    print(f"\n🧪 Testing checkpoint loading...")
    
    try:
        from model import GPTConfig, GPT
        
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'config' in checkpoint:
            config = checkpoint['config']
            gptconf = GPTConfig(
                block_size=config.get('block_size', 1024),
                vocab_size=config.get('vocab_size', 50304),
                n_layer=config.get('n_layer', 12),
                n_head=config.get('n_head', 12),
                n_embd=config.get('n_embd', 768),
                dropout=0.0,
                bias=config.get('bias', True),
                num_experts=config.get('num_experts', 1),
                top_k_experts=config.get('top_k_experts', 1),
                load_balancing_loss_coef=config.get('load_balancing_loss_coef', 0.01),
                expert_dropout=0.0,
                capacity_factor=config.get('capacity_factor', 1.0),
                expert_activation=config.get('expert_activation', 'gelu')
            )
        elif 'model_args' in checkpoint:
            gptconf = GPTConfig(**checkpoint['model_args'])
            gptconf.dropout = 0.0
            if hasattr(gptconf, 'expert_dropout'):
                gptconf.expert_dropout = 0.0
        else:
            print("❌ Cannot determine model configuration")
            return False
        
        # Create model
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        
        # Apply the same prefix removal logic as in the sampling scripts
        unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
        for prefix in unwanted_prefixes:
            keys_to_update = []
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    keys_to_update.append((k, k[len(prefix):]))
            
            for old_key, new_key in keys_to_update:
                state_dict[new_key] = state_dict.pop(old_key)
        
        # Try to load state dict
        model.load_state_dict(state_dict)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✅ Model loaded successfully!")
        print(f"   Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Debug and fix nanoGPT MoE checkpoints')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--fix', action='store_true', help='Fix the checkpoint by removing problematic prefixes')
    parser.add_argument('--output', help='Output path for fixed checkpoint')
    parser.add_argument('--test', action='store_true', help='Test loading the checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return 1
    
    # Always analyze first
    success = analyze_checkpoint(args.checkpoint)
    if not success:
        return 1
    
    # Fix if requested
    if args.fix:
        fixed_path = fix_checkpoint(args.checkpoint, args.output)
        if fixed_path and args.test:
            test_loading(fixed_path)
    elif args.test:
        test_loading(args.checkpoint)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
