#!/usr/bin/env python3
"""
Advanced Checkpoint Debug Tool for nanoGPT MoE

This script provides comprehensive analysis of checkpoint files and their compatibility
with the current model, including detailed prefix analysis and loading strategy recommendations.

Usage:
    python debug_checkpoint_advanced.py <checkpoint_path> [model_type]
    
Examples:
    python debug_checkpoint_advanced.py out/ckpt.pt
    python debug_checkpoint_advanced.py out/ckpt.pt gpt2
    python debug_checkpoint_advanced.py out/ckpt.pt moe
"""

import os
import sys
import torch
import argparse
from collections import defaultdict

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_key_patterns(keys, name="Keys"):
    """Comprehensive analysis of key patterns"""
    if not keys:
        return {
            'total_keys': 0,
            'patterns': {},
            'sample_keys': [],
            'summary': f"No {name.lower()} found"
        }
    
    key_list = list(keys)
    
    # Analyze prefixes
    prefix_patterns = {
        'module.': sum(1 for k in key_list if k.startswith('module.')),
        '_orig_mod.': sum(1 for k in key_list if k.startswith('_orig_mod.')),
        'module._orig_mod.': sum(1 for k in key_list if k.startswith('module._orig_mod.')),
        'no_prefix': sum(1 for k in key_list if not any(k.startswith(p) for p in ['module.', '_orig_mod.'])),
    }
    
    # Analyze key structure patterns
    structure_patterns = defaultdict(int)
    for key in key_list:
        parts = key.split('.')
        if len(parts) >= 2:
            structure_patterns[f"{parts[0]}.{parts[1]}"] += 1
        elif len(parts) == 1:
            structure_patterns[parts[0]] += 1
    
    # Find dominant patterns
    dominant_prefix = max(prefix_patterns.keys(), key=lambda k: prefix_patterns[k])
    dominant_structure = max(structure_patterns.keys(), key=lambda k: structure_patterns[k])
    
    return {
        'total_keys': len(key_list),
        'prefix_patterns': prefix_patterns,
        'structure_patterns': dict(structure_patterns),
        'dominant_prefix': dominant_prefix,
        'dominant_structure': dominant_structure,
        'sample_keys': key_list[:10],
        'summary': f"{len(key_list)} {name.lower()}, mostly '{dominant_prefix}' prefix"
    }

def compare_key_structures(model_keys, checkpoint_keys):
    """Compare model and checkpoint key structures to find compatibility"""
    comparisons = []
    
    # Direct comparison
    exact_matches = len(model_keys.intersection(checkpoint_keys))
    comparisons.append({
        'method': 'Exact match (no transformation)',
        'matches': exact_matches,
        'match_ratio': exact_matches / len(model_keys) if model_keys else 0,
        'transformation': None
    })
    
    # Try removing each prefix from checkpoint keys
    prefixes_to_remove = ['module.', '_orig_mod.', 'module._orig_mod.']
    
    for prefix in prefixes_to_remove:
        transformed_checkpoint_keys = set()
        transform_count = 0
        
        for key in checkpoint_keys:
            if key.startswith(prefix):
                transformed_checkpoint_keys.add(key[len(prefix):])
                transform_count += 1
            else:
                transformed_checkpoint_keys.add(key)
        
        if transform_count > 0:  # Only consider if we actually removed some prefixes
            matches = len(model_keys.intersection(transformed_checkpoint_keys))
            comparisons.append({
                'method': f'Remove "{prefix}" prefix from checkpoint',
                'matches': matches,
                'match_ratio': matches / len(model_keys) if model_keys else 0,
                'transformation': f'remove_prefix:{prefix}',
                'keys_transformed': transform_count
            })
    
    # Try adding each prefix to checkpoint keys
    for prefix in prefixes_to_remove:
        transformed_checkpoint_keys = set(prefix + key for key in checkpoint_keys)
        matches = len(model_keys.intersection(transformed_checkpoint_keys))
        comparisons.append({
            'method': f'Add "{prefix}" prefix to checkpoint',
            'matches': matches,
            'match_ratio': matches / len(model_keys) if model_keys else 0,
            'transformation': f'add_prefix:{prefix}',
            'keys_transformed': len(checkpoint_keys)
        })
    
    # Sort by match ratio (best first)
    comparisons.sort(key=lambda x: x['match_ratio'], reverse=True)
    
    return comparisons

def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file in detail"""
    print(f"🔍 Analyzing checkpoint: {checkpoint_path}")
    print("=" * 80)
    
    # Basic file info
    if not os.path.exists(checkpoint_path):
        print(f"❌ File does not exist: {checkpoint_path}")
        return
    
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
    print(f"📁 File size: {file_size:.1f} MB")
    
    try:
        # Load checkpoint
        print("📦 Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Analyze checkpoint structure
        print("\n📋 Checkpoint Structure:")
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                if key == 'model' and isinstance(value, dict):
                    print(f"  {key}: state_dict with {len(value)} keys")
                elif isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} keys")
                elif isinstance(value, (int, float)):
                    print(f"  {key}: {value}")
                elif hasattr(value, '__len__'):
                    print(f"  {key}: {type(value).__name__} with length {len(value)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print(f"  Direct state_dict with {len(checkpoint)} keys")
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            model_state_dict = checkpoint
        
        # Analyze keys
        print(f"\n🔑 Key Analysis:")
        checkpoint_keys = set(model_state_dict.keys())
        key_analysis = analyze_key_patterns(checkpoint_keys, "Checkpoint keys")
        
        print(f"  Total keys: {key_analysis['total_keys']}")
        print(f"  Prefix breakdown:")
        for prefix, count in key_analysis['prefix_patterns'].items():
            if count > 0:
                percentage = (count / key_analysis['total_keys']) * 100
                print(f"    {prefix}: {count} keys ({percentage:.1f}%)")
        
        print(f"  Sample keys:")
        for i, key in enumerate(key_analysis['sample_keys'][:5]):
            print(f"    {i+1}. {key}")
        
        print(f"  Top key structures:")
        sorted_structures = sorted(key_analysis['structure_patterns'].items(), 
                                 key=lambda x: x[1], reverse=True)
        for structure, count in sorted_structures[:5]:
            print(f"    {structure}: {count} keys")
        
        return model_state_dict, key_analysis
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None

def analyze_model_compatibility(checkpoint_path, model_type=None):
    """Analyze compatibility with different model types"""
    print(f"\n🤖 Model Compatibility Analysis")
    print("=" * 50)
    
    # Load checkpoint analysis
    model_state_dict, checkpoint_analysis = analyze_checkpoint(checkpoint_path)
    if model_state_dict is None:
        return
    
    checkpoint_keys = set(model_state_dict.keys())
    
    # Try to create models and compare
    models_to_try = []
    
    if model_type:
        models_to_try.append(model_type)
    else:
        # Try common model configurations
        models_to_try.extend(['gpt2', 'moe'])
    
    for model_name in models_to_try:
        print(f"\n📊 Testing compatibility with {model_name} model:")
        
        try:
            # Import and create model
            if model_name == 'gpt2':
                from model import GPTConfig, GPT
                config = GPTConfig(
                    block_size=1024,
                    vocab_size=50304,
                    n_layer=12,
                    n_head=12,
                    n_embd=768,
                )
                model = GPT(config)
            elif model_name == 'moe':
                try:
                    from model_moe import GPTConfig, GPT
                    config = GPTConfig(
                        block_size=1024,
                        vocab_size=50304,
                        n_layer=12,
                        n_head=12,
                        n_embd=768,
                        num_experts=8,
                        num_experts_active=2,
                    )
                    model = GPT(config)
                except ImportError:
                    print(f"  ⚠️ MoE model not available (model_moe.py not found)")
                    continue
            else:
                print(f"  ⚠️ Unknown model type: {model_name}")
                continue
            
            # Analyze model keys
            model_keys = set(model.state_dict().keys())
            model_analysis = analyze_key_patterns(model_keys, "Model keys")
            
            print(f"  Model keys: {model_analysis['total_keys']}")
            print(f"  Model structure: {model_analysis['dominant_prefix']} prefix")
            
            # Compare compatibility
            comparisons = compare_key_structures(model_keys, checkpoint_keys)
            
            print(f"  Compatibility analysis:")
            for i, comp in enumerate(comparisons[:3]):  # Top 3 methods
                match_percentage = comp['match_ratio'] * 100
                symbol = "✅" if match_percentage > 90 else "⚠️" if match_percentage > 50 else "❌"
                print(f"    {symbol} {comp['method']}: {comp['matches']}/{len(model_keys)} keys ({match_percentage:.1f}%)")
                if 'keys_transformed' in comp:
                    print(f"       (would transform {comp['keys_transformed']} keys)")
            
            # Show recommended loading strategy
            best_method = comparisons[0]
            if best_method['match_ratio'] > 0.9:
                print(f"  🎯 Recommended: {best_method['method']}")
                if best_method['transformation']:
                    print(f"     Transformation needed: {best_method['transformation']}")
            else:
                print(f"  ⚠️ Poor compatibility - may need manual intervention")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {e}")

def generate_loading_code(checkpoint_path, model_type=None):
    """Generate code snippet for loading this checkpoint"""
    print(f"\n💻 Loading Code Generator")
    print("=" * 40)
    
    model_state_dict, checkpoint_analysis = analyze_checkpoint(checkpoint_path)
    if model_state_dict is None:
        return
    
    checkpoint_keys = set(model_state_dict.keys())
    
    # Analyze the best loading strategy
    print("Based on the analysis, here's recommended loading code:")
    print("\n```python")
    print("import torch")
    print("from train_moe_advanced import load_checkpoint")
    print("")
    print("# Load your model")
    print("# model = ... (your model initialization)")
    print("")
    print(f"# Load checkpoint with robust handling")
    print(f"config, step, best_val_loss = load_checkpoint(")
    print(f"    '{checkpoint_path}',")
    print(f"    model,")
    print(f"    optimizer=None,  # or your optimizer")
    print(f"    device='cuda'")
    print(f")")
    print("```")
    
    # Also show manual loading if needed
    dominant_prefix = checkpoint_analysis['prefix_patterns'] if checkpoint_analysis else {}
    has_module = dominant_prefix.get('module.', 0) > 0
    has_orig_mod = dominant_prefix.get('_orig_mod.', 0) > 0
    
    if has_module or has_orig_mod:
        print("\nIf you need manual prefix handling:")
        print("\n```python")
        print("# Manual loading with prefix handling")
        print("checkpoint = torch.load(checkpoint_path, map_location='cpu')")
        print("state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint")
        print("")
        print("# Remove problematic prefixes")
        print("new_state_dict = {}")
        print("for key, value in state_dict.items():")
        
        if has_orig_mod:
            print("    if key.startswith('_orig_mod.'):")
            print("        new_key = key[len('_orig_mod.'):]")
        if has_module:
            print("    elif key.startswith('module.'):")
            print("        new_key = key[len('module.'):]")
        
        print("    else:")
        print("        new_key = key")
        print("    new_state_dict[new_key] = value")
        print("")
        print("model.load_state_dict(new_state_dict)")
        print("```")

def main():
    parser = argparse.ArgumentParser(description='Advanced Checkpoint Analysis Tool')
    parser.add_argument('checkpoint_path', help='Path to checkpoint file')
    parser.add_argument('model_type', nargs='?', help='Model type to test (gpt2, moe)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("🔧 nanoGPT Advanced Checkpoint Debugger")
    print("=" * 80)
    
    # Basic checkpoint analysis
    analyze_checkpoint(args.checkpoint_path)
    
    # Model compatibility analysis
    analyze_model_compatibility(args.checkpoint_path, args.model_type)
    
    # Generate loading code
    generate_loading_code(args.checkpoint_path, args.model_type)
    
    print("\n" + "=" * 80)
    print("✨ Analysis complete!")
    print("💡 Use the enhanced load_checkpoint() function in train_moe_advanced.py")
    print("   for automatic handling of most checkpoint compatibility issues.")

if __name__ == "__main__":
    main()
