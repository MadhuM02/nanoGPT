"""
Advanced sampling script for MoE models
Supports loading from various checkpoint formats and provides better error handling
Enhanced with KV cache support for efficient generation
"""
import os
import sys
import pickle
import argparse
from contextlib import nullcontext
import torch
import tiktoken

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GPTConfig, GPT
from kv_cache import add_kv_cache_support

def find_checkpoint(out_dir, checkpoint_type='best'):
    """Find the appropriate checkpoint file in the output directory"""
    possible_files = []
    
    if checkpoint_type == 'best':
        possible_files = ['best_model.pt', 'ckpt.pt']
    elif checkpoint_type == 'final':
        possible_files = ['final_model.pt', 'ckpt.pt']
    elif checkpoint_type == 'latest':
        # Find the most recent checkpoint file
        checkpoint_files = [f for f in os.listdir(out_dir) if f.endswith('.pt')]
        if checkpoint_files:
            # Sort by modification time, most recent first
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(out_dir, x)), reverse=True)
            possible_files = checkpoint_files
    else:
        # Specific checkpoint file name
        possible_files = [checkpoint_type]
    
    for filename in possible_files:
        ckpt_path = os.path.join(out_dir, filename)
        if os.path.exists(ckpt_path):
            return ckpt_path
    
    return None

def load_model_from_checkpoint(ckpt_path, device):
    """Load model from checkpoint with proper error handling"""
    print(f"Loading checkpoint from {ckpt_path}")
    
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        if 'config' in checkpoint:
            # New format with config object
            config = checkpoint['config']
            gptconf = GPTConfig(
                block_size=config.get('block_size', 1024),
                vocab_size=config.get('vocab_size', 50304),
                n_layer=config.get('n_layer', 12),
                n_head=config.get('n_head', 12),
                n_embd=config.get('n_embd', 768),
                dropout=config.get('dropout', 0.0),  # Set to 0 for sampling
                bias=config.get('bias', True),
                # MoE parameters
                num_experts=config.get('num_experts', 1),
                top_k_experts=config.get('top_k_experts', 1),
                load_balancing_loss_coef=config.get('load_balancing_loss_coef', 0.01),
                expert_dropout=0.0,  # Disable dropout for sampling
                capacity_factor=config.get('capacity_factor', 1.0),
                expert_activation=config.get('expert_activation', 'gelu')
            )
        elif 'model_args' in checkpoint:
            # Old format with model_args
            gptconf = GPTConfig(**checkpoint['model_args'])
            # Disable dropout for sampling
            gptconf.dropout = 0.0
            gptconf.expert_dropout = 0.0
        else:
            raise ValueError("Checkpoint format not recognized. Missing 'config' or 'model_args'")
        
        print(f"Model config: {gptconf}")
        
        # Create model
        model = GPT(gptconf)
        
        # Load state dict
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Handle compiled model artifacts and DDP prefixes
        unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
        for prefix in unwanted_prefixes:
            keys_to_update = []
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    keys_to_update.append((k, k[len(prefix):]))
            
            for old_key, new_key in keys_to_update:
                state_dict[new_key] = state_dict.pop(old_key)
        
        # Load the state dict
        model.load_state_dict(state_dict)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully. Total parameters: {total_params:,}")
        
        if hasattr(model, 'get_moe_info'):
            moe_info = model.get_moe_info()
            print(f"MoE info: {moe_info}")
        
        return model, checkpoint
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

def generate_with_kv_cache(model, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
    """
    Enhanced generation with KV cache support for better efficiency
    """
    import torch.nn.functional as F
    
    if not use_cache:
        # Fallback to original generate method
        return model.generate(idx, max_new_tokens, temperature, top_k)
    
    # Ensure model has KV cache support
    if not hasattr(model, 'forward_with_cache'):
        print("Adding KV cache support to model...")
        add_kv_cache_support(model)
    
    batch_size, seq_len = idx.shape
    device = idx.device
    
    # Initialize KV cache state
    past_key_values = None
    current_pos = 0
    
    # First forward pass with the full prompt
    with torch.no_grad():
        if seq_len > 0:
            logits, past_key_values = model.forward_with_cache(
                idx, past_key_values, use_cache=True, start_pos=0
            )
            current_pos = seq_len
        else:
            # Empty prompt case
            logits = None
    
    # Generate tokens one by one
    for step in range(max_new_tokens):
        if step == 0 and logits is not None:
            # Use logits from prompt processing for the first token
            next_logits = logits[:, -1, :] / temperature
        else:
            # For subsequent tokens, use the last generated token as input
            # The prompt has already been processed, so we only need the last token
            next_token_input = idx[:, -1:] if step > 0 else idx
            
            # Forward pass for a single token (or the full prompt if it was empty)
            logits, past_key_values = model.forward_with_cache(
                next_token_input, past_key_values, use_cache=True, start_pos=current_pos
            )
            
            # Update position for the next token
            if step == 0:
                current_pos = idx.size(1) # After processing prompt
            current_pos += 1
            
            next_logits = logits[:, -1, :] / temperature
        
        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float('Inf')
        
        # Sample next token
        probs = F.softmax(next_logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop if we hit max context length
        if idx.size(1) >= model.config.block_size:
            break
    
    return idx

def setup_tokenizer(checkpoint):
    """Setup tokenizer based on checkpoint metadata"""
    # Try to load meta from checkpoint
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        vocab_size = len(stoi)
    else:
        # Default to GPT-2 encodings
        print("No meta.pkl found, using GPT-2 encodings...")
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
        vocab_size = enc.n_vocab
    
    return encode, decode, vocab_size

def main():
    parser = argparse.ArgumentParser(description='Sample from a trained MoE model')
    parser.add_argument('--out_dir', type=str, default='checkpoints/out-moe-v100', 
                       help='Output directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default='best',
                       choices=['best', 'final', 'latest'], 
                       help='Which checkpoint to load')
    parser.add_argument('--start', type=str, default='Hello, I am a language model,',
                       help='Start prompt for generation')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to generate')
    parser.add_argument('--max_new_tokens', type=int, default=300,
                       help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k sampling parameter')
    parser.add_argument('--seed', type=int, default=1337,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for generation')
    parser.add_argument('--dtype', type=str, default='auto',
                       choices=['float32', 'float16', 'bfloat16', 'auto'],
                       help='Model dtype')
    parser.add_argument('--compile', action='store_true',
                       help='Compile model with PyTorch 2.0')
    parser.add_argument('--use_kv_cache', action='store_true',
                       help='Use KV cache for faster generation (experimental)')
    parser.add_argument('--profile', action='store_true',
                       help='Profile generation performance')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for generation')
    
    args = parser.parse_args()
    
    # Setup device and dtype
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    if args.dtype == 'auto':
        if device == 'cpu':
            dtype = 'float32'
        elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = 'bfloat16'
        else:
            dtype = 'float16'
    else:
        dtype = args.dtype
    
    print(f"Using device: {device}, dtype: {dtype}")
    if args.use_kv_cache:
        print("🚀 KV cache enabled for faster generation")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    if 'cuda' in device:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup autocast context
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Find and load checkpoint
    ckpt_path = find_checkpoint(args.out_dir, args.checkpoint)
    if ckpt_path is None:
        print(f"No checkpoint found in {args.out_dir}")
        print(f"Available files: {os.listdir(args.out_dir) if os.path.exists(args.out_dir) else 'Directory not found'}")
        return
    
    # Load model
    model, checkpoint = load_model_from_checkpoint(ckpt_path, device)
    model.eval()
    model.to(device)
    
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
    
    # Setup tokenizer
    encode, decode, vocab_size = setup_tokenizer(checkpoint)
    print(f"Vocabulary size: {vocab_size}")
    
    # Handle start prompt
    start = args.start
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
        print(f"Loaded prompt from file: {start[:100]}...")
    
    # Encode start prompt
    start_ids = encode(start)
    print(f"Prompt: '{start}' -> {len(start_ids)} tokens")
    
    # Prepare batched input if needed
    if args.batch_size > 1:
        x = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        x = x.repeat(args.batch_size, 1)
        print(f"Batched input shape: {x.shape}")
    else:
        x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    if args.use_kv_cache:
        print("Using KV cache for efficient generation")
    print("=" * 80)
    
    import time
    total_tokens = 0
    total_time = 0
    
    with torch.no_grad():
        with ctx:
            for k in range(args.num_samples):
                print(f"Sample {k+1}:")
                print("-" * 40)
                
                # Generate with timing
                start_time = time.time()
                
                if args.use_kv_cache:
                    y = generate_with_kv_cache(
                        model, x, args.max_new_tokens, 
                        temperature=args.temperature, 
                        top_k=args.top_k,
                        use_cache=True
                    )
                else:
                    y = model.generate(
                        x, args.max_new_tokens, 
                        temperature=args.temperature, 
                        top_k=args.top_k
                    )
                
                generation_time = time.time() - start_time
                total_time += generation_time
                
                # Decode and print
                for batch_idx in range(y.size(0)):
                    generated_text = decode(y[batch_idx].tolist())
                    print(f"Batch {batch_idx + 1}:")
                    print(generated_text)
                    if batch_idx < y.size(0) - 1:
                        print("-" * 20)
                
                # Performance metrics
                tokens_generated = (y.size(1) - x.size(1)) * y.size(0)
                total_tokens += tokens_generated
                tokens_per_second = tokens_generated / generation_time
                
                if args.profile:
                    print(f"\n📊 Performance metrics:")
                    print(f"  Tokens generated: {tokens_generated}")
                    print(f"  Time taken: {generation_time:.2f}s")
                    print(f"  Tokens/second: {tokens_per_second:.1f}")
                    if 'cuda' in device:
                        print(f"  GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
                
                print("=" * 80)
    
    # Final performance summary
    if args.profile and total_time > 0:
        avg_tokens_per_second = total_tokens / total_time
        print(f"\n📈 Overall Performance Summary:")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average tokens/second: {avg_tokens_per_second:.1f}")
        print(f"  Cache strategy: {'KV cache' if args.use_kv_cache else 'Standard'}")
        
        if 'cuda' in device:
            print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    
    print(f"\n✅ Generation complete! Generated {args.num_samples} samples with {args.max_new_tokens} max tokens each.")
    if args.use_kv_cache:
        print("💡 KV cache was used for potentially faster generation.")
    print("💡 Use --use_kv_cache for faster generation (when supported)")
    print("💡 Use --profile to see detailed performance metrics")

if __name__ == '__main__':
    main()
