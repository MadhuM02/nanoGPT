"""
Sample from a trained model
"""
import os
import sys
import pickle
from contextlib import nullcontext
import torch
import tiktoken

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'checkpoints/out-moe-v100' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        # Try other checkpoint names
        alt_names = ['final_model.pt', 'ckpt.pt']
        for alt_name in alt_names:
            alt_path = os.path.join(out_dir, alt_name)
            if os.path.exists(alt_path):
                ckpt_path = alt_path
                break
        else:
            print(f"No checkpoint found in {out_dir}")
            print(f"Available files: {os.listdir(out_dir) if os.path.exists(out_dir) else 'Directory not found'}")
            exit(1)
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Handle different checkpoint formats for MoE models
    if 'config' in checkpoint:
        # New format with config dict
        config = checkpoint['config']
        gptconf = GPTConfig(
            block_size=config.get('block_size', 1024),
            vocab_size=config.get('vocab_size', 50304),
            n_layer=config.get('n_layer', 12),
            n_head=config.get('n_head', 12),
            n_embd=config.get('n_embd', 768),
            dropout=0.0,  # Disable dropout for sampling
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
        if hasattr(gptconf, 'expert_dropout'):
            gptconf.expert_dropout = 0.0
    else:
        print("Error: Checkpoint format not recognized")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        exit(1)
    
    print(f"Model config: n_layer={gptconf.n_layer}, n_head={gptconf.n_head}, n_embd={gptconf.n_embd}")
    if hasattr(gptconf, 'num_experts') and gptconf.num_experts > 1:
        print(f"MoE config: num_experts={gptconf.num_experts}, top_k={gptconf.top_k_experts}")
    
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    
    # Handle compiled model artifacts and DDP prefixes
    unwanted_prefixes = ['_orig_mod.', 'module._orig_mod.', 'module.']
    for prefix in unwanted_prefixes:
        keys_to_update = []
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                keys_to_update.append((k, k[len(prefix):]))
        
        for old_key, new_key in keys_to_update:
            state_dict[new_key] = state_dict.pop(old_key)
    
    model.load_state_dict(state_dict)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully. Total parameters: {total_params:,}")
    
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
