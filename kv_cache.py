"""
KV Cache implementation for efficient autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

class KVCache:
    """Key-Value cache for efficient autoregressive generation"""
    
    def __init__(self, batch_size: int, seq_len: int, n_head: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.batch_size = batch_size
        self.seq_len = seq_len  # Maximum sequence length
        self.n_head = n_head
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        
        # Pre-allocate cache tensors
        # Shape: [batch_size, n_head, seq_len, head_dim]
        self.key_cache = torch.zeros(batch_size, n_head, seq_len, head_dim, device=device, dtype=dtype)
        self.value_cache = torch.zeros(batch_size, n_head, seq_len, head_dim, device=device, dtype=dtype)
        
        # Track the current position in the cache
        self.cache_position = 0
        
    def update(self, keys: torch.Tensor, values: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values
        
        Args:
            keys: New keys tensor [batch_size, n_head, seq_len, head_dim]
            values: New values tensor [batch_size, n_head, seq_len, head_dim]
            start_pos: Starting position in the cache
            
        Returns:
            Updated keys and values from cache
        """
        batch_size, n_head, seq_len, head_dim = keys.shape
        
        # Update cache
        end_pos = start_pos + seq_len
        self.key_cache[:batch_size, :, start_pos:end_pos] = keys
        self.value_cache[:batch_size, :, start_pos:end_pos] = values
        
        # Return the full cache up to current position
        self.cache_position = end_pos
        return (
            self.key_cache[:batch_size, :, :end_pos],
            self.value_cache[:batch_size, :, :end_pos]
        )
    
    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current cache contents"""
        return (
            self.key_cache[:, :, :self.cache_position],
            self.value_cache[:, :, :self.cache_position]
        )
    
    def reset(self):
        """Reset cache position"""
        self.cache_position = 0

class CausalSelfAttentionWithKVCache(nn.Module):
    """
    Causal Self Attention with KV Cache support
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, kv_cache: Optional[KVCache] = None, start_pos: int = 0):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Handle KV cache
        if kv_cache is not None:
            if start_pos == 0:
                # First forward pass - initialize cache
                k_cache, v_cache = kv_cache.update(k, v, start_pos)
            else:
                # Subsequent passes - use cached keys/values
                k_cache, v_cache = kv_cache.update(k, v, start_pos)
                k, v = k_cache, v_cache
        
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            # Create causal mask for the current sequence length
            seq_len = k.size(2)
            if start_pos == 0:
                # Standard causal mask for full sequence
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool))
                att = att.masked_fill(~causal_mask, float('-inf'))
            else:
                # For incremental generation, we only need to mask the new token
                # against future positions (which don't exist in this case)
                pass  # No additional masking needed for single token generation
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

def create_kv_caches(model, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype) -> Dict[str, KVCache]:
    """Create KV caches for all attention layers in the model"""
    caches = {}
    
    n_head = model.config.n_head
    head_dim = model.config.n_embd // n_head
    
    for i in range(model.config.n_layer):
        caches[f'layer_{i}'] = KVCache(
            batch_size=batch_size,
            seq_len=max_seq_len,
            n_head=n_head,
            head_dim=head_dim,
            device=device,
            dtype=dtype
        )
    
    return caches

def reset_kv_caches(caches: Dict[str, KVCache]):
    """Reset all KV caches"""
    for cache in caches.values():
        cache.reset()

# Monkey patch for existing models to add KV cache support
def add_kv_cache_support(model):
    """Add KV cache support to an existing model"""
    
    def forward_with_cache(self, idx, past_key_values=None, use_cache=False, start_pos=0):
        """Enhanced forward pass with KV cache support"""
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Create position embeddings
        pos = torch.arange(start_pos, start_pos + t, dtype=torch.long, device=device)
        
        # token embeddings of shape (b, t, n_embd)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Create or use existing KV caches
        if use_cache:
            if past_key_values is None:
                # Initialize caches
                past_key_values = create_kv_caches(
                    self, b, self.config.block_size, device, x.dtype
                )
                # Reset caches for new sequence
                reset_kv_caches(past_key_values)
        
        # Forward through transformer blocks
        for i, block in enumerate(self.transformer.h):
            if use_cache and past_key_values is not None:
                cache_key = f'layer_{i}'
                if hasattr(block.attn, 'forward') and cache_key in past_key_values:
                    # Temporarily replace the attention forward method
                    original_forward = block.attn.forward
                    
                    def cached_forward(attention_input):
                        return CausalSelfAttentionWithKVCache.forward(
                            block.attn, attention_input, 
                            past_key_values[cache_key], start_pos
                        )
                    
                    block.attn.forward = cached_forward
                    x = block(x)
                    block.attn.forward = original_forward  # Restore original
                else:
                    x = block(x)
            else:
                x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if hasattr(self.config, 'vocab_size') and self.config.vocab_size is not None:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) if start_pos > 0 else self.lm_head(x)
        else:
            logits = self.lm_head(x)
        
        return logits, past_key_values
    
    # Add the method to the model
    import types
    model.forward_with_cache = types.MethodType(forward_with_cache, model)
    
    return model
