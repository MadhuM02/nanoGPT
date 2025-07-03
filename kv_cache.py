"""
KV Cache implementation for efficient autoregressive generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

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

    def update(self, keys: torch.Tensor, values: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys and values

        Args:
            keys: New keys tensor [batch_size, n_head, seq_len, head_dim]
            values: New values tensor [batch_size, n_head, seq_len, head_dim]
            start_pos: Starting position in the cache

        Returns:
            Updated keys and values from cache up to the current length
        """
        batch_size, _, seq_len, _ = keys.shape
        end_pos = start_pos + seq_len
        self.key_cache[:batch_size, :, start_pos:end_pos] = keys
        self.value_cache[:batch_size, :, start_pos:end_pos] = values

        return (
            self.key_cache[:batch_size, :, :end_pos],
            self.value_cache[:batch_size, :, :end_pos]
        )

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
    """Reset all KV caches by clearing the tensors."""
    for cache in caches.values():
        cache.key_cache.zero_()
        cache.value_cache.zero_()

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
        
        # Forward through transformer blocks
        for i, block in enumerate(self.transformer.h):
            if use_cache and past_key_values is not None:
                cache_key = f'layer_{i}'
                x = self._forward_block_with_cache(block, x, past_key_values[cache_key], start_pos)
            else:
                x = block(x)

        x = self.transformer.ln_f(x)

        # inference-time mini-optimization: only forward the lm_head on the very last position
        logits = self.lm_head(x[:, [-1], :]) if start_pos > 0 else self.lm_head(x)

        return logits, past_key_values

    def _forward_block_with_cache(self, block, x, kv_cache, start_pos):
        """Forward pass through a transformer block with KV cache"""
        # Apply layer norm and attention with KV cache
        attn_input = block.ln_1(x)

        # Get attention components
        B, T, C = attn_input.size()
        attn = block.attn

        # Calculate query, key, values for all heads in batch
        q, k, v = attn.c_attn(attn_input).split(attn.n_embd, dim=2)
        k = k.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, attn.n_head, C // attn.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Update cache and get full sequence of keys and values
        k, v = kv_cache.update(k, v, start_pos)

        # Causal self-attention
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Flash attention
            # For prompt processing (T > 1), use causal mask.
            # For single-token generation (T = 1), do not use causal mask.
            is_causal = T > 1
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=attn.dropout if self.training else 0, is_causal=is_causal
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # Create causal mask
            seq_len_q = q.size(2)
            seq_len_k = k.size(2)
            mask = torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool)
            if seq_len_q == 1: # incremental generation
                # query can attend to all keys
                pass
            else: # prompt processing
                mask = torch.tril(mask)
            
            att = att.masked_fill(~mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = attn.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = attn.resid_dropout(attn.c_proj(y))

        # Residual connection
        x = x + y

        # MLP
        x = x + block.mlp(block.ln_2(x))

        return x

    # Add the methods to the model
    import types
    model.forward_with_cache = types.MethodType(forward_with_cache, model)
    model._forward_block_with_cache = types.MethodType(_forward_block_with_cache, model)

    return model
