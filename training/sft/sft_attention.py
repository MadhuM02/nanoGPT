"""
SFT-specific attention implementations for nanoGPT
Implements instruction-aware attention patterns for supervised fine-tuning
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SFTCausalSelfAttention(nn.Module):
    """Enhanced attention for SFT with instruction-response awareness"""

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
        
        # SFT-specific parameters
        self.sft_mode = getattr(config, 'sft_mode', False)
        self.instruction_weight = getattr(config, 'instruction_attention_weight', 1.5)
        self.response_focus = getattr(config, 'response_attention_focus', True)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def create_sft_attention_bias(self, x, instruction_positions=None):
        """Create attention bias for SFT training"""
        B, T, C = x.size()
        
        if not self.sft_mode or instruction_positions is None:
            return None
            
        # Create attention bias matrix
        bias = torch.zeros(B, 1, T, T, device=x.device, dtype=x.dtype)
        
        for b in range(B):
            if instruction_positions[b] is not None:
                instr_end = instruction_positions[b]
                
                # Boost attention from response tokens to instruction tokens
                if self.response_focus and instr_end < T:
                    # Response tokens (after instruction) attend more to instruction
                    bias[b, 0, instr_end:, :instr_end] = math.log(self.instruction_weight)
                    
                    # Slightly reduce attention within instruction for response tokens
                    bias[b, 0, instr_end:, instr_end:] = math.log(0.8)
        
        return bias

    def get_attention_weights(self, q, k, v, instruction_positions=None):
        """Compute attention weights with SFT-specific modifications"""
        B, nh, T, hs = q.shape
        
        # Standard attention computation
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Apply causal mask
        if self.flash:
            # Use flash attention with causal masking
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual attention with custom bias
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            
            # Add SFT-specific attention bias
            sft_bias = self.create_sft_attention_bias(
                q.transpose(1, 2).contiguous().view(B, T, -1), 
                instruction_positions
            )
            if sft_bias is not None:
                att = att + sft_bias
            
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            
            return y

    def forward(self, x, instruction_positions=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Apply SFT-aware attention
        y = self.get_attention_weights(q, k, v, instruction_positions)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MultiHeadSpecializedAttention(nn.Module):
    """Multi-head attention with role specialization for SFT"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % 3 == 0  # Divide heads into 3 roles
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Divide heads into specialized roles
        self.heads_per_role = config.n_head // 3
        self.instruction_heads = list(range(0, self.heads_per_role))
        self.response_heads = list(range(self.heads_per_role, 2 * self.heads_per_role))
        self.context_heads = list(range(2 * self.heads_per_role, config.n_head))
        
        # Projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def apply_role_specific_attention(self, q, k, v, head_idx, instruction_positions=None):
        """Apply role-specific attention patterns"""
        B, T, hs = q.shape
        
        if instruction_positions is None:
            # Standard causal attention
            return self.standard_attention(q, k, v)
        
        # Create role-specific masks
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        
        for b in range(B):
            if instruction_positions[b] is not None:
                instr_end = instruction_positions[b]
                
                if head_idx in self.instruction_heads:
                    # Instruction heads: focus on instruction tokens
                    att[b, instr_end:, instr_end:] *= 0.5  # Reduce response-to-response
                    
                elif head_idx in self.response_heads:
                    # Response heads: focus on response generation
                    att[b, :instr_end, :instr_end] *= 0.7  # Reduce instruction-internal
                    att[b, instr_end:, :instr_end] *= 1.3  # Boost instruction-to-response
                    
                # Context heads use standard attention
        
        return att

    def standard_attention(self, q, k, v):
        """Standard causal self-attention"""
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        return att

    def forward(self, x, instruction_positions=None):
        B, T, C = x.size()
        
        # Get Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply attention for each head with role specialization
        outputs = []
        for h in range(self.n_head):
            q_h, k_h, v_h = q[:, h], k[:, h], v[:, h]
            
            if self.flash:
                # Use flash attention (role specialization limited)
                y_h = torch.nn.functional.scaled_dot_product_attention(
                    q_h.unsqueeze(1), k_h.unsqueeze(1), v_h.unsqueeze(1),
                    attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
                ).squeeze(1)
            else:
                # Manual attention with role specialization
                att = self.apply_role_specific_attention(q_h, k_h, v_h, h, instruction_positions)
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y_h = att @ v_h
            
            outputs.append(y_h)
        
        # Concatenate all heads
        y = torch.stack(outputs, dim=1)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


def find_instruction_positions(input_ids, instruction_separators=None):
    """Find positions where instructions end and responses begin"""
    if instruction_separators is None:
        instruction_separators = [":", "Answer:", "Response:", "Category:", "Sentiment:"]
    
    batch_size, seq_len = input_ids.shape
    positions = []
    
    for b in range(batch_size):
        position = None
        
        # Convert token IDs to approximate text for separator detection
        # This is a simplified approach - in practice, you'd want to use the tokenizer
        for pos in range(seq_len - 1):
            # Look for separator patterns in token sequences
            # This is a placeholder - implement based on your tokenizer
            # For now, assume instruction ends at 70% of sequence (heuristic)
            if pos > seq_len * 0.3 and pos < seq_len * 0.8:
                position = pos
                break
        
        positions.append(position)
    
    return positions
