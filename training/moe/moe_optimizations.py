"""
Advanced MoE optimizations for multi-GPU training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

class ExpertParallelMoE(nn.Module):
    """Expert-parallel MoE where experts are distributed across GPUs"""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 8)
        self.top_k = getattr(config, 'top_k_experts', 2)
        self.n_embd = config.n_embd
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Distribute experts across GPUs
        self.experts_per_gpu = self.num_experts // self.world_size
        self.local_expert_start = self.rank * self.experts_per_gpu
        self.local_expert_end = (self.rank + 1) * self.experts_per_gpu
        
        # Only create local experts
        self.local_experts = nn.ModuleList([
            Expert(config) for _ in range(self.experts_per_gpu)
        ])
        
        # Shared gating network
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        # Compute gating scores
        gate_logits = self.gate(x_flat)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Prepare for all-to-all communication
        # Route tokens to appropriate GPUs based on expert assignment
        outputs = []
        
        for gpu_rank in range(self.world_size):
            expert_start = gpu_rank * self.experts_per_gpu
            expert_end = (gpu_rank + 1) * self.experts_per_gpu
            
            # Find tokens that need experts on this GPU
            gpu_mask = ((top_k_indices >= expert_start) & (top_k_indices < expert_end)).any(dim=1)
            
            if gpu_mask.any():
                gpu_tokens = x_flat[gpu_mask]
                gpu_top_k_indices = top_k_indices[gpu_mask] - expert_start  # Adjust indices
                gpu_top_k_scores = top_k_scores[gpu_mask]
                
                if gpu_rank == self.rank:
                    # Process locally
                    local_output = self._process_local_tokens(gpu_tokens, gpu_top_k_indices, gpu_top_k_scores)
                else:
                    # Send to remote GPU (simplified - would need proper all-to-all)
                    local_output = torch.zeros_like(gpu_tokens)
                
                outputs.append((gpu_mask, local_output))
        
        # Combine outputs
        final_output = torch.zeros_like(x_flat)
        for mask, output in outputs:
            final_output[mask] = output
            
        return final_output.view(batch_size, seq_len, hidden_dim)
    
    def _process_local_tokens(self, tokens, indices, scores):
        output = torch.zeros_like(tokens)
        
        for i in range(self.top_k):
            expert_indices = indices[:, i]
            expert_scores = scores[:, i:i+1]
            
            for local_expert_id in range(self.experts_per_gpu):
                expert_mask = (expert_indices == local_expert_id)
                if expert_mask.any():
                    expert_input = tokens[expert_mask]
                    expert_output = self.local_experts[local_expert_id](expert_input)
                    output[expert_mask] += expert_scores[expert_mask] * expert_output
        
        return output

class SwitchTransformer(nn.Module):
    """Switch Transformer - simpler MoE with expert capacity"""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 8)
        self.n_embd = config.n_embd
        self.capacity_factor = getattr(config, 'capacity_factor', 1.25)
        
        # Create experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        
        # Gating network
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        total_tokens = batch_size * seq_len
        
        # Calculate expert capacity
        capacity = int(self.capacity_factor * total_tokens / self.num_experts)
        
        x_flat = x.view(-1, hidden_dim)
        
        # Compute gating scores and select top expert for each token
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Route to top-1 expert
        expert_indices = torch.argmax(gate_probs, dim=-1)
        expert_scores = torch.max(gate_probs, dim=-1)[0]
        
        # Apply capacity constraints
        output = torch.zeros_like(x_flat)
        expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        
        for expert_id in range(self.num_experts):
            expert_mask = (expert_indices == expert_id)
            expert_tokens = expert_mask.sum().item()
            
            if expert_tokens > 0:
                # Apply capacity limit
                if expert_tokens > capacity:
                    # Randomly drop tokens that exceed capacity
                    indices = torch.where(expert_mask)[0]
                    selected_indices = indices[torch.randperm(expert_tokens, device=x.device)[:capacity]]
                    expert_mask = torch.zeros_like(expert_mask)
                    expert_mask[selected_indices] = True
                
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[expert_mask] = expert_scores[expert_mask:expert_mask].unsqueeze(-1) * expert_output
        
        return output.view(batch_size, seq_len, hidden_dim)

class GradientCheckpointingMoE(nn.Module):
    """MoE with gradient checkpointing to save memory"""
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = getattr(config, 'num_experts', 8)
        self.top_k = getattr(config, 'top_k_experts', 2)
        self.n_embd = config.n_embd
        
        # Create experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        
        # Gating network
        self.gate = nn.Linear(config.n_embd, self.num_experts, bias=False)
        
    def forward(self, x):
        return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
    
    def _forward_impl(self, x):
        # Standard MoE forward pass (same as before)
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        
        gate_logits = self.gate(x_flat)
        gate_scores = F.softmax(gate_logits, dim=-1)
        
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(x_flat)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_scores = top_k_scores[:, i:i+1]
            
            for expert_id in range(self.num_experts):
                expert_mask = (expert_indices == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[expert_mask] += expert_scores[expert_mask] * expert_output
        
        return output.view(batch_size, seq_len, hidden_dim)

class Expert(nn.Module):
    """Optimized Expert with optional expert dropout"""
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        
        # Use different activation functions for better performance
        activation = getattr(config, 'expert_activation', 'gelu')
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()
            
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        # Expert dropout - randomly drop entire expert during training
        self.expert_dropout = getattr(config, 'expert_dropout', 0.0)

    def forward(self, x):
        if self.training and self.expert_dropout > 0.0:
            if torch.rand(1).item() < self.expert_dropout:
                return torch.zeros_like(x)
        
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Memory efficient attention for longer sequences
class FlashAttentionMoE(nn.Module):
    """Combines Flash Attention with MoE for memory efficiency"""
    
    def __init__(self, config):
        super().__init__()
        # Standard attention components
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Use Flash Attention 2 if available
        try:
            import flash_attn
            self.flash_attn_func = flash_attn.flash_attn_func
            self.flash_available = True
        except ImportError:
            self.flash_available = False
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash_available:
            # Use Flash Attention 2 for better memory efficiency
            y = self.flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0, causal=True)
        else:
            # Fallback to standard attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
            
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
