# SFT Attention Analysis and Recommendations

## Current Issue: Using Pretraining Attention for SFT

Your SFT dataset has instruction-response format like:
- "Classify this news article: [TEXT] Category: [LABEL]"
- "Analyze sentiment: [REVIEW] Sentiment: [LABEL]"
- "Summarize: [TEXT] Summary: [SUMMARY]"

This requires different attention patterns than next-token prediction!

## Key SFT Attention Requirements

### 1. **Instruction-Response Separation**
- Model should attend differently to instruction vs response parts
- Need to distinguish between input context and target generation

### 2. **Cross-Attention Patterns**
- Strong attention from response tokens back to instruction tokens
- Weaker attention within instruction (already processed)
- Prevent response tokens from attending to future response tokens

### 3. **Task-Specific Attention**
- Classification: Focus on key discriminative features
- Summarization: Attention to salient content
- Q&A: Attention to relevant context parts

## Recommended SFT Attention Improvements

### 1. **Attention Mask Modifications**
Instead of pure causal masking, use structured masking:

```python
def create_sft_attention_mask(input_ids, instruction_separator="Category:", device="cuda"):
    """Create attention mask for SFT training"""
    batch_size, seq_len = input_ids.shape
    
    # Find separator positions
    separator_positions = find_separator_positions(input_ids, instruction_separator)
    
    # Create structured mask
    mask = torch.ones(batch_size, seq_len, seq_len, device=device)
    
    for b in range(batch_size):
        sep_pos = separator_positions[b]
        if sep_pos is not None:
            # Instruction tokens can attend to all instruction tokens
            mask[b, :sep_pos, :sep_pos] = 0
            # Response tokens can attend to instruction + previous response
            mask[b, sep_pos:, :] = 0
            # Causal masking within response
            for i in range(sep_pos, seq_len):
                mask[b, i, i+1:] = float('-inf')
    
    return mask
```

### 2. **Multi-Head Attention Specialization**
Different attention heads for different purposes:

```python
class SFTCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        # Specialized attention heads
        self.instruction_heads = config.n_head // 3  # Focus on instructions
        self.response_heads = config.n_head // 3     # Focus on response generation
        self.cross_heads = config.n_head - self.instruction_heads - self.response_heads  # Cross-attention
        
        # Rest of implementation...
```

### 3. **Position-Aware Attention**
Add position embeddings that distinguish instruction vs response:

```python
class SFTPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.role_emb = nn.Embedding(3, config.n_embd)  # [PAD, INSTRUCTION, RESPONSE]
    
    def forward(self, input_ids, role_ids):
        pos = torch.arange(input_ids.size(1), device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        role_emb = self.role_emb(role_ids)
        return pos_emb + role_emb
```

## Implementation Strategy

### Phase 1: Minimal Changes (Current Priority)
1. Add role-based masking in get_batch()
2. Modify attention dropout for SFT
3. Add instruction/response position encoding

### Phase 2: Advanced Attention (Future)
1. Multi-head specialization
2. Learned attention patterns
3. Task-specific attention mechanisms

### Phase 3: Full SFT Architecture (Long-term)
1. Separate encoders for instructions
2. Cross-attention layers
3. Dynamic attention patterns based on task type
