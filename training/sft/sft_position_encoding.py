"""
SFT Position Encoding Module
Adds instruction/response role awareness to position embeddings
"""

import torch
import torch.nn as nn


class SFTPositionalEmbedding(nn.Module):
    """
    Enhanced positional embedding that distinguishes between instruction and response tokens
    """
    
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        
        # Standard position embeddings
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Role embeddings: 0=instruction, 1=response
        self.role_emb = nn.Embedding(2, config.n_embd)
        
        # Optional: attention position bias for SFT
        self.use_position_bias = getattr(config, 'sft_position_bias', False)
        if self.use_position_bias:
            self.position_bias = nn.Parameter(torch.zeros(config.block_size, config.block_size))
    
    def create_role_ids(self, input_ids, instruction_positions):
        """
        Create role IDs tensor marking instruction vs response tokens
        
        Args:
            input_ids: (batch_size, seq_len) token IDs
            instruction_positions: list of instruction end positions for each batch item
            
        Returns:
            role_ids: (batch_size, seq_len) where 0=instruction, 1=response
        """
        batch_size, seq_len = input_ids.shape
        role_ids = torch.zeros_like(input_ids)
        
        if instruction_positions is not None:
            for b, instr_end in enumerate(instruction_positions):
                if instr_end is not None and instr_end < seq_len:
                    # Mark tokens after instruction_end as response tokens
                    role_ids[b, instr_end:] = 1
        
        return role_ids
    
    def forward(self, input_ids, instruction_positions=None):
        """
        Forward pass with SFT-aware position encoding
        
        Args:
            input_ids: (batch_size, seq_len) token IDs  
            instruction_positions: list of instruction end positions
            
        Returns:
            embeddings: (batch_size, seq_len, n_embd) position + role embeddings
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Standard position embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        pos_emb = self.pos_emb(pos)  # (seq_len, n_embd)
        
        # Role embeddings
        role_ids = self.create_role_ids(input_ids, instruction_positions)
        role_emb = self.role_emb(role_ids)  # (batch_size, seq_len, n_embd)
        
        # Combine position and role embeddings
        embeddings = pos_emb.unsqueeze(0) + role_emb  # (batch_size, seq_len, n_embd)
        
        return embeddings
    
    def get_position_bias(self, seq_len, instruction_positions=None):
        """
        Get position bias matrix for attention if enabled
        
        Args:
            seq_len: sequence length
            instruction_positions: list of instruction end positions
            
        Returns:
            bias: (seq_len, seq_len) attention bias matrix or None
        """
        if not self.use_position_bias:
            return None
            
        bias = self.position_bias[:seq_len, :seq_len].clone()
        
        # Apply instruction-response specific biases
        if instruction_positions is not None:
            for instr_end in instruction_positions:
                if instr_end is not None and instr_end < seq_len:
                    # Boost attention from response to instruction
                    bias[instr_end:, :instr_end] += 0.1
                    # Reduce attention within response
                    bias[instr_end:, instr_end:] -= 0.05
        
        return bias


class SFTEnhancedGPT(nn.Module):
    """
    GPT model enhanced with SFT-specific position encoding
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Enhanced position embeddings for SFT
        self.sft_pos_emb = SFTPositionalEmbedding(config)
        
        # Rest of the model (simplified for example)
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks (would use the modified Block class)
        # self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, input_ids, instruction_positions=None):
        """
        Forward pass with SFT position encoding
        """
        # Token embeddings
        token_emb = self.token_emb(input_ids)  # (batch_size, seq_len, n_embd)
        
        # SFT-aware position embeddings
        pos_emb = self.sft_pos_emb(input_ids, instruction_positions)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Process through transformer blocks
        # for block in self.blocks:
        #     x = block(x, instruction_positions)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits


def add_sft_position_encoding_to_existing_model(model, config):
    """
    Add SFT position encoding to an existing GPT model
    
    Args:
        model: existing GPT model
        config: model configuration with SFT parameters
        
    Returns:
        Enhanced model with SFT position encoding
    """
    # Replace the standard position embedding with SFT-aware version
    if hasattr(model.transformer, 'wpe'):
        # Save old position embedding weights
        old_pos_weights = model.transformer.wpe.weight.data.clone()
        
        # Create new SFT position embedding
        sft_pos_emb = SFTPositionalEmbedding(config)
        
        # Initialize with old weights for position component
        sft_pos_emb.pos_emb.weight.data = old_pos_weights
        
        # Initialize role embeddings to small values
        nn.init.normal_(sft_pos_emb.role_emb.weight, mean=0.0, std=0.01)
        
        # Replace the position embedding
        model.transformer.sft_wpe = sft_pos_emb
        
        print("✅ Added SFT position encoding to existing model")
    
    return model


if __name__ == "__main__":
    # Test the SFT position encoding
    from model import GPTConfig
    
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_embd=64,
        sft_position_bias=True
    )
    
    # Test position encoding
    pos_emb = SFTPositionalEmbedding(config)
    
    # Create test input
    input_ids = torch.randint(0, 100, (2, 16))
    instruction_positions = [8, 10]
    
    # Get embeddings
    embeddings = pos_emb(input_ids, instruction_positions)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Instruction positions: {instruction_positions}")
    
    # Check role IDs
    role_ids = pos_emb.create_role_ids(input_ids, instruction_positions)
    print(f"Role IDs for batch 0: {role_ids[0].tolist()}")
    print(f"Role IDs for batch 1: {role_ids[1].tolist()}")
    
    print("✅ SFT position encoding test completed!")
