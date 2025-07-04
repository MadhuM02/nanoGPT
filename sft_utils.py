"""
Utility functions for SFT attention and instruction position detection
"""

import torch
import tiktoken
import re
from typing import List, Optional, Tuple


def find_instruction_positions_simple(input_ids: torch.Tensor, separators: List[str] = None) -> List[Optional[int]]:
    """
    Simple heuristic to find instruction-response boundaries in tokenized sequences
    
    Args:
        input_ids: Tensor of shape (batch_size, seq_len) containing token ids
        separators: List of separator strings to look for
    
    Returns:
        List of instruction end positions for each sequence in the batch
    """
    if separators is None:
        separators = [":", "Category:", "Sentiment:", "Summary:", "Answer:", "Response:"]
    
    batch_size, seq_len = input_ids.shape
    positions = []
    
    # Initialize tokenizer for separator detection
    enc = tiktoken.get_encoding("gpt2")
    
    # Precompute separator token sequences
    separator_tokens = {}
    for sep in separators:
        separator_tokens[sep] = enc.encode(sep)
    
    for b in range(batch_size):
        sequence = input_ids[b].tolist()
        position = None
        
        # Look for separator patterns
        for sep, sep_tokens in separator_tokens.items():
            sep_len = len(sep_tokens)
            for i in range(seq_len - sep_len + 1):
                if sequence[i:i+sep_len] == sep_tokens:
                    position = i + sep_len  # Position after separator
                    break
            if position is not None:
                break
        
        # Fallback heuristic: assume instruction is first 60% of sequence
        if position is None:
            position = int(seq_len * 0.6)
        
        positions.append(position)
    
    return positions


def find_instruction_positions_regex(text_batch: List[str]) -> List[Optional[int]]:
    """
    More sophisticated regex-based instruction position detection
    
    Args:
        text_batch: List of text strings
    
    Returns:
        List of character positions where instructions end
    """
    patterns = [
        r'(Category|Sentiment|Summary|Answer|Response):\s*',
        r'(Classify|Analyze|Summarize|Answer).*?:\s*',
        r'\?\s*Answer:\s*',
        r'\.\s*(Category|Sentiment|Summary):\s*'
    ]
    
    positions = []
    
    for text in text_batch:
        position = None
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                position = match.end()  # Position after the match
                break
        
        # Fallback: look for question mark followed by answer
        if position is None:
            match = re.search(r'\?\s*([A-Z])', text)
            if match:
                position = match.start(1)
        
        # Final fallback: 60% through the text
        if position is None:
            position = int(len(text) * 0.6)
        
        positions.append(position)
    
    return positions


def convert_char_to_token_positions(text_batch: List[str], char_positions: List[int], 
                                  tokenizer=None) -> List[Optional[int]]:
    """
    Convert character positions to token positions
    
    Args:
        text_batch: List of text strings
        char_positions: List of character positions
        tokenizer: Tokenizer to use (defaults to GPT-2)
    
    Returns:
        List of token positions
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    
    token_positions = []
    
    for text, char_pos in zip(text_batch, char_positions):
        if char_pos is None:
            token_positions.append(None)
            continue
        
        # Encode text up to character position
        text_prefix = text[:char_pos]
        token_pos = len(tokenizer.encode(text_prefix))
        token_positions.append(token_pos)
    
    return token_positions


def create_instruction_aware_loss_mask(input_ids: torch.Tensor, instruction_positions: List[Optional[int]], 
                                     ignore_instruction_loss: bool = True) -> torch.Tensor:
    """
    Create a loss mask that optionally ignores instruction tokens during loss computation
    
    Args:
        input_ids: Tensor of shape (batch_size, seq_len)
        instruction_positions: List of instruction end positions
        ignore_instruction_loss: If True, set loss weight to 0 for instruction tokens
    
    Returns:
        Loss mask tensor of shape (batch_size, seq_len)
    """
    batch_size, seq_len = input_ids.shape
    loss_mask = torch.ones_like(input_ids, dtype=torch.float)
    
    if ignore_instruction_loss:
        for b, pos in enumerate(instruction_positions):
            if pos is not None and pos < seq_len:
                loss_mask[b, :pos] = 0.0  # Zero out instruction tokens
    
    return loss_mask


def log_attention_statistics(attention_weights: torch.Tensor, instruction_positions: List[Optional[int]], 
                           step: int = 0) -> dict:
    """
    Log statistics about attention patterns for analysis
    
    Args:
        attention_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        instruction_positions: List of instruction end positions
        step: Training step number
    
    Returns:
        Dictionary of attention statistics
    """
    stats = {}
    batch_size, n_heads, seq_len, _ = attention_weights.shape
    
    # Average across heads and batch
    avg_attention = attention_weights.mean(dim=(0, 1))  # (seq_len, seq_len)
    
    response_to_instruction_total = 0.0
    response_to_response_total = 0.0
    valid_samples = 0
    
    for b, pos in enumerate(instruction_positions):
        if pos is not None and pos < seq_len - 1:
            valid_samples += 1
            
            # Response tokens attending to instruction tokens
            response_to_instruction = attention_weights[b, :, pos:, :pos].mean().item()
            response_to_instruction_total += response_to_instruction
            
            # Response tokens attending to other response tokens
            response_to_response = attention_weights[b, :, pos:, pos:].mean().item()
            response_to_response_total += response_to_response
    
    if valid_samples > 0:
        stats['response_to_instruction_avg'] = response_to_instruction_total / valid_samples
        stats['response_to_response_avg'] = response_to_response_total / valid_samples
        stats['instruction_focus_ratio'] = stats['response_to_instruction_avg'] / (stats['response_to_response_avg'] + 1e-8)
    
    # Attention entropy (measure of focus)
    entropy = -torch.sum(avg_attention * torch.log(avg_attention + 1e-8), dim=-1).mean().item()
    stats['attention_entropy'] = entropy
    
    stats['step'] = step
    stats['valid_samples'] = valid_samples
    
    return stats


def apply_sft_attention_config(config, enable_sft: bool = True):
    """
    Apply SFT-specific attention configuration
    
    Args:
        config: Model configuration object
        enable_sft: Whether to enable SFT attention mode
    """
    if enable_sft:
        config.sft_mode = True
        config.instruction_attention_weight = getattr(config, 'instruction_attention_weight', 1.5)
        config.response_attention_dampening = getattr(config, 'response_attention_dampening', 0.8)
        config.sft_head_specialization = getattr(config, 'sft_head_specialization', False)
        config.sft_position_bias = getattr(config, 'sft_position_bias', False)
        
        # Ensure n_head is divisible by 3 for head specialization
        if config.sft_head_specialization and config.n_head % 3 != 0:
            print(f"Warning: n_head ({config.n_head}) not divisible by 3, disabling head specialization")
            config.sft_head_specialization = False
    else:
        config.sft_mode = False
