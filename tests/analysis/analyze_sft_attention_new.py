#!/usr/bin/env python3
"""
Comprehensive analysis of attention patterns for SFT vs Pretraining
Compare standard causal attention with SFT-specific attention improvements
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import tiktoken


def analyze_real_sft_data():
    """Analyze real SFT data to understand instruction-response patterns"""
    print("🔍 Analyzing Real SFT Data Patterns:")
    
    # Example SFT data patterns
    examples = [
        "Classify this news article: Scientists discover new planet in nearby galaxy. The planet shows signs of water. Category: Sci/Tech",
        "Analyze sentiment: This movie was absolutely amazing with great acting and plot! Sentiment: positive", 
        "Summarize this text: Machine learning is transforming industries worldwide. Companies invest billions in AI research. Summary: AI transforms industries with massive investment",
        "Answer this question: What is the capital of France? Answer: Paris"
    ]
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    for i, example in enumerate(examples):
        tokens = enc.encode(example)
        
        # Find separator positions (simplified heuristic)
        separators = [":", "Category:", "Sentiment:", "Summary:", "Answer:"]
        sep_pos = None
        
        for sep in separators:
            sep_tokens = enc.encode(sep)
            for j in range(len(tokens) - len(sep_tokens) + 1):
                if tokens[j:j+len(sep_tokens)] == sep_tokens:
                    sep_pos = j + len(sep_tokens)
                    break
            if sep_pos:
                break
        
        instruction_ratio = sep_pos / len(tokens) if sep_pos else 0.5
        
        print(f"\nExample {i+1}: {example[:50]}...")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Instruction tokens: {sep_pos if sep_pos else 'Unknown'}")
        print(f"  Response tokens: {len(tokens) - sep_pos if sep_pos else 'Unknown'}")
        print(f"  Instruction ratio: {instruction_ratio:.2f}")
    
    return examples


def standard_causal_attention(seq_len, n_heads=8, d_model=512):
    """Compute standard causal self-attention"""
    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    k = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    v = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    
    att = (q @ k.transpose(-2, -1)) * (1.0 / (d_model // n_heads) ** 0.5)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    
    return att[0].mean(dim=0)


def sft_aware_attention(seq_len, instruction_end, n_heads=8, d_model=512, instruction_weight=1.5):
    """Compute SFT-aware attention with instruction bias"""
    torch.manual_seed(42)
    q = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    k = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    v = torch.randn(1, n_heads, seq_len, d_model // n_heads)
    
    att = (q @ k.transpose(-2, -1)) * (1.0 / (d_model // n_heads) ** 0.5)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    att = att.masked_fill(causal_mask == 0, float('-inf'))
    
    # Add SFT-specific bias
    sft_bias = torch.zeros(seq_len, seq_len)
    sft_bias[instruction_end:, :instruction_end] = np.log(instruction_weight)
    sft_bias[instruction_end:, instruction_end:] = np.log(0.8)
    
    att = att + sft_bias.unsqueeze(0).unsqueeze(0)
    att = F.softmax(att, dim=-1)
    
    return att[0].mean(dim=0)


def specialized_heads_attention(seq_len, instruction_end, n_heads=9, d_model=512):
    """Compute attention with specialized heads for different roles"""
    torch.manual_seed(42)
    
    instruction_heads = list(range(0, 3))
    response_heads = list(range(3, 6))
    context_heads = list(range(6, 9))
    
    all_attention = []
    
    for head in range(n_heads):
        q = torch.randn(1, 1, seq_len, d_model // n_heads)
        k = torch.randn(1, 1, seq_len, d_model // n_heads)
        v = torch.randn(1, 1, seq_len, d_model // n_heads)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (d_model // n_heads) ** 0.5)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        
        role_bias = torch.zeros(seq_len, seq_len)
        
        if head in instruction_heads:
            role_bias[instruction_end:, instruction_end:] = np.log(0.5)
        elif head in response_heads:
            role_bias[:instruction_end, :instruction_end] = np.log(0.7)
            role_bias[instruction_end:, :instruction_end] = np.log(1.3)
        
        att = att + role_bias.unsqueeze(0).unsqueeze(0)
        att = F.softmax(att, dim=-1)
        all_attention.append(att[0, 0])
    
    return torch.stack(all_attention).mean(dim=0)


def plot_attention_comparison(seq_len, instruction_end):
    """Plot comparison of different attention mechanisms"""
    
    standard_att = standard_causal_attention(seq_len)
    sft_att = sft_aware_attention(seq_len, instruction_end)
    specialized_att = specialized_heads_attention(seq_len, instruction_end)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot attention matrices
    im1 = axes[0, 0].imshow(standard_att.numpy(), cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Standard Causal Attention')
    axes[0, 0].set_xlabel('Key Position')
    axes[0, 0].set_ylabel('Query Position')
    axes[0, 0].axhline(y=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].axvline(x=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(sft_att.numpy(), cmap='Blues', aspect='auto')
    axes[0, 1].set_title('SFT-Aware Attention')
    axes[0, 1].set_xlabel('Key Position')
    axes[0, 1].set_ylabel('Query Position')
    axes[0, 1].axhline(y=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axvline(x=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(specialized_att.numpy(), cmap='Blues', aspect='auto')
    axes[0, 2].set_title('Specialized Heads Attention')
    axes[0, 2].set_xlabel('Key Position')
    axes[0, 2].set_ylabel('Query Position')
    axes[0, 2].axhline(y=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    axes[0, 2].axvline(x=instruction_end-0.5, color='red', linestyle='--', alpha=0.7)
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot attention differences
    diff_sft = sft_att - standard_att
    diff_specialized = specialized_att - standard_att
    
    im4 = axes[1, 0].imshow(diff_sft.numpy(), cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 0].set_title('SFT - Standard (Difference)')
    axes[1, 0].set_xlabel('Key Position')
    axes[1, 0].set_ylabel('Query Position')
    axes[1, 0].axhline(y=instruction_end-0.5, color='black', linestyle='--', alpha=0.7)
    axes[1, 0].axvline(x=instruction_end-0.5, color='black', linestyle='--', alpha=0.7)
    plt.colorbar(im4, ax=axes[1, 0])
    
    im5 = axes[1, 1].imshow(diff_specialized.numpy(), cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 1].set_title('Specialized - Standard (Difference)')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    axes[1, 1].axhline(y=instruction_end-0.5, color='black', linestyle='--', alpha=0.7)
    axes[1, 1].axvline(x=instruction_end-0.5, color='black', linestyle='--', alpha=0.7)
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot attention flow analysis
    instruction_attention = sft_att[instruction_end:, :instruction_end].mean(dim=0)
    response_attention = sft_att[instruction_end:, instruction_end:].mean(dim=0)
    
    axes[1, 2].plot(instruction_attention.numpy(), label='Response→Instruction', linewidth=2)
    axes[1, 2].plot(range(instruction_end, seq_len), response_attention.numpy(), 
                   label='Response→Response', linewidth=2)
    axes[1, 2].set_title('Attention Flow Analysis')
    axes[1, 2].set_xlabel('Token Position')
    axes[1, 2].set_ylabel('Average Attention Weight')
    axes[1, 2].axvline(x=instruction_end-0.5, color='red', linestyle='--', alpha=0.7, label='Instruction End')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_attention_statistics(seq_len, instruction_end):
    """Analyze quantitative differences in attention patterns"""
    
    standard_att = standard_causal_attention(seq_len)
    sft_att = sft_aware_attention(seq_len, instruction_end)
    specialized_att = specialized_heads_attention(seq_len, instruction_end)
    
    print("=== ATTENTION PATTERN ANALYSIS ===\n")
    
    # 1. Response tokens attending to instruction tokens
    response_to_instruction_std = standard_att[instruction_end:, :instruction_end].mean().item()
    response_to_instruction_sft = sft_att[instruction_end:, :instruction_end].mean().item()
    response_to_instruction_spec = specialized_att[instruction_end:, :instruction_end].mean().item()
    
    print("1. Response → Instruction Attention:")
    print(f"   Standard:     {response_to_instruction_std:.4f}")
    print(f"   SFT-Aware:    {response_to_instruction_sft:.4f} ({response_to_instruction_sft/response_to_instruction_std:.2f}x)")
    print(f"   Specialized:  {response_to_instruction_spec:.4f} ({response_to_instruction_spec/response_to_instruction_std:.2f}x)\n")
    
    # 2. Response tokens attending to other response tokens
    response_to_response_std = standard_att[instruction_end:, instruction_end:].mean().item()
    response_to_response_sft = sft_att[instruction_end:, instruction_end:].mean().item()
    response_to_response_spec = specialized_att[instruction_end:, instruction_end:].mean().item()
    
    print("2. Response → Response Attention:")
    print(f"   Standard:     {response_to_response_std:.4f}")
    print(f"   SFT-Aware:    {response_to_response_sft:.4f} ({response_to_response_sft/response_to_response_std:.2f}x)")
    print(f"   Specialized:  {response_to_response_spec:.4f} ({response_to_response_spec/response_to_response_std:.2f}x)\n")
    
    # 3. Instruction tokens internal attention
    instruction_internal_std = standard_att[:instruction_end, :instruction_end].mean().item()
    instruction_internal_sft = sft_att[:instruction_end, :instruction_end].mean().item()
    instruction_internal_spec = specialized_att[:instruction_end, :instruction_end].mean().item()
    
    print("3. Instruction Internal Attention:")
    print(f"   Standard:     {instruction_internal_std:.4f}")
    print(f"   SFT-Aware:    {instruction_internal_sft:.4f} ({instruction_internal_sft/instruction_internal_std:.2f}x)")
    print(f"   Specialized:  {instruction_internal_spec:.4f} ({instruction_internal_spec/instruction_internal_std:.2f}x)\n")
    
    # 4. Attention entropy (measure of focus)
    def attention_entropy(att_matrix):
        entropy = -torch.sum(att_matrix * torch.log(att_matrix + 1e-8), dim=1)
        return entropy.mean().item()
    
    entropy_std = attention_entropy(standard_att)
    entropy_sft = attention_entropy(sft_att)
    entropy_spec = attention_entropy(specialized_att)
    
    print("4. Attention Entropy (lower = more focused):")
    print(f"   Standard:     {entropy_std:.4f}")
    print(f"   SFT-Aware:    {entropy_sft:.4f} ({entropy_sft/entropy_std:.2f}x)")
    print(f"   Specialized:  {entropy_spec:.4f} ({entropy_spec/entropy_std:.2f}x)\n")
    
    return {
        'response_to_instruction': {
            'standard': response_to_instruction_std,
            'sft': response_to_instruction_sft,
            'specialized': response_to_instruction_spec
        },
        'response_to_response': {
            'standard': response_to_response_std,
            'sft': response_to_response_sft,
            'specialized': response_to_response_spec
        },
        'entropy': {
            'standard': entropy_std,
            'sft': entropy_sft,
            'specialized': entropy_spec
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze SFT vs Pretraining Attention Patterns')
    parser.add_argument('--seq_len', type=int, default=18, help='Sequence length for analysis')
    parser.add_argument('--instruction_ratio', type=float, default=0.5, help='Fraction of sequence that is instruction')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for plots')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    instruction_end = int(args.seq_len * args.instruction_ratio)
    
    print(f"Analyzing attention patterns:")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Instruction length: {instruction_end}")
    print(f"  Response length: {args.seq_len - instruction_end}")
    print()
    
    # First analyze real SFT data patterns
    analyze_real_sft_data()
    print()
    
    # Analyze attention statistics
    stats = analyze_attention_statistics(args.seq_len, instruction_end)
    
    # Create and show plots
    fig = plot_attention_comparison(args.seq_len, instruction_end)
    
    if args.save_plots:
        output_path = Path(args.output_dir) / 'sft_attention_analysis.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()
    
    # Summary recommendations
    print("\n=== RECOMMENDATIONS FOR SFT ATTENTION ===")
    print()
    print("1. INSTRUCTION-RESPONSE BIAS:")
    sft_improvement = stats['response_to_instruction']['sft'] / stats['response_to_instruction']['standard']
    print(f"   - SFT attention increases response→instruction by {sft_improvement:.1f}x")
    print(f"   - This helps models focus on instruction context during generation")
    print()
    
    print("2. RESPONSE FOCUS:")
    response_reduction = stats['response_to_response']['sft'] / stats['response_to_response']['standard']
    print(f"   - SFT attention reduces response→response by {response_reduction:.1f}x")
    print(f"   - This prevents over-attention to already generated tokens")
    print()
    
    print("3. ATTENTION SPECIALIZATION:")
    print(f"   - Specialized heads provide more nuanced attention patterns")
    print(f"   - Different heads can focus on instruction understanding vs response generation")
    print()
    
    print("4. IMPLEMENTATION PRIORITY:")
    print(f"   - Start with instruction-aware attention bias (easiest to implement)")
    print(f"   - Add specialized head roles for more sophisticated patterns")
    print(f"   - Consider position-aware attention for different sequence regions")


if __name__ == "__main__":
    main()
