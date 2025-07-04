#!/usr/bin/env python3
"""
Comprehensive analysis of attention patterns for SFT vs Pretraining
Compare standard causal attention with SFT-specific attention improvements
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("=" * 60)
    
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}: {example[:50]}...")
        
        # Tokenize
        tokens = enc.encode_ordinary(example)
        token_strings = [enc.decode([t]) for t in tokens]
        
        # Find instruction-response boundary
        response_markers = ["Category:", "Sentiment:", "Summary:", "Answer:"]
        instruction_end = -1
        marker_found = ""
        
        for marker in response_markers:
            pos = example.find(marker)
            if pos != -1:
                instruction_end = pos + len(marker)
                marker_found = marker
                break
        
        if instruction_end > 0:
            instruction_text = example[:instruction_end]
            response_text = example[instruction_end:].strip()
            
            instruction_tokens = enc.encode_ordinary(instruction_text)
            instruction_token_end = len(instruction_tokens)
            
            print(f"  📝 Instruction ({len(instruction_tokens)} tokens): {instruction_text}")
            print(f"  💬 Response ({len(tokens) - instruction_token_end} tokens): {response_text}")
            print(f"  🎯 Marker: '{marker_found}' at position {instruction_token_end}")
            
            # Analyze ideal attention pattern
            print(f"  🧠 Attention Pattern Analysis:")
            print(f"     - Instruction tokens should attend to previous instruction tokens")
            print(f"     - Response tokens should attend to ALL instruction tokens")
            print(f"     - Response tokens should use causal attention within response")
            print(f"     - Loss should focus on response tokens only")
            
        else:
            print(f"  ⚠️  No clear instruction-response boundary found")
    
    print(f"\n📊 Summary Statistics:")
    total_tokens = sum(len(enc.encode_ordinary(ex)) for ex in examples)
    avg_tokens = total_tokens / len(examples)
    print(f"  - Total examples: {len(examples)}")
    print(f"  - Average tokens per example: {avg_tokens:.1f}")
    print(f"  - Total tokens: {total_tokens}")

def visualize_attention_pattern(seq_len=20, instruction_end=12):
    """Create a visualization of ideal SFT attention pattern"""
    
    # Create attention matrix
    attention_matrix = np.zeros((seq_len, seq_len))
    
    # Instruction tokens (0 to instruction_end) - can attend to previous instruction tokens
    for i in range(instruction_end):
        for j in range(i + 1):  # Causal attention within instruction
            attention_matrix[i, j] = 0.8
    
    # Response tokens (instruction_end to seq_len) 
    for i in range(instruction_end, seq_len):
        # Can attend to all instruction tokens (strongly)
        for j in range(instruction_end):
            attention_matrix[i, j] = 1.0
        # Causal attention within response
        for j in range(instruction_end, i + 1):
            attention_matrix[i, j] = 0.6
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, 
                cmap='Blues', 
                cbar_kws={'label': 'Attention Weight'},
                xticklabels=False,
                yticklabels=False)
    
    # Add labels and lines
    plt.axhline(y=instruction_end, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.axvline(x=instruction_end, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.title('Ideal SFT Attention Pattern\n(Red line separates instruction from response)', fontsize=14)
    plt.xlabel('Key/Value Position', fontsize=12)
    plt.ylabel('Query Position', fontsize=12)
    
    # Add text annotations
    plt.text(instruction_end/2, instruction_end/2, 'Instruction\nSelf-Attention', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(instruction_end/2, instruction_end + (seq_len-instruction_end)/2, 
             'Response→Instruction\nCross-Attention', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(instruction_end + (seq_len-instruction_end)/2, instruction_end + (seq_len-instruction_end)/2, 
             'Response\nCausal Attention', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/workspace/nanoGPT/sft_attention_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("💾 Attention pattern visualization saved as 'sft_attention_pattern.png'")

def compare_pretraining_vs_sft():
    """Compare pretraining vs SFT attention requirements"""
    
    print("\n🆚 Pretraining vs SFT Attention Comparison:")
    print("=" * 60)
    
    comparison = {
        "Aspect": [
            "Primary Goal",
            "Attention Pattern", 
            "Key Challenge",
            "Loss Focus",
            "Context Usage",
            "Token Relationships"
        ],
        "Pretraining": [
            "Next token prediction",
            "Pure causal (left-to-right)",
            "General language modeling", 
            "All tokens equally weighted",
            "Local context window",
            "Sequential dependencies"
        ],
        "SFT": [
            "Instruction following",
            "Instruction→Response + Causal",
            "Task-specific alignment",
            "Response tokens only",
            "Instruction-response structure",
            "Cross-modal dependencies"
        ]
    }
    
    for i, aspect in enumerate(comparison["Aspect"]):
        print(f"\n{aspect}:")
        print(f"  🔵 Pretraining: {comparison['Pretraining'][i]}")
        print(f"  🟢 SFT: {comparison['SFT'][i]}")

if __name__ == "__main__":
    print("🎯 SFT Attention Pattern Analysis")
    print("=" * 50)
    
    # Analyze SFT examples
    analyze_sft_attention_patterns()
    
    # Create visualization
    print(f"\n📈 Creating attention pattern visualization...")
    visualize_attention_pattern()
    
    # Compare approaches
    compare_pretraining_vs_sft()
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Key insight: SFT requires structured attention patterns, not just causal!")
    print(f"🚀 Consider implementing instruction-aware attention for better SFT performance.")
