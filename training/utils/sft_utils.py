"""
SFT (Supervised Fine-Tuning) utilities for nanoGPT.
Contains data loading, batch processing, loss analysis, and training utilities specific to SFT.
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from typing import Tuple, List, Optional, Dict, Any


class SFTDataManager:
    """Manages SFT dataset loading and batch processing."""
    
    def __init__(self, data_dir: str, tokenizer_name: str = "gpt2"):
        self.data_dir = data_dir
        self.train_examples = []
        self.val_examples = []
        self.enc = tiktoken.get_encoding(tokenizer_name)
        
    def load_examples(self, master_process: bool = True) -> None:
        """Load all examples from JSONL files."""
        # Load training examples
        train_path = os.path.join(self.data_dir, 'train.jsonl')
        if os.path.exists(train_path):
            self.train_examples = self._load_jsonl_examples(train_path)
        
        # Load validation examples  
        val_path = os.path.join(self.data_dir, 'val.jsonl')
        if os.path.exists(val_path):
            self.val_examples = self._load_jsonl_examples(val_path)
        
        # Only master process should print to avoid spam in distributed training
        if master_process:
            print(f"📚 Loaded {len(self.train_examples)} training examples, {len(self.val_examples)} validation examples")
            if len(self.train_examples) > 0:
                print(f"📋 First training example type: {type(self.train_examples[0])}")
                print(f"📋 First training example: {self.train_examples[0]}")
            if len(self.val_examples) > 0:
                print(f"📋 First validation example type: {type(self.val_examples[0])}")
                print(f"📋 First validation example: {self.val_examples[0]}")
    
    def _load_jsonl_examples(self, file_path: str) -> list:
        """Load examples from JSONL file."""
        examples = []
        
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    example = json.loads(line)
                    # Store the original prompt and completion separately for better control
                    examples.append({
                        'prompt': example['prompt'],
                        'completion': example['completion'],
                        'full_text': example['prompt'] + example['completion']  # Combined for convenience
                    })
        
        return examples
    
    def get_batch(self, split: str, batch_size: int, block_size: int, 
                  device: str, device_type: str) -> Tuple[torch.Tensor, ...]:
        """Get a batch of complete examples for SFT training with instruction-response awareness."""
        # Choose examples based on split
        examples = self.train_examples if split == 'train' else self.val_examples
        
        if len(examples) == 0:
            # Fallback to binary data if text examples not available
            if split == 'train':
                data = np.memmap(os.path.join(self.data_dir, 'train.bin'), dtype=np.uint16, mode='r')
            else:
                data = np.memmap(os.path.join(self.data_dir, 'val.bin'), dtype=np.uint16, mode='r')
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            
            if device_type == 'cuda':
                x = x.pin_memory().to(device, non_blocking=True)
                y = y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            return x, y
        else:
            # Sample random examples using Python's random module to avoid numpy string issues
            sampled_examples = random.choices(examples, k=batch_size)
            
            x_list = []
            y_list = []
            instruction_masks = []  # Track instruction vs response regions
            
            for example in sampled_examples:
                # Extract prompt and completion from the JSONL format
                prompt = example['prompt']
                completion = example['completion']
                full_text = example['full_text']
                
                x_tokens, y_tokens, instruction_mask = self._process_jsonl_example(prompt, completion, block_size)
                x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
                y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
                instruction_masks.append(torch.tensor(instruction_mask, dtype=torch.float32))
            
            x = torch.stack(x_list)
            y = torch.stack(y_list)
            instruction_mask_tensor = torch.stack(instruction_masks)
            
            # Create SFT loss mask: 0 for instruction tokens (no loss), 1 for response tokens (compute loss)
            # instruction_mask: 1 for instruction, 0 for response
            # loss_mask: 0 for instruction, 1 for response (inverted)
            loss_mask = 1.0 - instruction_mask_tensor
            
            # Apply loss mask to targets: set instruction tokens to -1 (ignored in cross_entropy)
            y_masked = y.clone()
            y_masked[loss_mask == 0] = -1  # Mask out instruction tokens
            
            # Clean up intermediate lists to free memory
            del x_list, y_list, instruction_masks
            
            if device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x = x.pin_memory().to(device, non_blocking=True)
                y_masked = y_masked.pin_memory().to(device, non_blocking=True)
                loss_mask = loss_mask.pin_memory().to(device, non_blocking=True)
                return x, y_masked, loss_mask
            else:
                x, y_masked = x.to(device), y_masked.to(device)
                loss_mask = loss_mask.to(device)
                return x, y_masked, loss_mask
    
    def _process_jsonl_example(self, prompt: str, completion: str, block_size: int) -> Tuple[List[int], List[int], List[float]]:
        """Process a JSONL example with separated prompt and completion."""
        # Tokenize prompt and completion separately for precise control
        prompt_tokens = self.enc.encode_ordinary(prompt)
        completion_tokens = self.enc.encode_ordinary(completion)
        
        # Combine tokens for the full sequence
        full_tokens = prompt_tokens + completion_tokens
        
        # Add end token to mark end of sequence (this is part of the actual completion)
        full_tokens = full_tokens + [self.enc.eot_token]
        original_length = len(full_tokens)
        
        # Create instruction mask aligned with the TARGET tokens (Y)
        # Since Y = tokens[1:], we need to align the mask accordingly
        prompt_length = len(prompt_tokens)
        
        # Create mask for the full sequence first
        # 1 = prompt/instruction (mask loss), 0 = completion/response (compute loss)
        full_mask = [1] * prompt_length + [0] * (original_length - prompt_length)
        
        # Ensure we have at least one token
        if len(full_tokens) == 0:
            full_tokens = [self.enc.eot_token]
            full_mask = [0]  # Single token is treated as completion
            original_length = 1
        
        # Truncate or pad to exactly block_size + 1 (since we'll shift)
        target_length = block_size + 1
        if len(full_tokens) >= target_length:
            full_tokens = full_tokens[:target_length]
            full_mask = full_mask[:target_length]
        else:
            # Pad with end tokens to reach target_length
            pad_length = target_length - len(full_tokens)
            full_tokens = full_tokens + [self.enc.eot_token] * pad_length
            # IMPORTANT: Padding tokens should be masked (not contribute to loss)
            full_mask = full_mask + [1] * pad_length  # 1 = mask loss for padding
        
        # Create input (x) and target (y) sequences
        # Standard language modeling: predict next token
        x_tokens = full_tokens[:-1]  # Input sequence (first block_size tokens)
        y_tokens = full_tokens[1:]   # Target sequence (shifted by 1)
        instruction_mask = full_mask[1:]  # Mask aligned with Y tokens
        
        # Ensure exactly block_size for all
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        instruction_mask = instruction_mask[:block_size]
        
        # Final safety check - pad if somehow still short and mask padding
        while len(x_tokens) < block_size:
            x_tokens.append(self.enc.eot_token)
        while len(y_tokens) < block_size:
            y_tokens.append(self.enc.eot_token)
        while len(instruction_mask) < block_size:
            instruction_mask.append(1)  # 1 = mask loss for any additional padding
        
        return x_tokens, y_tokens, instruction_mask

    def _process_example_with_prompt(self, example_str: str, prompt: str, block_size: int) -> Tuple[List[int], List[int], List[float]]:
        """Process a single example with separated prompt and completion."""
        # Tokenize the full example
        tokens = self.enc.encode_ordinary(example_str)
        
        # Tokenize just the prompt to find the boundary
        prompt_tokens = self.enc.encode_ordinary(prompt)
        prompt_token_end = len(prompt_tokens)
        
        # Ensure we have at least one token
        if len(tokens) == 0:
            tokens = [self.enc.eot_token]
        
        # Add end token to mark end of sequence
        tokens = tokens + [self.enc.eot_token]
        
        # Create instruction mask (1 for instruction/prompt, 0 for response/completion)
        instruction_mask = [1] * len(tokens)
        if prompt_token_end > 0 and prompt_token_end < len(tokens):
            # Mark response tokens (everything after the prompt)
            for i in range(prompt_token_end, len(tokens)):
                instruction_mask[i] = 0
        
        # Truncate or pad to exactly block_size
        if len(tokens) >= block_size:
            # Truncate to exactly block_size
            tokens = tokens[:block_size]
            instruction_mask = instruction_mask[:block_size]
        else:
            # Pad with end tokens to reach block_size
            pad_length = block_size - len(tokens)
            tokens = tokens + [self.enc.eot_token] * pad_length
            instruction_mask = instruction_mask + [1] * pad_length  # Mask padding tokens
        
        # Create input (x) and target (y) sequences
        x_tokens = tokens[:]  # Copy the full sequence
        y_tokens = tokens[1:] + [self.enc.eot_token]  # Shift by 1 and add eot at end
        
        # Ensure exactly block_size for both
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        instruction_mask = instruction_mask[:block_size]
        
        # Final safety check - pad if somehow still short and mask padding
        while len(x_tokens) < block_size:
            x_tokens.append(self.enc.eot_token)
        while len(y_tokens) < block_size:
            y_tokens.append(self.enc.eot_token)
        while len(instruction_mask) < block_size:
            instruction_mask.append(1)  # Mask any additional padding
        
        return x_tokens, y_tokens, instruction_mask

    def _process_example(self, example: str, block_size: int) -> Tuple[List[int], List[int], List[float]]:
        """Process a single example into tokenized sequences with instruction/response masking."""
        # Detect instruction-response boundary markers
        # Common patterns: "Category:", "Sentiment:", "Summary:", etc.
        response_markers = ["Category:", "Sentiment:", "Summary:", "Answer:", "Translation:", "Output:", 
                           "Rating:", "Emotion:", "Type:", "Classification:", "Result:"]
        
        # Find the first response marker
        instruction_end = -1
        for marker in response_markers:
            pos = example.find(marker)
            if pos != -1:
                instruction_end = pos + len(marker)
                break
        
        # Tokenize the example
        tokens = self.enc.encode_ordinary(example)
        
        # Find token boundary for instruction/response split
        instruction_token_end = -1
        if instruction_end > 0:
            # Tokenize up to the instruction end to find token boundary
            instruction_text = example[:instruction_end]
            instruction_tokens = self.enc.encode_ordinary(instruction_text)
            instruction_token_end = len(instruction_tokens)
        
        # Ensure we have at least one token
        if len(tokens) == 0:
            tokens = [self.enc.eot_token]
        
        # Add end token to mark end of sequence
        tokens = tokens + [self.enc.eot_token]
        
        # Create instruction mask (1 for instruction, 0 for response)
        instruction_mask = [1] * len(tokens)
        if instruction_token_end > 0 and instruction_token_end < len(tokens):
            # Mark response tokens
            for i in range(instruction_token_end, len(tokens)):
                instruction_mask[i] = 0
        
        # Truncate or pad to exactly block_size
        if len(tokens) >= block_size:
            # Truncate to exactly block_size
            tokens = tokens[:block_size]
            instruction_mask = instruction_mask[:block_size]
        else:
            # Pad with end tokens to reach block_size
            pad_length = block_size - len(tokens)
            tokens = tokens + [self.enc.eot_token] * pad_length
            instruction_mask = instruction_mask + [1] * pad_length  # Mask padding tokens
        
        # Create input (x) and target (y) sequences
        # x: tokens[:-1], y: tokens[1:]
        # But we need both to be exactly block_size
        x_tokens = tokens[:]  # Copy the full sequence
        y_tokens = tokens[1:] + [self.enc.eot_token]  # Shift by 1 and add eot at end
        
        # Ensure exactly block_size for both
        x_tokens = x_tokens[:block_size]
        y_tokens = y_tokens[:block_size]
        instruction_mask = instruction_mask[:block_size]
        
        # Final safety check - pad if somehow still short and mask padding
        while len(x_tokens) < block_size:
            x_tokens.append(self.enc.eot_token)
        while len(y_tokens) < block_size:
            y_tokens.append(self.enc.eot_token)
        while len(instruction_mask) < block_size:
            instruction_mask.append(1)  # Mask any additional padding
        
        return x_tokens, y_tokens, instruction_mask


class SFTLossAnalyzer:
    """Utilities for analyzing and debugging SFT training losses."""
    
    def __init__(self, tokenizer_name: str = "gpt2"):
        self.enc = tiktoken.get_encoding(tokenizer_name)
    
    def log_training_samples(self, X: torch.Tensor, Y: torch.Tensor, 
                           loss_per_sample: Optional[torch.Tensor] = None, 
                           step: Optional[int] = None, sample_limit: int = 3,
                           master_process: bool = True) -> None:
        """
        Log detailed information about training samples to debug loss explosion.
        
        Args:
            X: Input tensor (batch_size, seq_len)
            Y: Target tensor (batch_size, seq_len)
            loss_per_sample: Optional per-sample losses
            step: Current training step
            sample_limit: Maximum number of samples to log (to avoid spam)
            master_process: Whether this is the master process
        """
        if not master_process:  # Only log on master process
            return
            
        batch_size, seq_len = X.shape
        
        print(f"\n📋 Sample Logging (Step {step}):")
        print(f"   Batch shape: {X.shape}, Device: {X.device}")
        print(f"   X range: [{X.min().item()}, {X.max().item()}]")
        print(f"   Y range: [{Y.min().item()}, {Y.max().item()}]")
        
        # Log first few samples in detail
        for i in range(min(sample_limit, batch_size)):
            x_sample = X[i].cpu().numpy()
            y_sample = Y[i].cpu().numpy()
            
            # Check for problematic values
            x_invalid = np.sum((x_sample < 0) | (x_sample >= self.enc.n_vocab))
            y_invalid = np.sum((y_sample < 0) | (y_sample >= self.enc.n_vocab))
            
            print(f"\n   Sample {i}:")
            # print(f"     X tokens: {x_sample[:10].tolist()}...{x_sample[-10:].tolist()}")
            # print(f"     Y tokens: {y_sample[:10].tolist()}...{y_sample[-10:].tolist()}")
            # print(f"     Invalid X tokens: {x_invalid}, Invalid Y tokens: {y_invalid}")
            
            if loss_per_sample is not None:
                print(f"     Sample loss: {loss_per_sample[i].item():.4f}")
            
            # Try to decode samples (handle potential encoding errors)
            x_text = self.enc.decode([x for x in x_sample.tolist() if x >= 0 and x != self.enc.eot_token]) 
            print(f"     X text: {repr(x_text)}")
            try:
                y_text = self.enc.decode([y for y in y_sample.tolist() if y >=0 and y != self.enc.eot_token])
                print(f"     Y text: {repr(y_text)}")
            except Exception as e:
                print(f"     Decode error: {e}")
            
            # Check for suspicious patterns
            if len(np.unique(x_sample)) < 5:
                print(f"     ⚠️  WARNING: X sample has very low diversity ({len(np.unique(x_sample))} unique tokens)")
            if len(np.unique(y_sample)) < 5:
                print(f"     ⚠️  WARNING: Y sample has very low diversity ({len(np.unique(y_sample))} unique tokens)")
            
            # Check for extremely repetitive patterns
            if seq_len > 20:
                # Check if first 10 tokens repeat
                first_10 = x_sample[:10]
                repeats = 0
                for j in range(10, seq_len - 10, 10):
                    if np.array_equal(x_sample[j:j+10], first_10):
                        repeats += 1
                if repeats > seq_len // 20:  # More than 5% repetition
                    print(f"     ⚠️  WARNING: High repetition detected in X ({repeats} repeating blocks)")
    
    def compute_per_sample_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for each sample in the batch individually.
        
        Args:
            logits: Model output (batch_size, seq_len, vocab_size)
            targets: Target tokens (batch_size, seq_len)
        
        Returns:
            per_sample_losses: Tensor of shape (batch_size,)
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
        targets_flat = targets.view(-1)  # (batch_size * seq_len,)
        
        # Compute loss for each token
        token_losses = F.cross_entropy(logits_flat, targets_flat, ignore_index=-1, reduction='none')
        
        # Reshape back to (batch_size, seq_len)
        token_losses = token_losses.view(batch_size, seq_len)
        
        # Sum losses for each sample (ignoring -1 tokens)
        per_sample_losses = []
        for i in range(batch_size):
            mask = targets[i] != -1
            if mask.sum() > 0:
                sample_loss = token_losses[i][mask].mean()
            else:
                sample_loss = torch.tensor(0.0, device=token_losses.device)
            per_sample_losses.append(sample_loss)
        
        return torch.stack(per_sample_losses)


class SFTTrainingUtils:
    """General training utilities for SFT."""
    
    @staticmethod
    def get_cosine_lr_with_warmup(it: int, learning_rate: float, warmup_iters: int, 
                                 lr_decay_iters: int, min_lr: float) -> float:
        """
        Learning rate decay scheduler (cosine with warmup).
        
        Args:
            it: Current iteration
            learning_rate: Base learning rate
            warmup_iters: Number of warmup iterations
            lr_decay_iters: Total decay iterations
            min_lr: Minimum learning rate
        
        Returns:
            Current learning rate
        """
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    
    @staticmethod
    def create_estimate_loss_fn(model, data_manager: SFTDataManager, eval_iters: int,
                               batch_size: int, block_size: int, device: str, 
                               device_type: str, ctx):
        """
        Create a closure function for loss estimation.
        
        Returns:
            estimate_loss function that can be called without arguments
        """
        @torch.no_grad()
        def estimate_loss():
            out = {}
            model.eval()
            for split in ['train', 'val']:
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    batch_data = data_manager.get_batch(split, batch_size, block_size, device, device_type)
                    if len(batch_data) == 3:
                        X, Y, loss_mask = batch_data  # SFT data with loss mask
                    else:
                        X, Y = batch_data  # Binary data fallback
                        loss_mask = None
                    with ctx:
                        logits, loss = model(X, Y)
                    losses[k] = loss.item()
                    # Clean up intermediate tensors
                    del X, Y, logits, loss
                    if loss_mask is not None:
                        del loss_mask
                out[split] = losses.mean()
                # Clean up tensors
                del losses
            model.train()
            return out
        
        return estimate_loss
    
    @staticmethod
    def detect_loss_spike(loss_history: List[float], current_loss: float, 
                         spike_threshold: float = 2.0, min_history: int = 5) -> bool:
        """
        Detect if current loss is a spike compared to recent history.
        
        Args:
            loss_history: List of recent losses
            current_loss: Current loss value
            spike_threshold: Threshold multiplier for spike detection
            min_history: Minimum history length required for detection
        
        Returns:
            True if spike detected, False otherwise
        """
        if len(loss_history) < min_history:
            return False
        
        recent_avg_loss = sum(loss_history[-10:]) / min(len(loss_history), 10)
        return current_loss > recent_avg_loss * spike_threshold
    
    @staticmethod
    def apply_spike_recovery(optimizer, recovery_factor: float = 0.1) -> None:
        """
        Apply learning rate reduction for spike recovery.
        
        Args:
            optimizer: PyTorch optimizer
            recovery_factor: Factor to multiply learning rate by
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] *= recovery_factor


def create_sft_utilities(data_dir: str, tokenizer_name: str = "gpt2") -> Tuple[SFTDataManager, SFTLossAnalyzer, SFTTrainingUtils]:
    """
    Convenience function to create all SFT utility instances.
    
    Args:
        data_dir: Directory containing SFT data
        tokenizer_name: Name of tokenizer to use
    
    Returns:
        Tuple of (data_manager, loss_analyzer, training_utils)
    """
    data_manager = SFTDataManager(data_dir, tokenizer_name)
    loss_analyzer = SFTLossAnalyzer(tokenizer_name)
    training_utils = SFTTrainingUtils()
    
    return data_manager, loss_analyzer, training_utils
