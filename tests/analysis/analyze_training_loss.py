#!/usr/bin/env python3
"""
SFT Loss Analysis and Debugging Tool
Analyzes training logs to identify patterns in loss explosions
"""

import re
import numpy as np
import matplotlib.pyplot as plt

def analyze_training_log(log_text):
    """Analyze training log for loss patterns"""
    
    # Extract iterations and losses
    pattern = r'iter (\d+): loss ([\d.]+)'
    matches = re.findall(pattern, log_text)
    
    iterations = [int(m[0]) for m in matches]
    losses = [float(m[1]) for m in matches]
    
    if not losses:
        print("No loss values found in log")
        return
    
    print(f"📊 Training Analysis Summary")
    print(f"=" * 40)
    print(f"Iterations analyzed: {len(iterations)}")
    print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
    print(f"Average loss: {np.mean(losses):.4f}")
    print(f"Loss std dev: {np.std(losses):.4f}")
    
    # Detect loss spikes
    loss_diffs = np.diff(losses)
    spike_threshold = 2.0  # 2x increase
    spikes = []
    
    for i, (curr_loss, prev_loss) in enumerate(zip(losses[1:], losses[:-1])):
        if curr_loss > prev_loss * spike_threshold:
            spikes.append((iterations[i+1], curr_loss, prev_loss))
    
    print(f"\n🔍 Loss Spike Analysis")
    print(f"Spike threshold: {spike_threshold}x increase")
    print(f"Number of spikes detected: {len(spikes)}")
    
    for iter_num, curr, prev in spikes:
        print(f"  Iteration {iter_num}: {prev:.4f} → {curr:.4f} ({curr/prev:.2f}x increase)")
    
    # Analyze trends
    print(f"\n📈 Loss Trend Analysis")
    if len(losses) >= 10:
        early_avg = np.mean(losses[:10])
        late_avg = np.mean(losses[-10:])
        trend = "decreasing" if late_avg < early_avg else "increasing"
        change = abs(late_avg - early_avg) / early_avg * 100
        print(f"Overall trend: {trend} ({change:.1f}% change)")
        print(f"Early average (first 10): {early_avg:.4f}")
        print(f"Recent average (last 10): {late_avg:.4f}")
    
    # Recommendations
    print(f"\n💡 Recommendations")
    if len(spikes) > 0:
        print("⚠️  Loss spikes detected. Consider:")
        print("   - Reducing learning rate (try 1e-6 instead of 5e-6)")
        print("   - Increasing gradient clipping (try 0.5 instead of 1.0)")
        print("   - Extending warmup period")
        print("   - Adding loss spike detection and recovery")
    
    if np.std(losses) > np.mean(losses) * 0.1:
        print("⚠️  High loss variance detected. Consider:")
        print("   - Reviewing data quality and preprocessing")
        print("   - Using smaller learning rate")
        print("   - Increasing batch size or gradient accumulation")
    
    return {
        'iterations': iterations,
        'losses': losses,
        'spikes': spikes,
        'stats': {
            'mean': np.mean(losses),
            'std': np.std(losses),
            'min': min(losses),
            'max': max(losses)
        }
    }

if __name__ == "__main__":
    # Sample log from user's training
    sample_log = """
    iter 477: loss 40.4580, time 5297.17ms, mfu 2.35%, mem 5.83/13.86GB
    iter 478: loss 40.4865, time 5324.20ms, mfu 2.35%, mem 5.83/13.86GB
    iter 479: loss 40.1967, time 5322.43ms, mfu 2.35%, mem 5.83/13.86GB
    iter 480: loss 40.3984, time 5316.65ms, mfu 2.35%, mem 5.83/13.86GB
    iter 498: loss 41.9228, time 5293.77ms, mfu 2.36%, mem 5.83/13.87GB
    iter 499: loss 42.3715, time 5321.94ms, mfu 2.36%, mem 5.83/13.87GB
    iter 500: loss 41.8939, time 5346.81ms, mfu 2.35%, mem 5.83/13.87GB
    iter 501: loss 42.4953, time 5308.02ms, mfu 2.35%, mem 5.83/13.87GB
    iter 502: loss 42.9865, time 5285.10ms, mfu 2.36%, mem 5.83/13.87GB
    iter 508: loss 38.4209, time 5303.08ms, mfu 2.36%, mem 5.83/13.87GB
    iter 509: loss 35.3692, time 5350.46ms, mfu 2.36%, mem 5.83/13.87GB
    iter 510: loss 35.4251, time 5315.49ms, mfu 2.36%, mem 5.83/13.87GB
    iter 517: loss 35.9834, time 5270.39ms, mfu 2.36%, mem 5.83/13.88GB
    """
    
    analyze_training_log(sample_log)
