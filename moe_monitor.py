"""
Advanced monitoring and profiling for MoE training on V100s
"""

import time
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import json

class MoEMonitor:
    """Monitor MoE training metrics"""
    
    def __init__(self, num_experts, log_interval=100):
        self.num_experts = num_experts
        self.log_interval = log_interval
        self.reset_stats()
        
    def reset_stats(self):
        self.expert_usage = torch.zeros(self.num_experts)
        self.expert_load = torch.zeros(self.num_experts)
        self.routing_entropy = []
        self.load_balance_loss = []
        self.step_count = 0
        
    def update(self, expert_indices, gate_probs, load_balance_loss=None):
        """Update monitoring stats"""
        self.step_count += 1
        
        # Track expert usage
        for expert_id in range(self.num_experts):
            usage = (expert_indices == expert_id).float().sum()
            self.expert_usage[expert_id] += usage
            
        # Track routing entropy (diversity)
        if gate_probs is not None:
            entropy = -(gate_probs * torch.log(gate_probs + 1e-8)).sum(dim=-1).mean()
            self.routing_entropy.append(entropy.item())
            
        # Track load balance loss
        if load_balance_loss is not None:
            self.load_balance_loss.append(load_balance_loss.item())
            
    def log_stats(self, step):
        """Log current statistics"""
        if step % self.log_interval == 0:
            # Expert utilization
            total_usage = self.expert_usage.sum()
            if total_usage > 0:
                usage_pct = (self.expert_usage / total_usage * 100).tolist()
                print(f"Step {step} Expert Usage %: {[f'{u:.1f}' for u in usage_pct]}")
                
                # Check for expert collapse
                unused_experts = (self.expert_usage == 0).sum().item()
                if unused_experts > 0:
                    print(f"WARNING: {unused_experts} experts are unused!")
                    
            # Routing entropy
            if self.routing_entropy:
                avg_entropy = np.mean(self.routing_entropy[-self.log_interval:])
                print(f"Routing Entropy: {avg_entropy:.3f}")
                
            # Load balance loss
            if self.load_balance_loss:
                avg_lb_loss = np.mean(self.load_balance_loss[-self.log_interval:])
                print(f"Load Balance Loss: {avg_lb_loss:.6f}")
                
    def plot_stats(self, save_path=None):
        """Plot monitoring statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Expert usage distribution
        axes[0, 0].bar(range(self.num_experts), self.expert_usage.numpy())
        axes[0, 0].set_title('Expert Usage Distribution')
        axes[0, 0].set_xlabel('Expert ID')
        axes[0, 0].set_ylabel('Usage Count')
        
        # Routing entropy over time
        if self.routing_entropy:
            axes[0, 1].plot(self.routing_entropy)
            axes[0, 1].set_title('Routing Entropy Over Time')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Entropy')
            
        # Load balance loss over time
        if self.load_balance_loss:
            axes[1, 0].plot(self.load_balance_loss)
            axes[1, 0].set_title('Load Balance Loss Over Time')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            
        # Expert utilization percentage
        total_usage = self.expert_usage.sum()
        if total_usage > 0:
            usage_pct = (self.expert_usage / total_usage * 100).numpy()
            axes[1, 1].pie(usage_pct, labels=[f'E{i}' for i in range(self.num_experts)], autopct='%1.1f%%')
            axes[1, 1].set_title('Expert Utilization %')
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class GPUProfiler:
    """Profile GPU usage and memory"""
    
    def __init__(self):
        self.gpu_stats = defaultdict(list)
        self.timing_stats = defaultdict(list)
        
    def start_timing(self, name):
        """Start timing an operation"""
        torch.cuda.synchronize()
        setattr(self, f"{name}_start", time.time())
        
    def end_timing(self, name):
        """End timing an operation"""
        torch.cuda.synchronize()
        elapsed = time.time() - getattr(self, f"{name}_start")
        self.timing_stats[name].append(elapsed)
        return elapsed
        
    def log_gpu_memory(self, step):
        """Log GPU memory usage"""
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            self.gpu_stats[f'gpu_{i}_allocated'].append(memory_allocated)
            self.gpu_stats[f'gpu_{i}_reserved'].append(memory_reserved)
            
    def print_summary(self):
        """Print profiling summary"""
        print("\n=== GPU Memory Summary ===")
        for i in range(torch.cuda.device_count()):
            if f'gpu_{i}_allocated' in self.gpu_stats:
                allocated = self.gpu_stats[f'gpu_{i}_allocated']
                reserved = self.gpu_stats[f'gpu_{i}_reserved']
                print(f"GPU {i}: Allocated: {np.mean(allocated):.2f}±{np.std(allocated):.2f} GB, "
                      f"Reserved: {np.mean(reserved):.2f}±{np.std(reserved):.2f} GB")
                
        print("\n=== Timing Summary ===")
        for name, times in self.timing_stats.items():
            print(f"{name}: {np.mean(times)*1000:.2f}±{np.std(times)*1000:.2f} ms")

class PerformanceOptimizer:
    """Automatically tune performance parameters"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.best_throughput = 0
        self.best_params = None
        
    def tune_batch_size(self, min_batch=1, max_batch=32):
        """Find optimal batch size"""
        print("Tuning batch size...")
        throughputs = []
        batch_sizes = []
        
        for batch_size in range(min_batch, max_batch + 1):
            try:
                # Test throughput with this batch size
                throughput = self._measure_throughput(batch_size)
                throughputs.append(throughput)
                batch_sizes.append(batch_size)
                print(f"Batch size {batch_size}: {throughput:.2f} tokens/sec")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Batch size {batch_size}: OOM")
                    break
                else:
                    raise e
                    
        if throughputs:
            best_idx = np.argmax(throughputs)
            best_batch_size = batch_sizes[best_idx]
            best_throughput = throughputs[best_idx]
            print(f"Optimal batch size: {best_batch_size} ({best_throughput:.2f} tokens/sec)")
            return best_batch_size
        else:
            return min_batch
            
    def _measure_throughput(self, batch_size, num_steps=10):
        """Measure training throughput"""
        self.model.train()
        block_size = self.config.get('block_size', 1024)
        vocab_size = self.config.get('vocab_size', 50304)
        
        # Create dummy data
        x = torch.randint(0, vocab_size, (batch_size, block_size), device='cuda')
        y = torch.randint(0, vocab_size, (batch_size, block_size), device='cuda')
        
        # Warmup
        for _ in range(3):
            with torch.amp.autocast('cuda'):
                logits, loss = self.model(x, y)
                loss.backward()
                
        # Measure
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_steps):
            with torch.amp.autocast('cuda'):
                logits, loss = self.model(x, y)
                loss.backward()
                
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_tokens = batch_size * block_size * num_steps
        throughput = total_tokens / (end_time - start_time)
        
        # Clear gradients
        self.model.zero_grad()
        
        return throughput

def benchmark_moe_configurations():
    """Benchmark different MoE configurations"""
    configurations = [
        {'num_experts': 8, 'top_k_experts': 1, 'name': '8E_1K'},
        {'num_experts': 8, 'top_k_experts': 2, 'name': '8E_2K'},
        {'num_experts': 16, 'top_k_experts': 2, 'name': '16E_2K'},
        {'num_experts': 32, 'top_k_experts': 2, 'name': '32E_2K'},
        {'num_experts': 16, 'top_k_experts': 4, 'name': '16E_4K'},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\nBenchmarking {config['name']}...")
        try:
            # Create model with this config
            from model import GPT, GPTConfig
            
            model_config = GPTConfig(
                n_layer=12,
                n_head=12,
                n_embd=768,
                block_size=1024,
                num_experts=config['num_experts'],
                top_k_experts=config['top_k_experts']
            )
            
            model = GPT(model_config).cuda()
            optimizer = PerformanceOptimizer(model, model_config.__dict__)
            
            # Measure performance
            throughput = optimizer._measure_throughput(batch_size=8)
            memory_usage = torch.cuda.max_memory_allocated() / 1024**3
            
            results.append({
                'config': config['name'],
                'throughput': throughput,
                'memory_gb': memory_usage,
                'params': model.get_num_params() / 1e6
            })
            
            print(f"Throughput: {throughput:.2f} tokens/sec")
            print(f"Memory: {memory_usage:.2f} GB")
            print(f"Parameters: {model.get_num_params() / 1e6:.1f} M")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Failed to benchmark {config['name']}: {e}")
            
    # Print summary
    print("\n=== Benchmark Summary ===")
    for result in results:
        print(f"{result['config']}: {result['throughput']:.1f} tok/s, "
              f"{result['memory_gb']:.1f} GB, {result['params']:.1f}M params")
              
    return results

def save_profiling_report(monitor, profiler, save_path):
    """Save comprehensive profiling report"""
    report = {
        'expert_usage': monitor.expert_usage.tolist(),
        'routing_entropy': monitor.routing_entropy,
        'load_balance_loss': monitor.load_balance_loss,
        'gpu_stats': dict(profiler.gpu_stats),
        'timing_stats': dict(profiler.timing_stats),
        'timestamp': time.time()
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Profiling report saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    print("Running MoE benchmarks...")
    results = benchmark_moe_configurations()
    
    # Example monitoring
    monitor = MoEMonitor(num_experts=16)
    profiler = GPUProfiler()
    
    # Simulate some training steps
    for step in range(100):
        # Dummy expert assignments and gate probabilities
        expert_indices = torch.randint(0, 16, (32,))  # 32 tokens
        gate_probs = torch.softmax(torch.randn(32, 16), dim=-1)
        load_balance_loss = torch.tensor(0.01)
        
        monitor.update(expert_indices, gate_probs, load_balance_loss)
        monitor.log_stats(step)
        
    # Plot results
    monitor.plot_stats('moe_stats.png')
    profiler.print_summary()
