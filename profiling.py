from reformer.reformer_pytorch import ViRWithArcMargin, ViR
from vit_pytorch import ViT
import matplotlib.pyplot as plt
import numpy as np
import csv
from utils.datasets import get_lpfw_dataloaders
import os
from utils.profiler import ModelProfiler

# Create results directory if it doesn't exist
os.makedirs('hasil/profiling_results', exist_ok=True)

# Initialize CSV files with headers
csv_headers = ["Model", "Batch Size", "Patch Size", "Mean Batch Time (ms)", 
               "Standard Deviation (ms)", "Throughput (samples/second)", 
               "Peak GPU Memory (MB)", "Mean Batch Memory (MB)"]

with open('hasil/profiling_results/performance_metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# Define parameter ranges
batch_sizes = range(1, 8)
patch_sizes = [4, 8, 16, 32]  # Common patch sizes for vision transformers

def create_models(patch_size, num_classes):
    """Create both models with given patch size."""
    model_reformer = ViR(
        image_size=224,
        patch_size=patch_size,
        num_classes=2,
        dim=100,
        depth=12,
        heads=9,
        bucket_size=2,
        emb_dropout=0.1,
        num_mem_kv=3,
        n_hashes=1
    )

    model_vit = ViT(
        image_size=224,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=256,
        depth=12,
        heads=8,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    return model_reformer, model_vit

def save_results(model_name, batch_size, patch_size, results):
    """Save profiling results to CSV."""
    with open('profiling_results/performance_metrics.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            model_name,
            batch_size,
            patch_size,
            results['timing_statistics']['mean_batch_time_ms'],
            results['timing_statistics']['std_batch_time_ms'],
            results['timing_statistics']['samples_per_second'],
            results['memory_statistics']['peak_gpu_memory_mb'],
            results['memory_statistics']['mean_batch_memory_mb']
        ])

# Lists to store results for plotting
plot_data = {
    'reformer': {'batch_times': [], 'throughput': [], 'memory': []},
    'vit': {'batch_times': [], 'throughput': [], 'memory': []}
}

# Main profiling loop
for patch_size in patch_sizes:
    print(f"\nProfiling with patch size: {patch_size}")
    
    for batch_size in batch_sizes:
        print(f"Processing batch size: {batch_size}")
        
        # Get dataloaders
        train_loader, test_loader = get_lpfw_dataloaders(batch_size)
        num_classes = len(train_loader.dataset.dataset.class_to_idx)
        
        # Create models with current patch size
        model_reformer, model_vit = create_models(patch_size, num_classes)
        
        # Profile Reformer
        profiler_reformer = ModelProfiler(model_reformer)
        results_reformer = profiler_reformer.profile_with_loader(train_loader, num_batches=1)
        save_results('Reformer', batch_size, patch_size, results_reformer)
        
        # Profile ViT
        profiler_vit = ModelProfiler(model_vit)
        results_vit = profiler_vit.profile_with_loader(train_loader, num_batches=1)
        save_results('ViT', batch_size, patch_size, results_vit)
        
        # Store results for plotting
        for model_name, results in [('reformer', results_reformer), ('vit', results_vit)]:
            plot_data[model_name]['batch_times'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['timing_statistics']['mean_batch_time_ms']
            })
            plot_data[model_name]['throughput'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['timing_statistics']['samples_per_second']
            })
            plot_data[model_name]['memory'].append({
                'patch_size': patch_size,
                'batch_size': batch_size,
                'value': results['memory_statistics']['peak_gpu_memory_mb']
            })

# Create plots
fig, axs = plt.subplots(3, len(patch_sizes), figsize=(20, 15))
metrics = ['batch_times', 'throughput', 'memory']
titles = ['Batch Time', 'Throughput', 'Memory Usage']
ylabels = ['Time (ms)', 'Samples/Second', 'Memory (MB)']

for i, metric in enumerate(metrics):
    for j, patch_size in enumerate(patch_sizes):
        # Filter data for current patch size
        reformer_data = [x for x in plot_data['reformer'][metric] if x['patch_size'] == patch_size]
        vit_data = [x for x in plot_data['vit'][metric] if x['patch_size'] == patch_size]
        
        # Plot
        axs[i, j].plot(
            [x['batch_size'] for x in reformer_data],
            [x['value'] for x in reformer_data],
            label='Reformer'
        )
        axs[i, j].plot(
            [x['batch_size'] for x in vit_data],
            [x['value'] for x in vit_data],
            label='ViT'
        )
        
        axs[i, j].set_title(f'{titles[i]} (Patch Size {patch_size})')
        axs[i, j].set_xlabel('Batch Size')
        axs[i, j].set_ylabel(ylabels[i])
        axs[i, j].legend()

plt.tight_layout()
plt.savefig('hasil/profiling_results/performance_comparison.png')
plt.close()