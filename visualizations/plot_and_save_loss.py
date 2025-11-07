import matplotlib.pyplot as plt
import re
import numpy as np
import glob
import os
from collections import defaultdict

def extract_metrics_from_line(line):
    """
    Extract all numeric metrics from a log line using regex pattern matching.
    Now handles any metrics with '_loss' suffix as well as learning rate/weight decay parameters.
    """
    metrics = {}
    pattern = r'(\w+(?:_\w+)*): ([-\d.]+)(?: \(([-\d.]+)\))?'
    matches = re.finditer(pattern, line)
    
    for match in matches:
        metric_name, instant_value, avg_value = match.groups()
        if '_loss' in metric_name or any(term in metric_name for term in ['lr', 'weight_decay']):
            metrics[metric_name] = float(instant_value)
        if 'clustering_entropy' in metric_name:
            metrics[metric_name] = float(instant_value)
    
    return metrics


def parse_log_file(data, start_iteration=0):
    """
    Parse log file content to extract iterations, losses, and epoch boundaries.
    Handles both original and new log formats.
    """
    iterations = []
    losses = defaultdict(list)
    epoch_boundaries = []
    current_iter = start_iteration  # Only used for epoch format
    
    for line in data.split('\n'):
        # Try to match the original epoch format
        epoch_match = re.search(r'Epoch: \[(\d+)/(\d+)\]\s+\[\s*(\d+)/(\d+)\]', line)
        # Try to match the new iteration format
        iter_match = re.search(r'It\s+(\d+)\s*/\s*[\d\.]+k', line)
        
        if epoch_match:
            epoch = int(epoch_match.group(1))
            metrics = extract_metrics_from_line(line)
            
            if metrics:
                iterations.append(current_iter)
                for metric_name, value in metrics.items():
                    losses[metric_name].append(value)
                
                if len(epoch_boundaries) == 0 or epoch_boundaries[-1][1] != epoch:
                    epoch_boundaries.append((current_iter, epoch))
                
                current_iter += 1
        elif iter_match:
            # Extract the actual iteration number from the log
            actual_iter = int(iter_match.group(1))
            metrics = extract_metrics_from_line(line)
            
            if metrics:
                iterations.append(actual_iter)  # Use actual iteration from log
                for metric_name, value in metrics.items():
                    losses[metric_name].append(value)
    
    return iterations, losses, epoch_boundaries


def plot_training_losses(iterations, losses, epoch_boundaries, output_path):
    """
    Create a grid plot with 4 columns for all loss terms.
    """
    plt.style.use('dark_background')
    plt.rcParams.update({'font.size': 10})
    
    # Filter to only include loss metrics
    loss_metrics = [key for key in losses.keys() if '_loss' in key]
    
    # Calculate layout dimensions
    n_cols = 4
    n_rows = (len(loss_metrics) + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig = plt.figure(figsize=(5*n_cols, 3*n_rows))
    
    # Colors for different loss terms
    colors = ['#0088ff', '#00ff00', '#ff00ff', '#ffff00', '#ff8800', '#00ffff']
    
    # Plot each loss in its own subplot
    for idx, metric in enumerate(loss_metrics):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        values = np.array(losses[metric])
        
        # Ensure iterations and values have the same length
        min_len = min(len(iterations), len(values))
        iter_data = iterations[:min_len]
        value_data = values[:min_len]
        
        ax.plot(iter_data, value_data,
                   color=colors[idx % len(colors)],
                   alpha=0.8, linewidth=1.5)
        
        # Add epoch boundary markers
        ylim = ax.get_ylim()
        for iter_num, epoch in epoch_boundaries:
            if epoch % 10 == 0 and iter_num <= iter_data[-1]:
                ax.axvline(x=iter_num, color='gray', alpha=0.3, linestyle='--')
                ax.text(iter_num, ylim[1], f'E{epoch}',
                       rotation=90, va='top', ha='right',
                       color='gray', alpha=0.5)
        
        # Customize subplot appearance
        ax.set_title(metric.replace('_', ' ').title(),
                    color='white', pad=10)
        ax.grid(True, alpha=0.2, which='both')
        
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_facecolor('black')
        
        # Add y-label to leftmost plots
        if idx % n_cols == 0:
            ax.set_ylabel('Loss', color='white')
        
        # Add x-label to bottom row plots
        if idx >= len(loss_metrics) - n_cols:
            ax.set_xlabel('Iteration', color='white')
    
    plt.gcf().set_facecolor('black')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close()


def main():
    """
    Main function to process log files and generate visualization.
    Properly handles overlapping iterations between log files.
    """
    base_path = '../logs/*_0_log.out'
    output_path = 'training_losses.png'
    
    log_files = sorted(glob.glob(base_path))
    if not log_files:
        print("No log files found")
        return
    
    # Sort by modification time (oldest first, newest last)
    log_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Collect ALL data first - {iteration: {metric: value}}
    all_iteration_data = {}
    all_epoch_boundaries = []
    
    for file_path in log_files:
        print(f"Processing file: {file_path}")
        with open(file_path, 'r') as file:
            data = file.read()
            iterations, losses, boundaries = parse_log_file(data, 0)
            
            if not iterations:
                print(f"No data found in {file_path}, skipping")
                continue
            
            print(f"  Found iterations: {min(iterations)} to {max(iterations)}")
            
            # Store/overwrite data for each iteration
            for i, iter_num in enumerate(iterations):
                if iter_num not in all_iteration_data:
                    all_iteration_data[iter_num] = {}
                
                for metric, values in losses.items():
                    if i < len(values):
                        all_iteration_data[iter_num][metric] = values[i]
            
            # Collect epoch boundaries
            all_epoch_boundaries.extend(boundaries)
    
    if all_iteration_data:
        # Convert back to lists, sorted by iteration
        sorted_iters = sorted(all_iteration_data.keys())
        all_losses = defaultdict(list)
        
        for iter_num in sorted_iters:
            for metric, value in all_iteration_data[iter_num].items():
                all_losses[metric].append(value)
        
        # Print data length information
        print(f"\nTotal unique iterations: {len(sorted_iters)}")
        print(f"Iteration range: {min(sorted_iters)} to {max(sorted_iters)}")
        print("\nData lengths:")
        for metric, values in all_losses.items():
            print(f"{metric}: {len(values)}")
        
        # Deduplicate epoch boundaries
        unique_boundaries = list(set(all_epoch_boundaries))
        unique_boundaries.sort(key=lambda x: x[0])
        
        plot_training_losses(sorted_iters, all_losses, unique_boundaries, output_path)
        print(f"\nPlot saved to {output_path}")
        
        print("\nLoss Statistics:")
        for metric in sorted(all_losses.keys()):
            values = all_losses[metric]
            print(f"{metric:25s}: min = {min(values):.6f}, max = {max(values):.6f}, "
                  f"mean = {np.mean(values):.6f}")
    else:
        print("No data points were extracted from the log files")


if __name__ == "__main__":
    main()
