#!/usr/bin/env python3
"""
Plot clustering entropy comparison across variants.
Run with no arguments: python plot_clustering_entropy.py
"""

import matplotlib.pyplot as plt
import re
import numpy as np
import glob
import os
import argparse
from collections import defaultdict

def extract_clustering_entropy(line):
    """
    Extract clustering_entropy value from a log line.
    """
    pattern = r'clustering_entropy:\s+([-\d.]+)'
    match = re.search(pattern, line)
    if match:
        return float(match.group(1))
    return None


def parse_log_file(data):
    """
    Parse log file content to extract iterations and clustering_entropy values.
    """
    iteration_data = {}
    
    for line in data.split('\n'):
        # Match the iteration format
        iter_match = re.search(r'It\s+(\d+)\s*/\s*[\d\.]+k', line)
        
        if iter_match:
            actual_iter = int(iter_match.group(1))
            entropy = extract_clustering_entropy(line)
            
            if entropy is not None:
                iteration_data[actual_iter] = entropy
    
    return iteration_data


def load_variant_data(base_path, arch, variant, use_seqpacking=False):
    """
    Load data for a specific architecture-variant combination.
    Handles multiple log files with overlap using iteration-based stitching.
    
    Args:
        base_path: Base directory path
        arch: Architecture (S, B, L, H, G)
        variant: Variant identifier (e.g., B3)
        use_seqpacking: If True, look for _seqpacking directory
    """
    if use_seqpacking:
        log_pattern = f"{base_path}/TCGA_TMEDinov3_ViT-{arch}_{variant}_seqpacking/logs/*_0_log.out"
    else:
        log_pattern = f"{base_path}/TCGA_TMEDinov3_ViT-{arch}_{variant}/logs/*_0_log.out"
    
    log_files = sorted(glob.glob(log_pattern))
    
    if not log_files:
        return None, None
    
    # Sort by modification time (oldest first, newest last)
    log_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Collect all data - newer files overwrite older ones at same iteration
    all_iteration_data = {}
    
    for file_path in log_files:
        with open(file_path, 'r') as file:
            data = file.read()
            iteration_data = parse_log_file(data)
            
            # Store/overwrite data for each iteration
            all_iteration_data.update(iteration_data)
    
    if not all_iteration_data:
        return None, None
    
    # Convert to sorted lists
    sorted_iters = sorted(all_iteration_data.keys())
    entropies = [all_iteration_data[i] for i in sorted_iters]
    
    return sorted_iters, entropies


def plot_clustering_entropy(arch, variant_data, output_path):
    """
    Create a professional plot for clustering entropy across variants.
    Now includes both original and seqpacking versions.
    """
    plt.style.use('ggplot')
    
    # A4 landscape dimensions (adjusted for good proportions)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Add border around the figure
    fig.patch.set_edgecolor('black')
    fig.patch.set_linewidth(3)
    
    # Colorblind-friendly palette
    colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', 
              '#949494', '#ECE133', '#56B4E9', '#D55E00', '#F0E442']
    
    # Line styles for differentiation
    linestyles = {
        'original': '-',      # Solid line
        'seqpacking': '--'    # Dashed line
    }
    
    # Plot each variant
    idx = 0
    for variant_key in sorted(variant_data.keys()):
        iterations, entropies, is_seqpacking = variant_data[variant_key]
        
        # Create label
        if is_seqpacking:
            label = f"{variant_key}_seqpacking"
            linestyle = linestyles['seqpacking']
            linewidth = 2.5
        else:
            label = variant_key
            linestyle = linestyles['original']
            linewidth = 2.5
        
        ax.plot(iterations, entropies, 
                color=colors[idx % len(colors)],
                linewidth=linewidth,
                linestyle=linestyle,
                label=label,
                alpha=0.85)
        
        idx += 1
    
    # Custom formatter for x-axis (show values as 'k' for thousands)
    def format_thousands(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        else:
            return f'{int(x)}'
    
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
    
    # Styling
    ax.set_xlabel('Iteration', fontsize=13, fontweight='bold')
    ax.set_ylabel('Clustering Entropy', fontsize=13, fontweight='bold')
    ax.set_title(f'ViT-{arch} Clustering Entropy Comparison', fontsize=15, fontweight='bold', pad=15)
    
    # Let y-axis auto-scale
    # (removed fixed ylim)
    
    # Bold borders
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    # Grid
    ax.grid(True, alpha=0.3, linewidth=1)
    
    # Legend with custom handles to show linestyle legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=2.5, linestyle='-', label='Original'),
        Line2D([0], [0], color='gray', linewidth=2.5, linestyle='--', label='Seqpacking')
    ]
    
    # Get current legend handles
    handles, labels = ax.get_legend_handles_labels()
    
    # Add custom legend entries at the top
    all_handles = legend_elements + handles
    all_labels = ['', ''] + labels
    
    legend = ax.legend(all_handles, all_labels, loc='best', frameon=True, fontsize=9, 
                       framealpha=0.9, edgecolor='black', ncol=2)
    legend.get_frame().set_linewidth(1.5)
    
    # Tick parameters
    ax.tick_params(labelsize=10, width=2, length=6)
    
    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='black', pad_inches=0.1)
    plt.close()
    
    print(f"Saved plot: {output_path}")


def main():
    """
    Main function to process all architecture-variant combinations.
    Loads both original and seqpacking versions.
    """
    parser = argparse.ArgumentParser(description='Plot clustering entropy comparison')
    parser.add_argument('--base_path', type=str, default=None,
                       help='Base directory path (default: ../..)')
    parser.add_argument('--architectures', type=str, nargs='+', default=['S', 'B', 'L', 'H', 'G'],
                       help='Architectures to process (default: S B L H G)')
    args = parser.parse_args()
    
    # Default to relative path from script location
    if args.base_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.base_path = os.path.join(script_dir, '..', '..')
    
    for arch in args.architectures:
        print(f"\nProcessing ViT-{arch}...")
        
        variant_data = {}
        
        # Try to find all variants (checking up to 10 should be sufficient)
        for i in range(1, 11):
            variant = f"{arch}{i}"
            
            # Try original version (without _seqpacking)
            iterations, entropies = load_variant_data(args.base_path, arch, variant, use_seqpacking=False)
            if iterations is not None:
                variant_data[variant] = (iterations, entropies, False)
                print(f"  Found {variant}: {len(iterations)} iterations "
                      f"(range: {min(iterations)}-{max(iterations)})")
            
            # Try seqpacking version
            iterations_seq, entropies_seq = load_variant_data(args.base_path, arch, variant, use_seqpacking=True)
            if iterations_seq is not None:
                # Use a different key to distinguish
                variant_data[f"{variant}_sp"] = (iterations_seq, entropies_seq, True)
                print(f"  Found {variant}_seqpacking: {len(iterations_seq)} iterations "
                      f"(range: {min(iterations_seq)}-{max(iterations_seq)})")
        
        # Generate plot if any variants were found
        if variant_data:
            output_path = f'clustering_entropy_ViT-{arch}_comparison.png'
            plot_clustering_entropy(arch, variant_data, output_path)
            
            # Print statistics
            print(f"\n  Statistics for ViT-{arch}:")
            for variant_key in sorted(variant_data.keys()):
                _, entropies, is_seqpacking = variant_data[variant_key]
                label = variant_key if not is_seqpacking else variant_key.replace('_sp', '_seqpacking')
                print(f"    {label}: min={min(entropies):.4f}, "
                      f"max={max(entropies):.4f}, mean={np.mean(entropies):.4f}")
        else:
            print(f"  No variants found for ViT-{arch}")


if __name__ == "__main__":
    main()