#!/usr/bin/env python3
"""
PCA-based Prototype Weight Matrix Clustering
Side-by-side comparison of PCA-reduced vs raw first-N-dims similarity.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Publication settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10,
    'savefig.dpi': 300,
})


def main():
    parser = argparse.ArgumentParser(description='Plot prototype clustering analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: ../logs/checkpoint.pth)')
    parser.add_argument('--output', type=str, default='prototype_clustermap.png',
                       help='Output path (default: prototype_clustermap.png)')
    parser.add_argument('--n_components', type=int, default=10,
                       help='Number of PCA components (default: 10)')
    parser.add_argument('--n_raw_dims', type=int, default=10,
                       help='Number of raw dimensions for comparison (default: 10)')
    args = parser.parse_args()
    
    # Default to relative path from script location
    if args.checkpoint is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.checkpoint = os.path.join(script_dir, '..', 'logs', 'checkpoint.pth')
    
    print(f"Loading checkpoint: {args.checkpoint}")
    x = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # ========== LEFT PLOT: PCA-reduced similarity ==========
    X_full = F.normalize(x["prototype_bank"]["module.proto_layer.weight"].cpu().float(), dim=-1)
    
    print(f"Original prototype weight shape: {X_full.shape}")
    
    # Apply PCA
    print(f"Applying PCA to reduce {X_full.shape[1]}D → {args.n_components}D...")
    pca = PCA(n_components=args.n_components)
    X_pca = pca.fit_transform(X_full.numpy())
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained by {args.n_components} components: {variance_explained:.1%}")
    
    # Normalize PCA-reduced prototypes
    X_pca_norm = F.normalize(torch.from_numpy(X_pca).float(), dim=-1)
    
    # Compute similarity matrix
    similarity_pca = (X_pca_norm @ X_pca_norm.T).numpy()
    
    print(f"PCA prototype shape: {X_pca_norm.shape}")
    print(f"PCA similarity range: [{similarity_pca.min():.3f}, {similarity_pca.max():.3f}]")
    
    # ========== RIGHT PLOT: Raw first-N-dims similarity ==========
    X_raw = F.normalize(x["prototype_bank"]["module.proto_layer.weight"][:,:args.n_raw_dims].cpu().float(), dim=-1)
    similarity_raw = (X_raw @ X_raw.T).numpy()
    
    print(f"Raw {args.n_raw_dims}D similarity range: [{similarity_raw.min():.3f}, {similarity_raw.max():.3f}]")
    
    # ========== Create side-by-side plot ==========
    print("Creating side-by-side clustermaps...")
    
    fig = plt.figure(figsize=(20, 9))
    
    # Left: PCA clustermap
    ax1 = plt.subplot(1, 2, 1)
    g1 = sns.clustermap(
        similarity_pca, 
        vmin=-1, vmax=1, 
        cmap="coolwarm",
        figsize=(9, 9),
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity'},
    )
    
    # Position left plot
    heatmap_pos = g1.ax_heatmap.get_position()
    dendro_left_pos = g1.ax_row_dendrogram.get_position()
    dendro_top_pos = g1.ax_col_dendrogram.get_position()
    cbar_pos = g1.ax_cbar.get_position()
    
    g1.ax_heatmap.set_position([0.08, 0.12, 0.35, 0.72])
    g1.ax_row_dendrogram.set_position([0.03, 0.12, 0.04, 0.72])
    g1.ax_col_dendrogram.set_position([0.08, 0.85, 0.35, 0.06])
    g1.ax_cbar.set_position([0.44, 0.35, 0.01, 0.35])
    
    g1.ax_heatmap.set_title(f'PCA {args.n_components}D ({variance_explained:.1%} variance)', 
                            fontsize=13, pad=35, fontweight='bold')
    
    # Right: Raw dimension clustermap
    g2 = sns.clustermap(
        similarity_raw, 
        vmin=-1, vmax=1, 
        cmap="coolwarm",
        figsize=(9, 9),
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity'},
    )
    
    # Position right plot
    g2.ax_heatmap.set_position([0.55, 0.12, 0.35, 0.72])
    g2.ax_row_dendrogram.set_position([0.50, 0.12, 0.04, 0.72])
    g2.ax_col_dendrogram.set_position([0.55, 0.85, 0.35, 0.06])
    g2.ax_cbar.set_position([0.91, 0.35, 0.01, 0.35])
    
    g2.ax_heatmap.set_title(f'First {args.n_raw_dims}D (raw)', 
                            fontsize=13, pad=35, fontweight='bold')
    
    # Main title
    fig.suptitle('Prototype Weight Matrix Clustering', 
                fontsize=16, fontweight='bold', y=0.97)
    
    plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
    
    print(f"\n✓ Saved: {args.output}")
    
    # Print per-component variance
    print(f"\nVariance explained per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.2%}")


if __name__ == "__main__":
    main()