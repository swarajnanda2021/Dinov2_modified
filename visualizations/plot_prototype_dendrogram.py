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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

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
    
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Compute hierarchical clustering for PCA
    dist_pca = 1 - similarity_pca
    np.fill_diagonal(dist_pca, 0)
    condensed_pca = squareform(dist_pca, checks=False)
    linkage_pca = linkage(condensed_pca, method='average')
    dendro_pca = dendrogram(linkage_pca, no_plot=True)
    order_pca = dendro_pca['leaves']
    
    # Compute hierarchical clustering for raw
    dist_raw = 1 - similarity_raw
    np.fill_diagonal(dist_raw, 0)
    condensed_raw = squareform(dist_raw, checks=False)
    linkage_raw = linkage(condensed_raw, method='average')
    dendro_raw = dendrogram(linkage_raw, no_plot=True)
    order_raw = dendro_raw['leaves']
    
    # Reorder matrices
    similarity_pca_ordered = similarity_pca[order_pca, :][:, order_pca]
    similarity_raw_ordered = similarity_raw[order_raw, :][:, order_raw]
    
    # Create figure with 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: PCA heatmap
    im1 = ax1.imshow(similarity_pca_ordered, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'PCA {args.n_components}D\n({variance_explained:.1%} variance)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.colorbar(im1, ax=ax1, label='Cosine Similarity', fraction=0.046, pad=0.04)
    
    # Right: Raw heatmap
    im2 = ax2.imshow(similarity_raw_ordered, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title(f'First {args.n_raw_dims}D (raw)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.colorbar(im2, ax=ax2, label='Cosine Similarity', fraction=0.046, pad=0.04)
    
    # Main title
    fig.suptitle('Prototype Weight Matrix Clustering', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {args.output}")
    
    # Print per-component variance
    print(f"\nVariance explained per component:")
    for i, var in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {var:.2%}")


if __name__ == "__main__":
    main()