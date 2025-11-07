#!/usr/bin/env python3
"""
PCA-based Prototype Weight Matrix Clustering
Reduces 768D → 10D via PCA, then computes similarity clustermap.
"""

import os
import sys
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



checkpoint_path = "/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B_B3/logs/checkpoint.pth"
output_path = '/data1/vanderbc/nandas1/PostProc/prototype_clustermap_pca.png'
n_components = 10  # Reduce to 10 dimensions

print(f"Loading checkpoint: {checkpoint_path}")
x = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Normalize full-dimensional weights
X_full = F.normalize(x["prototype_bank"]["module.proto_layer.weight"].cpu().float(), dim=-1)

print(f"Original prototype weight shape: {X_full.shape}")

# Apply PCA
print(f"Applying PCA to reduce {X_full.shape[1]}D → {n_components}D...")
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_full.numpy())

variance_explained = pca.explained_variance_ratio_.sum()
print(f"Variance explained by {n_components} components: {variance_explained:.1%}")

# Normalize PCA-reduced prototypes
X_pca_norm = F.normalize(torch.from_numpy(X_pca).float(), dim=-1)

# Compute similarity matrix
similarity = (X_pca_norm @ X_pca_norm.T).numpy()

print(f"PCA prototype shape: {X_pca_norm.shape}")
print(f"Similarity matrix shape: {similarity.shape}")
print(f"Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")

# Plot clustermap
print("Creating clustermap...")
g = sns.clustermap(
    similarity,
    vmin=-1,
    vmax=1,
    cmap='coolwarm',
    figsize=(12, 12),
    xticklabels=False,
    yticklabels=False,
    cbar_kws={'label': 'Cosine Similarity (PCA space)'},
)

# Add title with variance explained
g.fig.suptitle(f'Prototype Similarity (PCA: {n_components}D, {variance_explained:.1%} variance)', 
                   fontsize=12, y=0.98)
    
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Saved: {output_path}")

# Print per-component variance
print(f"\nVariance explained per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.2%}")







X2 = F.normalize(x["prototype_bank"]["module.proto_layer.weight"][:,:10].cpu().float(), dim=-1)

g2 = sns.clustermap(X2@X2.T, vmin=-1, vmax=1, cmap="coolwarm")

plt.savefig("prototype_clustermap_10d.png", dpi=300, bbox_inches='tight')

