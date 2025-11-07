#!/usr/bin/env python3
"""
Prototype Assignment PCA Visualization
Reduces K-dimensional prototype assignments to RGB via PCA.
Shows spatial structure of prototype mixtures.

File organization:
  visualizations/
    prototype_pca/
      LUAD/
        LUAD_region_0_pca.png
        LUAD_region_0_comparison.png
      SARC/
        SARC_region_0_pca.png
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import openslide
from pathlib import Path
from scipy import ndimage
from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import models from relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

if not os.path.exists(os.path.join(project_root, 'models')):
    raise RuntimeError(f"Cannot find models directory. Project root resolved to: {project_root}")

from models.vision_transformer.modern_vit import VisionTransformer

# Publication-ready matplotlib settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def load_model_and_prototypes(checkpoint_path, device):
    """Load teacher backbone and prototype bank with architecture from checkpoint args"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # ===== Extract architecture parameters from saved args =====
    if 'args' not in checkpoint:
        raise KeyError("No 'args' found in checkpoint! Cannot determine model architecture.")
    
    args = checkpoint['args']
    
    # Extract ViT architecture parameters
    patch_size = getattr(args, 'patch_size', 16)
    embed_dim = getattr(args, 'embeddingdim', 768)
    depth = getattr(args, 'vitdepth', 12)
    num_heads = getattr(args, 'vitheads', 6)
    
    print(f"Model architecture from checkpoint:")
    print(f"  patch_size: {patch_size}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  depth: {depth}")
    print(f"  num_heads: {num_heads}")
    
    # ===== Load Teacher Backbone with correct architecture =====
    teacher_encoder = VisionTransformer(
        img_size=224,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.4,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    teacher_state = checkpoint['teacher']
    
    # Extract backbone weights
    backbone_state = {}
    for k, v in teacher_state.items():
        if k.startswith('module.backbone.'):
            new_key = k.replace('module.backbone.', '')
            backbone_state[new_key] = v
    
    teacher_encoder.load_state_dict(backbone_state, strict=False)
    teacher_encoder = teacher_encoder.to(device)
    teacher_encoder.eval()
    
    # ===== Reconstruct LinearPrototypeBank =====
    if 'prototype_bank' not in checkpoint:
        raise KeyError("No 'prototype_bank' found in checkpoint!")
    
    proto_state = checkpoint['prototype_bank']
    
    # Handle DDP wrapping
    weight_key = None
    if 'module.proto_layer.weight' in proto_state:
        weight_key = 'module.proto_layer.weight'
        bias_key = 'module.proto_layer.bias'
    elif 'proto_layer.weight' in proto_state:
        weight_key = 'proto_layer.weight'
        bias_key = 'proto_layer.bias'
    else:
        raise KeyError(f"Could not find proto_layer.weight! Keys: {list(proto_state.keys())}")
    
    weight = proto_state[weight_key]
    num_prototypes, proto_embed_dim = weight.shape
    has_bias = bias_key in proto_state
    
    # Verify dimensions match
    if proto_embed_dim != embed_dim:
        raise ValueError(f"Prototype dimension mismatch! "
                        f"ViT embed_dim={embed_dim}, but prototype weights have dim={proto_embed_dim}")
    
    # Create linear layer
    proto_layer = nn.Linear(embed_dim, num_prototypes, bias=has_bias)
    proto_layer.weight.data = weight
    if has_bias:
        proto_layer.bias.data = proto_state[bias_key]
    
    proto_layer = proto_layer.to(device)
    proto_layer.eval()
    
    print(f"Loaded LinearPrototypeBank: {num_prototypes} prototypes, dim={embed_dim}, bias={has_bias}")
    print(f"Operating on backbone patch tokens ({embed_dim}-dim)")
    
    return teacher_encoder, proto_layer


def find_tissue_regions_diverse(slide, tissue_threshold=240, min_tissue_ratio=0.5, 
                                n_regions=10, thumbnail_size=2000):
    """Find diverse tissue regions"""
    thumbnail = slide.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_np = np.array(thumbnail)
    
    gray = np.dot(thumbnail_np[...,:3], [0.299, 0.587, 0.114])
    tissue_mask = gray < tissue_threshold
    
    tissue_mask = ndimage.binary_erosion(tissue_mask, iterations=2)
    tissue_mask = ndimage.binary_dilation(tissue_mask, iterations=2)
    
    scale_x = slide.dimensions[0] / thumbnail.size[0]
    scale_y = slide.dimensions[1] / thumbnail.size[1]
    
    patch_size = 3584
    patch_size_thumb = int(patch_size / scale_x)
    
    valid_regions = []
    step = 20
    
    for y in range(0, tissue_mask.shape[0] - patch_size_thumb, step):
        for x in range(0, tissue_mask.shape[1] - patch_size_thumb, step):
            patch_mask = tissue_mask[y:y+patch_size_thumb, x:x+patch_size_thumb]
            tissue_ratio = np.mean(patch_mask)
            
            if tissue_ratio >= min_tissue_ratio:
                patch_gray = gray[y:y+patch_size_thumb, x:x+patch_size_thumb]
                tissue_pixels = patch_gray[patch_mask]
                
                if len(tissue_pixels) > 0:
                    darkness_score = np.mean(tissue_pixels)
                    if darkness_score > 230:
                        continue
                    darkness_std = np.std(tissue_pixels)
                else:
                    continue
                
                patch_edges = ndimage.sobel(patch_mask.astype(float))
                edge_score = np.std(patch_edges)
                
                valid_regions.append({
                    'location': (int(x * scale_x), int(y * scale_y)),
                    'tissue_ratio': tissue_ratio,
                    'darkness_score': darkness_score,
                    'darkness_std': darkness_std,
                    'edge_score': edge_score,
                    'thumb_coords': (x, y),
                    'center_x': x + patch_size_thumb // 2,
                    'center_y': y + patch_size_thumb // 2
                })
    
    if not valid_regions:
        raise ValueError("No valid tissue regions found!")
    
    for region in valid_regions:
        darkness_norm = 1.0 - (region['darkness_score'] / 240.0)
        region['combined_score'] = (darkness_norm * 0.5 + region['tissue_ratio'] * 0.3 + 
                                   min(region['edge_score'], 1.0) * 0.2)
    
    print(f"Found {len(valid_regions)} valid tissue regions")
    valid_regions.sort(key=lambda x: x['combined_score'], reverse=True)
    
    selected_regions = []
    min_distance_thumb = patch_size_thumb * 0.5
    
    for region in valid_regions:
        far_enough = True
        for selected in selected_regions:
            distance = np.sqrt((region['center_x'] - selected['center_x'])**2 + 
                             (region['center_y'] - selected['center_y'])**2)
            if distance < min_distance_thumb:
                far_enough = False
                break
        
        if far_enough:
            selected_regions.append(region)
            if len(selected_regions) >= n_regions:
                break
    
    if len(selected_regions) < n_regions:
        for region in valid_regions:
            if region not in selected_regions:
                selected_regions.append(region)
                if len(selected_regions) >= n_regions:
                    break
    
    print(f"Selected {len(selected_regions)} regions")
    
    return selected_regions


def extract_region_and_patches(slide, location):
    """Extract 3584x3584 region and divide into patches"""
    x, y = location
    region = slide.read_region((x, y), 0, (3584, 3584))
    region = np.array(region)[:, :, :3]
    
    # Divide into 16x16 grid of 224x224 patches
    patches = []
    for i in range(16):
        for j in range(16):
            patch = region[i*224:(i+1)*224, j*224:(j+1)*224]
            patches.append(patch)
    
    return region, patches


def normalize_image(image):
    """Normalize image for model input"""
    mean = np.array([0.6816, 0.5640, 0.7232])
    std = np.array([0.1617, 0.1714, 0.1389])
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def extract_patch_tokens(backbone, patches, device):
    """Extract patch tokens directly from backbone (768-dim)"""
    all_patch_tokens = []
    batch_size = 8
    
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i+batch_size]
        batch_tensors = []
        
        for patch in batch_patches:
            patch_normalized = normalize_image(patch)
            patch_tensor = torch.from_numpy(patch_normalized).permute(2, 0, 1).float()
            batch_tensors.append(patch_tensor)
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            output = backbone(batch)
            if isinstance(output, dict):
                patch_tokens = output['patchtokens']
            else:
                num_registers = 4
                patch_tokens = output[:, 1+num_registers:, :]
        
        all_patch_tokens.append(patch_tokens.cpu())
    
    all_patch_tokens = torch.cat(all_patch_tokens, dim=0)
    print(f"Extracted backbone patch tokens: {all_patch_tokens.shape}")
    
    return all_patch_tokens


def compute_prototype_assignments(patch_tokens, proto_layer, teacher_temp=0.07):
    """Compute prototype assignments using the nn.Linear layer"""
    B, N, D = patch_tokens.shape
    patch_tokens_flat = patch_tokens.reshape(-1, D)
    
    # Normalize patch tokens
    patch_tokens_norm = F.normalize(patch_tokens_flat, p=2, dim=-1)
    
    # Get logits from linear layer
    with torch.no_grad():
        logits = proto_layer(patch_tokens_norm)
    
    # Softmax assignments
    assignments = F.softmax(logits / teacher_temp, dim=-1)
    
    # Reshape back
    assignments = assignments.reshape(B, N, -1)
    
    print(f"Computed assignments: {assignments.shape}")
    
    return assignments


def create_prototype_pca_map(assignments, method='pca', n_components=3):
    """
    Reduce K-dimensional prototype assignments to 3D via PCA.
    
    Args:
        assignments: [B, N, K] prototype assignment probabilities
        method: 'pca' or 'tsne'
        n_components: 2 or 3 (for grayscale/RGB)
    
    Returns:
        pca_map: [H, W, n_components] spatial map
        reducer: Fitted dimensionality reduction object
        variance: Explained variance (PCA only)
    """
    B, N, K = assignments.shape
    
    # Reshape assignments to spatial grid (16×16 patches × 14×14 tokens each)
    H, W = 224, 224  # 16 patches × 14 tokens per patch
    
    # Flatten to [H*W, K]
    assignments_flat = torch.zeros(H * W, K)
    
    for patch_idx in range(B):
        patch_i = patch_idx // 16
        patch_j = patch_idx % 16
        
        patch_assignments = assignments[patch_idx].cpu()  # [196, K]
        patch_reshaped = patch_assignments.reshape(14, 14, K)
        
        # Place in full spatial grid
        for ti in range(14):
            for tj in range(14):
                spatial_i = patch_i * 14 + ti
                spatial_j = patch_j * 14 + tj
                idx = spatial_i * W + spatial_j
                assignments_flat[idx] = patch_reshaped[ti, tj]
    
    assignments_np = assignments_flat.numpy()
    
    # Apply dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        reduced = reducer.fit_transform(assignments_np)
        variance = reducer.explained_variance_ratio_.sum()
        print(f"  PCA: {n_components} components explain {variance:.1%} of assignment variance")
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=42, n_iter=1000)
        reduced = reducer.fit_transform(assignments_np)
        variance = None
        print(f"  t-SNE: Computed {n_components}D embedding")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Reshape to spatial grid
    pca_map = reduced.reshape(H, W, n_components)
    
    # Normalize to [0, 1] for visualization
    for c in range(n_components):
        channel = pca_map[:, :, c]
        channel_min = channel.min()
        channel_max = channel.max()
        pca_map[:, :, c] = (channel - channel_min) / (channel_max - channel_min + 1e-8)
    
    return pca_map, reducer, variance


def plot_prototype_pca_detailed(original_image, assignments, output_path):
    """
    Create detailed PCA visualization.
    
    Layout: Original | PCA-RGB | PC1 | PC2 | PC3
    """
    fig = plt.figure(figsize=(20, 4.5), facecolor='white')
    
    gs = gridspec.GridSpec(1, 5, figure=fig,
                          hspace=0.02, wspace=0.15,
                          left=0.04, right=0.96, top=0.85, bottom=0.12)
    
    # Get PCA map
    pca_map, reducer, variance = create_prototype_pca_map(
        assignments, method='pca', n_components=3
    )
    
    # Upsample to match original resolution (224 → 3584)
    scale = 3584 / 224
    pca_map_upsampled = zoom(pca_map, (scale, scale, 1), order=1)
    
    # 1. Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(original_image, extent=[0, 3584, 3584, 0])
    ax_orig.set_title('Original\nImage', fontsize=12, fontweight='bold', pad=10)
    ax_orig.set_xlabel('Pixels', fontsize=9)
    ax_orig.set_ylabel('Pixels', fontsize=9)
    ax_orig.set_xticks([0, 1792, 3584])
    ax_orig.set_yticks([0, 1792, 3584])
    ax_orig.tick_params(labelsize=8)
    
    for spine in ax_orig.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    # 2. PCA-RGB composite
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_rgb.imshow(pca_map_upsampled, extent=[0, 3584, 3584, 0])
    ax_rgb.set_title(f'PCA-RGB\n({variance:.1%} variance)', 
                    fontsize=12, fontweight='bold', pad=10)
    ax_rgb.set_xlabel('Pixels', fontsize=9)
    ax_rgb.set_xticks([0, 1792, 3584])
    ax_rgb.set_yticks([])
    ax_rgb.tick_params(labelsize=8)
    
    for spine in ax_rgb.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    # 3-5. Individual components
    component_names = ['PC1', 'PC2', 'PC3']
    component_vars = reducer.explained_variance_ratio_
    
    for c in range(3):
        ax_comp = fig.add_subplot(gs[0, c+2])
        im = ax_comp.imshow(pca_map_upsampled[:, :, c], 
                           cmap='viridis', extent=[0, 3584, 3584, 0])
        ax_comp.set_title(f'{component_names[c]}\n({component_vars[c]:.1%} var.)', 
                        fontsize=12, fontweight='bold', pad=10)
        ax_comp.set_xlabel('Pixels', fontsize=9)
        ax_comp.set_xticks([0, 1792, 3584])
        ax_comp.set_yticks([])
        ax_comp.tick_params(labelsize=8)
        
        for spine in ax_comp.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_comp, fraction=0.046, pad=0.04)
        cbar.set_label('Intensity', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.suptitle('Prototype Assignment PCA Visualization', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_method_comparison(original_image, assignments, output_path):
    """
    Compare PCA vs t-SNE for prototype assignment visualization.
    
    Layout: Original | PCA-RGB | t-SNE-RGB
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    
    # 1. Original
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image', fontsize=13, fontweight='bold', pad=10)
    axes[0].axis('off')
    
    # 2. PCA
    try:
        pca_map, _, variance_pca = create_prototype_pca_map(
            assignments, method='pca', n_components=3
        )
        scale = 3584 / 224
        pca_map_up = zoom(pca_map, (scale, scale, 1), order=1)
        
        axes[1].imshow(pca_map_up)
        axes[1].set_title(f'PCA Visualization\n({variance_pca:.1%} variance)', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[1].axis('off')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'PCA Failed:\n{str(e)}',
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=10)
        axes[1].axis('off')
    
    # 3. t-SNE
    try:
        tsne_map, _, _ = create_prototype_pca_map(
            assignments, method='tsne', n_components=3
        )
        scale = 3584 / 224
        tsne_map_up = zoom(tsne_map, (scale, scale, 1), order=1)
        
        axes[2].imshow(tsne_map_up)
        axes[2].set_title('t-SNE Visualization', 
                         fontsize=13, fontweight='bold', pad=10)
        axes[2].axis('off')
    except Exception as e:
        axes[2].text(0.5, 0.5, f't-SNE Failed:\n{str(e)}',
                    ha='center', va='center', transform=axes[2].transAxes,
                    fontsize=10)
        axes[2].axis('off')
    
    plt.suptitle('Prototype Assignment: Method Comparison',
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def process_region(slide, region_info, region_index, backbone, proto_layer, 
                   device, cancer_dir, cancer_type):
    """Process a single region and create visualizations"""
    print(f"\n{'='*60}")
    print(f"PROCESSING {cancer_type} - REGION {region_index}")
    print(f"{'='*60}")
    
    # Extract region and patches
    print("Extracting patches...")
    original_image, patches = extract_region_and_patches(slide, region_info['location'])
    
    # Extract backbone patch tokens (768-dim)
    print("Extracting patch tokens from backbone...")
    patch_tokens = extract_patch_tokens(backbone, patches, device)
    patch_tokens = patch_tokens.to(device)
    
    # Compute prototype assignments
    print("Computing prototype assignments...")
    assignments = compute_prototype_assignments(patch_tokens, proto_layer)
    
    # Create detailed PCA visualization
    print("Creating detailed PCA visualization...")
    output_path_detailed = os.path.join(cancer_dir, 
                                       f'{cancer_type}_region_{region_index}_pca.png')
    plot_prototype_pca_detailed(original_image, assignments, output_path_detailed)
    
    # Create method comparison
    print("Creating method comparison...")
    output_path_comp = os.path.join(cancer_dir, 
                                   f'{cancer_type}_region_{region_index}_comparison.png')
    plot_method_comparison(original_image, assignments, output_path_comp)
    
    # Clean up
    del patch_tokens, assignments
    torch.cuda.empty_cache()


def process_wsi(wsi_path, cancer_type, backbone, proto_layer, device, 
                main_output_dir, n_regions=3):
    """Process a WSI with multiple regions"""
    print(f"\n{'#'*60}")
    print(f"# {cancer_type}")
    print(f"{'#'*60}")
    
    # Create output directory for this cancer type
    cancer_dir = os.path.join(main_output_dir, cancer_type)
    os.makedirs(cancer_dir, exist_ok=True)
    
    slide = None
    try:
        # Open slide
        slide = openslide.OpenSlide(wsi_path)
        
        # Find diverse regions
        print(f"\nFinding {n_regions} diverse tissue regions...")
        regions = find_tissue_regions_diverse(slide, min_tissue_ratio=0.5, 
                                             n_regions=n_regions)
        
        # Process each region
        for i, region in enumerate(regions):
            try:
                process_region(slide, region, i, backbone, proto_layer, 
                             device, cancer_dir, cancer_type)
            except Exception as e:
                print(f"Error processing region {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        return True
        
    except Exception as e:
        print(f"Error processing {cancer_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if slide:
            slide.close()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description='Generate prototype assignment PCA visualizations'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: ../logs/checkpoint.pth)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../visualizations/prototype_pca)')
    parser.add_argument('--n_regions', type=int, default=3,
                       help='Number of regions per WSI (default: 3)')
    args = parser.parse_args()
    
    # Default to relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, '..', 'logs', 'checkpoint.pth')
    
    if args.output_dir is None:
        # Create main folder for prototype PCA outputs
        args.output_dir = os.path.join(script_dir, '..', 'visualizations', 'prototype_pca')
    
    # WSI paths
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
    }
    
    # Create main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model and prototypes ONCE
    print("="*60)
    print("LOADING BACKBONE AND PROTOTYPES")
    print("="*60)
    backbone, proto_layer = load_model_and_prototypes(args.checkpoint, device)
    
    # Process each WSI
    for cancer_type, wsi_path in wsi_paths.items():
        if not os.path.exists(wsi_path):
            print(f"\n⚠️  Skipping {cancer_type}: File not found")
            continue
        
        process_wsi(wsi_path, cancer_type, backbone, proto_layer, device, 
                   args.output_dir, n_regions=args.n_regions)
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    LUAD/")
    print(f"      LUAD_region_0_pca.png           (Original | PCA-RGB | PC1 | PC2 | PC3)")
    print(f"      LUAD_region_0_comparison.png    (Original | PCA | t-SNE)")
    print(f"      LUAD_region_1_pca.png")
    print(f"      LUAD_region_1_comparison.png")
    print(f"    SARC/")
    print(f"      SARC_region_0_pca.png")
    print(f"      SARC_region_0_comparison.png")
    print(f"      ...")
    print(f"\n✓ PCA on K-dimensional prototype assignments")
    print(f"✓ RGB = first 3 principal components of assignment distributions")


if __name__ == "__main__":
    main()