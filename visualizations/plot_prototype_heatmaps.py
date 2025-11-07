#!/usr/bin/env python3
"""
Top-K Prototype Heatmap Visualization
Shows which prototypes are most active in different spatial regions of tissue.
Works with LinearPrototypeBank (nn.Linear) architecture.
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
import warnings
warnings.filterwarnings('ignore')

# Import models from relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Verify models directory exists
if not os.path.exists(os.path.join(project_root, 'models')):
    raise RuntimeError(f"Cannot find models directory. Project root resolved to: {project_root}")

from models.vision_transformer.modern_vit import VisionTransformer
from models.vision_transformer.auxiliary_models import DINOHead, MaskModel_SpectralNorm

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
    """Load teacher backbone and prototype bank (as nn.Linear layer)"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # ===== Load Teacher Backbone =====
    teacher_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=6,
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
    num_prototypes, embed_dim = weight.shape
    has_bias = bias_key in proto_state
    
    # Create linear layer
    proto_layer = nn.Linear(embed_dim, num_prototypes, bias=has_bias)
    proto_layer.weight.data = weight
    if has_bias:
        proto_layer.bias.data = proto_state[bias_key]
    
    proto_layer = proto_layer.to(device)
    proto_layer.eval()
    
    print(f"Loaded LinearPrototypeBank: {num_prototypes} prototypes, dim={embed_dim}, bias={has_bias}")
    print(f"Operating on backbone patch tokens (768-dim)")
    
    return teacher_encoder, proto_layer


def find_tissue_regions_diverse(slide, tissue_threshold=240, min_tissue_ratio=0.5, 
                                n_regions=10, thumbnail_size=2000, output_dir=None):
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
    """
    Compute prototype assignments using the nn.Linear layer.
    Matches training exactly (includes bias if present).
    """
    B, N, D = patch_tokens.shape
    patch_tokens_flat = patch_tokens.reshape(-1, D)
    
    # Normalize patch tokens
    patch_tokens_norm = F.normalize(patch_tokens_flat, p=2, dim=-1)
    
    # Get logits from linear layer (xW^T + b)
    with torch.no_grad():
        logits = proto_layer(patch_tokens_norm)
    
    # Softmax assignments
    assignments = F.softmax(logits / teacher_temp, dim=-1)
    
    # Reshape back
    assignments = assignments.reshape(B, N, -1)
    
    print(f"Computing assignments via nn.Linear (teacher_temp={teacher_temp})")
    
    return assignments

def find_top_k_prototypes(assignments, k=5):
    """
    Simple and interpretable: Find prototypes that "own" the most tokens.
    
    For each token, assign it to the prototype with highest probability (argmax).
    Then count which prototypes are used most frequently.
    """
    B, N, K = assignments.shape
    
    # Argmax: which prototype owns each token?
    prototype_assignments = assignments.argmax(dim=-1)  # [B, N]
    
    # Count frequency of each prototype
    prototype_counts = torch.bincount(
        prototype_assignments.flatten(), 
        minlength=K
    ).float()  # [K]
    
    # Get top-K most frequent
    top_k_values, top_k_indices = torch.topk(prototype_counts, k)
    
    print(f"\nTop-{k} prototypes by token ownership:")
    for i, (idx, count) in enumerate(zip(top_k_indices, top_k_values)):
        percentage = (count / (B * N)) * 100
        max_prob = assignments[:, :, idx].max().item()
        mean_prob = assignments[:, :, idx].mean().item()
        print(f"  {i+1}. Proto {idx.item():5d}: "
              f"{count.item():.0f} tokens ({percentage:.1f}%), "
              f"max_prob={max_prob:.3f}, mean_prob={mean_prob:.4f}")
    
    return top_k_indices, top_k_values


def create_prototype_heatmaps(assignments, top_k_indices):
    """Create spatial heatmaps for top-K prototypes"""
    B, N, K = assignments.shape
    n_prototypes = len(top_k_indices)
    
    heatmaps = []
    
    for proto_idx in top_k_indices:
        # Extract assignments for this prototype
        proto_assignments = assignments[:, :, proto_idx]  # [B, N]
        
        # Reshape to spatial grid (16x16 patches, 14x14 tokens per patch)
        heatmap_full = np.zeros((224, 224))
        
        for patch_idx in range(B):
            patch_i = patch_idx // 16
            patch_j = patch_idx % 16
            
            # Get this patch's assignments
            patch_assignments = proto_assignments[patch_idx].cpu().numpy()  # [196]
            
            # Reshape to 14x14
            patch_heatmap = patch_assignments.reshape(14, 14)
            
            # Place in full heatmap
            heatmap_full[patch_i*14:(patch_i+1)*14, patch_j*14:(patch_j+1)*14] = patch_heatmap
        
        heatmaps.append(heatmap_full)
    
    return heatmaps


def plot_prototype_heatmaps(original_image, heatmaps, top_k_indices, top_k_values, output_path):
    """
    Plot original image + top-K prototype heatmaps with single shared colorbar.
    Publication-ready layout.
    """
    k = len(heatmaps)
    
    # Create figure
    fig_width = 3.2 * (k + 1)
    fig = plt.figure(figsize=(fig_width, 4.5), facecolor='white')
    
    # Create gridspec
    gs = gridspec.GridSpec(2, k+1, figure=fig, 
                          height_ratios=[0.05, 1],
                          hspace=0.02, wspace=0.25,
                          left=0.04, right=0.96, top=0.90, bottom=0.12)
    
    # Find global vmax for consistent colorbar
    vmax = max(hm.max() for hm in heatmaps)
    
    # Plot original image (row 1, col 0)
    ax_orig = fig.add_subplot(gs[1, 0])
    ax_orig.imshow(original_image, extent=[0, 3584, 3584, 0], interpolation='nearest')
    ax_orig.set_title('Original Image\n(3584×3584)', fontsize=11, fontweight='bold', pad=8)
    ax_orig.set_xlabel('Pixels', fontsize=9)
    ax_orig.set_ylabel('Pixels', fontsize=9)
    ax_orig.set_xticks([0, 1792, 3584])
    ax_orig.set_yticks([0, 1792, 3584])
    ax_orig.tick_params(axis='both', labelsize=8)
    
    for spine in ax_orig.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.5)
    
    # Plot heatmaps (row 1, cols 1 to k)
    for i, (heatmap, proto_idx, proto_value) in enumerate(zip(heatmaps, top_k_indices, top_k_values)):
        ax = fig.add_subplot(gs[1, i+1])
        
        # Plot heatmap with viridis colormap
        im = ax.imshow(heatmap, cmap='viridis', extent=[0, 3584, 3584, 0], 
                      interpolation='bilinear', vmin=0, vmax=vmax)
        
        # Title with prototype info
        ax.set_title(f'Prototype #{proto_idx.item()}\n(Score: {proto_value.item():.0f})', 
                    fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Pixels', fontsize=9)
        ax.set_xticks([0, 1792, 3584])
        ax.set_yticks([0, 1792, 3584])
        ax.tick_params(axis='both', labelsize=8)
        
        # Only show ylabel on first heatmap
        if i == 0:
            ax.set_ylabel('Pixels', fontsize=9)
        else:
            ax.set_yticklabels([])
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(1.5)
    
    # Add single colorbar spanning all heatmaps (row 0, cols 1 to k)
    cbar_ax = fig.add_subplot(gs[0, 1:])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Assignment Probability', fontsize=10, labelpad=5)
    cbar.ax.tick_params(labelsize=8)
    cbar_ax.xaxis.set_ticks_position('top')
    cbar_ax.xaxis.set_label_position('top')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved: {output_path}")


def process_region(slide, region_info, region_index, backbone, proto_layer, 
                   device, output_dir, cancer_type, k=5):
    """Process a single region and create visualization"""
    print(f"\n{'='*60}")
    print(f"PROCESSING REGION {region_index}")
    print(f"{'='*60}")
    
    # Extract region and patches
    original_image, patches = extract_region_and_patches(slide, region_info['location'])
    
    # Extract backbone patch tokens (768-dim)
    print("Extracting patch tokens from backbone...")
    patch_tokens = extract_patch_tokens(backbone, patches, device)
    patch_tokens = patch_tokens.to(device)
    
    # Compute prototype assignments
    print("Computing prototype assignments...")
    assignments = compute_prototype_assignments(patch_tokens, proto_layer)
    
    # Find top-K prototypes
    top_k_indices, top_k_values = find_top_k_prototypes(assignments, k=k)
    
    # Create heatmaps
    print("Creating spatial heatmaps...")
    heatmaps = create_prototype_heatmaps(assignments, top_k_indices)
    
    # Plot
    output_path = os.path.join(output_dir, f'{cancer_type}_region_{region_index}.png')
    plot_prototype_heatmaps(original_image, heatmaps, top_k_indices, top_k_values, output_path)
    
    # Clean up
    del patch_tokens, assignments, heatmaps
    torch.cuda.empty_cache()


def process_wsi(wsi_path, cancer_type, backbone, proto_layer, device, 
                output_dir, n_regions=3, k=5):
    """Process a WSI with multiple regions"""
    print(f"\n{'#'*60}")
    print(f"# {cancer_type}")
    print(f"{'#'*60}")
    
    # Create output directory for this cancer type
    cancer_dir = os.path.join(output_dir, cancer_type)
    os.makedirs(cancer_dir, exist_ok=True)
    
    slide = None
    try:
        # Open slide
        slide = openslide.OpenSlide(wsi_path)
        
        # Find diverse regions
        print(f"\nFinding {n_regions} diverse tissue regions...")
        regions = find_tissue_regions_diverse(slide, min_tissue_ratio=0.5, 
                                             n_regions=n_regions, output_dir=cancer_dir)
        
        # Process each region
        for i, region in enumerate(regions):
            try:
                process_region(slide, region, i, backbone, proto_layer, 
                             device, cancer_dir, cancer_type, k=k)
            except Exception as e:
                print(f"Error processing region {i}: {str(e)}")
                continue
        
        return True
        
    except Exception as e:
        print(f"Error processing {cancer_type}: {str(e)}")
        return False
        
    finally:
        if slide:
            slide.close()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Generate prototype heatmap visualizations')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: ../logs/checkpoint.pth)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: current directory)')
    parser.add_argument('--n_regions', type=int, default=3,
                       help='Number of regions per WSI (default: 3)')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of top prototypes to visualize (default: 5)')
    args = parser.parse_args()
    
    # Default to relative paths from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, '..', 'logs', 'checkpoint.pth')
    
    if args.output_dir is None:
        args.output_dir = script_dir
    
    # WSI paths
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
    }
    
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
                   args.output_dir, n_regions=args.n_regions, k=args.k)
    
    print("\n" + "="*60)
    print("ALL PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    LUAD/")
    print(f"      LUAD_region_0.png")
    print(f"      LUAD_region_1.png")
    print(f"      LUAD_region_2.png")
    print(f"    SARC/")
    print(f"      SARC_region_0.png")
    print(f"      ...")
    print(f"\nEach PNG shows: Original | Proto #X | Proto #Y | Proto #Z | ... (top-{args.k})")
    print(f"✓ Operating on backbone patch tokens (768-dim)")
    print(f"✓ Works with LinearPrototypeBank (nn.Linear)")
    print(f"✓ Single shared colorbar at top")


if __name__ == "__main__":
    main()