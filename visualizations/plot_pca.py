#!/usr/bin/env python3
"""
Combined PCA + Hue Spectra Visualization
Uses robust checkpoint loading and creates Original | PCA-RGB | Polar Hue plot
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter1d
import openslide
from scipy import ndimage
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from models.vision_transformer.modern_vit import VisionTransformer

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 15  # 1.5x standard (10 * 1.5)
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.5


def load_model_from_checkpoint(checkpoint_path, device):
    """Load teacher backbone using architecture from checkpoint args"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'args' not in checkpoint:
        raise KeyError("No 'args' found in checkpoint!")
    
    args = checkpoint['args']
    
    # Extract architecture parameters
    patch_size = getattr(args, 'patch_size', 16)
    embed_dim = getattr(args, 'embeddingdim', 768)
    depth = getattr(args, 'vitdepth', 12)
    num_heads = getattr(args, 'vitheads', 12)
    
    print(f"Model architecture from checkpoint:")
    print(f"  patch_size: {patch_size}")
    print(f"  embed_dim: {embed_dim}")
    print(f"  depth: {depth}")
    print(f"  num_heads: {num_heads}")
    
    # Create teacher backbone
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
    
    print(f"✓ Loaded teacher backbone")
    
    return teacher_encoder


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


def extract_3584_region(slide, location):
    """Extract a 3584x3584 region"""
    x, y = location
    region = slide.read_region((x, y), 0, (3584, 3584))
    region = np.array(region)[:, :, :3]
    return region


def extract_patches_from_region(region_3584):
    """Divide into 256 patches"""
    patches = []
    for i in range(16):
        for j in range(16):
            patch = region_3584[i*224:(i+1)*224, j*224:(j+1)*224]
            patches.append(patch)
    return patches


def normalize_image(image):
    """Normalize image"""
    mean = np.array([0.6816, 0.5640, 0.7232])
    std = np.array([0.1617, 0.1714, 0.1389])
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    return image


def extract_features_from_patches(model, patches, device):
    """Extract features"""
    all_features = []
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
            output = model(batch)
            if isinstance(output, dict):
                patch_tokens = output['patchtokens']
            else:
                num_registers = 4
                patch_tokens = output[:, 1+num_registers:, :]
        
        for j in range(patch_tokens.shape[0]):
            all_features.append(patch_tokens[j].cpu().numpy())
    
    return np.vstack(all_features)


def compute_foreground_mask(features, patches):
    """Compute foreground/background mask using 3D PCA clustering"""
    from sklearn.cluster import KMeans
    
    print("  Computing background/foreground separation (3D PCA clustering)")
    
    # Stage 1: PCA to 3D
    pca = PCA(n_components=3)
    pca_features_3d = pca.fit_transform(features)
    
    print(f"    PCA: {features.shape[1]} -> 3 dimensions")
    print(f"    Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"    Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Stage 2: Clustering in 3D space
    n_clusters = 4
    print(f"\n  Clustering in 3D PCA space with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features_3d)
    cluster_centers = kmeans.cluster_centers_
    
    # Stage 3: Compute actual brightness for validation
    actual_brightness = []
    for patch in patches:
        for i in range(14):
            for j in range(14):
                region = patch[i*16:(i+1)*16, j*16:(j+1)*16]
                gray_value = np.mean(np.dot(region, [0.299, 0.587, 0.114]))
                actual_brightness.append(gray_value)
    actual_brightness = np.array(actual_brightness)
    
    # Stage 4: Identify which cluster is background
    print("\n  Analyzing clusters to identify background...")
    
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_pixels = np.where(cluster_mask)[0]
        
        if len(cluster_pixels) > 0:
            cluster_brightness = actual_brightness[cluster_mask]
            mean_brightness = np.mean(cluster_brightness)
            std_brightness = np.std(cluster_brightness)
            
            cluster_pca_values = pca_features_3d[cluster_mask]
            mean_pc1 = np.mean(cluster_pca_values[:, 0])
            mean_pc2 = np.mean(cluster_pca_values[:, 1])
            mean_pc3 = np.mean(cluster_pca_values[:, 2])
            
            cluster_mask_2d = cluster_mask.reshape(256, 196).any(axis=1).reshape(16, 16)
            labeled, n_components = ndimage.label(cluster_mask_2d)
            
            periphery_mask = np.zeros((16, 16), dtype=bool)
            periphery_mask[0, :] = True
            periphery_mask[-1, :] = True
            periphery_mask[:, 0] = True
            periphery_mask[:, -1] = True
            
            periphery_ratio = np.sum(cluster_mask_2d & periphery_mask) / max(np.sum(cluster_mask_2d), 1)
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_pixels),
                'mean_brightness': mean_brightness,
                'std_brightness': std_brightness,
                'mean_pc1': mean_pc1,
                'mean_pc2': mean_pc2,
                'mean_pc3': mean_pc3,
                'n_components': n_components,
                'periphery_ratio': periphery_ratio,
                'cluster_center': cluster_centers[cluster_id]
            })
            
            print(f"    Cluster {cluster_id}:")
            print(f"      Size: {len(cluster_pixels)} pixels ({len(cluster_pixels)/len(cluster_labels)*100:.1f}%)")
            print(f"      Mean brightness: {mean_brightness:.1f} ± {std_brightness:.1f}")
            print(f"      PCA center: [{mean_pc1:.2f}, {mean_pc2:.2f}, {mean_pc3:.2f}]")
            print(f"      Spatial: {n_components} components, {periphery_ratio:.1%} at periphery")
    
    # Stage 5: Score each cluster for likelihood of being background
    print("\n  Scoring clusters for background likelihood...")
    
    background_scores = []
    for stats in cluster_stats:
        score = 0
        reasons = []
        
        if stats['mean_brightness'] > 245:
            score += 3
            reasons.append("very bright (>245)")
        elif stats['mean_brightness'] > 235:
            score += 2
            reasons.append("bright (>235)")
        elif stats['mean_brightness'] > 225:
            score += 1
            reasons.append("moderately bright (>225)")
        
        if stats['std_brightness'] < 10:
            score += 1
            reasons.append("uniform brightness")
        
        if stats['periphery_ratio'] > 0.5:
            score += 1
            reasons.append("peripheral location")
        
        if stats['n_components'] <= 2:
            score += 1
            reasons.append("spatially coherent")
        
        all_pc1_values = [s['mean_pc1'] for s in cluster_stats]
        if stats['mean_pc1'] == max(all_pc1_values) or stats['mean_pc1'] == min(all_pc1_values):
            score += 1
            reasons.append("extreme PC1 value")
        
        background_scores.append({
            'cluster_id': stats['cluster_id'],
            'score': score,
            'reasons': reasons,
            'stats': stats
        })
        
        print(f"    Cluster {stats['cluster_id']}: score={score}, reasons={reasons}")
    
    # Stage 6: Select background cluster(s)
    background_threshold_score = 4
    
    background_clusters = [bs for bs in background_scores if bs['score'] >= background_threshold_score]
    
    if not background_clusters:
        print("\n  ⚠️ No cluster meets background criteria (score >= 4)")
        print("     Falling back to most likely candidate...")
        best_candidate = max(background_scores, key=lambda x: x['score'])
        if best_candidate['stats']['mean_brightness'] > 220:
            background_clusters = [best_candidate]
            print(f"     Selected cluster {best_candidate['cluster_id']} (score={best_candidate['score']}, brightness={best_candidate['stats']['mean_brightness']:.1f})")
        else:
            print("     No suitable background cluster found - keeping all pixels as foreground")
            return np.ones(len(features), dtype=bool)
    
    # Stage 7: Create background mask
    background_mask = np.zeros(len(features), dtype=bool)
    for bg_cluster in background_clusters:
        cluster_id = bg_cluster['cluster_id']
        background_mask |= (cluster_labels == cluster_id)
        print(f"\n  ✓ Cluster {cluster_id} identified as background")
        print(f"     Reasons: {', '.join(bg_cluster['reasons'])}")
    
    # Stage 8: Additional validation
    removed_dark_pixels = np.sum(background_mask & (actual_brightness < 200))
    if removed_dark_pixels > 100:
        print(f"\n  ⚠️ WARNING: Background mask would remove {removed_dark_pixels} dark pixels!")
        print("     Applying morphological operations to preserve tissue...")
        
        background_mask_2d = background_mask.reshape(256, 196)
        
        for patch_idx in range(256):
            patch_bg_mask = background_mask_2d[patch_idx]
            patch_bg_mask_2d = patch_bg_mask.reshape(14, 14)
            
            labeled, n_features = ndimage.label(patch_bg_mask_2d)
            if n_features > 0:
                sizes = ndimage.sum(patch_bg_mask_2d, labeled, range(1, n_features + 1))
                if len(sizes) > 0:
                    max_size = max(sizes)
                    if max_size < 49:
                        patch_bg_mask_2d[:] = False
                    else:
                        max_label = np.argmax(sizes) + 1
                        patch_bg_mask_2d[labeled != max_label] = False
            
            background_mask_2d[patch_idx] = patch_bg_mask_2d.flatten()
        
        background_mask = background_mask_2d.flatten()
    
    # Final statistics
    foreground_mask = ~background_mask
    n_background = np.sum(background_mask)
    n_foreground = np.sum(foreground_mask)
    
    actual_whitespace = actual_brightness > 240
    actual_tissue = actual_brightness < 200
    whitespace_recall = np.sum(background_mask & actual_whitespace) / max(np.sum(actual_whitespace), 1)
    tissue_preserved = 1.0 - (np.sum(background_mask & actual_tissue) / max(np.sum(actual_tissue), 1))
    
    print(f"\n  Final separation statistics:")
    print(f"    Background pixels: {n_background} ({n_background/len(background_mask)*100:.1f}%)")
    print(f"    Foreground pixels: {n_foreground} ({n_foreground/len(foreground_mask)*100:.1f}%)")
    print(f"    Whitespace recall: {whitespace_recall:.1%}")
    print(f"    Tissue preserved: {tissue_preserved:.1%}")
    
    return foreground_mask


def apply_pca_and_extract_hues(features, foreground_mask):
    """Apply PCA using foreground mask and get hues"""
    n_foreground = np.sum(foreground_mask)
    
    if n_foreground < 10:
        return None, None
    
    pca = PCA(n_components=3)
    foreground_features = features[foreground_mask]
    pca_features = pca.fit_transform(foreground_features)
    
    pca_features_normalized = pca_features.copy()
    for i in range(3):
        if pca_features_normalized[:, i].max() > pca_features_normalized[:, i].min():
            pca_features_normalized[:, i] = (
                (pca_features_normalized[:, i] - pca_features_normalized[:, i].min()) /
                (pca_features_normalized[:, i].max() - pca_features_normalized[:, i].min())
            )
    
    pca_features_rgb = np.zeros((len(foreground_mask), 3))
    pca_features_rgb[~foreground_mask] = 0
    pca_features_rgb[foreground_mask] = pca_features_normalized
    
    hsv_pixels = np.zeros_like(pca_features_normalized)
    for i in range(len(pca_features_normalized)):
        rgb_single = pca_features_normalized[i].reshape(1, 1, 3)
        hsv_single = mcolors.rgb_to_hsv(rgb_single)
        hsv_pixels[i] = hsv_single.reshape(3)
    
    hue_values = hsv_pixels[:, 0] * 360
    
    return pca_features_rgb, hue_values


def create_rgb_grid(pca_features_rgb):
    """Create RGB grid"""
    rgb_grid = np.zeros((224, 224, 3))
    patch_idx = 0
    
    for i in range(16):
        for j in range(16):
            start_idx = patch_idx * 196
            end_idx = start_idx + 196
            patch_rgb = pca_features_rgb[start_idx:end_idx].reshape(14, 14, 3)
            rgb_grid[i*14:(i+1)*14, j*14:(j+1)*14] = patch_rgb
            patch_idx += 1
    
    return rgb_grid


def plot_polar(ax, hue_values):
    """Polar plot for hue distribution with color ring"""
    if hue_values is None or len(hue_values) == 0:
        return
    
    n_bins = 72
    theta_edges = np.linspace(0, 2*np.pi, n_bins + 1)
    theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
    
    hue_radians = hue_values * np.pi / 180
    counts, _ = np.histogram(hue_radians, bins=theta_edges)
    
    # Individual min-max normalization
    min_count = counts.min()
    max_count = counts.max()
    if max_count > min_count:
        counts_norm = (counts - min_count) / (max_count - min_count)
    else:
        counts_norm = np.ones_like(counts) * 0.5
    
    theta_plot = np.append(theta_centers, theta_centers[0])
    counts_plot = np.append(counts_norm, counts_norm[0])
    
    # Scale down to fit within limits
    counts_plot_scaled = counts_plot * 0.75
    ax.plot(theta_plot, counts_plot_scaled, color='#1E88E5', linewidth=2.5, alpha=0.9)
    ax.fill(theta_plot, counts_plot_scaled, color='#1E88E5', alpha=0.15)
    
    # Formatting
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    
    # Remove angular text labels but keep tick marks
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    ax.set_xticks(angles)
    ax.set_xticklabels([])
    
    # Radial axis
    ax.set_ylim(0, 0.9)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'], fontsize=16.5, alpha=0.7)  # 1.5x of 11
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    ax.xaxis.grid(True, linestyle='-', linewidth=0.5, alpha=0.3, color='gray')
    
    # Add color ring
    n_segments = 360
    theta_ring = np.linspace(0, 2*np.pi, n_segments, endpoint=False)
    
    ring_bottom = 0.78
    ring_height = 0.06
    
    for i, angle in enumerate(theta_ring):
        hue_degrees = (90 - np.degrees(angle)) % 360
        hsv_color = np.array([[[hue_degrees / 360.0, 1.0, 1.0]]])
        rgb_color = mcolors.hsv_to_rgb(hsv_color)[0, 0]
        
        ax.bar(angle, ring_height, width=2*np.pi/n_segments, 
               bottom=ring_bottom, color=rgb_color, 
               edgecolor='none', linewidth=0)
    
    # Add border circles for the ring
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta_circle, np.ones_like(theta_circle) * ring_bottom, 
            'k-', linewidth=0.5, alpha=0.5)
    ax.plot(theta_circle, np.ones_like(theta_circle) * (ring_bottom + ring_height), 
            'k-', linewidth=0.5, alpha=0.5)


def create_final_figure(original_image, rgb_grid, hue_values, output_path):
    """Create final figure: Original | PCA-RGB | Polar plot"""
    
    # Calculate dimensions
    fig_width = 16
    fig_height = 6
    
    # Create figure with WHITE background
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Create 1x3 grid
    gs = fig.add_gridspec(1, 3, 
                          hspace=0.04,
                          wspace=0.08,
                          left=0.04, right=0.96, 
                          top=0.92, bottom=0.08)
    
    labels = ['(a)', '(b)', '(c)']
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image, extent=[0, 3584, 3584, 0], interpolation='nearest')
    ax1.set_xlabel('Pixels', fontsize=21)  # 1.5x of 14
    ax1.set_ylabel('Pixels', fontsize=21)
    ax1.set_xticks([0, 1792, 3584])
    ax1.set_yticks([0, 1792, 3584])
    ax1.tick_params(axis='both', labelsize=19.5)  # 1.5x of 13
    

    
    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(2)
    
    # 2. PCA-RGB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(rgb_grid, extent=[0, 3584, 3584, 0], 
             interpolation='nearest', aspect='equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(2)
    
    # 3. Polar plot
    ax3 = fig.add_subplot(gs[0, 2], projection='polar')
    plot_polar(ax3, hue_values)
    
    
    # Save with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def process_region(slide, region_info, region_index, model, device, output_dir, cancer_type):
    """Process single region"""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING REGION {region_index}")
    print(f"{'='*60}")
    
    original_image = extract_3584_region(slide, region_info['location'])
    patches = extract_patches_from_region(original_image)
    
    print("Extracting features...")
    features = extract_features_from_patches(model, patches, device)
    
    print("Computing foreground mask...")
    foreground_mask = compute_foreground_mask(features, patches)
    
    print("Applying PCA and extracting hues...")
    pca_features_rgb, hue_values = apply_pca_and_extract_hues(features, foreground_mask)
    
    if pca_features_rgb is None:
        print("  Skipping region due to insufficient foreground pixels")
        return
    
    print("Creating RGB grid...")
    rgb_grid = create_rgb_grid(pca_features_rgb)
    
    # Create final figure
    output_path = os.path.join(output_dir, f"{cancer_type}_{region_index}.png")
    create_final_figure(original_image, rgb_grid, hue_values, output_path)


def process_wsi(wsi_path, cancer_type, output_dir, model, device, n_regions=10):
    """Process WSI"""
    
    print(f"\n{'#'*60}")
    print(f"# {cancer_type}")
    print(f"{'#'*60}")
    
    slide = None
    try:
        slide = openslide.OpenSlide(wsi_path)
        regions = find_tissue_regions_diverse(slide, min_tissue_ratio=0.5, 
                                             n_regions=n_regions, output_dir=output_dir)
        
        for i in range(len(regions)):
            process_region(slide, regions[i], i, model, device, output_dir, cancer_type)
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False
        
    finally:
        if slide:
            slide.close()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Combined PCA + Hue Spectra Visualization')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (default: ../logs/checkpoint.pth)')
    parser.add_argument('--output_dir', type=str, default='pca_hue_visualizations',
                       help='Output directory (default: pca_hue_visualizations)')
    parser.add_argument('--n_regions', type=int, default=1,
                       help='Number of regions per WSI (default: 1)')
    args = parser.parse_args()
    
    # Default to relative path from script location
    if args.checkpoint is None:
        args.checkpoint = os.path.join(script_dir, '..', 'logs', 'checkpoint.pth')
    
    wsi_paths = {
        'LUAD': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-LUAD_svs/svs/862a0948-7481-48d5-b127-8e56be1c1e92/TCGA-MP-A4TH-01Z-00-DX1.E89D2C19-F9B2-4BF2-AA5F-6104CBC076D1.svs",
        'SARC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-SARC_svs/svs/ff832ed6-f547-4e7d-b5f2-79f4b2a16d4e/TCGA-IF-A4AJ-01Z-00-DX1.A6CE6AEC-B645-4885-A995-99FF7A4B26A5.svs",
        'ACC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-ACC_svs/svs/fe92d4f9-3bf0-4ee5-9eae-558155f5be06/TCGA-OR-A5LR-01Z-00-DX4.0AF1F52B-222F-4D41-94A1-AA7D9CFBC70C.svs",
        'BLCA': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-BLCA_svs/svs/fed5f7ea-43b0-4a72-92b6-3ec43fac6b60/TCGA-FJ-A3Z7-01Z-00-DX6.28B723F7-1035-4DC2-8DB1-87F08166A9FA.svs",
        'KIRC': "/data1/vanderbc/foundation_model_training_images/TCGA/TCGA-KIRC_svs/svs/fffdfd4f-a579-4377-aa11-0aab83b644be/TCGA-DV-5576-01Z-00-DX1.ddd18b71-fc48-40f7-bc87-fb50d9ff468c.svs",
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model once
    print("="*60)
    print("LOADING MODEL")
    print("="*60)
    model = load_model_from_checkpoint(args.checkpoint, device)
    
    # Process each WSI
    for cancer_type, wsi_path in wsi_paths.items():
        cancer_dir = os.path.join(args.output_dir, cancer_type)
        os.makedirs(cancer_dir, exist_ok=True)
        process_wsi(wsi_path, cancer_type, cancer_dir, model, device, n_regions=args.n_regions)
    
    print(f"\nOutput files:")
    print(f"  {args.output_dir}/CANCER_TYPE/CANCER_TYPE_0.png, CANCER_TYPE_1.png, ...")
    print("  Layout: Original Image | PCA-RGB | Polar Hue Distribution")


if __name__ == "__main__":
    main()