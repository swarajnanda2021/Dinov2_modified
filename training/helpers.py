"""
Helper functions for DINOv2 training.
Includes mask generation, visualization, and data loading utilities.
"""

import os
import gc
import random
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import random

from models.vision_transformer.auxiliary_models import MaskModel_SpectralNorm


def load_pretrained_mask_model(checkpoint_path, num_masks=3):
    """
    Load pre-trained mask model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_masks: Number of masks
        
    Returns:
        Loaded mask model
    """
    print(f"Loading pre-trained mask model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    from models.vision_transformer.modern_vit import VisionTransformer
    
    mask_encoder = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.1,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    mask_model = MaskModel_SpectralNorm(
        encoder=mask_encoder,
        num_masks=num_masks,
        encoder_dim=192,
        drop_rate=0.2
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'mask_model' in checkpoint:
        mask_state_dict = checkpoint['mask_model']
    else:
        raise KeyError("No 'mask_model' found in checkpoint")
    
    cleaned_state_dict = {}
    for k, v in mask_state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k.replace('module.', '')] = v
        else:
            cleaned_state_dict[k] = v
    
    mask_model.load_state_dict(cleaned_state_dict, strict=False)
    
    print(f"Successfully loaded mask model weights")
    
    if 'iteration' in checkpoint:
        print(f"Checkpoint was saved at iteration: {checkpoint['iteration']}")
    
    return mask_model


def apply_masks_to_images(images, masks):
    """
    Apply masks to images.
    
    Args:
        images: [B, C, H, W]
        masks: [B, num_masks, H, W]
        
    Returns:
        List of [B, C, H, W] masked images
    """
    B, num_masks, H, W = masks.shape
    masked_images = []
    
    for i in range(num_masks):
        mask = masks[:, i:i+1, :, :]
        masked = images * (1 - mask)
        masked_images.append(masked)
    
    return masked_images


def extract_local_crops_from_masked(masked_img, n_crops, crop_size=96):
    """
    Extract n_crops local crops from a masked image.
    
    Args:
        masked_img: [B, C, H, W] masked image
        n_crops: Number of crops to extract
        crop_size: Size of each crop
        
    Returns:
        List of crops
    """
    B, C, H, W = masked_img.shape
    crops = []
    
    for _ in range(n_crops):
        crops_batch = []
        for b in range(B):
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            crop = masked_img[b:b+1, :, top:top+crop_size, left:left+crop_size]
            crops_batch.append(crop)
        crops.append(torch.cat(crops_batch, dim=0))
    
    return crops


def generate_random_token_masks(batch_size, n_patches_h, n_patches_w, mask_ratio, device):
    """
    Generate random token masks for iBOT training.
    
    Args:
        batch_size: Batch size
        n_patches_h: Number of patches in height
        n_patches_w: Number of patches in width
        mask_ratio: Ratio of tokens to mask
        device: Device to create masks on
        
    Returns:
        Boolean mask where True indicates masked tokens [B, N]
    """
    n_patches = n_patches_h * n_patches_w
    token_masks = torch.bernoulli(
        torch.ones(batch_size, n_patches) * mask_ratio
    ).bool().to(device)
    return token_masks

def generate_random_image_masks(batch_size, num_masks, height, width, device):
    """
    Generate random rectangular masks for image-level augmentation.
    
    Args:
        batch_size: Number of images
        num_masks: Number of masks per image
        height: Image height
        width: Image width
        device: Device to create masks on
        
    Returns:
        masks: [B, num_masks, H, W] float masks with 1.0 in masked regions
    """
    masks = torch.zeros(batch_size, num_masks, height, width, device=device)
    
    for b in range(batch_size):
        for m in range(num_masks):
            # Random rectangle parameters
            h_start = torch.randint(0, height//2, (1,)).item()
            w_start = torch.randint(0, width//2, (1,)).item()
            h_size = torch.randint(height//4, height//2, (1,)).item()
            w_size = torch.randint(width//4, width//2, (1,)).item()
            
            h_end = min(h_start + h_size, height)
            w_end = min(w_start + w_size, width)
            
            masks[b, m, h_start:h_end, w_start:w_end] = 1.0
    
    return masks

def calculate_total_student_views(args):
    """
    Calculate total number of student views based on configuration.
    Handles adversarial, CellViT, and random mask augmentations independently.
    """
    total = 0
    
    # Global views (always included unless explicitly set to 0)
    if args.global_views > 0:
        total += args.global_views
    
    # Standard local crops
    total += args.n_standard_local_crops
    
    # Adversarial mask augmentation (3-channel semantic masks)
    if hasattr(args, 'use_adversarial_mask_augmentation') and args.use_adversarial_mask_augmentation:
        if args.num_masks > 0:
            total += args.num_masks  # Masked global views
            total += args.num_masks * args.crops_per_mask  # Masked local crops
    
    # CellViT augmentation (2-channel nuclei/background)
    if hasattr(args, 'use_cellvit_augmentation') and args.use_cellvit_augmentation:
        total += 2  # 1 nuclei global + 1 background global
        total += 2 * args.cellvit_crops_per_channel  # Crops from both channels
    
    # Random mask augmentation
    if hasattr(args, 'use_random_mask_augmentation') and args.use_random_mask_augmentation:
        if args.random_num_masks > 0:
            total += args.random_num_masks  # Masked global views
            total += args.random_num_masks * args.random_crops_per_mask  # Masked local crops
    
    return total

def save_iteration_masks_efficient(
    images, 
    masks, 
    iteration, 
    save_dir, 
    reconstructed_images=None, 
    num_samples=4, 
    timeout_seconds=30
):
    """
    Efficient mask visualization that prevents hanging.
    
    Args:
        images: Input images [B, C, H, W]
        masks: Generated masks [B, num_masks, H, W]
        iteration: Current iteration
        save_dir: Directory to save visualizations
        reconstructed_images: Optional reconstructed images
        num_samples: Number of samples to visualize
        timeout_seconds: Timeout for visualization (unused, kept for compatibility)
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        images_cpu = images.detach().cpu().float()
        masks_cpu = masks.detach().cpu().float()
        
        torch.cuda.empty_cache()
        
        batch_size = images_cpu.size(0)
        num_samples = min(num_samples, batch_size, 4)
        
        torch.manual_seed(42)
        if batch_size > num_samples:
            indices = torch.randperm(batch_size)[:num_samples]
            images_cpu = images_cpu[indices]
            masks_cpu = masks_cpu[indices]
        
        mean_cpu = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1)
        std_cpu = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1)
        
        images_norm = images_cpu * std_cpu + mean_cpu
        images_norm = torch.clamp(images_norm, 0, 1)
        
        num_masks = masks_cpu.size(1)
        
        cols = min(num_masks + 2, 5)
        fig, axes = plt.subplots(num_samples, cols, figsize=(3*cols, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        for i in range(num_samples):
            col_idx = 0
            
            img_np = images_norm[i].permute(1, 2, 0).numpy()
            axes[i, col_idx].imshow(img_np)
            axes[i, col_idx].axis('off')
            if i == 0:
                axes[i, col_idx].set_title('Original', fontsize=10)
            col_idx += 1
            
            for j in range(min(num_masks, cols - 2)):
                mask_np = masks_cpu[i, j].numpy()
                axes[i, col_idx].imshow(mask_np, cmap='viridis', vmin=0, vmax=1)
                axes[i, col_idx].axis('off')
                if i == 0:
                    axes[i, col_idx].set_title(f'Mask {j+1}', fontsize=10)
                col_idx += 1
            
            if num_masks >= 3:
                rgb_masks = torch.stack([
                    masks_cpu[i, 0], 
                    masks_cpu[i, 1], 
                    masks_cpu[i, 2]
                ], dim=0).permute(1, 2, 0).numpy()
                axes[i, col_idx].imshow(rgb_masks, vmin=0, vmax=1)
                title = 'RGB Combined'
            else:
                avg_mask = masks_cpu[i].mean(dim=0).numpy()
                axes[i, col_idx].imshow(avg_mask, cmap='viridis', vmin=0, vmax=1)
                title = 'Avg Masks'
            
            axes[i, col_idx].axis('off')
            if i == 0:
                axes[i, col_idx].set_title(title, fontsize=10)
        
        save_path = os.path.join(save_dir, f'iter_{iteration:06d}_masks.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=100, facecolor='white')
        
        plt.close(fig)
        plt.clf()
        
        del images_cpu, masks_cpu, images_norm
        gc.collect()
        
        print(f"Mask visualization saved to {save_path}")
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        plt.close('all')
        plt.clf()


def worker_init_fn(worker_id):
    """
    Initialize worker with proper random seeding.
    
    Args:
        worker_id: Worker ID
    """
    import numpy as np
    import torch
    from torch.utils.data import get_worker_info
    
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    
    if hasattr(dataset, 'base_dataset'):
        dataset.base_dataset.set_worker_info(worker_info.id, worker_info.num_workers)
        seed = dataset.base_dataset.seed
    else:
        dataset.worker_id = worker_info.id
        seed = dataset.seed
    
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def setup_ddp_model(model, args, find_unused=False):
    """
    Setup model for distributed data parallel training.
    
    Args:
        model: Model to wrap    
        args: Arguments with GPU info
        find_unused: Whether to find unused parameters
        
    Returns:
        DDP-wrapped model
    """
    # Enable gradient checkpointing BEFORE DDP wrapping
    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
        if hasattr(model, 'set_grad_checkpointing'):
            model.set_grad_checkpointing(True)
            print(f"✓ Enabled gradient checkpointing before DDP wrapping")

    ddp_model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=find_unused,
        broadcast_buffers=True
    )
    
    return ddp_model


def load_pretrained_cellvit_model(checkpoint_path, device='cuda'):
    """
    Load pre-trained CellViT model for nuclei/background segmentation.
    Returns 2-channel mask output: [B, 2, H, W] where channel 0=nuclei, channel 1=background.
    
    Args:
        checkpoint_path: Path to trained CellViT checkpoint (.pth file)
        device: Device to load model on
        
    Returns:
        Frozen CellViT model in eval mode
    """
    print(f"Loading pre-trained CellViT model from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CellViT checkpoint not found at {checkpoint_path}")
    
    # Import from local models package
    from models.vision_transformer import VisionTransformer as ModernViT, CellViT
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        magnification = checkpoint.get('magnification', '40x')
        feature_dim = checkpoint.get('feature_dim', 768)
    else:
        raise KeyError("No 'model_state_dict' found in checkpoint")
    
    print(f"CellViT checkpoint info: magnification={magnification}, feature_dim={feature_dim}")
    
    # Create encoder
    encoder = ModernViT(
        img_size=224,
        patch_size=16,
        embed_dim=feature_dim,
        depth=12,
        num_heads=feature_dim // 64,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.4,
        pre_norm=False,
        num_register_tokens=4,
    )
    
    # Create CellViT model
    cellvit_model = CellViT(
        encoder=encoder,
        encoder_dim=feature_dim,
        drop_rate=0.2
    )
    
    # Load weights - handle potential module. prefix from DDP
    cleaned_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k.replace('module.', '')] = v
        else:
            cleaned_state_dict[k] = v
    
    # Load state dict - use strict=False since we only need the binary decoder
    missing_keys, unexpected_keys = cellvit_model.load_state_dict(cleaned_state_dict, strict=False)
    
    if missing_keys:
        print(f"  Missing keys (expected for simplified model): {len(missing_keys)} keys")
    if unexpected_keys:
        # Filter out hv_map_decoder keys since we don't use that branch
        unexpected_filtered = [k for k in unexpected_keys if 'hv_map' not in k]
        if unexpected_filtered:
            print(f"  Warning: Unexpected keys: {unexpected_filtered[:5]}...")
    
    # Move to device and set to eval mode
    cellvit_model = cellvit_model.to(device)
    cellvit_model = cellvit_model.to(torch.bfloat16)
    cellvit_model.eval()
    
    # Freeze all parameters
    for param in cellvit_model.parameters():
        param.requires_grad = False
    
    print(f"Successfully loaded CellViT model (2-channel nuclei/background segmentation)")
    
    return cellvit_model


def apply_cellvit_masks(images, masks):
    """
    Apply 2-channel CellViT masks to create nuclei and background views.
    
    Args:
        images: [B, C, H, W] input images
        masks: [B, 2, H, W] CellViT output (channel 0=nuclei, channel 1=background)
        
    Returns:
        nuclei_images: [B, C, H, W] - images with only nuclei visible
        background_images: [B, C, H, W] - images with only background visible
    """
    B, _, H, W = images.shape
    
    # Extract individual channels
    nuclei_mask = masks[:, 0:1, :, :]  # [B, 1, H, W]
    background_mask = masks[:, 1:2, :, :]  # [B, 1, H, W]
    
    # Apply masks (element-wise multiplication)
    nuclei_images = images * nuclei_mask
    background_images = images * background_mask
    
    return nuclei_images, background_images


def extract_crops_from_cellvit_channel(masked_img, n_crops, crop_size=96):
    """
    Extract n_crops random 96x96 crops from a masked image.
    
    Args:
        masked_img: [B, C, H, W] masked image
        n_crops: Number of crops to extract
        crop_size: Size of each crop (default 96)
        
    Returns:
        List of [B, C, crop_size, crop_size] crops
    """
    B, C, H, W = masked_img.shape
    crops = []
    
    for _ in range(n_crops):
        crops_batch = []
        for b in range(B):
            # Random crop location
            top = random.randint(0, H - crop_size)
            left = random.randint(0, W - crop_size)
            crop = masked_img[b:b+1, :, top:top+crop_size, left:left+crop_size]
            crops_batch.append(crop)
        crops.append(torch.cat(crops_batch, dim=0))
    
    return crops