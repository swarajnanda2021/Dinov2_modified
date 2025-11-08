"""
Complete Pipeline Parallelism for DINOv2 with Projection Heads.

Supports:
- Multi-stage model partitioning across GPUs
- Projection heads on last stage
- Returns both projected outputs AND raw features
- Gradient checkpointing
- Sequence packing with attention bias
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List
from copy import deepcopy


def get_model_config(model_size: str):
    """Get ViT configuration for different model sizes."""
    configs = {
        'base': {'depth': 12, 'embed_dim': 768, 'num_heads': 12},
        'large': {'depth': 24, 'embed_dim': 1024, 'num_heads': 16},
        'huge': {'depth': 32, 'embed_dim': 1280, 'num_heads': 16},
        'giant': {'depth': 40, 'embed_dim': 1408, 'num_heads': 16},
        'giant2b': {'depth': 48, 'embed_dim': 1664, 'num_heads': 16},
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    return configs[model_size]


def get_layer_split_points(depth: int, gpus_per_node: int):
    """Calculate layer split points for even distribution."""
    layers_per_gpu = depth // gpus_per_node
    
    if depth % gpus_per_node != 0:
        print(f"Warning: {depth} layers not evenly divisible by {gpus_per_node} GPUs")
    
    split_points = []
    for i in range(1, gpus_per_node):
        split_points.append(i * layers_per_gpu)
    
    return split_points


def create_pipeline_student_teacher(
    args,
    local_rank: int,
    device: torch.device,
    pipeline_group,
):
    """
    Create pipeline-parallelized student and teacher models with projection heads.
    
    Strategy:
    1. Create full model on device
    2. Initialize weights
    3. Partition: delete layers we don't need
    4. Create projection heads on last stage only
    5. Create identical teacher via deepcopy
    
    Returns:
        student_stage, teacher_stage, prototype_bank (or None if not last GPU)
    """
    from models import ModernViT, DINOHead, LinearPrototypeBank

    # Get model configuration
    config = get_model_config(args.model_size)
    depth = config['depth']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']

    print(f"Creating {args.model_size} model: depth={depth}, dim={embed_dim}, heads={num_heads}")
    
    # Calculate split points
    split_points = get_layer_split_points(depth, args.gpus_per_node)
    
    # Determine which layers this GPU handles
    if local_rank == 0:
        layer_start = 0
        layer_end = split_points[0] if split_points else depth
        has_patch_embed = True
        has_heads = False
    elif local_rank == args.gpus_per_node - 1:
        layer_start = split_points[-1] if split_points else 0
        layer_end = depth
        has_patch_embed = False
        has_heads = True
    else:
        layer_start = split_points[local_rank - 1]
        layer_end = split_points[local_rank]
        has_patch_embed = False
        has_heads = False
    
    print(f"GPU {local_rank}: Layers [{layer_start}:{layer_end}], "
          f"patch_embed={has_patch_embed}, heads={has_heads}")
    
    # ========== Create FULL model ==========
    full_student_encoder = ModernViT(
        img_size=224,
        patch_size=args.patch_size,
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
    ).to(device)
    
    full_student_encoder._init_weights()
    
    if args.grad_checkpointing:
        full_student_encoder.set_grad_checkpointing(True)
        print(f"✓ Enabled gradient checkpointing in pipeline student encoder")
    
    # ========== Partition backbone ==========
    # Keep only layers we need
    if not has_patch_embed:
        full_student_encoder.patch_embed = None
        full_student_encoder.cls_token = None
        full_student_encoder.register_tokens = None
        full_student_encoder.pos_embed = None
        print("  Removed patch_embed")
    
    # Keep only our transformer blocks
    blocks_to_keep = []
    for layer_idx in range(layer_start, layer_end):
        blocks_to_keep.append(full_student_encoder.blocks[layer_idx])
    
    full_student_encoder.blocks = nn.Sequential(*blocks_to_keep)
    print(f"  Kept {len(blocks_to_keep)} blocks [{layer_start}:{layer_end}]")
    
    # Keep or remove final norm
    if not has_heads:
        full_student_encoder.norm = None
        print("  Removed final norm")
    
    # ========== Create projection heads (ONLY on last stage) ==========
    if has_heads:
        print("  Creating projection heads on last stage")
        
        full_student_classhead = DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ).to(device)
        
        full_student_patchhead = DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        ).to(device)
        
        print(f"  Created DINO heads: {embed_dim} -> {args.out_dim}")
    else:
        full_student_classhead = None
        full_student_patchhead = None
    
    # ========== Create prototype bank (ONLY on last stage) ==========
    if has_heads:
        full_prototype_bank = LinearPrototypeBank(
            num_prototypes=args.num_prototypes,
            embed_dim=embed_dim,
            bias=True,
        ).to(device)
        print(f"  Created prototype bank: {args.num_prototypes} prototypes")
    else:
        full_prototype_bank = None
    
    # ========== Wrap in PipelineStageWrapper ==========
    student_stage = PipelineStageWrapper(
        backbone=full_student_encoder,
        classhead=full_student_classhead,
        patchhead=full_student_patchhead,
        local_rank=local_rank,
        num_stages=args.gpus_per_node,
        has_patch_embed=has_patch_embed,
        has_heads=has_heads,
        grad_checkpointing=args.grad_checkpointing,
    )

    # ========== Create teacher (identical copy) ==========
    teacher_stage = deepcopy(student_stage)
    teacher_stage.requires_grad_(False)
    
    print(f"✓ Created student and teacher pipeline stages for GPU {local_rank}")
    
    return student_stage, teacher_stage, full_prototype_bank


class PipelineStageWrapper(nn.Module):
    """
    Wrapper for a single pipeline stage with heads support.
    
    Returns both projected outputs AND raw features for downstream losses.
    """
    def __init__(
        self,
        backbone,
        classhead,
        patchhead,
        local_rank: int,
        num_stages: int,
        has_patch_embed: bool,
        has_heads: bool,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.classhead = classhead
        self.patchhead = patchhead
        self.local_rank = local_rank
        self.num_stages = num_stages
        self.has_patch_embed = has_patch_embed
        self.has_heads = has_heads
        self.grad_checkpointing = grad_checkpointing
        
        # Cache for attention bias
        self._cached_attn_bias = None

    def _apply_blocks_with_checkpointing(self, x, attn_bias=None):
        # Let the backbone handle its own checkpointing
        # (since we already enabled it via set_grad_checkpointing)
        
        if attn_bias is not None:
            # Custom path for attention bias
            if self.grad_checkpointing and self.training:
                # Checkpoint in reasonable chunks
                chunk_size = 4
                for i in range(0, len(self.backbone.blocks), chunk_size):
                    def create_chunk_forward(start, end):
                        def forward(x):
                            for j in range(start, end):
                                x = self.backbone.blocks[j](x, attn_bias=attn_bias)
                            return x
                        return forward
                    
                    end_idx = min(i + chunk_size, len(self.backbone.blocks))
                    x = torch.utils.checkpoint.checkpoint(
                        create_chunk_forward(i, end_idx),
                        x,
                        use_reentrant=False
                    )
            else:
                for block in self.backbone.blocks:
                    x = block(x, attn_bias=attn_bias)
        else:
            # No attention bias: use backbone's native checkpointing
            # This hits ModernViT.forward_features_list or forward
            # which already handles checkpointing efficiently
            for block in self.backbone.blocks:
                x = block(x, attn_bias=None)
        
        return x

    def forward(self, x, token_masks=None, attn_bias=None):
        """
        Forward through this pipeline stage.
        
        Returns:
        - First/middle stages: (processed_features, attn_bias) tuple
        - Last stage: Dict or List[Dict] with BOTH projected AND raw features
        """
        is_multi_crop = isinstance(x, list)
        
        # ========== FIRST STAGE ==========
        if self.local_rank == 0:
            if is_multi_crop:
                # Multi-crop: pack sequences
                x_processed = []
                
                for crop in x:
                    if token_masks is not None and isinstance(token_masks, list):
                        crop_idx = x.index(crop)
                        mask = token_masks[crop_idx] if crop_idx < len(token_masks) else None
                        tokens = self.backbone.prepare_tokens_with_masks(crop, mask)
                    else:
                        tokens = self.backbone.prepare_tokens(crop)
                    x_processed.append(tokens)
                
                # Create attention bias for sequence packing
                from models.vision_transformer.modern_vit import get_attn_bias_and_cat
                attn_bias, x_cat = get_attn_bias_and_cat(x_processed)
                self._cached_attn_bias = attn_bias
                
                x_cat = self._apply_blocks_with_checkpointing(x_cat, attn_bias)
                
                return x_cat, attn_bias
                
            else:
                # Single image
                if token_masks is not None:
                    x_processed = self.backbone.prepare_tokens_with_masks(x, token_masks)
                else:
                    x_processed = self.backbone.prepare_tokens(x)
                
                x_processed = self._apply_blocks_with_checkpointing(x_processed, attn_bias=None)
                
                return x_processed, None
        
        # ========== LAST STAGE ==========
        elif self.has_heads:
            if is_multi_crop or attn_bias is not None:
                # Packed multi-crop case
                x = self._apply_blocks_with_checkpointing(x, attn_bias)
                x = self.backbone.norm(x)
                
                # Unpack sequences
                if attn_bias is not None:
                    outputs_list = attn_bias.split(x)
                else:
                    outputs_list = [x]
                
                # Extract features and apply heads
                results = []
                for tokens in outputs_list:
                    cls_token = tokens[:, 0]
                    patch_tokens = tokens[:, 5:]  # Skip cls + 4 registers
                    
                    # Apply projection heads
                    cls_output = self.classhead(cls_token)
                    patch_output = self.patchhead(patch_tokens)
                    
                    results.append({
                        'cls_output': cls_output,          # Projected [B, out_dim]
                        'patch_output': patch_output,      # Projected [B, N, out_dim]
                        'cls_features': cls_token,         # Raw [B, embed_dim]
                        'patch_features': patch_tokens,    # Raw [B, N, embed_dim]
                    })
                
                return results
                
            else:
                # Single image case
                x = self._apply_blocks_with_checkpointing(x, attn_bias=None)
                x = self.backbone.norm(x)
                
                cls_token = x[:, 0]
                patch_tokens = x[:, 5:]
                
                # Apply projection heads
                cls_output = self.classhead(cls_token)
                patch_output = self.patchhead(patch_tokens)
                
                return {
                    'cls_output': cls_output,
                    'patch_output': patch_output,
                    'cls_features': cls_token,
                    'patch_features': patch_tokens,
                }
        
        # ========== MIDDLE STAGE ==========
        else:
            x = self._apply_blocks_with_checkpointing(x, attn_bias)
            return x, attn_bias