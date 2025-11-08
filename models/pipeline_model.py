"""
Pipeline parallelism wrapper for DINOv2.
Supports 4-GPU and 8-GPU nodes with hybrid data+pipeline parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List
from copy import deepcopy


def get_model_config(model_size: str):
    """
    Get ViT configuration for different model sizes.
    
    Args:
        model_size: 'base', 'large', 'huge', 'giant', 'giant2b'
    
    Returns:
        Dictionary with depth, embed_dim, num_heads
    """
    configs = {
        'base': {'depth': 12, 'embed_dim': 768, 'num_heads': 12},      # ~86M
        'large': {'depth': 24, 'embed_dim': 1024, 'num_heads': 16},     # ~307M
        'huge': {'depth': 32, 'embed_dim': 1280, 'num_heads': 16},      # ~632M
        'giant': {'depth': 40, 'embed_dim': 1408, 'num_heads': 16},     # ~1.01B
        'giant2b': {'depth': 48, 'embed_dim': 1664, 'num_heads': 16},   # ~2.04B
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    return configs[model_size]


def get_layer_split_points(depth: int, gpus_per_node: int):
    """
    Calculate layer split points for even distribution across GPUs.
    
    Args:
        depth: Total number of transformer layers
        gpus_per_node: Number of GPUs per node (4 or 8)
    
    Returns:
        List of split indices (exclusive end points)
    """
    layers_per_gpu = depth // gpus_per_node
    
    if depth % gpus_per_node != 0:
        print(f"Warning: {depth} layers not evenly divisible by {gpus_per_node} GPUs")
        print(f"Using {layers_per_gpu} layers per GPU with last GPU getting extra layers")
    
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
    Create pipeline-parallelized student and teacher models.
    
    Strategy:
    1. Create full model on device
    2. Initialize weights properly
    3. Delete layers we don't need
    4. Create identical teacher via deepcopy
    
    Args:
        args: Training arguments
        local_rank: GPU rank within the node (0 to gpus_per_node-1)
        device: CUDA device
        pipeline_group: Process group for pipeline parallelism
    
    Returns:
        student_stage, teacher_stage, prototype_bank (or None if not on last GPU)
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
    
    # ========== Create FULL model on device ==========
    print("Creating full model on device...")
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
    
    # Initialize weights
    print("Initializing weights...")
    full_student_encoder._init_weights()
    
    # ========== Partition model: Delete what we don't need ==========
    print("Partitioning model...")
    
    # 1. Handle patch embedding (only first stage keeps it)
    if not has_patch_embed:
        full_student_encoder.patch_embed = None
        full_student_encoder.cls_token = None
        full_student_encoder.register_tokens = None
        full_student_encoder.pos_embed = None
        print("  Removed patch_embed, cls_token, register_tokens, pos_embed")
    
    # 2. Keep only our transformer blocks
    blocks_to_keep = []
    for layer_idx in range(layer_start, layer_end):
        blocks_to_keep.append(full_student_encoder.blocks[layer_idx])
    
    full_student_encoder.blocks = nn.Sequential(*blocks_to_keep)
    print(f"  Kept {len(blocks_to_keep)} transformer blocks [{layer_start}:{layer_end}]")
    
    # 3. Handle final norm and heads (only last stage keeps them)
    if not has_heads:
        full_student_encoder.norm = None
        print("  Removed final norm")
    
    # ========== Create projection heads (only on last GPU) ==========
    if has_heads:
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
        
        print("  Created projection heads")
    else:
        full_student_classhead = None
        full_student_patchhead = None
    
    # ========== Create prototype bank (only on last GPU) ==========
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
    Wrapper for a single pipeline stage with gradient checkpointing support.
    Handles partial model execution with send/recv between stages.
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
        
        # For sequence packing: cache attention bias between forward calls
        self._cached_attn_bias = None

    def _apply_blocks_with_checkpointing(self, x, attn_bias=None):
        """
        Apply transformer blocks with optional gradient checkpointing.
        
        Gradient checkpointing is applied per-block to save memory during backward pass.
        """
        if self.grad_checkpointing and self.training:
            # Apply gradient checkpointing to each transformer block
            for block in self.backbone.blocks:
                # Checkpoint each block individually
                def create_forward_fn(module):
                    def forward_fn(*inputs):
                        # Handle both (x,) and (x, attn_bias) signatures
                        if len(inputs) == 1:
                            return module(inputs[0], attn_bias=None)
                        else:
                            return module(inputs[0], attn_bias=inputs[1])
                    return forward_fn
                
                if attn_bias is not None:
                    x = torch.utils.checkpoint.checkpoint(
                        create_forward_fn(block),
                        x,
                        attn_bias,
                        use_reentrant=False
                    )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_forward_fn(block),
                        x,
                        use_reentrant=False
                    )
        else:
            # No checkpointing: normal forward pass
            for block in self.backbone.blocks:
                x = block(x, attn_bias=attn_bias)
        
        return x

    def forward(self, x, token_masks=None, attn_bias=None):
        """
        Forward through this pipeline stage.
        
        Handles three cases:
        1. First stage: Apply patch embed + blocks
        2. Middle stages: Apply blocks only
        3. Last stage: Apply blocks + norm + heads
        
        Args:
            x: Input tensor(s) - can be list for multi-crop or single tensor
            token_masks: Optional token masking for iBOT
            attn_bias: Optional attention bias for sequence packing
            
        Returns:
            - First/middle stages: Processed features to send to next stage
            - Last stage: Dict with clstoken, patchtokens, etc.
        """
        is_multi_crop = isinstance(x, list)
        
        # ========== FIRST STAGE: Patch Embedding + Initial Blocks ==========
        if self.local_rank == 0:
            if is_multi_crop:
                # Multi-crop: pack sequences together
                x_processed = []
                
                # Process each crop through patch embedding
                for crop in x:
                    if token_masks is not None and isinstance(token_masks, list):
                        # Find corresponding mask (if provided)
                        crop_idx = x.index(crop)
                        mask = token_masks[crop_idx] if crop_idx < len(token_masks) else None
                        tokens = self.backbone.prepare_tokens_with_masks(crop, mask)
                    else:
                        tokens = self.backbone.prepare_tokens(crop)
                    x_processed.append(tokens)
                
                # Create attention bias for sequence packing
                from models.vision_transformer.modern_vit import get_attn_bias_and_cat
                attn_bias, x_cat = get_attn_bias_and_cat(x_processed)
                
                # Cache attention bias for backward pass
                self._cached_attn_bias = attn_bias
                
                # Apply transformer blocks with gradient checkpointing
                x_cat = self._apply_blocks_with_checkpointing(x_cat, attn_bias)
                
                # Return packed tensor and attention bias
                return x_cat, attn_bias
                
            else:
                # Single image
                if token_masks is not None:
                    x_processed = self.backbone.prepare_tokens_with_masks(x, token_masks)
                else:
                    x_processed = self.backbone.prepare_tokens(x)
                
                # Apply transformer blocks with gradient checkpointing
                x_processed = self._apply_blocks_with_checkpointing(x_processed, attn_bias=None)
                
                return x_processed, None
        
        # ========== LAST STAGE: Final Blocks + Norm + Heads ==========
        elif self.has_heads:
            if is_multi_crop or attn_bias is not None:
                # Packed multi-crop case
                # x is already concatenated: [1, total_tokens, D]
                
                # Apply transformer blocks with gradient checkpointing
                x = self._apply_blocks_with_checkpointing(x, attn_bias)
                
                # Apply final norm
                x = self.backbone.norm(x)
                
                # Unpack sequences using attention bias
                if attn_bias is not None:
                    outputs_list = attn_bias.split(x)
                else:
                    # If no attention bias, assume single sequence
                    outputs_list = [x]
                
                # Extract CLS and patch tokens for each crop
                results = []
                for tokens in outputs_list:
                    results.append({
                        'clstoken': tokens[:, 0],
                        'regtokens': tokens[:, 1:5],
                        'patchtokens': tokens[:, 5:],
                    })
                
                return results
                
            else:
                # Single image case
                # Apply transformer blocks with gradient checkpointing
                x = self._apply_blocks_with_checkpointing(x, attn_bias=None)
                
                # Apply final norm
                x = self.backbone.norm(x)
                
                return {
                    'clstoken': x[:, 0],
                    'regtokens': x[:, 1:5],
                    'patchtokens': x[:, 5:],
                }
        
        # ========== MIDDLE STAGE: Blocks Only ==========
        else:
            # Apply transformer blocks with gradient checkpointing
            x = self._apply_blocks_with_checkpointing(x, attn_bias)
            
            # Return processed features (preserving attn_bias for next stage)
            return x, attn_bias

