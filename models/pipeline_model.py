"""
Pipeline parallelism wrapper for DINOv2.
Supports 4-GPU and 8-GPU nodes with hybrid data+pipeline parallelism.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
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
    if local_rank == 0: # first GPU handles the patch embedding and the first set of blocks
        layer_start = 0
        layer_end = split_points[0] if split_points else depth
        has_patch_embed = True
        has_heads = False
    elif local_rank == args.gpus_per_node - 1: # last GPU handles the last set of blocks, the prototypes, the projection heads, and losses
        layer_start = split_points[-1] if split_points else 0
        layer_end = depth
        has_patch_embed = False
        has_heads = True
    else: # middle layers are only for intermediate transformer blocks
        layer_start = split_points[local_rank - 1]
        layer_end = split_points[local_rank]
        has_patch_embed = False
        has_heads = False
    
    print(f"GPU {local_rank}: Layers [{layer_start}:{layer_end}], "
          f"patch_embed={has_patch_embed}, heads={has_heads}")
    
    # ========== Create FULL model on meta device (to avoid OOM) ==========
    with torch.device('meta'):
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
        )
        
        full_student_classhead = DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
        
        full_student_patchhead = DINOHead(
            embed_dim,
            args.out_dim,
            use_bn=args.use_bn_in_head,
            norm_last_layer=args.norm_last_layer,
        )
    
        # Prototype bank (only on last GPU)
        if has_heads:
            full_prototype_bank = LinearPrototypeBank(
                num_prototypes=args.num_prototypes,
                embed_dim=embed_dim,
                bias=True,
            )
        else:
            full_prototype_bank = None
    
    # ========== Partition model for this GPU ==========
    # Delete layers we don't need
    
    # 1. Handle patch embedding
    if not has_patch_embed:
        full_student_encoder.patch_embed = None
        full_student_encoder.cls_token = None
        full_student_encoder.register_tokens = None
        full_student_encoder.pos_embed = None
    
    # 2. Handle transformer blocks - keep only our range
    for layer_idx in range(depth):
        if layer_idx < layer_start or layer_idx >= layer_end:
            del full_student_encoder.blocks[layer_idx]
    
    # 3. Handle final norm and heads
    if not has_heads:
        full_student_encoder.norm = None
        full_student_classhead = None
        full_student_patchhead = None
        full_prototype_bank = None
    
    # ========== Move to device and materialize ==========
    # This will only allocate memory for the layers this GPU owns
    full_student_encoder = full_student_encoder.to_empty(device=device)
    
    if full_student_classhead is not None:
        full_student_classhead = full_student_classhead.to_empty(device=device)
    
    if full_student_patchhead is not None:
        full_student_patchhead = full_student_patchhead.to_empty(device=device)
    
    if full_prototype_bank is not None:
        full_prototype_bank = full_prototype_bank.to_empty(device=device)
    
    # Initialize weights properly
    if has_patch_embed:
        full_student_encoder._init_weights()
    else:
        # Only init the blocks we own
        for block in full_student_encoder.blocks:
            full_student_encoder._init_module_weights(block)
    
    if has_heads:
        full_student_encoder._init_module_weights(full_student_encoder.norm)
        # Heads have their own initialization
    
    # ========== Create PipelineStage wrapper ==========
    student_stage = PipelineStageWrapper(
        backbone=full_student_encoder,
        classhead=full_student_classhead,
        patchhead=full_student_patchhead,
        local_rank=local_rank,
        num_stages=args.gpus_per_node,
        has_patch_embed=has_patch_embed,
        has_heads=has_heads,
    )

    # ========== Create teacher (same structure) ==========
    teacher_stage = deepcopy(student_stage)
    teacher_stage.requires_grad_(False)

    # ========== Wrap in PipelineStage ==========
    from torch.distributed.pipelining import PipelineStage as TorchPipelineStage
    
    # Note: We'll actually use our custom wrapper for now
    # and handle scheduling manually for full control
    
    return student_stage, teacher_stage, full_prototype_bank


class PipelineStageWrapper(nn.Module):
    """
    Wrapper for a single pipeline stage.
    Handles partial model execution with send/recv.
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
    ):
        super().__init__()
        self.backbone = backbone
        self.classhead = classhead
        self.patchhead = patchhead
        self.local_rank = local_rank
        self.num_stages = num_stages
        self.has_patch_embed = has_patch_embed
        self.has_heads = has_heads

    
    def forward(self, x, token_masks=None):
        """
        Forward through this pipeline stage.
        
        For first stage: Apply patch embed + initial blocks
        For middle stages: Apply blocks
        For last stage: Apply final blocks + norm + heads
        """

        # Handle different input formats (single crop vs multi-crop)
        is_multi_crop = isinstance(x, list)
        
        if self.local_rank == 0:
            # First stage: patch embedding
            if is_multi_crop:
                x_processed = []
                for crop in x:
                    tokens = self.backbone.prepare_tokens(crop)
                    x_processed.append(tokens)
            else:
                if token_masks is not None:
                    x_processed = self.backbone.prepare_tokens_with_masks(x, token_masks)
                else:
                    x_processed = self.backbone.prepare_tokens(x)
            
            # Apply our transformer blocks
            if is_multi_crop:
                for i, tokens in enumerate(x_processed):
                    for block in self.backbone.blocks:
                        tokens = block(tokens)
                    x_processed[i] = tokens
            else:
                for block in self.backbone.blocks:
                    x_processed = block(x_processed)
            
            return x_processed
        
        elif self.has_heads:
            # Last stage: apply final blocks + norm + heads
            if is_multi_crop:
                outputs = []
                for tokens in x:
                    for block in self.backbone.blocks:
                        tokens = block(tokens)
                    tokens = self.backbone.norm(tokens)
                    outputs.append({
                        'clstoken': tokens[:, 0],
                        'regtokens': tokens[:, 1:5],
                        'patchtokens': tokens[:, 5:],
                    })
                
                return outputs
            else:
                for block in self.backbone.blocks:
                    x = block(x)
                x = self.backbone.norm(x)
                
                return {
                    'clstoken': x[:, 0],
                    'regtokens': x[:, 1:5],
                    'patchtokens': x[:, 5:],
                }
        
        else:
            # Middle stage: just apply blocks
            if is_multi_crop:
                for i, tokens in enumerate(x):
                    for block in self.backbone.blocks:
                        tokens = block(tokens)
                    x[i] = tokens
            else:
                for block in self.backbone.blocks:
                    x = block(x)
            
            return x


def create_pipeline_schedule(
    student_stage,
    teacher_stage,
    local_rank: int,
    num_microbatches: int,
):
    """
    Create 1F1B schedule for pipeline execution.
    
    Note: For now we'll use manual scheduling.
    Full torch.distributed.pipelining integration coming in Phase 2.
    """
    # This will be expanded to use Schedule1F1B from torch.distributed.pipelining
    # For now, return stages
    return student_stage, teacher_stage