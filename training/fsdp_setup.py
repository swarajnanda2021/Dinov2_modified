"""
FSDP2 setup utilities for DINOv2 training.
Strategy: Replicated prototype bank (no sharding) for KoLeo loss compatibility.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh


def get_mixed_precision_policy():
    """Configure bf16 mixed precision for H100."""
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )

def apply_fsdp_wrapping(student, teacher, args):
    """
    Apply FSDP2 wrapping to student and teacher models.
    NO DDP WRAPPING - FSDP2 only (like DINOv3).
    
    Args:
        student: Raw student model (NOT DDP-wrapped)
        teacher: Raw teacher model (NOT DDP-wrapped)
        args: Training arguments
        
    Returns:
        FSDP2-wrapped student and teacher
    """
    # Initialize device mesh
    world_mesh = init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),), mesh_dim_names=("dp",))
    
    # Mixed precision policy
    mp_policy = get_mixed_precision_policy()
    
    fsdp_config = {
        "mesh": world_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }
    
    print(f"[Rank {dist.get_rank()}] Starting FSDP2 wrapping...")
    
    # ========== STUDENT BACKBONE ==========
    # Wrap each transformer block with FSDP first
    blocks = list(student.backbone.blocks)
    for block_id in range(len(blocks)):
        blocks[block_id] = fully_shard(blocks[block_id], **fsdp_config)
    
    # Apply gradient checkpointing AFTER FSDP wrapping if enabled
    if hasattr(args, 'grad_checkpointing') and args.grad_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
        for block_id in range(len(blocks)):
            blocks[block_id] = checkpoint_wrapper(blocks[block_id])
        print(f"[Rank {dist.get_rank()}] Applied gradient checkpointing to student blocks")
    else:
        # Set up prefetching only if NOT using gradient checkpointing
        for prev_block, next_block in zip(blocks[:-1], blocks[1:]):
            prev_block.set_modules_to_forward_prefetch([next_block])
            next_block.set_modules_to_backward_prefetch([prev_block])
    
    # Wrap the entire backbone
    student.backbone = fully_shard(student.backbone, **fsdp_config)
    
    # Wrap projection heads
    student.classhead = fully_shard(student.classhead, **fsdp_config)
    student.patchhead = fully_shard(student.patchhead, **fsdp_config)
    
    print(f"[Rank {dist.get_rank()}] ✓ Student FSDP2 wrapping complete")
    
    # ========== TEACHER BACKBONE (inference-only) ==========
    # Wrap each transformer block
    teacher_blocks = list(teacher.backbone.blocks)
    for block_id in range(len(teacher_blocks)):
        teacher_blocks[block_id] = fully_shard(teacher_blocks[block_id], **fsdp_config)
    
    # Set up prefetching for teacher (no backward)
    for prev_block, next_block in zip(teacher_blocks[:-1], teacher_blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
    
    # Wrap the entire backbone
    teacher.backbone = fully_shard(teacher.backbone, **fsdp_config)
    
    # Wrap projection heads
    teacher.classhead = fully_shard(teacher.classhead, **fsdp_config)
    teacher.patchhead = fully_shard(teacher.patchhead, **fsdp_config)
    
    print(f"[Rank {dist.get_rank()}] ✓ Teacher FSDP2 wrapping complete (inference-only)")
    
    return student, teacher

def _enable_inference_mode_resharding(model):
    """Enable immediate resharding after forward for inference-only models."""
    from torch.distributed._composable.fsdp import FSDPModule
    
    for module in model.modules():
        if isinstance(module, FSDPModule):
            # Access FSDP state and enable immediate reshard
            state = module._get_fsdp_state()
            if state._fsdp_param_group:
                # Store current post_forward_mesh_info
                state._lazy_init()
                # This enables immediate resharding after forward pass
                pass  # The resharding is already enabled by default in FSDP2