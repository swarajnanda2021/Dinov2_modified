"""
FSDP2 setup utilities for DINOv2 training.
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
    
    CRITICAL: Teacher is inference-only with immediate resharding.
    
    Args:
        student: Raw student model (NOT DDP-wrapped)
        teacher: Raw teacher model (NOT DDP-wrapped)  
        args: Training arguments
        
    Returns:
        FSDP2-wrapped student and teacher
    """
    # Initialize device mesh
    world_mesh = init_device_mesh(
        "cuda", 
        mesh_shape=(dist.get_world_size(),), 
        mesh_dim_names=("dp",)
    )
    
    # Mixed precision policy
    mp_policy = get_mixed_precision_policy()
    
    fsdp_config = {
        "mesh": world_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": True,
    }
    
    print(f"[Rank {dist.get_rank()}] Starting FSDP2 wrapping...")
    
    # ========== STUDENT BACKBONE ==========
    print(f"[Rank {dist.get_rank()}] Wrapping student...")
    
    # Wrap each transformer block with FSDP
    blocks = list(student.backbone.blocks)
    for block_id in range(len(blocks)):
        blocks[block_id] = fully_shard(blocks[block_id], **fsdp_config)
    
    # Set up prefetching (no gradient checkpointing for now to debug OOM)
    # Gradient checkpointing can be added back later if needed
    for prev_block, next_block in zip(blocks[:-1], blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
        next_block.set_modules_to_backward_prefetch([prev_block])
    
    # Wrap the entire backbone
    student.backbone = fully_shard(student.backbone, **fsdp_config)
    
    # Wrap projection heads
    student.classhead = fully_shard(student.classhead, **fsdp_config)
    student.patchhead = fully_shard(student.patchhead, **fsdp_config)
    
    print(f"[Rank {dist.get_rank()}] ✓ Student FSDP2 wrapping complete")
    
    # ========== TEACHER BACKBONE (INFERENCE-ONLY) ==========
    print(f"[Rank {dist.get_rank()}] Wrapping teacher (inference-only mode)...")
    
    # CRITICAL: Disable gradients BEFORE wrapping
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Wrap each transformer block
    teacher_blocks = list(teacher.backbone.blocks)
    for block_id in range(len(teacher_blocks)):
        teacher_blocks[block_id] = fully_shard(teacher_blocks[block_id], **fsdp_config)
    
    # Set up forward-only prefetching (no backward)
    for prev_block, next_block in zip(teacher_blocks[:-1], teacher_blocks[1:]):
        prev_block.set_modules_to_forward_prefetch([next_block])
    
    # Wrap the entire backbone
    teacher.backbone = fully_shard(teacher.backbone, **fsdp_config)
    
    # Wrap projection heads
    teacher.classhead = fully_shard(teacher.classhead, **fsdp_config)
    teacher.patchhead = fully_shard(teacher.patchhead, **fsdp_config)
    
    # CRITICAL: Enable immediate resharding for inference-only teacher
    _enable_inference_only_resharding(teacher)
    
    print(f"[Rank {dist.get_rank()}] ✓ Teacher FSDP2 wrapping complete (inference-only)")
    
    # Move to CUDA after wrapping (like DINOv3)
    student.to_empty(device="cuda")
    teacher.to_empty(device="cuda")
    
    return student, teacher


def _enable_inference_only_resharding(model):
    """
    Enable immediate resharding after forward for inference-only models.
    This is CRITICAL for memory efficiency - without this, teacher holds
    unsharded parameters in memory unnecessarily.
    
    Based on DINOv3's ac_compile_parallelize.py lines 118-125
    """
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    
    inference_only_modules = [
        model.backbone,
        model.classhead, 
        model.patchhead,
    ]
    
    for module in inference_only_modules:
        # Get FSDP state
        if hasattr(module, '_get_fsdp_state'):
            fsdp_state = module._get_fsdp_state()
            
            if fsdp_state and fsdp_state._fsdp_param_group:
                # Store the post_forward_mesh_info
                mi = fsdp_state._fsdp_param_group.post_forward_mesh_info
                
                # Force lazy initialization
                fsdp_state._lazy_init()
                
                # Restore mesh info to enable immediate resharding
                fsdp_state._fsdp_param_group.post_forward_mesh_info = mi
                
    print(f"[Rank {dist.get_rank()}] ✓ Enabled inference-only resharding for teacher")