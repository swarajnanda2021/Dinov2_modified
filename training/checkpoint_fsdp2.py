"""
Checkpointing for FSDP2 models - saves consolidated .pth files.
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, 
    set_model_state_dict,
    get_optimizer_state_dict, 
    set_optimizer_state_dict,
    StateDictOptions
)


def save_checkpoint_fsdp2(
    ckpt_dir,
    iteration,
    student,
    teacher,
    prototype_bank,
    optimizer_student,
    optimizer_prototypes,
    args,
    dino_class_loss=None,
    patch_prototype_loss=None,
    fp16_scaler=None,
):
    """
    Save FSDP2 checkpoint as consolidated .pth file (rank 0 only).
    Creates both numbered and 'latest' checkpoints.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Only rank 0 saves the checkpoint
    if rank != 0:
        if dist.is_initialized():
            dist.barrier()  # Sync with rank 0
        return
    
    print(f"[Rank {rank}] Starting checkpoint save at iteration {iteration}")
    
    # Create checkpoint directory
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get full state dicts (gathered on rank 0)
        print(f"[Rank {rank}] Gathering full model state dicts...")
        
        student_state = get_model_state_dict(
            student,
            options=StateDictOptions(full_state_dict=True)
        )
        
        teacher_state = get_model_state_dict(
            teacher,
            options=StateDictOptions(full_state_dict=True)
        )
        
        print(f"[Rank {rank}] Gathering optimizer state dicts...")
        
        optimizer_student_state = get_optimizer_state_dict(
            student,
            optimizer_student,
            options=StateDictOptions(full_state_dict=True)
        )
        
        # Build checkpoint dictionary
        checkpoint = {
            "iteration": iteration,
            "student": student_state,
            "teacher": teacher_state,
            "optimizer_student": optimizer_student_state,
            "args": vars(args) if hasattr(args, '__dict__') else args,
        }
        
        # DDP models - regular state dict
        if prototype_bank is not None:
            checkpoint["prototype_bank"] = prototype_bank.state_dict()
        if optimizer_prototypes is not None:
            checkpoint["optimizer_prototypes"] = optimizer_prototypes.state_dict()
        
        # Optional components
        if dino_class_loss is not None:
            checkpoint["dino_class_loss"] = dino_class_loss.state_dict()
        if patch_prototype_loss is not None:
            checkpoint["patch_prototype_loss"] = patch_prototype_loss.state_dict()
        if fp16_scaler is not None:
            checkpoint["fp16_scaler"] = fp16_scaler.state_dict()
        
        # Save numbered checkpoint
        numbered_path = ckpt_dir / f"checkpoint_iter_{iteration:08d}.pth"
        print(f"[Rank {rank}] Saving checkpoint to {numbered_path}")
        torch.save(checkpoint, numbered_path)
        
        # Save as latest checkpoint
        latest_path = ckpt_dir / "checkpoint.pth"
        print(f"[Rank {rank}] Saving latest checkpoint to {latest_path}")
        torch.save(checkpoint, latest_path)
        
        print(f"[Rank {rank}] Checkpoint save completed successfully")
        
        # Clean up old checkpoints (keep only last 3 numbered checkpoints)
        existing_checkpoints = sorted(ckpt_dir.glob("checkpoint_iter_*.pth"))
        if len(existing_checkpoints) > 3:
            for old_ckpt in existing_checkpoints[:-3]:
                os.remove(old_ckpt)
                print(f"[Rank {rank}] Removed old checkpoint: {old_ckpt}")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR during checkpoint save: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()


def load_checkpoint_fsdp2(
    ckpt_dir,
    student,
    teacher,
    prototype_bank,
    optimizer_student,
    optimizer_prototypes,
    args,
    dino_class_loss=None,
    patch_prototype_loss=None,
    fp16_scaler=None,
):
    """
    Load FSDP2 checkpoint from consolidated .pth file.
    Tries 'checkpoint.pth' first, then looks for latest numbered checkpoint.
    
    Returns:
        iteration: Loaded iteration number, or 0 if checkpoint doesn't exist
    """
    ckpt_dir = Path(ckpt_dir)
    
    if not ckpt_dir.exists():
        print(f"No checkpoint directory found at {ckpt_dir}")
        return 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Find checkpoint file
    latest_path = ckpt_dir / "checkpoint.pth"
    
    if latest_path.exists():
        ckpt_path = latest_path
        print(f"[Rank {rank}] Loading from latest checkpoint: {ckpt_path}")
    else:
        # Look for numbered checkpoints
        checkpoint_files = sorted(ckpt_dir.glob("checkpoint_iter_*.pth"))
        if not checkpoint_files:
            print(f"[Rank {rank}] No checkpoint files found in {ckpt_dir}")
            return 0
        
        ckpt_path = checkpoint_files[-1]  # Get most recent
        print(f"[Rank {rank}] Loading from numbered checkpoint: {ckpt_path}")
    
    try:
        # Load checkpoint (all ranks load the same file)
        print(f"[Rank {rank}] Loading checkpoint file...")
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        iteration = checkpoint.get("iteration", 0)
        
        # Prepare state dicts for loading
        print(f"[Rank {rank}] Setting model state dicts...")
        
        # Set FSDP2 model states
        set_model_state_dict(
            student,
            model_state_dict=checkpoint["student"],
            options=StateDictOptions(strict=True)
        )
        
        set_model_state_dict(
            teacher,
            model_state_dict=checkpoint["teacher"],
            options=StateDictOptions(strict=True)
        )
        
        # Set optimizer state
        set_optimizer_state_dict(
            student,
            optimizer_student,
            optim_state_dict=checkpoint["optimizer_student"],
            options=StateDictOptions(strict=True)
        )
        
        # Load DDP model states
        if prototype_bank is not None and "prototype_bank" in checkpoint:
            prototype_bank.load_state_dict(checkpoint["prototype_bank"])
        
        if optimizer_prototypes is not None and "optimizer_prototypes" in checkpoint:
            optimizer_prototypes.load_state_dict(checkpoint["optimizer_prototypes"])
        
        # Load optional components
        if dino_class_loss is not None and "dino_class_loss" in checkpoint:
            dino_class_loss.load_state_dict(checkpoint["dino_class_loss"])
        
        if patch_prototype_loss is not None and "patch_prototype_loss" in checkpoint:
            patch_prototype_loss.load_state_dict(checkpoint["patch_prototype_loss"])
        
        if fp16_scaler is not None and "fp16_scaler" in checkpoint:
            fp16_scaler.load_state_dict(checkpoint["fp16_scaler"])
        
        print(f"[Rank {rank}] Successfully loaded checkpoint from iteration {iteration}")
        
        return iteration
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print(f"[Rank {rank}] Starting training from scratch")
        return 0