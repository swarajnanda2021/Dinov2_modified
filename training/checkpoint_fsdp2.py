"""
Hybrid checkpointing for FSDP2 models.
Saves both DCP (fast) and consolidated .pth (portable) formats.
"""

import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
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
    Hybrid checkpoint saving:
    1. Save DCP checkpoint (fast, all ranks in parallel)
    2. Reconstitute to .pth on CPU (rank 0 only, no GPU memory pressure)
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    print(f"[Rank {rank}] Starting checkpoint save at iteration {iteration}")
    
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: Save DCP checkpoint (all ranks) ==========
    dcp_path = ckpt_dir / f"dcp_iter_{iteration}"
    
    try:
        print(f"[Rank {rank}] Saving DCP checkpoint to {dcp_path}")
        
        to_save = {
            "iteration": iteration,
            "args": vars(args) if hasattr(args, '__dict__') else args,
        }
        
        # FSDP2 models - use DCP state dict functions
        to_save["student"] = get_model_state_dict(student)
        to_save["teacher"] = get_model_state_dict(teacher)
        to_save["optimizer_student"] = get_optimizer_state_dict(student, optimizer_student)
        
        # DDP models - regular state dict (rank 0 only saves these)
        if rank == 0:
            if prototype_bank is not None:
                to_save["prototype_bank"] = prototype_bank.state_dict()
            if optimizer_prototypes is not None:
                to_save["optimizer_prototypes"] = optimizer_prototypes.state_dict()
            if dino_class_loss is not None:
                to_save["dino_class_loss"] = dino_class_loss.state_dict()
            if patch_prototype_loss is not None:
                to_save["patch_prototype_loss"] = patch_prototype_loss.state_dict()
            if fp16_scaler is not None:
                to_save["fp16_scaler"] = fp16_scaler.state_dict()
        
        # Save with DCP (parallel across ranks)
        dcp.save(
            to_save,
            storage_writer=FileSystemWriter(str(dcp_path)),
        )
        
        print(f"[Rank {rank}] DCP checkpoint saved successfully")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR during DCP checkpoint save: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Barrier to ensure all ranks finished DCP save
    if dist.is_initialized():
        dist.barrier()
    
    # ========== STEP 2: Reconstitute to .pth on CPU (rank 0 only) ==========
    if rank == 0:
        try:
            print(f"[Rank 0] Reconstituting DCP to consolidated .pth on CPU...")
            
            # Load DCP checkpoint on CPU
            to_load = {}
            dcp.load(
                to_load,
                storage_reader=FileSystemReader(str(dcp_path)),
            )
            
            # Build consolidated checkpoint dict
            checkpoint = {
                "iteration": to_load["iteration"],
                "student": to_load["student"],
                "teacher": to_load["teacher"],
                "optimizer_student": to_load["optimizer_student"],
                "args": to_load["args"],
            }
            
            # Add optional components
            if "prototype_bank" in to_load:
                checkpoint["prototype_bank"] = to_load["prototype_bank"]
            if "optimizer_prototypes" in to_load:
                checkpoint["optimizer_prototypes"] = to_load["optimizer_prototypes"]
            if "dino_class_loss" in to_load:
                checkpoint["dino_class_loss"] = to_load["dino_class_loss"]
            if "patch_prototype_loss" in to_load:
                checkpoint["patch_prototype_loss"] = to_load["patch_prototype_loss"]
            if "fp16_scaler" in to_load:
                checkpoint["fp16_scaler"] = to_load["fp16_scaler"]
            
            # Save consolidated .pth files
            numbered_path = ckpt_dir / f"checkpoint_iter_{iteration:08d}.pth"
            latest_path = ckpt_dir / "checkpoint.pth"
            
            print(f"[Rank 0] Saving consolidated checkpoint to {numbered_path}")
            torch.save(checkpoint, numbered_path)
            
            print(f"[Rank 0] Saving latest checkpoint to {latest_path}")
            torch.save(checkpoint, latest_path)
            
            print(f"[Rank 0] Consolidated checkpoints saved successfully")
            
            # Clean up old checkpoints (keep last 3 of each type)
            _cleanup_old_checkpoints(ckpt_dir, keep_last=3)
            
        except Exception as e:
            print(f"[Rank 0] WARNING: Failed to create consolidated checkpoint: {e}")
            print(f"[Rank 0] DCP checkpoint is still available for resuming")
            import traceback
            traceback.print_exc()
    
    # Final barrier
    if dist.is_initialized():
        dist.barrier()
    
    print(f"[Rank {rank}] Checkpoint save completed")


def _cleanup_old_checkpoints(ckpt_dir, keep_last=3):
    """Clean up old checkpoints, keeping only the most recent ones."""
    # Clean DCP checkpoints
    dcp_checkpoints = sorted(ckpt_dir.glob("dcp_iter_*"))
    if len(dcp_checkpoints) > keep_last:
        for old_ckpt in dcp_checkpoints[:-keep_last]:
            shutil.rmtree(old_ckpt)
            print(f"[Rank 0] Removed old DCP checkpoint: {old_ckpt}")
    
    # Clean .pth checkpoints
    pth_checkpoints = sorted(ckpt_dir.glob("checkpoint_iter_*.pth"))
    if len(pth_checkpoints) > keep_last:
        for old_ckpt in pth_checkpoints[:-keep_last]:
            os.remove(old_ckpt)
            print(f"[Rank 0] Removed old .pth checkpoint: {old_ckpt}")


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
    Load checkpoint with fallback priority:
    1. Try DCP format (fastest, for normal training resume)
    2. Fall back to consolidated .pth (for flexibility)
    
    Returns:
        iteration: Loaded iteration number, or 0 if checkpoint doesn't exist
    """
    ckpt_dir = Path(ckpt_dir)
    
    if not ckpt_dir.exists():
        print(f"No checkpoint directory found at {ckpt_dir}")
        return 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # ========== Try DCP checkpoint first (fastest) ==========
    dcp_checkpoints = sorted(ckpt_dir.glob("dcp_iter_*"))
    if dcp_checkpoints:
        latest_dcp = dcp_checkpoints[-1]
        print(f"[Rank {rank}] Loading from DCP checkpoint: {latest_dcp}")
        
        try:
            return _load_from_dcp(
                latest_dcp,
                student, teacher, prototype_bank,
                optimizer_student, optimizer_prototypes,
                dino_class_loss, patch_prototype_loss, fp16_scaler
            )
        except Exception as e:
            print(f"[Rank {rank}] Failed to load DCP checkpoint: {e}")
            print(f"[Rank {rank}] Falling back to .pth checkpoint...")
    
    # ========== Fall back to consolidated .pth checkpoint ==========
    latest_path = ckpt_dir / "checkpoint.pth"
    
    if not latest_path.exists():
        # Try numbered checkpoints
        pth_checkpoints = sorted(ckpt_dir.glob("checkpoint_iter_*.pth"))
        if not pth_checkpoints:
            print(f"[Rank {rank}] No checkpoints found in {ckpt_dir}")
            return 0
        latest_path = pth_checkpoints[-1]
    
    print(f"[Rank {rank}] Loading from consolidated .pth checkpoint: {latest_path}")
    
    try:
        return _load_from_pth(
            latest_path,
            student, teacher, prototype_bank,
            optimizer_student, optimizer_prototypes,
            dino_class_loss, patch_prototype_loss, fp16_scaler
        )
    except Exception as e:
        print(f"[Rank {rank}] ERROR loading .pth checkpoint: {e}")
        import traceback
        traceback.print_exc()
        print(f"[Rank {rank}] Starting training from scratch")
        return 0


def _load_from_dcp(
    dcp_path,
    student, teacher, prototype_bank,
    optimizer_student, optimizer_prototypes,
    dino_class_loss, patch_prototype_loss, fp16_scaler
):
    """Load from DCP checkpoint."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    to_load = {
        "iteration": 0,
        "student": get_model_state_dict(student),
        "teacher": get_model_state_dict(teacher),
        "optimizer_student": get_optimizer_state_dict(student, optimizer_student),
    }
    
    if prototype_bank is not None:
        to_load["prototype_bank"] = prototype_bank.state_dict()
    if optimizer_prototypes is not None:
        to_load["optimizer_prototypes"] = optimizer_prototypes.state_dict()
    if dino_class_loss is not None:
        to_load["dino_class_loss"] = dino_class_loss.state_dict()
    if patch_prototype_loss is not None:
        to_load["patch_prototype_loss"] = patch_prototype_loss.state_dict()
    if fp16_scaler is not None:
        to_load["fp16_scaler"] = fp16_scaler.state_dict()
    
    # Load with DCP
    dcp.load(
        to_load,
        storage_reader=FileSystemReader(str(dcp_path)),
    )
    
    # Set loaded states
    iteration = to_load["iteration"]
    set_model_state_dict(student, to_load["student"])
    set_model_state_dict(teacher, to_load["teacher"])
    
    if prototype_bank is not None:
        prototype_bank.load_state_dict(to_load["prototype_bank"])
    
    set_optimizer_state_dict(student, optimizer_student, to_load["optimizer_student"])
    
    if optimizer_prototypes is not None:
        optimizer_prototypes.load_state_dict(to_load["optimizer_prototypes"])
    
    # Load optional components
    if dino_class_loss is not None and "dino_class_loss" in to_load:
        dino_class_loss.load_state_dict(to_load["dino_class_loss"])
    if patch_prototype_loss is not None and "patch_prototype_loss" in to_load:
        patch_prototype_loss.load_state_dict(to_load["patch_prototype_loss"])
    if fp16_scaler is not None and "fp16_scaler" in to_load:
        fp16_scaler.load_state_dict(to_load["fp16_scaler"])
    
    print(f"[Rank {rank}] Loaded DCP checkpoint from iteration {iteration}")
    return iteration


def _load_from_pth(
    pth_path,
    student, teacher, prototype_bank,
    optimizer_student, optimizer_prototypes,
    dino_class_loss, patch_prototype_loss, fp16_scaler
):
    """Load from consolidated .pth checkpoint."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Load checkpoint on CPU
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    iteration = checkpoint.get("iteration", 0)
    
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
    
    print(f"[Rank {rank}] Loaded .pth checkpoint from iteration {iteration}")
    return iteration
