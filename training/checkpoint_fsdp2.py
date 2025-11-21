"""
DCP-based checkpointing for FSDP2 models.
"""

import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict


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
    Save FSDP2 checkpoint using DCP with better error handling.
    Handles optional prototype_bank and optimizer_prototypes (can be None).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    print(f"[Rank {rank}] Starting checkpoint save at iteration {iteration}")
    
    # Create checkpoint directory if it doesn't exist
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Use iteration number in the checkpoint name
    ckpt_path = ckpt_dir / f"dcp_iter_{iteration}"
    
    try:
        # Prepare state dict
        print(f"[Rank {rank}] Preparing state dicts...")
        
        to_save = {
            "iteration": iteration,
            "args": vars(args) if hasattr(args, '__dict__') else args,
        }
        
        # FSDP2 models - use DCP state dict functions
        print(f"[Rank {rank}] Getting FSDP2 model states...")
        to_save["student"] = get_model_state_dict(student)
        to_save["teacher"] = get_model_state_dict(teacher)
        
        print(f"[Rank {rank}] Getting optimizer states...")
        to_save["optimizer_student"] = get_optimizer_state_dict(student, optimizer_student)
        
        # DDP model - use regular state dict (rank 0 only)
        if rank == 0:
            print(f"[Rank {rank}] Getting prototype bank state...")
            if prototype_bank is not None:
                to_save["prototype_bank"] = prototype_bank.state_dict()
            if optimizer_prototypes is not None:
                to_save["optimizer_prototypes"] = optimizer_prototypes.state_dict()
                
            # Optional components
            if dino_class_loss is not None:
                to_save["dino_class_loss"] = dino_class_loss.state_dict()
            if patch_prototype_loss is not None:
                to_save["patch_prototype_loss"] = patch_prototype_loss.state_dict()
            if fp16_scaler is not None:
                to_save["fp16_scaler"] = fp16_scaler.state_dict()
        
        # Save with DCP
        print(f"[Rank {rank}] Calling DCP save to {ckpt_path}...")
        dcp.save(
            to_save,
            storage_writer=FileSystemWriter(str(ckpt_path)),
        )
        
        print(f"[Rank {rank}] DCP save completed")
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR during checkpoint save: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Synchronize all ranks
    if dist.is_initialized():
        print(f"[Rank {rank}] Waiting at barrier...")
        dist.barrier()
    
    print(f"[Rank {rank}] Checkpoint save completed successfully")


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
    Load FSDP2 checkpoint using DCP.
    
    Returns:
        iteration: Loaded iteration number, or 0 if checkpoint doesn't exist
    """
    ckpt_dir = Path(ckpt_dir)
    
    # Look for DCP checkpoints
    dcp_checkpoints = sorted(ckpt_dir.glob("dcp_iter_*"))
    if not dcp_checkpoints:
        dcp_checkpoints = sorted(ckpt_dir.glob("iter_*"))
    
    if not dcp_checkpoints:
        print(f"No checkpoint found in {ckpt_dir}")
        return 0
    
    latest_ckpt = dcp_checkpoints[-1]
    print(f"Loading checkpoint from {latest_ckpt}")
    
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
        
    # Add optional components
    if dino_class_loss is not None:
        to_load["dino_class_loss"] = dino_class_loss.state_dict()
    if patch_prototype_loss is not None:
        to_load["patch_prototype_loss"] = patch_prototype_loss.state_dict()
    if fp16_scaler is not None:
        to_load["fp16_scaler"] = fp16_scaler.state_dict()
    
    # Load with DCP - wrap in try/except to handle missing keys
    try:
        dcp.load(
            to_load,
            storage_reader=FileSystemReader(str(latest_ckpt)),
        )
    except RuntimeError as e:
        if "Missing key in checkpoint state_dict" in str(e):
            print(f"[Rank {rank}] WARNING: Checkpoint has mismatched keys (likely due to frozen parameters)")
            print(f"[Rank {rank}] Attempting to load with no_dist_check=True...")
            
            # Try loading with more lenient settings
            # First, load just the model states
            model_to_load = {
                "iteration": 0,
                "student": get_model_state_dict(student),
                "teacher": get_model_state_dict(teacher),
            }
            
            if prototype_bank is not None:
                model_to_load["prototype_bank"] = prototype_bank.state_dict()
            
            dcp.load(
                model_to_load,
                storage_reader=FileSystemReader(str(latest_ckpt)),
            )
            
            # Set model states
            iteration = model_to_load["iteration"]
            set_model_state_dict(student, model_to_load["student"])
            set_model_state_dict(teacher, model_to_load["teacher"])
            
            if prototype_bank is not None:
                prototype_bank.load_state_dict(model_to_load["prototype_bank"])
            
            print(f"[Rank {rank}] ✓ Loaded model states successfully")
            print(f"[Rank {rank}] ⚠ Skipped optimizer state due to mismatch - training will continue with fresh optimizer state")
            
            return iteration
        else:
            raise
    
    # Normal path - all states loaded successfully
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
    
    print(f"[Rank {rank}] Loaded FSDP2 checkpoint from iteration {iteration}")
    
    return iteration
