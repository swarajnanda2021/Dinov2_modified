"""
DCP-based checkpointing for FSDP2 models.
"""

import os
import tempfile
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
    Save FSDP2 checkpoint using DCP.
    
    Args:
        ckpt_dir: Directory to save checkpoint
        iteration: Current iteration
        student: FSDP2-wrapped student model
        teacher: FSDP2-wrapped teacher model
        prototype_bank: DDP-wrapped prototype bank
        optimizer_student: Student optimizer
        optimizer_prototypes: Prototype optimizer
        args: Training arguments
        dino_class_loss: DINO loss state
        patch_prototype_loss: Prototype loss state
        fp16_scaler: Optional gradient scaler
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Create temp directory
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir_tmp = Path(tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=f"{ckpt_dir.name}_"))
    
    # Prepare state dict
    to_save = {
        "iteration": iteration,
        "student": get_model_state_dict(student),
        "teacher": get_model_state_dict(teacher),
        "prototype_bank": prototype_bank.state_dict(),  # Regular DDP state dict
        "optimizer_student": get_optimizer_state_dict(student, optimizer_student),
        "optimizer_prototypes": optimizer_prototypes.state_dict(),
        "args": args,
    }
    
    # Add optional components
    if dino_class_loss is not None:
        to_save["dino_class_loss"] = dino_class_loss.state_dict()
    if patch_prototype_loss is not None:
        to_save["patch_prototype_loss"] = patch_prototype_loss.state_dict()
    if fp16_scaler is not None:
        to_save["fp16_scaler"] = fp16_scaler.state_dict()
    
    # Save with DCP
    dcp.save(
        to_save,
        storage_writer=FileSystemWriter(ckpt_dir_tmp),
    )
    
    # Atomically rename temp to final
    if rank == 0:
        if ckpt_dir.exists():
            import shutil
            shutil.rmtree(ckpt_dir)
        ckpt_dir_tmp.rename(ckpt_dir)
    
    if dist.is_initialized():
        dist.barrier()
    
    print(f"[Rank {rank}] Saved FSDP2 checkpoint at iteration {iteration}")


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
    if not ckpt_dir.exists():
        print(f"No checkpoint found at {ckpt_dir}")
        return 0
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Prepare state dict
    to_load = {
        "iteration": 0,
        "student": get_model_state_dict(student),
        "teacher": get_model_state_dict(teacher),
        "prototype_bank": prototype_bank.state_dict(),
        "optimizer_student": get_optimizer_state_dict(student, optimizer_student),
        "optimizer_prototypes": optimizer_prototypes.state_dict(),
    }
    
    # Add optional components
    if dino_class_loss is not None:
        to_load["dino_class_loss"] = dino_class_loss.state_dict()
    if patch_prototype_loss is not None:
        to_load["patch_prototype_loss"] = patch_prototype_loss.state_dict()
    if fp16_scaler is not None:
        to_load["fp16_scaler"] = fp16_scaler.state_dict()
    
    # Load with DCP
    dcp.load(
        to_load,
        storage_reader=FileSystemReader(ckpt_dir),
    )
    
    # Set loaded states
    iteration = to_load["iteration"]
    set_model_state_dict(student, to_load["student"])
    set_model_state_dict(teacher, to_load["teacher"])
    prototype_bank.load_state_dict(to_load["prototype_bank"])
    set_optimizer_state_dict(student, optimizer_student, to_load["optimizer_student"])
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