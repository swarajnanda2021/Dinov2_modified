#!/usr/bin/env python3
"""
Convert DCP checkpoints to regular PyTorch checkpoints (DDP format).

Usage:
    python convert_dcp_to_pth.py --iteration 8000
"""

import os
import sys
import argparse
import pickle
import gc
from pathlib import Path
from copy import deepcopy
import torch
import torch.distributed as dist

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import DCP utilities
try:
    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
except ImportError:
    print("ERROR: torch.distributed.checkpoint not available (need PyTorch 2.0+)")
    sys.exit(1)

# Import model components
try:
    from models import CombinedModelDINO, ModernViT, DINOHead
except ImportError as e:
    print(f"ERROR: Could not import models: {e}")
    sys.exit(1)


def init_distributed():
    """Initialize minimal distributed context."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group(backend='gloo', rank=0, world_size=1)


def clean_fsdp_key(key: str) -> str:
    """Remove FSDP2/compile prefixes from parameter names."""
    key = key.replace("_fsdp_wrapped_module.", "")
    key = key.replace("_checkpoint_wrapped_module.", "")
    key = key.replace("parametrizations.", "")
    key = key.replace("_orig_mod.", "")
    key = key.removesuffix(".original")
    return key


def convert_state_dict_to_ddp_format(fsdp_state_dict: dict) -> dict:
    """
    Convert FSDP2 state dict to DDP format.
    - Cleans FSDP prefixes
    - Adds 'module.' prefix (DDP wrapping)
    """
    ddp_state_dict = {}
    
    for key, value in fsdp_state_dict.items():
        # Handle nested FSDP state format
        if isinstance(value, dict) and "state" in value:
            inner = value["state"]
            if isinstance(inner, dict):
                for inner_key, inner_value in inner.items():
                    clean_key = clean_fsdp_key(inner_key)
                    ddp_key = f"module.{clean_key}"
                    ddp_state_dict[ddp_key] = inner_value
            else:
                clean_key = clean_fsdp_key(key)
                ddp_key = f"module.{clean_key}"
                ddp_state_dict[ddp_key] = inner
        else:
            clean_key = clean_fsdp_key(key)
            ddp_key = f"module.{clean_key}"
            ddp_state_dict[ddp_key] = value
    
    return ddp_state_dict


def dict_to_namespace(d: dict) -> argparse.Namespace:
    """Convert dict to argparse.Namespace."""
    return argparse.Namespace(**d)


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    torch.cuda.empty_cache()
    gc.collect()


def apply_fsdp_wrapping(student, teacher):
    """Apply FSDP2 wrapping to match checkpoint structure."""
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    
    world_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy, "reshard_after_forward": True}
    
    # Wrap student
    for block in student.backbone.blocks:
        block = fully_shard(block, **fsdp_config)
    student.backbone = fully_shard(student.backbone, **fsdp_config)
    student.classhead = fully_shard(student.classhead, **fsdp_config)
    student.patchhead = fully_shard(student.patchhead, **fsdp_config)
    
    # Wrap teacher
    for block in teacher.backbone.blocks:
        block = fully_shard(block, **fsdp_config)
    teacher.backbone = fully_shard(teacher.backbone, **fsdp_config)
    teacher.classhead = fully_shard(teacher.classhead, **fsdp_config)
    teacher.patchhead = fully_shard(teacher.patchhead, **fsdp_config)
    
    return student, teacher


def create_models(dcp_dir):
    """Create models with architecture from checkpoint."""
    # Try loading args
    state_dict = {"iteration": 0, "args": {}}
    dcp.load(state_dict, storage_reader=FileSystemReader(str(dcp_dir)))
    
    iteration = state_dict.get("iteration", 0)
    args = state_dict.get("args", {})
    
    # Default to ViT-L if no args
    if len(args) == 0:
        print(f"  No args in checkpoint, using ViT-L defaults")
        args = {
            'patch_size': 16, 'embeddingdim': 1024, 'vitheads': 16, 
            'vitdepth': 24, 'out_dim': 65536, 'norm_last_layer': True,
            'use_bn_in_head': False, 'num_masks': 0
        }
    
    # Extract architecture params
    patch_size = args.get('patch_size', 16)
    embed_dim = args.get('embeddingdim', 1024)
    num_heads = args.get('vitheads', 16)
    depth = args.get('vitdepth', 24)
    out_dim = args.get('out_dim', 65536)
    
    print(f"  Architecture: ViT embed_dim={embed_dim}, depth={depth}, heads={num_heads}")
    
    # Create models
    student_encoder = ModernViT(
        img_size=224, patch_size=patch_size, embed_dim=embed_dim,
        depth=depth, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
        qk_norm=False, dual_norm=False, drop_path_rate=0.4,
        pre_norm=False, num_register_tokens=4
    )
    
    teacher_encoder = deepcopy(student_encoder)
    
    student = CombinedModelDINO(
        backbone=student_encoder,
        classhead=DINOHead(embed_dim, out_dim, use_bn=args.get('use_bn_in_head', False), 
                          norm_last_layer=args.get('norm_last_layer', True)),
        patchhead=DINOHead(embed_dim, out_dim, use_bn=args.get('use_bn_in_head', False),
                          norm_last_layer=args.get('norm_last_layer', True)),
        num_masks=args.get('num_masks', 0),
        patch_size=patch_size
    )
    
    teacher = CombinedModelDINO(
        backbone=teacher_encoder,
        classhead=DINOHead(embed_dim, out_dim, use_bn=args.get('use_bn_in_head', False)),
        patchhead=DINOHead(embed_dim, out_dim, use_bn=args.get('use_bn_in_head', False)),
        num_masks=args.get('num_masks', 0),
        patch_size=patch_size
    )
    
    if torch.cuda.is_available():
        student = student.cuda()
        teacher = teacher.cuda()
    
    return student, teacher, iteration, args


def convert_checkpoint(dcp_dir, output_dir):
    """Convert DCP checkpoint to regular PyTorch checkpoint (DDP format)."""
    iteration = int(dcp_dir.name.replace("dcp_iter_", ""))
    output_path = output_dir / f"checkpoint_iter_{iteration:08d}.pth"
    
    print(f"\n{'='*70}")
    print(f"Converting iteration {iteration}")
    print(f"{'='*70}")
    
    if output_path.exists():
        print(f"  Output exists: {output_path.name}")
        response = input("  Overwrite? [y/N]: ")
        if response.lower() != 'y':
            return True
    
    # Variables to track for cleanup
    student = None
    teacher = None
    to_load = None
    student_state = None
    teacher_state = None
    checkpoint = None
    
    try:
        # Initialize
        init_distributed()
        
        # Create models
        print("  Creating models...")
        student, teacher, checkpoint_iter, args = create_models(dcp_dir)
        
        # Apply FSDP2
        print("  Applying FSDP2 wrapping...")
        student, teacher = apply_fsdp_wrapping(student, teacher)
        
        # Load checkpoint
        print("  Loading from DCP...")
        to_load = {
            "iteration": 0,
            "args": {},
            "student": get_model_state_dict(student),
            "teacher": get_model_state_dict(teacher),
        }
        dcp.load(to_load, storage_reader=FileSystemReader(str(dcp_dir)))
        
        # Convert to DDP format
        print("  Converting to DDP format...")
        student_state = convert_state_dict_to_ddp_format(to_load["student"])
        teacher_state = convert_state_dict_to_ddp_format(to_load["teacher"])
        
        # Convert args dict to Namespace
        args_dict = to_load.get("args", args)
        if isinstance(args_dict, dict):
            args_namespace = dict_to_namespace(args_dict)
        else:
            args_namespace = args_dict
        
        checkpoint = {
            "iteration": to_load["iteration"],
            "args": args_namespace,
            "student": student_state,
            "teacher": teacher_state,
        }
        
        # Save
        print("  Saving...")
        torch.save(checkpoint, output_path)
        
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"  ✓ Saved: {output_path.name} ({size_gb:.2f} GB)")
        print(f"    Iteration: {checkpoint['iteration']}")
        print(f"    Parameters: {len(checkpoint['student'])} student, {len(checkpoint['teacher'])} teacher")
        
        # Print sample keys to verify format
        sample_keys = list(checkpoint['student'].keys())[:5]
        print(f"    Sample keys: {sample_keys}")
        
        # Cleanup GPU memory
        del student, teacher, to_load, student_state, teacher_state, checkpoint
        cleanup_gpu_memory()
        print("  ✓ GPU memory cleaned up")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        if student is not None:
            del student
        if teacher is not None:
            del teacher
        if to_load is not None:
            del to_load
        if student_state is not None:
            del student_state
        if teacher_state is not None:
            del teacher_state
        if checkpoint is not None:
            del checkpoint
        cleanup_gpu_memory()
        print("  ✓ GPU memory cleaned up after error")
        
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert DCP to PyTorch checkpoints (DDP format)')
    parser.add_argument('--iteration', type=int, help='Specific iteration to convert')
    parser.add_argument('--output_dir', type=str, help='Output directory (default: ../logs/)')
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = project_root / "logs" / "checkpoint_fsdp2"
    if not checkpoint_dir.exists():
        print(f"ERROR: No checkpoint directory: {checkpoint_dir}")
        return
    
    # Find checkpoints
    dcp_checkpoints = sorted(checkpoint_dir.glob("dcp_iter_*"))
    if not dcp_checkpoints:
        print("No DCP checkpoints found")
        return
    
    # Filter by iteration
    if args.iteration:
        dcp_checkpoints = [d for d in dcp_checkpoints if int(d.name.replace("dcp_iter_", "")) == args.iteration]
        if not dcp_checkpoints:
            print(f"No checkpoint found for iteration {args.iteration}")
            return
    
    print(f"Found {len(dcp_checkpoints)} checkpoint(s)")
    print(f"Output: {output_dir}")
    
    # Convert
    success = 0
    for dcp_dir in dcp_checkpoints:
        if convert_checkpoint(dcp_dir, output_dir):
            success += 1
    
    print(f"\n{'='*70}")
    print(f"Converted {success}/{len(dcp_checkpoints)} checkpoints")
    print(f"{'='*70}\n")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()