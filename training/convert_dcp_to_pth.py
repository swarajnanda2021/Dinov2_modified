#!/usr/bin/env python3
"""
Convert DCP checkpoints to regular PyTorch checkpoints.
Properly handles FSDP2 state dicts by creating actual model objects.

Usage:
    cd training/
    python convert_dcp_to_pth.py
    
    # Or convert specific iteration
    python convert_dcp_to_pth.py --iteration 8000
    
    # Or from project root
    python training/convert_dcp_to_pth.py
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}\n")

# Import DCP utilities
try:
    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
    print("✓ DCP imports successful\n")
except ImportError:
    print("ERROR: torch.distributed.checkpoint not available!")
    print("You need PyTorch 2.0+ with DCP support")
    sys.exit(1)

# Import model components
try:
    from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
    from training.fsdp_setup import apply_fsdp_wrapping
    print("✓ Model imports successful\n")
except ImportError as e:
    print(f"ERROR: Could not import model components: {e}")
    print("Make sure you're running from the project root or training/ folder")
    sys.exit(1)


def init_fake_distributed():
    """Initialize minimal distributed context for DCP loading."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12357'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            print("✓ Initialized fake distributed context (required for DCP)")
            return True
        except Exception as e:
            print(f"ERROR: Could not init distributed: {e}")
            return False
    return True


def extract_iteration_from_path(dcp_path):
    """Extract iteration number from dcp_iter_XXXXX path."""
    try:
        return int(dcp_path.name.replace("dcp_iter_", ""))
    except ValueError:
        return None


def load_args_from_metadata(metadata_file):
    """Load args from DCP metadata file."""
    print(f"\n{'='*70}")
    print("STEP 1: Reading checkpoint metadata")
    print(f"{'='*70}")
    
    print(f"Loading metadata from: {metadata_file}")
    
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"✓ Metadata loaded successfully")
        print(f"  Type: {type(metadata)}")
        
        # Extract args from metadata
        if hasattr(metadata, 'state_dict_metadata'):
            state_dict_meta = metadata.state_dict_metadata
            
            # Count components
            args_keys = [k for k in state_dict_meta.keys() if k.startswith('args.')]
            student_keys = [k for k in state_dict_meta.keys() if k.startswith('student.')]
            teacher_keys = [k for k in state_dict_meta.keys() if k.startswith('teacher.')]
            
            print(f"  Total keys in checkpoint: {len(state_dict_meta)}")
            print(f"    - Args: {len(args_keys)}")
            print(f"    - Student parameters: {len(student_keys)}")
            print(f"    - Teacher parameters: {len(teacher_keys)}")
            
            return True
        else:
            print("ERROR: Could not find state_dict_metadata in metadata")
            return False
            
    except Exception as e:
        print(f"ERROR loading metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_models_from_checkpoint(dcp_dir):
    """Create model objects based on checkpoint configuration."""
    print(f"\n{'='*70}")
    print("STEP 2: Creating model objects")
    print(f"{'='*70}")
    
    # Try to load args from checkpoint
    print("Attempting to load args from checkpoint...")
    
    state_dict = {
        "iteration": 0,
        "args": {},
    }
    
    try:
        dcp.load(
            state_dict,
            storage_reader=FileSystemReader(str(dcp_dir)),
        )
        
        iteration = state_dict.get("iteration", 0)
        args_dict = state_dict.get("args", {})
        
        print(f"✓ Loaded checkpoint info")
        print(f"  Iteration: {iteration}")
        print(f"  Args available: {len(args_dict)} items")
        
        if len(args_dict) == 0:
            print("WARNING: No args found in checkpoint, using default ViT-L config")
            # Default to ViT-L configuration
            args_dict = {
                'patch_size': 16,
                'embeddingdim': 1024,
                'vitheads': 16,
                'vitdepth': 24,
                'out_dim': 65536,
                'norm_last_layer': True,
                'use_bn_in_head': False,
                'num_masks': 0,
            }
        
        # Extract key architecture parameters
        patch_size = args_dict.get('patch_size', 16)
        embed_dim = args_dict.get('embeddingdim', 1024)
        num_heads = args_dict.get('vitheads', 16)
        depth = args_dict.get('vitdepth', 24)
        out_dim = args_dict.get('out_dim', 65536)
        norm_last_layer = args_dict.get('norm_last_layer', True)
        use_bn_in_head = args_dict.get('use_bn_in_head', False)
        num_masks = args_dict.get('num_masks', 0)
        
        print(f"\nModel architecture:")
        print(f"  Patch size: {patch_size}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Num heads: {num_heads}")
        print(f"  Depth: {depth}")
        print(f"  Output dim: {out_dim}")
        print(f"  Num masks: {num_masks}")
        
        # Create models
        print("\nCreating student encoder...")
        student_encoder = ModernViT(
            img_size=224,
            patch_size=patch_size,
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
        print("✓ Student encoder created")
        
        print("Creating teacher encoder...")
        teacher_encoder = deepcopy(student_encoder)
        print("✓ Teacher encoder created")
        
        print("Creating projection heads...")
        student_classhead = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        
        teacher_classhead = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
        )
        
        student_patchhead = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
            norm_last_layer=norm_last_layer,
        )
        
        teacher_patchhead = DINOHead(
            embed_dim,
            out_dim,
            use_bn=use_bn_in_head,
        )
        print("✓ Projection heads created")
        
        print("Wrapping in CombinedModelDINO...")
        student = CombinedModelDINO(
            backbone=student_encoder,
            classhead=student_classhead,
            patchhead=student_patchhead,
            num_masks=num_masks,
            patch_size=patch_size,
        )
        
        teacher = CombinedModelDINO(
            backbone=teacher_encoder,
            classhead=teacher_classhead,
            patchhead=teacher_patchhead,
            num_masks=num_masks,
            patch_size=patch_size,
        )
        print("✓ Models wrapped")
        
        # Move to CUDA if available
        if torch.cuda.is_available():
            print("Moving models to CUDA...")
            student = student.cuda()
            teacher = teacher.cuda()
            print("✓ Models on CUDA")
        
        return student, teacher, iteration, args_dict
        
    except Exception as e:
        print(f"ERROR creating models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0, {}


def convert_dcp_to_regular(dcp_dir, output_dir):
    """
    Convert a DCP checkpoint to regular PyTorch checkpoint.
    
    Args:
        dcp_dir: Path to DCP checkpoint directory
        output_dir: Directory to save output checkpoint
    """
    iteration = extract_iteration_from_path(dcp_dir)
    if iteration is None:
        print(f"ERROR: Could not extract iteration from {dcp_dir}")
        return False
    
    print(f"\n{'='*70}")
    print(f"CONVERTING CHECKPOINT FROM ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Source: {dcp_dir}")
    print(f"Output directory: {output_dir}")
    
    # Output filename
    output_filename = f"checkpoint_iter_{iteration:08d}.pth"
    output_path = output_dir / output_filename
    
    # Skip if already exists
    if output_path.exists():
        print(f"\n⚠️  Output file already exists: {output_path}")
        response = input("Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Skipping...")
            return True
    
    # Initialize distributed
    if not init_fake_distributed():
        return False
    
    # Load metadata
    metadata_file = dcp_dir / ".metadata"
    if not metadata_file.exists():
        print(f"ERROR: Metadata file not found: {metadata_file}")
        return False
    
    if not load_args_from_metadata(metadata_file):
        return False
    
    # Create models
    student, teacher, checkpoint_iteration, args_dict = create_models_from_checkpoint(dcp_dir)
    
    if student is None or teacher is None:
        print("ERROR: Failed to create models")
        return False
    
    print(f"\n✓ Models created successfully for iteration {checkpoint_iteration}")
    
    # Apply FSDP2 wrapping (required to match checkpoint structure)
    print(f"\n{'='*70}")
    print("STEP 3: Applying FSDP2 wrapping")
    print(f"{'='*70}")
    
    try:
        # Create a minimal args object for FSDP wrapping
        class Args:
            pass
        
        args = Args()
        args.grad_checkpointing = False
        
        print("Applying FSDP2 wrapping to match checkpoint structure...")
        student, teacher = apply_fsdp_wrapping(student, teacher, args)
        print("✓ FSDP2 wrapping applied")
        
    except Exception as e:
        print(f"ERROR during FSDP wrapping: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load checkpoint
    print(f"\n{'='*70}")
    print("STEP 4: Loading checkpoint data")
    print(f"{'='*70}")
    
    try:
        print("Preparing state dict structures...")
        
        to_load = {
            "iteration": 0,
            "args": {},
            "student": get_model_state_dict(student),
            "teacher": get_model_state_dict(teacher),
        }
        
        print(f"  - iteration: scalar")
        print(f"  - args: dict")
        print(f"  - student: {len(to_load['student'])} keys")
        print(f"  - teacher: {len(to_load['teacher'])} keys")
        
        print(f"\nLoading from DCP checkpoint...")
        dcp.load(
            to_load,
            storage_reader=FileSystemReader(str(dcp_dir)),
        )
        
        print(f"✓ DCP load complete")
        print(f"  Iteration loaded: {to_load['iteration']}")
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Extract regular state dicts
    print(f"\n{'='*70}")
    print("STEP 5: Extracting and unwrapping state dicts")
    print(f"{'='*70}")
    
    try:
        regular_checkpoint = {
            "iteration": to_load["iteration"],
            "args": to_load.get("args", {}),
        }
        
        print(f"Iteration: {regular_checkpoint['iteration']}")
        print(f"Args: {len(regular_checkpoint['args'])} items")
        
        # Unwrap FSDP2 state dicts
        print("\nUnwrapping student state dict...")
        if isinstance(to_load["student"], dict) and "state" in to_load["student"]:
            regular_checkpoint["student"] = to_load["student"]["state"]
            print(f"  ✓ Student unwrapped: {len(regular_checkpoint['student'])} parameters")
        else:
            regular_checkpoint["student"] = to_load["student"]
            print(f"  ✓ Student: {len(regular_checkpoint['student'])} parameters")
        
        print("Unwrapping teacher state dict...")
        if isinstance(to_load["teacher"], dict) and "state" in to_load["teacher"]:
            regular_checkpoint["teacher"] = to_load["teacher"]["state"]
            print(f"  ✓ Teacher unwrapped: {len(regular_checkpoint['teacher'])} parameters")
        else:
            regular_checkpoint["teacher"] = to_load["teacher"]
            print(f"  ✓ Teacher: {len(regular_checkpoint['teacher'])} parameters")
        
        # Show sample keys
        print("\nSample student parameters:")
        for i, key in enumerate(list(regular_checkpoint['student'].keys())[:5]):
            value = regular_checkpoint['student'][key]
            if hasattr(value, 'shape'):
                print(f"  [{i+1}] {key}: {value.shape}")
            else:
                print(f"  [{i+1}] {key}: {type(value)}")
        
    except Exception as e:
        print(f"ERROR extracting state dicts: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save checkpoint
    print(f"\n{'='*70}")
    print("STEP 6: Saving regular PyTorch checkpoint")
    print(f"{'='*70}")
    
    try:
        print(f"Saving to: {output_path}")
        torch.save(regular_checkpoint, output_path)
        
        # Verify file was created
        if output_path.exists():
            size_gb = output_path.stat().st_size / (1024**3)
            print(f"\n✓ Checkpoint saved successfully!")
            print(f"  File: {output_path}")
            print(f"  Size: {size_gb:.2f} GB")
            print(f"  Iteration: {regular_checkpoint['iteration']}")
            
            return True
        else:
            print(f"ERROR: File was not created")
            return False
        
    except Exception as e:
        print(f"ERROR saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def find_dcp_checkpoints():
    """Find all DCP checkpoint directories."""
    checkpoint_dir = script_dir.parent / "logs" / "checkpoint_fsdp2"
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        return []
    
    dcp_checkpoints = sorted(checkpoint_dir.glob("dcp_iter_*"))
    return dcp_checkpoints


def main():
    parser = argparse.ArgumentParser(
        description='Convert DCP checkpoints to regular PyTorch checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all DCP checkpoints
  python convert_dcp_to_pth.py
  
  # Convert specific iteration
  python convert_dcp_to_pth.py --iteration 8000
  
  # Specify custom output directory
  python convert_dcp_to_pth.py --output_dir ../converted_checkpoints/
        """
    )
    parser.add_argument('--iteration', type=int, default=None,
                       help='Convert only this specific iteration')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../logs/)')
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir.parent / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("DCP TO PYTORCH CHECKPOINT CONVERTER")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}\n")
    
    # Find DCP checkpoints
    print("Searching for DCP checkpoints...")
    dcp_checkpoints = find_dcp_checkpoints()
    
    if not dcp_checkpoints:
        print("No DCP checkpoints found!")
        print(f"Expected location: {script_dir.parent / 'logs' / 'checkpoint_fsdp2' / 'dcp_iter_*'}")
        return
    
    print(f"✓ Found {len(dcp_checkpoints)} DCP checkpoints\n")
    
    # Show available checkpoints
    print("Available checkpoints:")
    for ckpt in dcp_checkpoints:
        iteration = extract_iteration_from_path(ckpt)
        size = sum(f.stat().st_size for f in ckpt.glob("*")) / (1024**3)
        print(f"  - Iteration {iteration:6d}: {size:.2f} GB")
    
    # Filter by iteration if specified
    if args.iteration is not None:
        dcp_checkpoints = [
            ckpt for ckpt in dcp_checkpoints 
            if extract_iteration_from_path(ckpt) == args.iteration
        ]
        
        if not dcp_checkpoints:
            print(f"\nERROR: No checkpoint found for iteration {args.iteration}")
            return
        
        print(f"\nConverting only iteration {args.iteration}")
    
    # Convert each checkpoint
    success_count = 0
    for i, dcp_dir in enumerate(dcp_checkpoints, 1):
        print(f"\n{'#'*70}")
        print(f"CHECKPOINT {i}/{len(dcp_checkpoints)}")
        print(f"{'#'*70}")
        
        if convert_dcp_to_regular(dcp_dir, output_dir):
            success_count += 1
        else:
            print(f"\n❌ Conversion failed for {dcp_dir.name}")
    
    # Summary
    print(f"\n{'='*70}")
    print("CONVERSION SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully converted: {success_count}/{len(dcp_checkpoints)} checkpoints")
    print(f"Output directory: {output_dir}")
    print(f"Output format: checkpoint_iter_XXXXXXXX.pth")
    
    if success_count == len(dcp_checkpoints):
        print(f"\n✓ All checkpoints converted successfully!")
    elif success_count > 0:
        print(f"\n⚠️  Some checkpoints failed to convert")
    else:
        print(f"\n❌ No checkpoints were converted")
    
    print(f"{'='*70}\n")
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
