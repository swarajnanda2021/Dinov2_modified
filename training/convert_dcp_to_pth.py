#!/usr/bin/env python3
"""
Convert DCP checkpoints to regular PyTorch checkpoints.
Place in training/ folder and run from there or project root.

Usage:
    cd training/
    python convert_dcp_to_pth.py
    
    # Or convert specific iteration
    python convert_dcp_to_pth.py --iteration 54000
    
    # Or from project root
    python training/convert_dcp_to_pth.py
"""

import os
import sys
import argparse
from pathlib import Path
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
    print("ERROR: torch.distributed.checkpoint not available!")
    print("You need PyTorch 2.0+ with DCP support")
    sys.exit(1)


def init_fake_distributed():
    """Initialize minimal distributed context for DCP loading."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            print("✓ Initialized fake distributed context for DCP loading")
        except Exception as e:
            print(f"Warning: Could not init distributed: {e}")
            print("Attempting to load without distributed context...")


def find_dcp_checkpoints():
    """Find all DCP checkpoint directories relative to script location."""
    # From training/ folder, logs/ is at ../logs/
    logs_dir = script_dir.parent / "logs"
    checkpoint_dir = logs_dir / "checkpoint_fsdp2"
    
    if not checkpoint_dir.exists():
        print(f"ERROR: Checkpoint directory not found: {checkpoint_dir}")
        print(f"Expected structure: project_root/logs/checkpoint_fsdp2/dcp_iter_*/")
        return []
    
    # Find all dcp_iter_* directories
    dcp_checkpoints = sorted(checkpoint_dir.glob("dcp_iter_*"))
    
    return dcp_checkpoints


def extract_iteration_from_path(dcp_path):
    """Extract iteration number from dcp_iter_XXXXX path."""
    try:
        return int(dcp_path.name.replace("dcp_iter_", ""))
    except ValueError:
        return None


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
    
    # Output filename: checkpoint_iter_00054000.pth
    output_filename = f"checkpoint_iter_{iteration:08d}.pth"
    output_path = output_dir / output_filename
    
    print(f"\n{'='*60}")
    print(f"Converting DCP checkpoint from iteration {iteration}")
    print(f"  Source: {dcp_dir}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")
    
    # Skip if already exists
    if output_path.exists():
        print(f"⚠️  Output file already exists, skipping...")
        return True
    
    try:
        # Initialize distributed if needed
        init_fake_distributed()
        
        # Prepare empty state dict structure
        state_dict = {
            "iteration": 0,
            "student": {},
            "teacher": {},
            "prototype_bank": {},
            "optimizer_student": {},
            "optimizer_prototypes": {},
            "args": {},
        }
        
        # Load from DCP
        print("Loading DCP checkpoint...")
        dcp.load(
            state_dict,
            storage_reader=FileSystemReader(str(dcp_dir)),
        )
        
        print("✓ DCP load complete")
        
        # Extract and restructure
        regular_checkpoint = {
            "iteration": state_dict["iteration"],
            "args": state_dict.get("args", {}),
        }
        
        # Extract model states (unwrap DCP structure)
        if "student" in state_dict and state_dict["student"]:
            if isinstance(state_dict["student"], dict) and "state" in state_dict["student"]:
                regular_checkpoint["student"] = state_dict["student"]["state"]
            else:
                regular_checkpoint["student"] = state_dict["student"]
            print(f"  ✓ Student: {len(regular_checkpoint['student'])} keys")
        
        if "teacher" in state_dict and state_dict["teacher"]:
            if isinstance(state_dict["teacher"], dict) and "state" in state_dict["teacher"]:
                regular_checkpoint["teacher"] = state_dict["teacher"]["state"]
            else:
                regular_checkpoint["teacher"] = state_dict["teacher"]
            print(f"  ✓ Teacher: {len(regular_checkpoint['teacher'])} keys")
        
        # Prototype bank (DDP model - already unwrapped)
        if "prototype_bank" in state_dict and state_dict["prototype_bank"]:
            regular_checkpoint["prototype_bank"] = state_dict["prototype_bank"]
            print(f"  ✓ Prototype bank: {len(regular_checkpoint['prototype_bank'])} keys")
        
        # Extract optimizer states (unwrap DCP structure)
        if "optimizer_student" in state_dict and state_dict["optimizer_student"]:
            if isinstance(state_dict["optimizer_student"], dict) and "state" in state_dict["optimizer_student"]:
                regular_checkpoint["optimizer_student"] = state_dict["optimizer_student"]["state"]
            else:
                regular_checkpoint["optimizer_student"] = state_dict["optimizer_student"]
            print(f"  ✓ Optimizer student")
        
        if "optimizer_prototypes" in state_dict and state_dict["optimizer_prototypes"]:
            regular_checkpoint["optimizer_prototypes"] = state_dict["optimizer_prototypes"]
            print(f"  ✓ Optimizer prototypes")
        
        # Optional components
        for key in ["dino_class_loss", "patch_prototype_loss", "fp16_scaler"]:
            if key in state_dict and state_dict[key]:
                regular_checkpoint[key] = state_dict[key]
                print(f"  ✓ {key}")
        
        # Save as regular checkpoint
        print(f"\nSaving to {output_path}...")
        torch.save(regular_checkpoint, output_path)
        
        # Print size
        size_gb = output_path.stat().st_size / (1024**3)
        print(f"✓ Conversion complete! File size: {size_gb:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert DCP checkpoints to regular PyTorch checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all DCP checkpoints
  python convert_dcp_to_pth.py
  
  # Convert specific iteration
  python convert_dcp_to_pth.py --iteration 54000
  
  # Specify custom output directory
  python convert_dcp_to_pth.py --output_dir ../converted_checkpoints/
        """
    )
    parser.add_argument('--iteration', type=int, default=None,
                       help='Convert only this specific iteration')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ../logs/)')
    args = parser.parse_args()
    
    # Set output directory (default: ../logs/)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir.parent / "logs"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Find DCP checkpoints
    print("Searching for DCP checkpoints...")
    dcp_checkpoints = find_dcp_checkpoints()
    
    if not dcp_checkpoints:
        print("No DCP checkpoints found!")
        print(f"Expected location: {script_dir.parent / 'logs' / 'checkpoint_fsdp2' / 'dcp_iter_*'}")
        return
    
    print(f"Found {len(dcp_checkpoints)} DCP checkpoints")
    
    # Filter by iteration if specified
    if args.iteration is not None:
        dcp_checkpoints = [
            ckpt for ckpt in dcp_checkpoints 
            if extract_iteration_from_path(ckpt) == args.iteration
        ]
        
        if not dcp_checkpoints:
            print(f"ERROR: No checkpoint found for iteration {args.iteration}")
            return
        
        print(f"Converting only iteration {args.iteration}")
    
    # Convert each checkpoint
    success_count = 0
    for dcp_dir in dcp_checkpoints:
        if convert_dcp_to_regular(dcp_dir, output_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Converted: {success_count}/{len(dcp_checkpoints)} checkpoints")
    print(f"Output directory: {output_dir}")
    print(f"Files: checkpoint_iter_XXXXXXXX.pth")
    print(f"{'='*60}\n")
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
