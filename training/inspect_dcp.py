#!/usr/bin/env python3
"""
Inspect DCP checkpoint files using proper DCP APIs.
"""

import os
import sys
import pickle
from pathlib import Path
import torch
import torch.distributed as dist

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

# Import DCP utilities
try:
    from torch.distributed.checkpoint import FileSystemReader
    import torch.distributed.checkpoint as dcp
except ImportError:
    print("ERROR: torch.distributed.checkpoint not available!")
    print("You need PyTorch 2.0+ with DCP support")
    sys.exit(1)


def init_fake_distributed():
    """Initialize minimal distributed context for DCP inspection."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        
        try:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
            print("✓ Initialized fake distributed context\n")
        except Exception as e:
            print(f"Warning: Could not init distributed: {e}\n")


def inspect_metadata_file(metadata_file):
    """Inspect .metadata file (pickle format)."""
    print(f"\n{'='*60}")
    print("METADATA FILE ANALYSIS")
    print(f"{'='*60}\n")
    
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Type: {type(metadata)}")
        print(f"Size: {metadata_file.stat().st_size / (1024**2):.2f} MB\n")
        
        if hasattr(metadata, 'state_dict_metadata'):
            state_dict_meta = metadata.state_dict_metadata
            print(f"State dict keys: {len(state_dict_meta)}")
            
            # Categorize keys
            param_keys = []
            args_keys = []
            scalar_keys = []
            optimizer_keys = []
            
            for key in state_dict_meta.keys():
                if key.startswith('student.') or key.startswith('teacher.'):
                    param_keys.append(key)
                elif key.startswith('args.'):
                    args_keys.append(key)
                elif key.startswith('optimizer_'):
                    optimizer_keys.append(key)
                else:
                    scalar_keys.append(key)
            
            print(f"\nKey categories:")
            print(f"  Student/Teacher parameters: {len(param_keys)}")
            print(f"  Args: {len(args_keys)}")
            print(f"  Optimizer states: {len(optimizer_keys)}")
            print(f"  Other/Scalar: {len(scalar_keys)}")
            
            print(f"\nSample student parameters (first 10):")
            for key in sorted(param_keys)[:10]:
                print(f"  - {key}")
            
            print(f"\nSample args (first 10):")
            for key in sorted(args_keys)[:10]:
                print(f"  - {key}")
                
            print(f"\nScalar keys:")
            for key in sorted(scalar_keys):
                print(f"  - {key}")
            
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        import traceback
        traceback.print_exc()


def inspect_dcp_checkpoint(dcp_dir):
    """Inspect DCP checkpoint using DCP API."""
    print(f"\n{'='*60}")
    print(f"Inspecting DCP Checkpoint: {dcp_dir}")
    print(f"{'='*60}\n")
    
    # List files
    files = sorted(dcp_dir.glob("*"))
    print("Files in checkpoint:")
    total_size = 0
    for f in files:
        size = f.stat().st_size / (1024**2)
        total_size += size
        print(f"  {f.name:30s} {size:8.1f} MB")
    print(f"\nTotal size: {total_size/1024:.2f} GB\n")
    
    # Inspect metadata file
    metadata_file = dcp_dir / ".metadata"
    if metadata_file.exists():
        inspect_metadata_file(metadata_file)
    
    # Try loading checkpoint content without model objects
    print(f"\n{'='*60}")
    print("LOADING SCALAR VALUES")
    print(f"{'='*60}\n")
    
    try:
        init_fake_distributed()
        
        # Load just scalar values (iteration, args, etc.)
        state_dict = {
            "iteration": 0,
            "args": {},
        }
        
        # Try to load what we can
        print("Attempting to load scalars and args...")
        try:
            dcp.load(
                state_dict,
                storage_reader=FileSystemReader(str(dcp_dir)),
            )
            
            print(f"✓ Successfully loaded\n")
            
            if "iteration" in state_dict:
                print(f"Iteration: {state_dict['iteration']}")
            
            if "args" in state_dict and isinstance(state_dict['args'], dict):
                print(f"\nArgs ({len(state_dict['args'])} items):")
                print(f"\nTraining configuration:")
                for key in sorted(state_dict['args'].keys())[:30]:
                    value = state_dict['args'][key]
                    print(f"  {key}: {value}")
                
                if len(state_dict['args']) > 30:
                    print(f"  ... and {len(state_dict['args']) - 30} more args")
        
        except Exception as e:
            print(f"Could not load scalars: {e}")
        
        # Summary from metadata
        print(f"\n{'='*60}")
        print("CHECKPOINT CONTENT SUMMARY (from metadata)")
        print(f"{'='*60}\n")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        state_dict_meta = metadata.state_dict_metadata
        
        # Count parameters by module
        student_params = [k for k in state_dict_meta.keys() if k.startswith('student.')]
        teacher_params = [k for k in state_dict_meta.keys() if k.startswith('teacher.')]
        optimizer_params = [k for k in state_dict_meta.keys() if k.startswith('optimizer_')]
        prototype_params = [k for k in state_dict_meta.keys() if k.startswith('prototype_bank.')]
        
        print(f"Model components:")
        print(f"  Student parameters: {len(student_params)}")
        print(f"  Teacher parameters: {len(teacher_params)}")
        print(f"  Optimizer states: {len(optimizer_params)}")
        print(f"  Prototype bank: {len(prototype_params)}")
        
        # Show student structure
        print(f"\nStudent model structure (top-level modules):")
        student_modules = {}
        for key in student_params:
            parts = key.split('.')
            if len(parts) >= 2:
                module = parts[1]  # After 'student.'
                student_modules[module] = student_modules.get(module, 0) + 1
        
        for module, count in sorted(student_modules.items()):
            print(f"  student.{module}: {count} parameters")
        
        print(f"\nCheckpoint successfully analyzed!")
        print(f"Note: Full parameter inspection requires loading with actual model objects")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect DCP checkpoint structure')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Specific iteration to inspect (default: latest)')
    args = parser.parse_args()
    
    # Determine checkpoint directory
    if args.checkpoint:
        dcp_dir = Path(args.checkpoint)
    else:
        checkpoint_base = project_root / "logs" / "checkpoint_fsdp2"
        
        if args.iteration:
            dcp_dir = checkpoint_base / f"dcp_iter_{args.iteration}"
        else:
            # Find latest
            dcp_checkpoints = sorted(checkpoint_base.glob("dcp_iter_*"))
            if not dcp_checkpoints:
                print(f"ERROR: No checkpoints found in {checkpoint_base}")
                sys.exit(1)
            dcp_dir = dcp_checkpoints[-1]
    
    if not dcp_dir.exists():
        print(f"ERROR: {dcp_dir} not found!")
        sys.exit(1)
    
    inspect_dcp_checkpoint(dcp_dir)
