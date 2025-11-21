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
    print("METADATA FILE")
    print(f"{'='*60}\n")
    
    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Type: {type(metadata)}")
        print(f"Size: {metadata_file.stat().st_size / (1024**2):.2f} MB\n")
        
        if hasattr(metadata, '__dict__'):
            print("Attributes:")
            for key, value in metadata.__dict__.items():
                print(f"  {key}: {type(value).__name__}")
                if isinstance(value, dict):
                    print(f"    Dict with {len(value)} keys")
                    if len(value) <= 10:
                        for k in list(value.keys()):
                            print(f"      - {k}")
                    else:
                        print(f"      First 10: {list(value.keys())[:10]}")
                elif isinstance(value, list):
                    print(f"    List with {len(value)} items")
                    if len(value) <= 5:
                        for item in value:
                            print(f"      - {item}")
                elif isinstance(value, (int, float, str, bool)):
                    print(f"    Value: {value}")
                print()
        else:
            print(f"Content: {metadata}")
            
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
    
    # Try loading with DCP
    print(f"\n{'='*60}")
    print("LOADING WITH DCP API")
    print(f"{'='*60}\n")
    
    try:
        init_fake_distributed()
        
        # Create empty state dict to load into
        state_dict = {}
        
        print("Loading checkpoint...")
        dcp.load(
            state_dict,
            storage_reader=FileSystemReader(str(dcp_dir)),
        )
        
        print(f"✓ Successfully loaded checkpoint\n")
        print(f"Top-level keys: {list(state_dict.keys())}\n")
        
        # Analyze each top-level key
        for key in state_dict.keys():
            value = state_dict[key]
            print(f"\n{'='*60}")
            print(f"Key: {key}")
            print(f"{'='*60}")
            print(f"Type: {type(value)}")
            
            if isinstance(value, dict):
                if "state" in value:
                    # FSDP2 wrapped state
                    print(f"Structure: FSDP2 wrapped")
                    inner_state = value["state"]
                    print(f"Inner state type: {type(inner_state)}")
                    
                    if isinstance(inner_state, dict):
                        print(f"Parameters: {len(inner_state)}")
                        
                        # Show first few keys
                        print(f"\nFirst 10 parameters:")
                        for i, (param_key, param_value) in enumerate(list(inner_state.items())[:10]):
                            if hasattr(param_value, 'shape'):
                                size_mb = param_value.numel() * param_value.element_size() / (1024**2)
                                print(f"  [{i+1}] {param_key}")
                                print(f"      Shape: {param_value.shape}, Dtype: {param_value.dtype}, Size: {size_mb:.2f} MB")
                            else:
                                print(f"  [{i+1}] {param_key}: {type(param_value)}")
                        
                        # Count total parameters and size
                        tensor_params = {k: v for k, v in inner_state.items() if hasattr(v, 'shape')}
                        if tensor_params:
                            total_params = sum(v.numel() for v in tensor_params.values())
                            total_size_gb = sum(v.numel() * v.element_size() for v in tensor_params.values()) / (1024**3)
                            print(f"\nTotal parameters: {total_params:,}")
                            print(f"Total size: {total_size_gb:.2f} GB")
                else:
                    # Regular dict
                    print(f"Dict with {len(value)} keys")
                    print(f"Keys: {list(value.keys())[:20]}")
                    
                    # Check if it contains tensors
                    tensor_keys = [k for k, v in value.items() if hasattr(v, 'shape')]
                    if tensor_keys:
                        print(f"\nTensor parameters: {len(tensor_keys)}")
                        print(f"First 5 tensors:")
                        for k in tensor_keys[:5]:
                            v = value[k]
                            size_mb = v.numel() * v.element_size() / (1024**2)
                            print(f"  {k}: {v.shape}, {v.dtype}, {size_mb:.2f} MB")
            
            elif hasattr(value, 'shape'):
                # Direct tensor
                size_mb = value.numel() * value.element_size() / (1024**2)
                print(f"Shape: {value.shape}")
                print(f"Dtype: {value.dtype}")
                print(f"Size: {size_mb:.2f} MB")
            
            elif isinstance(value, (int, float, str, bool)):
                print(f"Value: {value}")
            
            else:
                print(f"Unknown type: {type(value)}")
        
        # Overall summary
        print(f"\n\n{'='*60}")
        print("CHECKPOINT SUMMARY")
        print(f"{'='*60}\n")
        print(f"Top-level keys: {len(state_dict)}")
        print(f"Keys: {list(state_dict.keys())}")
        
        # Check for iteration
        if 'iteration' in state_dict:
            print(f"\nIteration: {state_dict['iteration']}")
        
        # Check for args
        if 'args' in state_dict:
            print(f"\nArgs available: Yes")
            if isinstance(state_dict['args'], dict):
                print(f"  Sample args: {list(state_dict['args'].keys())[:10]}")
        
    except Exception as e:
        print(f"❌ Error loading with DCP: {e}")
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
