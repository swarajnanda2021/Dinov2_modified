#!/usr/bin/env python3
"""
Inspect raw DCP checkpoint files to understand structure.
"""

import os
import sys
import json
from pathlib import Path
import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent


def inspect_dcp_files(dcp_dir):
    """Inspect raw files in DCP directory."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {dcp_dir}")
    print(f"{'='*60}\n")
    
    # List all files
    files = sorted(dcp_dir.glob("*"))
    print(f"Files in checkpoint directory:")
    for f in files:
        size = f.stat().st_size / (1024**2)  # MB
        print(f"  {f.name:30s} {size:8.1f} MB")
    
    # Look for .metadata file
    metadata_file = dcp_dir / ".metadata"
    if metadata_file.exists():
        print(f"\n\n{'='*60}")
        print("METADATA FILE")
        print(f"{'='*60}")
        
        # Read as binary first to avoid decode errors
        with open(metadata_file, 'rb') as f:
            raw_content = f.read()
        
        # Try to decode as JSON
        try:
            content_str = raw_content.decode('utf-8')
            metadata = json.loads(content_str)
            print("Format: JSON\n")
            print(f"Metadata keys: {list(metadata.keys())}\n")
            
            for key, value in metadata.items():
                print(f"{key}:")
                if isinstance(value, dict):
                    print(f"  Dict with {len(value)} keys")
                    if len(value) < 20:
                        for k, v in list(value.items())[:20]:
                            if isinstance(v, dict):
                                print(f"    - {k}: {type(v).__name__} with {len(v)} keys")
                            else:
                                v_str = str(v)[:100]
                                print(f"    - {k}: {v_str}")
                    else:
                        print(f"    First 10 keys:")
                        for k in list(value.keys())[:10]:
                            print(f"    - {k}")
                elif isinstance(value, list):
                    print(f"  List with {len(value)} items")
                    if len(value) < 10:
                        for item in value:
                            print(f"    - {item}")
                else:
                    print(f"  {value}")
                print()
                
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            # Binary format - not JSON
            print(f"Format: Binary (not JSON)")
            print(f"Error: {e}\n")
            print(f"Size: {metadata_file.stat().st_size} bytes")
            print(f"First 100 bytes (hex):\n{raw_content[:100].hex()}")
            print(f"\nFirst 100 bytes (repr):\n{repr(raw_content[:100])}")
            
            # Try to detect format
            if raw_content[:2] == b'\x80\x02':
                print("\n⚠️  Looks like Python pickle format")
            elif raw_content[:4] == b'PK\x03\x04':
                print("\n⚠️  Looks like ZIP format")
            else:
                print("\n⚠️  Unknown binary format")
    
    # Look for individual shard files
    print(f"\n\n{'='*60}")
    print("SHARD FILES")
    print(f"{'='*60}\n")
    
    shard_files = sorted(dcp_dir.glob("__*_*.distcp"))
    if shard_files:
        print(f"Found {len(shard_files)} shard files\n")
        print("Loading first shard to inspect structure...\n")
        
        first_shard = shard_files[0]
        print(f"File: {first_shard.name}")
        print(f"Size: {first_shard.stat().st_size / (1024**2):.1f} MB\n")
        
        try:
            shard_data = torch.load(first_shard, map_location='cpu', weights_only=False)
            print(f"Type: {type(shard_data)}")
            
            if isinstance(shard_data, dict):
                print(f"Keys: {len(shard_data)} total\n")
                print(f"First 10 keys with details:\n")
                
                for i, (key, value) in enumerate(list(shard_data.items())[:10]):
                    print(f"[{i+1}] {key}:")
                    print(f"    Type: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"    Shape: {value.shape}")
                        print(f"    Dtype: {value.dtype}")
                        print(f"    Size: {value.numel() * value.element_size() / (1024**2):.1f} MB")
                    elif isinstance(value, dict):
                        print(f"    Dict with {len(value)} keys")
                        if len(value) < 5:
                            print(f"    Keys: {list(value.keys())}")
                    elif isinstance(value, (int, float, str, bool)):
                        print(f"    Value: {value}")
                    print()
                
                # Summary statistics
                print(f"\n{'='*60}")
                print("SUMMARY")
                print(f"{'='*60}\n")
                
                tensor_keys = [k for k, v in shard_data.items() if hasattr(v, 'shape')]
                dict_keys = [k for k, v in shard_data.items() if isinstance(v, dict)]
                scalar_keys = [k for k, v in shard_data.items() if isinstance(v, (int, float, str, bool))]
                
                print(f"Tensor parameters: {len(tensor_keys)}")
                print(f"Dict entries: {len(dict_keys)}")
                print(f"Scalar values: {len(scalar_keys)}")
                
                if tensor_keys:
                    total_params = sum(shard_data[k].numel() for k in tensor_keys)
                    total_size = sum(shard_data[k].numel() * shard_data[k].element_size() for k in tensor_keys) / (1024**3)
                    print(f"\nTotal parameters: {total_params:,}")
                    print(f"Total size: {total_size:.2f} GB")
                    
        except Exception as e:
            print(f"❌ Error loading shard: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No shard files found")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect DCP checkpoint structure')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (default: latest in logs/)')
    parser.add_argument('--iteration', type=int, default=None,
                       help='Specific iteration to inspect')
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
    
    inspect_dcp_files(dcp_dir)
