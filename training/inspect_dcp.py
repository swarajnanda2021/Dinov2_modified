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
        
        # Try reading as JSON first
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print("Format: JSON")
            print(f"\nMetadata keys: {list(metadata.keys())}")
            
            for key, value in metadata.items():
                print(f"\n{key}:")
                if isinstance(value, dict):
                    print(f"  Dict with {len(value)} keys")
                    if len(value) < 20:
                        for k, v in value.items():
                            if isinstance(v, dict):
                                print(f"    - {k}: {type(v).__name__} with {len(v)} keys")
                            else:
                                print(f"    - {k}: {v}")
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
        except json.JSONDecodeError:
            # Try as binary
            print("Format: Binary (not JSON)")
            print(f"Size: {metadata_file.stat().st_size} bytes")
            
            # Read first 100 bytes to see structure
            with open(metadata_file, 'rb') as f:
                header = f.read(100)
            print(f"First 100 bytes (hex): {header.hex()[:200]}")
            print(f"First 100 bytes (repr): {repr(header[:100])}")
    
    # Look for individual shard files
    print(f"\n\n{'='*60}")
    print("SHARD FILES")
    print(f"{'='*60}")
    
    shard_files = sorted(dcp_dir.glob("__*_*.distcp"))
    if shard_files:
        print(f"Found {len(shard_files)} shard files")
        print("\nLoading first shard to inspect structure...")
        
        first_shard = shard_files[0]
        print(f"  File: {first_shard.name}")
        
        shard_data = torch.load(first_shard, map_location='cpu', weights_only=False)
        print(f"  Type: {type(shard_data)}")
        
        if isinstance(shard_data, dict):
            print(f"  Keys: {len(shard_data)} total")
            print(f"\nFirst 10 keys with details:")
            
            for i, (key, value) in enumerate(list(shard_data.items())[:10]):
                print(f"\n  [{i+1}] {key}:")
                print(f"      Type: {type(value)}")
                if hasattr(value, 'shape'):
                    print(f"      Shape: {value.shape}")
                    print(f"      Dtype: {value.dtype}")
                    print(f"      Size: {value.numel() * value.element_size() / (1024**2):.1f} MB")
                elif isinstance(value, dict):
                    print(f"      Dict with keys: {list(value.keys())}")


if __name__ == "__main__":
    dcp_dir = project_root / "logs" / "checkpoint_fsdp2" / "dcp_iter_8000"
    
    if not dcp_dir.exists():
        print(f"ERROR: {dcp_dir} not found!")
        sys.exit(1)
    
    inspect_dcp_files(dcp_dir)
