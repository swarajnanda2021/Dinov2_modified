#!/usr/bin/env python3
"""
Convert DCP checkpoints to regular PyTorch checkpoints (DDP format).

Usage:
    python convert_dcp_to_pth.py                  # convert all (subprocess per checkpoint)
    python convert_dcp_to_pth.py --iteration 8000 # convert single (worker mode)
"""

import os
import sys
import argparse
import ast
import gc
import subprocess
from pathlib import Path
from copy import deepcopy

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


def parse_config_file(logs_dir: Path) -> argparse.Namespace:
    """
    Parse the latest *_config.txt file from logs directory.
    Returns argparse.Namespace with all training args.
    """
    config_files = sorted(logs_dir.glob("*_config.txt"), key=lambda p: p.stat().st_mtime)
    if not config_files:
        raise FileNotFoundError(f"No *_config.txt found in {logs_dir}")
    
    config_path = config_files[-1]
    print(f"  Parsing args from: {config_path.name}")
    
    args_dict = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            
            if value.lower() == 'true':
                args_dict[key] = True
            elif value.lower() == 'false':
                args_dict[key] = False
            elif value.startswith('[') and value.endswith(']'):
                args_dict[key] = ast.literal_eval(value)
            else:
                try:
                    args_dict[key] = int(value)
                except ValueError:
                    try:
                        args_dict[key] = float(value)
                    except ValueError:
                        args_dict[key] = value
    
    return argparse.Namespace(**args_dict)


def get_pending_checkpoints(logs_dir: Path, output_dir: Path):
    """Find DCP checkpoints that haven't been converted yet."""
    checkpoint_dir = logs_dir / "checkpoint_fsdp2"
    if not checkpoint_dir.exists():
        return []
    
    dcp_checkpoints = sorted(checkpoint_dir.glob("dcp_iter_*"))
    
    pending = []
    for dcp_dir in dcp_checkpoints:
        iteration = int(dcp_dir.name.replace("dcp_iter_", ""))
        output_path = output_dir / f"checkpoint_iter_{iteration:08d}.pth"
        if not output_path.exists():
            pending.append(iteration)
    
    return pending


def run_orchestrator(output_dir: Path):
    """Orchestrator mode: spawn subprocess per checkpoint."""
    logs_dir = project_root / "logs"
    
    pending = get_pending_checkpoints(logs_dir, output_dir)
    
    if not pending:
        print("All checkpoints already converted")
        return
    
    print(f"Found {len(pending)} checkpoint(s) to convert: {pending}")
    print(f"Output: {output_dir}")
    print()
    
    success = 0
    failed = []
    
    for iteration in pending:
        print(f"{'='*70}")
        print(f"Spawning subprocess for iteration {iteration}")
        print(f"{'='*70}")
        
        cmd = [
            sys.executable,
            str(script_dir / "convert_dcp_to_pth.py"),
            "--iteration", str(iteration),
            "--output_dir", str(output_dir)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            success += 1
            print(f"✓ Iteration {iteration} completed\n")
        else:
            failed.append(iteration)
            print(f"✗ Iteration {iteration} failed (exit code {result.returncode})\n")
    
    print(f"{'='*70}")
    print(f"Summary: {success}/{len(pending)} converted successfully")
    if failed:
        print(f"Failed iterations: {failed}")
    print(f"{'='*70}")


def run_worker(iteration: int, output_dir: Path):
    """Worker mode: convert single checkpoint."""
    import torch
    import torch.distributed as dist
    
    try:
        from torch.distributed.checkpoint import FileSystemReader
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
    except ImportError:
        print("ERROR: torch.distributed.checkpoint not available")
        sys.exit(1)
    
    try:
        from models import CombinedModelDINO, ModernViT, DINOHead
    except ImportError as e:
        print(f"ERROR: Could not import models: {e}")
        sys.exit(1)
    
    logs_dir = project_root / "logs"
    dcp_dir = logs_dir / "checkpoint_fsdp2" / f"dcp_iter_{iteration}"
    output_path = output_dir / f"checkpoint_iter_{iteration:08d}.pth"
    
    if not dcp_dir.exists():
        print(f"ERROR: DCP checkpoint not found: {dcp_dir}")
        sys.exit(1)
    
    if output_path.exists():
        print(f"Already converted: {output_path.name}")
        sys.exit(0)
    
    print(f"Converting iteration {iteration}")
    
    # Parse config
    args = parse_config_file(logs_dir)
    
    # Initialize distributed
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    # Create models
    print(f"  Architecture: ViT embed_dim={args.embeddingdim}, depth={args.vitdepth}, heads={args.vitheads}")
    
    student_encoder = ModernViT(
        img_size=224, patch_size=args.patch_size, embed_dim=args.embeddingdim,
        depth=args.vitdepth, num_heads=args.vitheads, mlp_ratio=4.0, qkv_bias=True,
        qk_norm=False, dual_norm=False, drop_path_rate=0.4,
        pre_norm=False, num_register_tokens=4
    )
    teacher_encoder = deepcopy(student_encoder)
    
    student = CombinedModelDINO(
        backbone=student_encoder,
        classhead=DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head, 
                          norm_last_layer=args.norm_last_layer),
        patchhead=DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head,
                          norm_last_layer=args.norm_last_layer),
        num_masks=args.num_masks,
        patch_size=args.patch_size
    ).cuda()
    
    teacher = CombinedModelDINO(
        backbone=teacher_encoder,
        classhead=DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head),
        patchhead=DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head),
        num_masks=args.num_masks,
        patch_size=args.patch_size
    ).cuda()
    
    # Apply FSDP2 wrapping
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh
    
    world_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("dp",))
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fsdp_config = {"mesh": world_mesh, "mp_policy": mp_policy, "reshard_after_forward": True}
    
    for block in student.backbone.blocks:
        fully_shard(block, **fsdp_config)
    fully_shard(student.backbone, **fsdp_config)
    fully_shard(student.classhead, **fsdp_config)
    fully_shard(student.patchhead, **fsdp_config)
    
    for block in teacher.backbone.blocks:
        fully_shard(block, **fsdp_config)
    fully_shard(teacher.backbone, **fsdp_config)
    fully_shard(teacher.classhead, **fsdp_config)
    fully_shard(teacher.patchhead, **fsdp_config)
    
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
    
    def clean_fsdp_key(key: str) -> str:
        key = key.replace("_fsdp_wrapped_module.", "")
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("parametrizations.", "")
        key = key.replace("_orig_mod.", "")
        if key.endswith(".original"):
            key = key[:-9]
        return key
    
    def tensor_to_regular(tensor):
        if hasattr(tensor, 'full_tensor'):
            tensor = tensor.full_tensor()
        elif hasattr(tensor, 'to_local'):
            tensor = tensor.to_local()
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu()
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.clone().detach()
        return tensor
    
    def convert_state_dict(fsdp_state_dict):
        ddp_state_dict = {}
        for key, value in fsdp_state_dict.items():
            if isinstance(value, dict) and "state" in value:
                inner = value["state"]
                if isinstance(inner, dict):
                    for inner_key, inner_value in inner.items():
                        clean_key = clean_fsdp_key(inner_key)
                        ddp_state_dict[f"module.{clean_key}"] = tensor_to_regular(inner_value)
                else:
                    clean_key = clean_fsdp_key(key)
                    ddp_state_dict[f"module.{clean_key}"] = tensor_to_regular(inner)
            else:
                clean_key = clean_fsdp_key(key)
                ddp_state_dict[f"module.{clean_key}"] = tensor_to_regular(value)
        return ddp_state_dict
    
    student_state = convert_state_dict(to_load["student"])
    teacher_state = convert_state_dict(to_load["teacher"])
    
    checkpoint = {
        "iteration": to_load["iteration"],
        "args": args,
        "student": student_state,
        "teacher": teacher_state,
    }
    
    # Save
    print("  Saving...")
    torch.save(checkpoint, output_path)
    
    size_gb = output_path.stat().st_size / (1024**3)
    print(f"  ✓ Saved: {output_path.name} ({size_gb:.2f} GB)")
    print(f"    Iteration: {checkpoint['iteration']}")
    print(f"    Parameters: {len(student_state)} student, {len(teacher_state)} teacher")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Convert DCP to PyTorch checkpoints')
    parser.add_argument('--iteration', type=int, help='Specific iteration (worker mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory (default: ../logs/)')
    cli_args = parser.parse_args()
    
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else project_root / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if cli_args.iteration is not None:
        run_worker(cli_args.iteration, output_dir)
    else:
        run_orchestrator(output_dir)


if __name__ == "__main__":
    main()
