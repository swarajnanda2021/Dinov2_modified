"""
Submitit launcher for DINOv2 training on SLURM clusters.
Supports both standard DDP and pipeline parallelism.
"""

import argparse
import os
import uuid
import datetime
from pathlib import Path

import submitit

from configs import get_args_parser
from training import train_dinov2


def parse_args():
    """Parse submitit and training arguments."""
    parser = argparse.ArgumentParser(
        "Submitit for DINOv2", 
        parents=[get_args_parser()]
    )
    
    # Submitit specific arguments
    parser.add_argument("--ngpus", default=4, type=int, 
                        help="Number of GPUs per node")
    parser.add_argument("--nodes", default=1, type=int, 
                        help="Number of nodes")
    parser.add_argument("--timeout", default=10000, type=int, 
                        help="Job duration in minutes")
    parser.add_argument("--partition", default="vanderbc_gpu", type=str, 
                        help="Partition name")
    parser.add_argument("--constraint", default="h100", type=str,
                        help="GPU constraint (h100, a100, etc.)")
    parser.add_argument("--mem_gb", default=256, type=int,
                        help="Memory per node in GB")
    parser.add_argument("--cpus_per_task", default=8, type=int,
                        help="CPUs per task")
    
    return parser.parse_args()


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints."""
    p = Path("/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B_pipelineparallel/logs")
    p.mkdir(exist_ok=True, parents=True)
    return p


def get_init_file():
    """Create unique init file for distributed training."""
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    """Wrapper class for submitit job."""
    def __init__(self, args):
        self.args = args

    def __call__(self):
        """Main training call."""
        self._setup_gpu_args()
        train_dinov2(self.args)

    def checkpoint(self):
        """Checkpoint for job preemption."""
        self.args.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        """Setup GPU arguments from submitit environment."""
        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    """Main submitit launcher."""
    args = parse_args()
    
    # Set output directory
    args.output_dir = str(get_shared_folder())
    
    # Setup executor
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    # Job name with timestamp
    mode = "pipeline" if args.use_pipeline_parallel else "ddp"
    model_str = f"{args.model_size}" if args.use_pipeline_parallel else f"ViT-{args.embeddingdim}"
    job_name = f"dinov2_{mode}_{model_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Slurm parameters
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    # Verify pipeline configuration
    if args.use_pipeline_parallel:
        if args.gpus_per_node != num_gpus_per_node:
            print(f"Warning: Setting gpus_per_node={num_gpus_per_node} to match --ngpus")
            args.gpus_per_node = num_gpus_per_node
        if args.num_nodes != nodes:
            print(f"Warning: Setting num_nodes={nodes} to match --nodes")
            args.num_nodes = nodes

    executor.update_parameters(
        mem_gb=args.mem_gb,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{num_gpus_per_node}',
        slurm_constraint=args.constraint,
        slurm_setup=[
            f'export OMP_NUM_THREADS={args.cpus_per_task}',
            f'export NCCL_DEBUG=INFO',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={num_gpus_per_node * nodes}',
        ]
    )
    
    executor.update_parameters(name=job_name)

    args.dist_url = get_init_file().as_uri()

    # ========== SET YOUR TRAINING PARAMETERS HERE ==========
    # This is where you configure everything!
    
    # Architecture (for standard DDP)
    args.patch_size = 16
    args.embeddingdim = 768
    args.vitheads = 12
    args.vitdepth = 12
    
    # For pipeline parallel, use model_size instead
    args.model_size = 'base' # Choices: 'base', 'large', 'huge', 'giant', 'giant2b'
    args.gpus_per_node = 4
    args.num_nodes = 1

    # Mask model checkpoint
    args.mask_checkpoint = "/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B/checkpoint_saved_mask_model.pth"

    # Augmentation
    args.global_views = 2
    args.n_standard_local_crops = 8
    args.local_crop_size = 96
    args.num_masks = 0
    args.crops_per_mask = 0
    
    # DINO parameters
    args.out_dim = 65536
    args.norm_last_layer = True
    args.use_bn_in_head = False
    
    # DINOv2 parameters
    args.koleo_loss_weight = 0.1
    args.ibot_loss_weight = 1.0
    args.token_mask_ratio = 0.4
    
    # Prototype clustering
    args.num_prototypes = 4096
    args.clustering_weight = 1.0
    args.clustering_teacher_temp = 0.07
    args.clustering_student_temp = 0.1
    
    # Teacher parameters
    args.momentum_teacher = 0.996
    args.teacher_temp = 0.07
    args.warmup_teacher_temp = 0.04
    args.teacher_temp_warmup_iters = 37_500
    
    # Optimization
    args.batch_size_per_gpu = 192
    args.warmup_iterations = 12_500
    args.total_iterations = 300_000
    args.freeze_last_layer_iters = 1_250
    args.lr = 5e-5
    args.min_lr = 1e-6
    args.weight_decay = 0.04
    args.weight_decay_end = 0.4
    
    # Training setup
    args.use_fp16 = True
    args.clip_grad = 1.0
    args.save_checkpoint_freq = 2_000
    args.num_workers = 10
    args.visualization_freq = 100
    args.grad_checkpointing = True
    
    # Dataset
    args.base_dir = "/data1/vanderbc/foundation_model_training_images/TCGA"
    
    # =======================================================

    # Save configuration
    with open(os.path.join(args.output_dir, f"{job_name}_config.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write(f"Job: {job_name}\n")
        f.write(f"Mode: {mode.upper()}\n")
        f.write("="*80 + "\n\n")
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    # Create and submit trainer
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("\n" + "="*80)
    print("JOB SUBMITTED")
    print("="*80)
    print(f"Job ID: {job.job_id}")
    print(f"Job name: {job_name}")
    print(f"Mode: {mode.upper()}")
    print(f"Logs and checkpoints: {args.output_dir}")
    
    print("\n" + "="*80)
    print("CONFIGURATION SUMMARY")
    print("="*80)
    
    # Distributed setup
    print(f"\nDistributed Setup:")
    print(f"  Nodes: {nodes}")
    print(f"  GPUs per node: {num_gpus_per_node}")
    print(f"  Total GPUs: {num_gpus_per_node * nodes}")
    print(f"  World size: {num_gpus_per_node * nodes}")
    
    if args.use_pipeline_parallel:
        print(f"\nPipeline Parallelism:")
        print(f"  Model: {args.model_size}")
        print(f"  Pipeline stages: {args.gpus_per_node}")
        print(f"  Data parallel replicas: {args.num_nodes}")
    else:
        print(f"\nStandard DDP:")
        print(f"  Model: ViT with embed_dim={args.embeddingdim}")
        print(f"  Depth: {args.vitdepth}")
        print(f"  Heads: {args.vitheads}")
    
    # Training config
    print(f"\nTraining:")
    print(f"  Batch size per GPU: {args.batch_size_per_gpu}")
    print(f"  Effective batch size: {args.batch_size_per_gpu * num_gpus_per_node * nodes}")
    print(f"  Total iterations: {args.total_iterations:,}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay} -> {args.weight_decay_end}")
    
    # Augmentation
    print(f"\nAugmentation:")
    print(f"  Global crops: {args.global_views}")
    print(f"  Local crops: {args.n_standard_local_crops}")
    print(f"  Local crop size: {args.local_crop_size}")
    print(f"  Num masks: {args.num_masks}")
    

    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()