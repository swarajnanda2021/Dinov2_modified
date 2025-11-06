"""
Submitit launcher for DINOv2 training on SLURM clusters.
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
    
    return parser.parse_args()


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints."""
    p = Path("/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B_B2_seqpacking/logs")
    p.mkdir(exist_ok=True)
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
    job_name = f"dinov2_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Slurm parameters
    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    executor.update_parameters(
        mem_gb=256,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,
        cpus_per_task=8,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=args.partition,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{args.ngpus}',
        slurm_constraint='h100',
        slurm_setup=[
            f'export OMP_NUM_THREADS=8',
            f'export NCCL_DEBUG=INFO',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={num_gpus_per_node * nodes}',
        ]
    )
    
    executor.update_parameters(name=job_name)

    args.dist_url = get_init_file().as_uri()

    # ========== Default training configuration ==========
    # Architecture
    args.patch_size = 16
    args.embeddingdim = 768
    args.vitheads = 12
    args.vitdepth = 12

    # Mask model
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
    
    # Save configuration
    with open(os.path.join(args.output_dir, f"{job_name}_config.txt"), "w") as f:
        for arg, value in sorted(vars(args).items()):
            f.write(f"{arg}: {value}\n")

    # Create and submit trainer
    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Job name: {job_name}")
    print(f"Logs and checkpoints: {args.output_dir}")
    
    print("\n" + "="*80)
    print("Configuration Summary:")
    print(f"  Global crops: {args.global_views}")
    print(f"  Local crops: {args.n_standard_local_crops}")
    print(f"  Masked crops: {args.num_masks}")
    total_views = args.global_views + args.n_standard_local_crops + args.num_masks
    print(f"  Total student views: {total_views}")
    print(f"  Batch size per GPU: {args.batch_size_per_gpu}")
    print(f"  Total GPUs: {args.ngpus * args.nodes}")
    print(f"  Effective batch size: {args.batch_size_per_gpu * args.ngpus * args.nodes}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()