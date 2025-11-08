"""
Submitit launcher for DINOv2 training on SLURM clusters.
Edit the configuration section below and run: python run_with_submitit.py
"""

import os
import uuid
import datetime
from pathlib import Path
import submitit
from configs import get_args_parser
from training import train_dinov2


# ============================================================================
# SLURM CONFIGURATION
# ============================================================================
NGPUS = 4                          # GPUs per node
NODES = 1                          # Number of nodes
TIMEOUT = 10000                    # Job duration in minutes
PARTITION = "vanderbc_gpu"         # Partition name
CONSTRAINT = "h100"                # GPU constraint (h100, a100, etc.)
MEM_GB = 256                       # Memory per node in GB
CPUS_PER_TASK = 8                  # CPUs per task


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# --- Log path ---
LOGPATH = "/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B_pipelineparallel/logs"

# --- Architecture (for standard DDP) ---
PATCH_SIZE = 16
EMBEDDING_DIM = 768
VIT_HEADS = 12
VIT_DEPTH = 12

# --- Pipeline Parallel (alternative to standard DDP) ---
USE_PIPELINE_PARALLEL = False
MODEL_SIZE = 'base'                # 'base', 'large', 'huge', 'giant', 'giant2b'
GPUS_PER_NODE = 4                  # Must match NGPUS if using pipeline parallel
NUM_NODES = 1                      # Must match NODES if using pipeline parallel

# --- Mask Model ---
MASK_CHECKPOINT = "/data1/vanderbc/nandas1/TCGA_TMEDinov3_ViT-B/checkpoint_saved_mask_model.pth"

# --- Augmentation ---
GLOBAL_VIEWS = 2
N_STANDARD_LOCAL_CROPS = 8
LOCAL_CROP_SIZE = 96
NUM_MASKS = 0
CROPS_PER_MASK = 0

# --- DINO Parameters ---
OUT_DIM = 65536
NORM_LAST_LAYER = True
USE_BN_IN_HEAD = False

# --- DINOv2 Parameters ---
KOLEO_LOSS_WEIGHT = 0.1
IBOT_LOSS_WEIGHT = 1.0
TOKEN_MASK_RATIO = 0.4

# --- Prototype Clustering ---
NUM_PROTOTYPES = 4096
CLUSTERING_WEIGHT = 1.0
CLUSTERING_TEACHER_TEMP = 0.07
CLUSTERING_STUDENT_TEMP = 0.1

# --- Teacher Parameters ---
MOMENTUM_TEACHER = 0.996
TEACHER_TEMP = 0.07
WARMUP_TEACHER_TEMP = 0.04
TEACHER_TEMP_WARMUP_ITERS = 37_500

# --- Optimization ---
BATCH_SIZE_PER_GPU = 192
WARMUP_ITERATIONS = 12_500
TOTAL_ITERATIONS = 300_000
FREEZE_LAST_LAYER_ITERS = 1_250
LR = 5e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 0.04
WEIGHT_DECAY_END = 0.4

# --- Training Setup ---
USE_FP16 = True
CLIP_GRAD = 1.0
SAVE_CHECKPOINT_FREQ = 2_000
NUM_WORKERS = 10
VISUALIZATION_FREQ = 100
GRAD_CHECKPOINTING = True

# --- Dataset ---
BASE_DIR = "/data1/vanderbc/foundation_model_training_images/TCGA"

# ============================================================================
# END CONFIGURATION
# ============================================================================


def get_shared_folder() -> Path:
    """Get shared folder for logs and checkpoints."""
    p = Path(LOGPATH)
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
    
    # Create args object from config constants
    parser = get_args_parser()
    args = parser.parse_args([])  # Parse empty args to get defaults
    
    # Set output directory
    args.output_dir = str(get_shared_folder())
    
    # Apply all configuration
    args.patch_size = PATCH_SIZE
    args.embeddingdim = EMBEDDING_DIM
    args.vitheads = VIT_HEADS
    args.vitdepth = VIT_DEPTH
    
    args.use_pipeline_parallel = USE_PIPELINE_PARALLEL
    args.model_size = MODEL_SIZE
    args.gpus_per_node = GPUS_PER_NODE if USE_PIPELINE_PARALLEL else NGPUS
    args.num_nodes = NUM_NODES if USE_PIPELINE_PARALLEL else NODES
    
    args.mask_checkpoint = MASK_CHECKPOINT
    
    args.global_views = GLOBAL_VIEWS
    args.n_standard_local_crops = N_STANDARD_LOCAL_CROPS
    args.local_crop_size = LOCAL_CROP_SIZE
    args.num_masks = NUM_MASKS
    args.crops_per_mask = CROPS_PER_MASK
    
    args.out_dim = OUT_DIM
    args.norm_last_layer = NORM_LAST_LAYER
    args.use_bn_in_head = USE_BN_IN_HEAD
    
    args.koleo_loss_weight = KOLEO_LOSS_WEIGHT
    args.ibot_loss_weight = IBOT_LOSS_WEIGHT
    args.token_mask_ratio = TOKEN_MASK_RATIO
    
    args.num_prototypes = NUM_PROTOTYPES
    args.clustering_weight = CLUSTERING_WEIGHT
    args.clustering_teacher_temp = CLUSTERING_TEACHER_TEMP
    args.clustering_student_temp = CLUSTERING_STUDENT_TEMP
    
    args.momentum_teacher = MOMENTUM_TEACHER
    args.teacher_temp = TEACHER_TEMP
    args.warmup_teacher_temp = WARMUP_TEACHER_TEMP
    args.teacher_temp_warmup_iters = TEACHER_TEMP_WARMUP_ITERS
    
    args.batch_size_per_gpu = BATCH_SIZE_PER_GPU
    args.warmup_iterations = WARMUP_ITERATIONS
    args.total_iterations = TOTAL_ITERATIONS
    args.freeze_last_layer_iters = FREEZE_LAST_LAYER_ITERS
    args.lr = LR
    args.min_lr = MIN_LR
    args.weight_decay = WEIGHT_DECAY
    args.weight_decay_end = WEIGHT_DECAY_END
    
    args.use_fp16 = USE_FP16
    args.clip_grad = CLIP_GRAD
    args.save_checkpoint_freq = SAVE_CHECKPOINT_FREQ
    args.num_workers = NUM_WORKERS
    args.visualization_freq = VISUALIZATION_FREQ
    args.grad_checkpointing = GRAD_CHECKPOINTING
    
    args.base_dir = BASE_DIR
    
    # Setup executor
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    # Job name with timestamp
    mode = "pipeline" if USE_PIPELINE_PARALLEL else "ddp"
    model_str = f"{MODEL_SIZE}" if USE_PIPELINE_PARALLEL else f"ViT-{EMBEDDING_DIM}"
    job_name = f"dinov2_{mode}_{model_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Slurm parameters
    executor.update_parameters(
        mem_gb=MEM_GB,
        gpus_per_node=NGPUS,
        tasks_per_node=NGPUS,
        cpus_per_task=CPUS_PER_TASK,
        nodes=NODES,
        timeout_min=TIMEOUT,
        slurm_partition=PARTITION,
        slurm_signal_delay_s=120,
        slurm_gres=f'gpu:{NGPUS}',
        slurm_constraint=CONSTRAINT,
        slurm_setup=[
            f'export OMP_NUM_THREADS={CPUS_PER_TASK}',
            f'export NCCL_DEBUG=INFO',
            f'export NCCL_SOCKET_IFNAME=ib,bond',
            f'export MASTER_PORT=23468',
            f'export WORLD_SIZE={NGPUS * NODES}',
        ]
    )
    
    executor.update_parameters(name=job_name)
    args.dist_url = get_init_file().as_uri()

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
    print(f"  Nodes: {NODES}")
    print(f"  GPUs per node: {NGPUS}")
    print(f"  Total GPUs: {NGPUS * NODES}")
    print(f"  World size: {NGPUS * NODES}")
    
    if USE_PIPELINE_PARALLEL:
        print(f"\nPipeline Parallelism:")
        print(f"  Model: {MODEL_SIZE}")
        print(f"  Pipeline stages: {GPUS_PER_NODE}")
        print(f"  Data parallel replicas: {NUM_NODES}")
    else:
        print(f"\nStandard DDP:")
        print(f"  Model: ViT with embed_dim={EMBEDDING_DIM}")
        print(f"  Depth: {VIT_DEPTH}")
        print(f"  Heads: {VIT_HEADS}")
    
    # Training config
    print(f"\nTraining:")
    print(f"  Batch size per GPU: {BATCH_SIZE_PER_GPU}")
    print(f"  Effective batch size: {BATCH_SIZE_PER_GPU * NGPUS * NODES}")
    print(f"  Total iterations: {TOTAL_ITERATIONS:,}")
    print(f"  Learning rate: {LR}")
    print(f"  Weight decay: {WEIGHT_DECAY} -> {WEIGHT_DECAY_END}")
    
    # Augmentation
    print(f"\nAugmentation:")
    print(f"  Global crops: {GLOBAL_VIEWS}")
    print(f"  Local crops: {N_STANDARD_LOCAL_CROPS}")
    print(f"  Local crop size: {LOCAL_CROP_SIZE}")
    print(f"  Num masks: {NUM_MASKS}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()