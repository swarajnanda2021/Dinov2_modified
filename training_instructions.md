# SSL Foundation Model Training Guide

This guide covers monitoring, managing, and post-processing self-supervised learning (SSL) foundation model training runs on the cluster.

---

## Table of Contents

1. [Current Training Jobs](#1-current-training-jobs)
2. [Environment Setup](#2-environment-setup)
3. [Monitoring Training Jobs](#3-monitoring-training-jobs)
4. [Checking Training Progress](#4-checking-training-progress)
5. [Visualizing Training Curves](#5-visualizing-training-curves)
6. [Checkpoint Handling](#6-checkpoint-handling)
7. [Crash Recovery](#7-crash-recovery)
8. [Expected Training Times](#8-expected-training-times)
9. [Post-Processing and Benchmarking](#9-post-processing-and-benchmarking)
10. [Preparing Results for Papers](#10-preparing-results-for-papers)
11. [Mutation Prediction (MIL)](#11-mutation-prediction-mil)

---

## 1. Current Training Jobs

The following jobs are queued/running for the upcoming ICML paper submission. These train various Vision Transformer (ViT) architectures using DINOv2 self-supervised learning.

### Job Overview

| Job ID | Node | Status | Folder | Architecture | Training Strategy |
|--------|------|--------|--------|--------------|-------------------|
| 19030041 | iscn001 | Running | `FoundationModel_ViT-L_p16_b2048` | ViT-L (Large) | **DDP** |
| 19021731 | isci[001-002] | Running | `FoundationModel_ViT-L_p16_b2048_adios` | ViT-L + ADIOS masks | **FSDP** |
| 19022303 | — | Pending | `FoundationModel_ViT-H_p16_b2048` | ViT-H (Huge) | **FSDP** |
| 19022308 | — | Pending | `FoundationModel_ViT-H_p16_b2048_PatchReg` | ViT-H + Patch Prototype Regularization | **FSDP** |
| 19022368 | — | Pending | `FoundationModel_ViT-L_p16_b2048_PatchReg` | ViT-L + Patch Prototype Regularization | **FSDP** |
| 19022210 | — | Pending | `FoundationModel_ViT-B_p16_b1024_PatchReg` | ViT-B + Patch Prototype Regularization | **FSDP** |

All folders are located under `/data1/vanderbc/nandas1/`.

### Training Strategy Differences

| Strategy | Description | Checkpoint Format | Nodes |
|----------|-------------|-------------------|-------|
| **DDP** (Distributed Data Parallel) | Standard PyTorch distributed training. Simpler, works on single multi-GPU node. | Standard `.pth` files — **no conversion needed** | 1 node |
| **FSDP** (Fully Sharded Data Parallel) | Shards model parameters across GPUs for memory efficiency. Required for larger models (ViT-H) and multi-node training. | Sharded DCP directories — **requires conversion before post-processing** | 1–2 nodes |

### Naming Convention

- `ViT-B/L/H` — Model size (Base/Large/Huge)
- `p16` — Patch size 16×16
- `b1024/b2048` — Effective batch size
- `_adios` — Uses ADIOS adversarial mask augmentation
- `_PatchReg` — Uses Patch Prototype Regularization loss

---

## 2. Environment Setup

Before any operations, activate the conda environment:

```bash
conda activate ssl-v1
```

> **Important:** Use a hyphen (`ssl-v1`), not an underscore.

---

## 3. Monitoring Training Jobs

### Recommended: Use `rsqueue` Instead of `squeue`

The standard `squeue` command provides limited information. Use the `rsqueue` alias instead, which displays comprehensive job details including working directories:

```bash
rsqueue
```

If the alias is not set, add it to your `~/.bashrc`:

```bash
alias rsqueue='squeue --format="%.18i %.9P %.8j %.8u %.2t %.10M %.6D %.8Q %.20R %.75Z" --sort=-i'
```

### Example Output

```
          JOBID PARTITION     NAME     USER ST       TIME  NODES PRIORITY NODELIST(REASON)         WORK_DIR
       19030041 vanderbc_ dinov2_2 vanderbc  R    9:05:21      1      409 iscn001                  /data1/.../FoundationModel_ViT-L_p16_b2048
       19022368 vanderbc_ dinov2_2 vanderbc PD       0:00      2      421 (Priority)               /data1/.../FoundationModel_ViT-H_p16_b2048_PatchReg
```

**Status codes:**
- `R` = Running
- `PD` = Pending (with reason in parentheses)

---

## 4. Checking Training Progress

To monitor real-time training progress, use `tail` on the log file:

```bash
tail <folder_name>/logs/*0_log.out
```

For continuous monitoring:

```bash
tail -f <folder_name>/logs/*0_log.out
```

### Example Output

```
It 150230/300k (ETA 9:04:29), Progress: 50.1%, max mem: 74.2 GB : student_loss: 7.958070 (7.677960) dino_class_loss: 7.889459 (7.677960) koleo_loss: -0.285916 (-0.285904) ibot_loss: 0.097202 (0.097943) lr: 0.000427 (0.000440) wd: 0.208149 (0.202066)
```

**Key metrics:**
- `It X/300k` — Current iteration / total iterations
- `ETA` — Estimated time remaining
- `Progress` — Completion percentage
- `max mem` — Peak GPU memory usage
- Loss values shown as: `current_value (running_average)`

---

## 5. Visualizing Training Curves

To generate training loss plots:

```bash
cd <folder_name>/visualizations
python plot_and_save_loss.py
```

This creates `training_losses.png` with a grid of all loss components over training iterations.

---

## 6. Checkpoint Handling

### Two Checkpoint Formats

The codebase uses two distributed training strategies with different checkpoint formats:

| Strategy | Jobs Using This | Checkpoint Format | Location | Conversion Needed? |
|----------|-----------------|-------------------|----------|-------------------|
| **DDP** | `FoundationModel_ViT-L_p16_b2048` (iscn001) | Standard `.pth` files | `logs/checkpoint_iter_*.pth` | **No** |
| **FSDP** | All other jobs (multi-node, `_adios`, `_PatchReg`) | Sharded DCP directories | `logs/checkpoint_fsdp2/dcp_iter_*` | **Yes** |

### DDP Checkpoints (No Conversion Needed)

For the DDP job running on `iscn001` (`FoundationModel_ViT-L_p16_b2048`), checkpoints are saved as standard PyTorch files. You can use them directly for post-processing:

```bash
ls <folder>/logs/checkpoint_iter_*.pth
```

### FSDP Checkpoints (Conversion Required)

All other jobs use FSDP and save sharded checkpoints that **must be converted** before any post-processing (benchmarking, visualization, etc.).

#### Step 1: Get an Interactive Session

```bash
get_interactive
```

If the alias is not set, add it to your `~/.bashrc`:

```bash
alias get_interactive='salloc -n 1 --mem=24G --gres=gpu:a100:1 -p interactive -t 3:00:00'
```

#### Step 2: Activate Environment and Convert

```bash
conda activate ssl-v1
cd <folder_name>/training
python convert_dcp_to_pth.py
```

This will:
- Find all DCP checkpoints in `logs/checkpoint_fsdp2/`
- Convert each to a standard `.pth` file
- Save converted checkpoints to `logs/checkpoint_iter_XXXXXXXX.pth`

> **Note:** Conversion is relatively fast and does not require the full training resources.

---

## 7. Crash Recovery

### Why Crashes Happen

Long-running jobs (2–4 weeks) may occasionally crash due to:
- GPU power fluctuations
- Node failures
- Network issues

### How to Resume Training

Simply re-run the submitit launcher from the training folder:

```bash
cd <folder_name>
python run_with_submitit.py
```

The training will automatically:
1. Detect the latest checkpoint
2. Resume from that iteration
3. Continue training to completion

---

## 8. Expected Training Times

| Architecture | Approximate Duration |
|--------------|---------------------|
| ViT-B | ~1 day |
| ViT-L | ~2 weeks |
| ViT-H | ~4 weeks |

These estimates assume standard cluster availability. Actual times may vary based on queue priority and resource availability.

---

## Quick Reference

```bash
# Environment
conda activate ssl-v1

# Monitor jobs
rsqueue

# Check progress
tail -f <folder>/logs/*0_log.out

# Visualize losses
cd <folder>/visualizations && python plot_and_save_loss.py

# Convert FSDP checkpoints (in interactive session)
get_interactive
conda activate ssl-v1
cd <folder>/training && python convert_dcp_to_pth.py

# Resume crashed job
cd <folder> && python run_with_submitit.py
```

---

*For post-processing and benchmarking procedures, see below.*

---

## 9. Post-Processing and Benchmarking

### Overview

Once training is complete (or at intermediate checkpoints), run benchmarking to evaluate model performance on downstream tasks. All post-processing happens in:

```
/data1/vanderbc/nandas1/PostProc_FoundationModels/
```

### Running Benchmarks

Each model has a corresponding shell script. Submit with `sbatch`:

```bash
cd /data1/vanderbc/nandas1/PostProc_FoundationModels/
sbatch run_FoundationModel_ViT-L_p16_b2048.sh
```

Benchmarks typically complete in **< 2 days** (often much faster) using a single GPU.

### Benchmark Scripts by Paper

| Script | Paper Target |
|--------|--------------|
| `run_FoundationModel_ViT-L_p16_b2048.sh` | **ICML + NeurIPS** |
| `run_FoundationModel_ViT-L_p16_b2048_adios.sh` | **ICML only** |
| `run_FoundationModel_ViT-L_p16_b2048_PatchReg.sh` | NeurIPS only |
| `run_FoundationModel_ViT-H_p16_b2048.sh` | NeurIPS only |
| `run_FoundationModel_ViT-H_p16_b2048_PatchReg.sh` | NeurIPS only |
| `run_FoundationModel_ViT-B_p16_b1024_PatchReg.sh` | NeurIPS only |

### Results Location

Benchmark results are saved to:

```
/data1/vanderbc/nandas1/PostProc_FoundationModels/benchmark_results/<model_name>/
```

Example:
```bash
ls /data1/vanderbc/nandas1/PostProc_FoundationModels/benchmark_results/
# FoundationModel_ViT-L_p16_b2048/
# FoundationModel_ViT-L_p16_b2048_adios/
# ...
```

---

## 10. Preparing Results for Papers

### ICML Paper

#### Step 1: Generate `.dat` Files

```bash
cd /data1/vanderbc/nandas1/PostProc_FoundationModels/
python convert_to_dat.py
```

This creates consolidated results in:
```
/data1/vanderbc/nandas1/PostProc_FoundationModels/dat_files/
```

#### Step 2: Update Overleaf

Copy the entire `dat_files/` folder to the Overleaf document, replacing the existing folder.

### NeurIPS Paper

For NeurIPS, the set of models to include will differ. Modify `convert_to_dat.py` or create a new consolidation script to include/exclude the appropriate benchmark result folders.

---

## 11. Mutation Prediction (MIL)

For mutation prediction experiments using Multiple Instance Learning (MIL), add these checkpoints to the MIL processing pipeline:

### ICML Checkpoints

| Model | Checkpoint Path |
|-------|-----------------|
| ViT-L (baseline) | `/data1/vanderbc/nandas1/FoundationModel_ViT-L_p16_b2048/logs/checkpoint_iter_00298000.pth` |
| ViT-L + ADIOS | `/data1/vanderbc/nandas1/FoundationModel_ViT-L_p16_b2048_adios/logs/checkpoint_iter_00298000.pth` |

Add these paths to your checkpoint list in the MIL code configuration.

---

## Quick Reference (Updated)

```bash
# Environment
conda activate ssl-v1

# Monitor jobs
rsqueue

# Check progress
tail -f <folder>/logs/*0_log.out

# Visualize losses
cd <folder>/visualizations && python plot_and_save_loss.py

# Convert FSDP checkpoints (in interactive session)
get_interactive
conda activate ssl-v1
cd <folder>/training && python convert_dcp_to_pth.py

# Resume crashed job
cd <folder> && python run_with_submitit.py

# Run benchmarks
cd /data1/vanderbc/nandas1/PostProc_FoundationModels/
sbatch run_FoundationModel_ViT-L_p16_b2048.sh

# Generate paper results (ICML)
python convert_to_dat.py
# Then copy dat_files/ to Overleaf
```
