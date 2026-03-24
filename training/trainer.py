"""
Main training loop for DINOv2 with iBOT and prototype clustering.
"""

import os
import sys
import time
import datetime
import json
from pathlib import Path
from copy import deepcopy
import gc
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, PatchPrototypeLoss
from data import DINOv2PathologyDataset, ProportionalMultiDatasetWrapper
from .helpers import (
    load_pretrained_mask_model,
    apply_masks_to_images,
    extract_local_crops_from_masked,
    generate_random_token_masks,
    generate_block_masks,
    convert_semantic_masks_to_token_masks,
    calculate_total_student_views,
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)


def _gather_and_compute_weights(patch_tokens, mask, masks_weight=None):
    """
    Gather masked tokens from [B, N, D] using boolean mask [B, N].
    Returns gathered tokens [M, D], per-token weights [M], and batch size B.
    
    Args:
        patch_tokens: [B, N, D] patch tokens
        mask: [B, N] boolean mask (True = masked)
        masks_weight: Optional [B] per-sample weights. If None, uses 1/num_masked.
        
    Returns:
        gathered: [M, D] masked tokens
        weights: [M] per-token weights
        B: batch size
    """
    B, N, D = patch_tokens.shape
    
    mask_flat = mask.reshape(-1)  # [B*N]
    masked_indices = mask_flat.nonzero(as_tuple=True)[0]  # [M]
    M = masked_indices.numel()
    
    if M == 0:
        return None, None, B
    
    gathered = patch_tokens.reshape(B * N, D)[masked_indices]  # [M, D]
    
    # Per-token weights
    sample_idx = masked_indices // N  # [M]
    if masks_weight is not None:
        weights = masks_weight[sample_idx]  # [M]
    else:
        num_masked_per_sample = mask.sum(dim=1).float().clamp(min=1.0)  # [B]
        weights = 1.0 / num_masked_per_sample[sample_idx]  # [M]
    
    return gathered, weights, B


def train_dinov2(args):
    """
    Main training function for DINOv2 with iBOT and prototype clustering.

    Args:
        args: Training arguments namespace
    """
    # ============ Setup ============
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # Augmentation configuration
    augmentation_free_mode = (args.global_views == 0)

    print("\n========== Augmentation Configuration ==========")
    print(f"Global views (teacher/student): {args.global_views}")
    print(f"Standard local crops: {args.n_standard_local_crops}")
    print(f"Local crop size: {args.local_crop_size}x{args.local_crop_size}")

    if args.use_semantic_ibot:
        print("Semantic iBOT: ENABLED")
        print(f"  Mask model: {args.mask_model_arch}")
        print(f"  Semantic channels: {args.num_masks}")
        print(f"  Channels per iteration: {args.semantic_masks_per_iteration}")
        print(f"  Semantic iBOT weight: {args.semantic_ibot_weight}")
        if args.use_semantic_prototypes:
            print(f"  Semantic prototype loss: ENABLED (weight={args.semantic_clustering_weight})")
    else:
        print("Semantic iBOT: DISABLED")

    total_student_views = calculate_total_student_views(args)
    print(f"\nTotal student views: {total_student_views}")
    print("================================================\n")

    # ============ Load pre-trained mask model for semantic iBOT (if enabled) ============
    mask_model_frozen = None
    if args.use_semantic_ibot:
        if args.mask_checkpoint is None:
            raise ValueError("--use_semantic_ibot is True but --mask_checkpoint not provided")

        if args.num_masks <= 0:
            raise ValueError("--use_semantic_ibot is True but --num_masks must be > 0")

        mask_model_frozen = load_pretrained_mask_model(
                                                        args.mask_checkpoint,
                                                        args.num_masks,
                                                        mask_model_arch=args.mask_model_arch,
                                                        mask_encoder_dim=args.mask_encoder_dim
                                                    )
        mask_model_frozen = mask_model_frozen.cuda()
        mask_model_frozen.eval()

        for param in mask_model_frozen.parameters():
            param.requires_grad = False

        print(f"Loaded and froze mask model with {args.num_masks} semantic channels for semantic iBOT")
    else:
        print("Semantic iBOT disabled (--use_semantic_ibot=False)")

    # ============ Create dataset ============
    dataset_configs = []
    for source in args.dataset_sources:
        parts = source.split(':')
        name, base_dir, index_file = parts
        index_path = os.path.join(base_dir, index_file)
        metadata_path = index_path.replace('.pkl', '_metadata.pkl')
        dataset_configs.append({
            'name': name,
            'base_dir': base_dir,
            'index_file': index_file
        })

    trainset = ProportionalMultiDatasetWrapper(
        dataset_configs=dataset_configs,
        batch_size_per_gpu=args.batch_size_per_gpu,
        n_standard_local_crops=args.n_standard_local_crops,
        global_views=args.global_views,
        local_crop_size=args.local_crop_size,
        worker_id=0,
        num_workers=args.num_workers,
        rank=args.gpu,
        world_size=dist.get_world_size(),
        seed=args.seed,
        global_size=224,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_init_fn
    )

    # ============ Initialize models ============
    student_encoder = ModernViT(
        img_size=224,
        patch_size=args.patch_size,
        embed_dim=args.embeddingdim,
        depth=args.vitdepth,
        num_heads=args.vitheads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_norm=False,
        dual_norm=False,
        drop_path_rate=0.4,
        pre_norm=False,
        num_register_tokens=4,
    )

    teacher_encoder = deepcopy(student_encoder)

    student_classhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    teacher_classhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
    )

    student_patchhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    )

    teacher_patchhead = DINOHead(
        args.embeddingdim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
    )

    student = CombinedModelDINO(
        backbone=student_encoder,
        classhead=student_classhead,
        patchhead=student_patchhead,
        num_masks=args.num_masks,
        patch_size=args.patch_size,
    )

    teacher = CombinedModelDINO(
        backbone=teacher_encoder,
        classhead=teacher_classhead,
        patchhead=teacher_patchhead,
        num_masks=args.num_masks,
        patch_size=args.patch_size,
    )

    # ============ Create Prototype Bank (Optional) ============
    prototype_bank = None
    if args.use_prototype_clustering:
        prototype_bank = LinearPrototypeBank(
            num_prototypes=args.num_prototypes,
            embed_dim=args.embeddingdim,
            bias=True
        )
        prototype_bank = prototype_bank.cuda()

        print(f"Created LinearPrototypeBank with {args.num_prototypes} soft prototypes")
    else:
        print("Prototype clustering disabled (--use_prototype_clustering=False)")

    student = student.cuda()
    teacher = teacher.cuda()

    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

    student = setup_ddp_model(student, args, find_unused=True)
    teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])

    if args.use_prototype_clustering:
        prototype_bank = nn.parallel.DistributedDataParallel(prototype_bank, device_ids=[args.gpu])

    teacher_without_ddp = teacher.module

    student._set_static_graph()
    print("Set static graph for student model")

    teacher_without_ddp.backbone.load_state_dict(student.module.backbone.state_dict())
    teacher_without_ddp.classhead.load_state_dict(student.module.classhead.state_dict())
    teacher_without_ddp.patchhead.load_state_dict(student.module.patchhead.state_dict())
    teacher.requires_grad_(False)

    # ============ Initialize losses ============
    dino_class_loss = DINOLoss(
        ncrops=total_student_views,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_temp_iters=args.teacher_temp_warmup_iters,
        n_iterations=5,
        student_temp=0.1,
    ).cuda()

    ibot_patch_loss = iBOTPatchLoss(
        student_temp=0.1,
        n_iterations=3,
    ).cuda()

    dino_koleo_loss = KoLeoLoss().cuda()

    patch_prototype_loss = None
    if args.use_prototype_clustering:
        patch_prototype_loss = PatchPrototypeLoss(
            num_prototypes=args.num_prototypes,
            embed_dim=args.embeddingdim,
            teacher_temp=args.clustering_teacher_temp,
            student_temp=args.clustering_student_temp,
        ).cuda()

        print(f"Initialized PatchPrototypeLoss with {args.num_prototypes} prototypes")
    else:
        print("Patch prototype clustering disabled")

    # ============ Create fp16_scaler ============
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    # ============ Create optimizers ============
    backbone_params = utils.get_params_groups_with_layer_decay(
        student.module.backbone,
        lr_decay_rate=args.lr_decay_rate,
        num_layers=args.vitdepth,
    )

    classhead_params = utils.get_params_groups_with_decay_for_heads(student.module.classhead)
    patchhead_params = utils.get_params_groups_with_decay_for_heads(student.module.patchhead)

    all_param_groups = backbone_params + classhead_params + patchhead_params

    optimizer_student = torch.optim.AdamW(all_param_groups)

    if utils.is_main_process():
        print(f"\n=== Layer-wise LR Decay (rate={args.lr_decay_rate}) ===")
        for i, pg in enumerate(backbone_params):
            print(f"  Group {i}: lr_mult={pg['lr_multiplier']:.4f}, wd_mult={pg['wd_multiplier']}, params={len(pg['params'])}")
        print(f"  Head groups: {len(classhead_params) + len(patchhead_params)} groups with lr_mult=1.0")
        print(f"  Total param groups: {len(all_param_groups)}")
        print("=" * 50 + "\n")

    optimizer_prototypes = None
    if args.use_prototype_clustering:
        optimizer_prototypes = torch.optim.AdamW(
            prototype_bank.module.parameters(),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        print(f"Created optimizers (including prototype optimizer)")
    else:
        print(f"Created optimizer (student only)")

    # ============ Create schedulers ============
    student_lr_schedule = utils.cosine_scheduler(
        base_value=args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        final_value=args.min_lr,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
        start_warmup_value=0
    )

    proto_lr_schedule = None
    if args.use_prototype_clustering:
        proto_lr_schedule = utils.cosine_scheduler(
            base_value=args.lr * 0.5,
            final_value=0,
            total_iters=args.total_iterations,
            warmup_iters=args.warmup_iterations,
            start_warmup_value=0
        )

    wd_schedule = utils.cosine_scheduler(
        base_value=args.weight_decay,
        final_value=args.weight_decay_end,
        total_iters=args.total_iterations,
        warmup_iters=args.warmup_iterations,
        start_warmup_value=args.weight_decay
    )

    momentum_schedule = utils.cosine_scheduler(
        base_value=args.momentum_teacher,
        final_value=1.0,
        total_iters=args.total_iterations,
        warmup_iters=0,
        start_warmup_value=args.momentum_teacher
    )

    # ============ Load checkpoint ============
    to_restore = {"iteration": 0, "dataset_position": 0}

    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    loaded_checkpoint = None
    if os.path.exists(checkpoint_path):
        try:
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Pre-loaded checkpoint from iteration {loaded_checkpoint.get('iteration', 'N/A')}.")
        except Exception as e:
            print(f"Could not pre-load checkpoint. Starting fresh. Error: {e}")
            loaded_checkpoint = None

    checkpoint_kwargs = {
        'student': student,
        'teacher': teacher,
        'optimizer_student': optimizer_student,
        'fp16_scaler': fp16_scaler,
        'dino_class_loss': dino_class_loss,
    }

    if args.use_prototype_clustering:
        checkpoint_kwargs['prototype_bank'] = prototype_bank
        checkpoint_kwargs['optimizer_prototypes'] = optimizer_prototypes
        checkpoint_kwargs['patch_prototype_loss'] = patch_prototype_loss

    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        **checkpoint_kwargs
    )

    current_iteration = to_restore["iteration"]
    dataset_position = to_restore.get("dataset_position", 0)

    # ============ Set resume position in dataset ============
    if current_iteration > 0:
        global_samples_processed = current_iteration * args.batch_size_per_gpu * dist.get_world_size()
        trainset.set_resume_position(global_samples_processed)
        print(f"Resuming from iteration {current_iteration}")

    # ============ Restore RNGs ============
    if loaded_checkpoint and 'torch_rng_state' in loaded_checkpoint:
        try:
            print("Restoring RNG states from checkpoint...")
            torch.set_rng_state(loaded_checkpoint['torch_rng_state'])
            torch.cuda.set_rng_state_all(loaded_checkpoint['cuda_rng_state'])
            np.random.set_state(loaded_checkpoint['numpy_rng_state'])
            random.setstate(loaded_checkpoint['random_rng_state'])
            print(f"Successfully restored all RNG states to iteration {current_iteration}.")
        except Exception as e:
            print(f"WARNING: Failed to restore RNG states. Re-seeding. Error: {e}")
            utils.fix_random_seeds(args.seed + utils.get_rank())
    else:
        if current_iteration == 0:
            print("Starting from scratch. Fixing random seeds.")
        else:
            print(f"WARNING: Checkpoint found but no RNG state. Re-seeding.")
        utils.fix_random_seeds(args.seed + utils.get_rank())

    # ============ Verify checkpoint ============
    if utils.is_main_process() and current_iteration > 0:
        print(f"\n=== Checkpoint Loaded at Iteration {current_iteration} ===")
        if args.use_prototype_clustering:
            proto_stats = prototype_bank.module.get_stats()
            print(f"Prototype Bank Statistics:")
            print(f"  Weight norm mean: {proto_stats['weight_norm_mean']:.6f}")
            print(f"  Weight norm std: {proto_stats['weight_norm_std']:.6f}")
        print("="*50 + "\n")

    metric_logger = utils.IterationMetricLogger(total_iterations=args.total_iterations)
    metric_logger.start_time = time.time()

    data_iterator = iter(train_loader)

    loader_len = len(train_loader) if len(train_loader) > 0 else 1
    dataset_passes = dataset_position // loader_len
    max_passes = 5

    if utils.is_main_process():
        print(f"Starting training at iteration {current_iteration}")

    # ============ Training loop ============
    print("Starting training!")

    while current_iteration < args.total_iterations:
        # ========== Get batch ==========
        try:
            batch_data = next(data_iterator)
            dataset_position += 1
        except StopIteration:
            dataset_passes += 1
            if dataset_passes >= max_passes:
                print(f"Reached maximum passes ({max_passes}). Stopping.")
                break

            data_iterator = iter(train_loader)
            batch_data = next(data_iterator)
            dataset_position = dataset_passes * loader_len
            print(f"Starting pass {dataset_passes + 1} at iteration {current_iteration}")

        # === 1. Extract crops from batch ===
        idx = 0

        teacher_global_crops = []
        for i in range(args.global_views):
            teacher_global_crops.append(batch_data[idx].cuda(non_blocking=True))
            idx += 1

        student_all_crops = []
        for crop in teacher_global_crops:
            student_all_crops.append(crop)

        student_local_crops = []
        for i in range(args.n_standard_local_crops):
            crop = batch_data[idx].cuda(non_blocking=True)
            student_local_crops.append(crop)
            student_all_crops.append(crop)
            idx += 1

        # === 2. Generate block masks for standard iBOT ===
        batch_size = teacher_global_crops[0].shape[0]
        n_patches_h = n_patches_w = 224 // args.patch_size

        block_masks_1, masks_weight_1 = generate_block_masks(
            batch_size, n_patches_h, n_patches_w,
            mask_ratio_min=args.mask_ratio_min,
            mask_ratio_max=args.mask_ratio_max,
            mask_sample_probability=args.mask_sample_probability,
            device=teacher_global_crops[0].device
        )

        block_masks_2, masks_weight_2 = generate_block_masks(
            batch_size, n_patches_h, n_patches_w,
            mask_ratio_min=args.mask_ratio_min,
            mask_ratio_max=args.mask_ratio_max,
            mask_sample_probability=args.mask_sample_probability,
            device=teacher_global_crops[0].device
        )

        # === 3. Generate semantic token masks (if enabled) ===
        semantic_token_masks = None
        semantic_masks_weights = None
        selected_channels = None

        if args.use_semantic_ibot and mask_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                    mask_output = mask_model_frozen(teacher_global_crops[0])
                    soft_masks = mask_output['masks'].float()

            all_token_masks, all_masks_weights = convert_semantic_masks_to_token_masks(
                soft_masks, patch_size=args.patch_size
            )

            if args.semantic_masks_per_iteration >= args.num_masks:
                selected_channels = list(range(args.num_masks))
            else:
                selected_channels = random.sample(
                    range(args.num_masks), args.semantic_masks_per_iteration
                )

            semantic_token_masks = [all_token_masks[:, c, :] for c in selected_channels]
            semantic_masks_weights = [all_masks_weights[:, c] for c in selected_channels]

        # ========== Debug: Print shapes on first iteration ==========
        if current_iteration == 0 and utils.is_main_process():
            print("\n=== Crop Organization (First Iteration) ===")
            print(f"Teacher global crops: {len(teacher_global_crops)} crops")
            for i, crop in enumerate(teacher_global_crops):
                print(f"  Teacher crop {i}: {crop.shape}")
            print(f"Student total crops: {len(student_all_crops)} crops")
            for i, crop in enumerate(student_all_crops):
                print(f"  Student crop {i}: {crop.shape}")
            print(f"Block masks 1: {block_masks_1.shape}, masks_weight_1: {masks_weight_1.shape}")
            print(f"Block masks 2: {block_masks_2.shape}, masks_weight_2: {masks_weight_2.shape}")

            if args.use_semantic_ibot and semantic_token_masks is not None:
                print(f"\nSemantic iBOT:")
                print(f"  Selected channels: {selected_channels}")
                for ch_idx, (sm, sw) in enumerate(zip(semantic_token_masks, semantic_masks_weights)):
                    n_masked = sm.sum(dim=1).float().mean().item()
                    print(f"  Channel {selected_channels[ch_idx]}: mask shape {sm.shape}, avg masked patches: {n_masked:.1f}")

            print("="*50 + "\n")

        # ========== Update learning rates ==========
        for i, param_group in enumerate(optimizer_student.param_groups):
            base_lr = student_lr_schedule[current_iteration]
            lr_mult = param_group.get("lr_multiplier", 1.0)
            param_group["lr"] = base_lr * lr_mult

            wd_mult = param_group.get("wd_multiplier", 1.0)
            if wd_mult > 0:
                param_group["weight_decay"] = wd_schedule[current_iteration] * wd_mult

        if args.use_prototype_clustering and optimizer_prototypes is not None:
            for param_group in optimizer_prototypes.param_groups:
                param_group["lr"] = proto_lr_schedule[current_iteration]

        optimizer_student.zero_grad()
        if args.use_prototype_clustering and optimizer_prototypes is not None:
            optimizer_prototypes.zero_grad()

        # ========== Forward passes and loss computation ==========
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
            # ========== DINO Loss with Sequence Packing ==========

            student_masks = [block_masks_1, block_masks_2] + [None] * len(student_local_crops)

            # Teacher forward (unmasked targets)
            with torch.no_grad():
                teacher_output = teacher(teacher_global_crops, token_masks=[None, None], mode='dino')
                teacher_cls_outputs = teacher_output['cls_outputs']
                teacher_patch_tokens_g1 = teacher_output['features_list'][0]['patchtokens']
                teacher_patch_tokens_g2 = teacher_output['features_list'][1]['patchtokens']

            # Student forward (all crops, masks on global crops)
            student_output = student(student_all_crops, token_masks=student_masks, mode='dino')
            student_cls_outputs = student_output['cls_outputs']
            student_patch_tokens_g1 = student_output['features_list'][0]['patchtokens']
            student_patch_tokens_g2 = student_output['features_list'][1]['patchtokens']

            # DINO CLS loss
            dino_class_loss_val = dino_class_loss(
                student_cls_outputs,
                teacher_cls_outputs,
                current_iteration
            )

            # ========== KoLeo Loss ==========
            global_cls_tokens = [student_output['features_list'][i]['clstoken'] for i in range(args.global_views)]

            koleo_loss_val = torch.tensor(0.0).cuda()
            if len(global_cls_tokens) > 0:
                koleo_loss_val = sum(dino_koleo_loss(token) for token in global_cls_tokens) / len(global_cls_tokens)

            # ================================================================
            # iBOT Loss — gather-then-project to avoid [B, N, 65536] tensors.
            # Projects only masked tokens (~7k) instead of all tokens (~50k).
            # Semantic iBOT integrated here, reusing teacher backbone tokens.
            # ================================================================
            current_teacher_temp_ibot = dino_class_loss.teacher_temp_schedule(current_iteration)

            semantic_ibot_loss_val = torch.tensor(0.0, device='cuda')
            semantic_backbone_outputs = []  # Raw [B, N, D] tokens for prototype section

            # ---------- Block iBOT: Global crop 1 ----------
            s_gathered_1, weights_1, B = _gather_and_compute_weights(
                student_patch_tokens_g1, block_masks_1, masks_weight_1
            )

            if s_gathered_1 is not None:
                with torch.no_grad():
                    t_gathered_1 = teacher_patch_tokens_g1.reshape(-1, teacher_patch_tokens_g1.shape[-1])[
                        block_masks_1.reshape(-1).nonzero(as_tuple=True)[0]
                    ]
                    t_proj_1 = teacher.module.patchhead(t_gathered_1)

                s_proj_1 = student.module.patchhead(s_gathered_1)

                ibot_loss_g1 = ibot_patch_loss.forward_gathered(
                    s_proj_1, t_proj_1, weights_1, B, current_teacher_temp_ibot
                )
                del s_proj_1, t_proj_1, s_gathered_1, t_gathered_1
            else:
                ibot_loss_g1 = torch.tensor(0.0, device='cuda')

            # ---------- Semantic iBOT on global crop 1 ----------
            if args.use_semantic_ibot and semantic_token_masks is not None:
                ibot_accum = 0.0

                for sem_mask, sem_weight in zip(semantic_token_masks, semantic_masks_weights):
                    # Backbone forward with semantic mask tokens
                    sem_backbone_out = student.module.backbone(
                        teacher_global_crops[0], token_masks=sem_mask
                    )
                    sem_patch_raw = sem_backbone_out['patchtokens']  # [B, N, D]

                    # Store for prototype section (backbone-dim, ~0.4 GB each)
                    semantic_backbone_outputs.append((sem_patch_raw, sem_mask, sem_weight))

                    # Gather only masked tokens, then project
                    sem_s_gathered, sem_weights, _ = _gather_and_compute_weights(
                        sem_patch_raw, sem_mask, sem_weight
                    )

                    if sem_s_gathered is not None:
                        with torch.no_grad():
                            sem_t_gathered = teacher_patch_tokens_g1.reshape(-1, teacher_patch_tokens_g1.shape[-1])[
                                sem_mask.reshape(-1).nonzero(as_tuple=True)[0]
                            ]
                            sem_t_proj = teacher.module.patchhead(sem_t_gathered)

                        sem_s_proj = student.module.patchhead(sem_s_gathered)

                        loss_ibot = ibot_patch_loss.forward_gathered(
                            sem_s_proj, sem_t_proj, sem_weights, B, current_teacher_temp_ibot
                        )
                        ibot_accum += loss_ibot
                        del sem_s_proj, sem_t_proj, sem_s_gathered, sem_t_gathered

                    del sem_backbone_out

                n_ch = len(semantic_token_masks)
                semantic_ibot_loss_val = ibot_accum / n_ch

            # ---------- Block iBOT: Global crop 2 ----------
            s_gathered_2, weights_2, _ = _gather_and_compute_weights(
                student_patch_tokens_g2, block_masks_2, masks_weight_2
            )

            if s_gathered_2 is not None:
                with torch.no_grad():
                    t_gathered_2 = teacher_patch_tokens_g2.reshape(-1, teacher_patch_tokens_g2.shape[-1])[
                        block_masks_2.reshape(-1).nonzero(as_tuple=True)[0]
                    ]
                    t_proj_2 = teacher.module.patchhead(t_gathered_2)

                s_proj_2 = student.module.patchhead(s_gathered_2)

                ibot_loss_g2 = ibot_patch_loss.forward_gathered(
                    s_proj_2, t_proj_2, weights_2, B, current_teacher_temp_ibot
                )
                del s_proj_2, t_proj_2, s_gathered_2, t_gathered_2
            else:
                ibot_loss_g2 = torch.tensor(0.0, device='cuda')

            ibot_loss_val = (ibot_loss_g1 + ibot_loss_g2) / 2.0

        # ================================================================
        # Patch Prototype Clustering
        # Operates on backbone-dim [B, N, 768] — no memory concern.
        # Semantic prototype loss integrated here.
        # ================================================================
        semantic_clustering_loss = torch.tensor(0.0, device='cuda')
        semantic_teacher_proto_loss = torch.tensor(0.0, device='cuda')
        semantic_koleo_proto_loss = torch.tensor(0.0, device='cuda')

        if args.use_prototype_clustering:
            current_teacher_temp = dino_class_loss.teacher_temp_schedule(current_iteration)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                # ---------- Block mask prototype: Global crop 1 ----------
                clust_loss_g1, proto_loss_g1, koleo_proto_g1 = patch_prototype_loss(
                    teacher_patch_tokens_g1,
                    student_patch_tokens_g1,
                    block_masks_1,
                    prototype_bank,
                    current_iteration,
                    current_teacher_temp,
                    masks_weight=masks_weight_1
                )

                # ---------- Semantic prototype on global crop 1 ----------
                if args.use_semantic_prototypes and len(semantic_backbone_outputs) > 0:
                    clust_accum = 0.0
                    proto_accum = 0.0
                    koleo_accum = 0.0

                    for sem_patch_raw, sem_mask, sem_weight in semantic_backbone_outputs:
                        clust_ch, proto_ch, koleo_ch = patch_prototype_loss(
                            teacher_patch_tokens_g1,
                            sem_patch_raw,
                            sem_mask,
                            prototype_bank,
                            current_iteration,
                            current_teacher_temp,
                            masks_weight=sem_weight
                        )
                        clust_accum += clust_ch
                        proto_accum += proto_ch
                        koleo_accum += koleo_ch

                    n_ch = len(semantic_backbone_outputs)
                    semantic_clustering_loss = clust_accum / n_ch
                    semantic_teacher_proto_loss = proto_accum / n_ch
                    semantic_koleo_proto_loss = koleo_accum / n_ch

                # Free semantic backbone outputs
                del semantic_backbone_outputs
                semantic_backbone_outputs = []

                # ---------- Block mask prototype: Global crop 2 ----------
                clust_loss_g2, proto_loss_g2, koleo_proto_g2 = patch_prototype_loss(
                    teacher_patch_tokens_g2,
                    student_patch_tokens_g2,
                    block_masks_2,
                    prototype_bank,
                    current_iteration,
                    current_teacher_temp,
                    masks_weight=masks_weight_2
                )

                clustering_loss = (clust_loss_g1 + clust_loss_g2) / 2.0
                teacher_proto_loss = (proto_loss_g1 + proto_loss_g2) / 2.0
                koleo_proto_loss = (koleo_proto_g1 + koleo_proto_g2) / 2.0

            prototype_loss = teacher_proto_loss + koleo_proto_loss
            if args.use_semantic_prototypes:
                prototype_loss = prototype_loss + semantic_teacher_proto_loss + semantic_koleo_proto_loss
        else:
            del semantic_backbone_outputs

            clustering_loss = torch.tensor(0.0).cuda()
            teacher_proto_loss = torch.tensor(0.0).cuda()
            koleo_proto_loss = torch.tensor(0.0).cuda()
            prototype_loss = torch.tensor(0.0).cuda()

        # ========== Compute Total Losses ==========
        student_loss = (
            dino_class_loss_val +
            args.koleo_loss_weight * koleo_loss_val +
            args.ibot_loss_weight * ibot_loss_val +
            args.clustering_weight * clustering_loss +
            args.semantic_ibot_weight * semantic_ibot_loss_val +
            args.semantic_clustering_weight * semantic_clustering_loss
        )

        # ========== Backward and optimizer steps ==========
        if fp16_scaler is None:
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                optimizer_prototypes.step()

            optimizer_student.zero_grad()
            student_loss.backward()

            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.module.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.module.patchhead, args.freeze_last_layer_iters)

            optimizer_student.step()

        else:
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                optimizer_prototypes.step()

            optimizer_student.zero_grad()
            fp16_scaler.scale(student_loss).backward()
            fp16_scaler.unscale_(optimizer_student)

            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.module.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.module.patchhead, args.freeze_last_layer_iters)

            fp16_scaler.step(optimizer_student)
            fp16_scaler.update()

        # ========== EMA update teacher ==========
        with torch.no_grad():
            m = momentum_schedule[current_iteration]

            for param_q, param_k in zip(student.module.backbone.parameters(),
                                    teacher_without_ddp.backbone.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            for param_q, param_k in zip(student.module.classhead.parameters(),
                                    teacher_without_ddp.classhead.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            for param_q, param_k in zip(student.module.patchhead.parameters(),
                                    teacher_without_ddp.patchhead.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # ========== Clean cache periodically ==========
        if current_iteration % 100 == 0:
            torch.cuda.empty_cache()

        # ========== Visualize masks ==========
        if current_iteration % args.visualization_freq == 0 and current_iteration < 5000:
            if args.use_semantic_ibot and mask_model_frozen is not None:
                sample_image = teacher_global_crops[0][:1]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                        vis_masks = mask_model_frozen(sample_image)['masks']
                    save_iteration_masks_efficient(
                        sample_image,
                        vis_masks,
                        current_iteration,
                        os.path.join(args.output_dir, 'semantic_mask_visualizations'),
                        num_samples=1
                    )
                    vis_token_masks, _ = convert_semantic_masks_to_token_masks(
                        vis_masks.float(), patch_size=args.patch_size
                    )
                    h_patches = w_patches = 224 // args.patch_size
                    vis_token_grid = vis_token_masks.float().reshape(1, args.num_masks, h_patches, w_patches)
                    vis_token_upsampled = F.interpolate(
                        vis_token_grid, size=(224, 224), mode='nearest'
                    )
                    save_iteration_masks_efficient(
                        sample_image,
                        vis_token_upsampled,
                        current_iteration,
                        os.path.join(args.output_dir, 'semantic_token_mask_visualizations'),
                        num_samples=1
                    )

        # ========== Logging ==========
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(dino_class_loss=dino_class_loss_val.item())
        metric_logger.update(koleo_loss=koleo_loss_val.item())
        metric_logger.update(ibot_loss=ibot_loss_val.item())

        if args.use_prototype_clustering:
            metric_logger.update(clustering_loss=clustering_loss.item())
            metric_logger.update(proto_koleo_loss=koleo_proto_loss.item())
            metric_logger.update(teacher_proto_arrangement_loss=teacher_proto_loss.item())
            metric_logger.update(clustering_entropy=patch_prototype_loss.last_entropy)

        if args.use_semantic_ibot:
            metric_logger.update(semantic_ibot_loss=semantic_ibot_loss_val.item())
        if args.use_semantic_prototypes:
            metric_logger.update(semantic_clustering_loss=semantic_clustering_loss.item())
            metric_logger.update(semantic_proto_arrangement=semantic_teacher_proto_loss.item())
            metric_logger.update(semantic_proto_koleo=semantic_koleo_proto_loss.item())

        metric_logger.update(lr=optimizer_student.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer_student.param_groups[0]["weight_decay"])

        if utils.is_main_process() and current_iteration % 10 == 0:
            elapsed = time.time() - metric_logger.start_time
            progress = current_iteration / args.total_iterations
            eta_seconds = elapsed / max(progress, 1e-8) * (1 - progress)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if torch.cuda.is_available():
                memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            else:
                memory = 0

            metric_logger.synchronize_between_processes()
            print(f"It {current_iteration}/{args.total_iterations/1000:.0f}k (ETA {eta_string}), "
                f"Progress: {progress*100:.1f}%, max mem: {memory/1000:.1f} GB : {metric_logger}")

        # ========== Write to log file ==========
        if utils.is_main_process() and current_iteration % 100 == 0:
            log_stats = {
                **{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()},
                'iteration': current_iteration,
                'total_iterations': args.total_iterations,
                'progress_percentage': (current_iteration / args.total_iterations) * 100,
                'augmentation_config': {
                    'global_views': args.global_views,
                    'n_standard_local_crops': args.n_standard_local_crops,
                    'use_semantic_ibot': args.use_semantic_ibot,
                    'use_semantic_prototypes': args.use_semantic_prototypes,
                    'semantic_ibot_weight': args.semantic_ibot_weight if args.use_semantic_ibot else 0,
                    'semantic_clustering_weight': args.semantic_clustering_weight if args.use_semantic_prototypes else 0,
                    'semantic_masks_per_iteration': args.semantic_masks_per_iteration if args.use_semantic_ibot else 0,
                    'num_masks': args.num_masks if args.use_semantic_ibot else 0,
                    'total_student_views': total_student_views,
                }
            }

            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # ========== Save checkpoints ==========
        if current_iteration % args.save_checkpoint_freq == 0:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'dino_class_loss': dino_class_loss.state_dict(),
                'optimizer_student': optimizer_student.state_dict(),
                'iteration': current_iteration,
                'dataset_position': dataset_position,
                'args': args,
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'numpy_rng_state': np.random.get_state(),
                'random_rng_state': random.getstate(),
            }

            if args.use_prototype_clustering:
                if prototype_bank is not None:
                    save_dict['prototype_bank'] = prototype_bank.state_dict()
                if patch_prototype_loss is not None:
                    save_dict['patch_prototype_loss'] = patch_prototype_loss.state_dict()
                if optimizer_prototypes is not None:
                    save_dict['optimizer_prototypes'] = optimizer_prototypes.state_dict()

            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()

            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint_iter_{current_iteration:08d}.pth'))
            utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))

        current_iteration += 1

        if current_iteration % 100 == 0:
            if dist.is_initialized():
                dist.barrier()

    # ========== Final checkpoint and log ==========
    if utils.is_main_process():
        final_log_stats = {
            **{f'train_{k}': v.global_avg for k, v in metric_logger.meters.items()},
            'iteration': args.total_iterations,
            'training_completed': True,
        }

        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(final_log_stats) + "\n")

    print("Training Complete!")