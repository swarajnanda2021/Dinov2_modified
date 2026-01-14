"""
Main training loop for DINOv2 with iBOT and prototype clustering.
FSDP2 version with multi-augmentation support.
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
import numpy as np
import math

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import utils
from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, PatchPrototypeLoss
from data import ProportionalMultiDatasetWrapper
from .helpers import (
    load_pretrained_mask_model,
    load_pretrained_cellvit_model,
    apply_masks_to_images,
    apply_cellvit_masks,
    extract_local_crops_from_masked,
    extract_crops_from_cellvit_channel,
    generate_random_token_masks,
    generate_random_image_masks,
    calculate_total_student_views,
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)

from .fsdp_setup import apply_fsdp_wrapping
from .checkpoint_fsdp2 import save_checkpoint_fsdp2, load_checkpoint_fsdp2
from .param_groups_fsdp2 import get_params_groups_fsdp2


def train_dinov2(args):
    """
    Main training function for DINOv2 with iBOT and prototype clustering.
    FSDP2 version with multi-augmentation support.
    
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

    # Adversarial mask augmentation
    if args.use_adversarial_mask_augmentation:
        print(f"\nAdversarial Mask Augmentation: ENABLED")
        print(f"  Number of masks: {args.num_masks}")
        print(f"  Crops per mask: {args.crops_per_mask}")
        print(f"  Architecture: {args.mask_model_arch}")
    else:
        print(f"\nAdversarial Mask Augmentation: DISABLED")

    # CellViT augmentation
    if args.use_cellvit_augmentation:
        print(f"\nCellViT Augmentation: ENABLED")
        print(f"  Crops per channel: {args.cellvit_crops_per_channel}")
    else:
        print(f"\nCellViT Augmentation: DISABLED")

    # Random mask augmentation
    if args.use_random_mask_augmentation:
        print(f"\nRandom Mask Augmentation: ENABLED")
        print(f"  Number of masks: {args.random_num_masks}")
        print(f"  Crops per mask: {args.random_crops_per_mask}")
    else:
        print(f"\nRandom Mask Augmentation: DISABLED")

    total_student_views = calculate_total_student_views(args)
    print(f"\nTotal student views: {total_student_views}")
    print("================================================\n")
    
    # ============ Load pre-trained adversarial mask model (if enabled) ============
    mask_model_frozen = None
    if args.use_adversarial_mask_augmentation:
        if args.mask_checkpoint is None:
            raise ValueError("--use_adversarial_mask_augmentation is True but --mask_checkpoint not provided")
        
        if args.num_masks <= 0:
            raise ValueError("--use_adversarial_mask_augmentation is True but --num_masks must be > 0")
        
        mask_model_frozen = load_pretrained_mask_model(
            args.mask_checkpoint, 
            args.num_masks,
            mask_model_arch=args.mask_model_arch,
            mask_encoder_dim=getattr(args, 'mask_encoder_dim', 192)
        )
        mask_model_frozen = mask_model_frozen.cuda()
        mask_model_frozen.eval()
        
        for param in mask_model_frozen.parameters():
            param.requires_grad = False
        
        print(f"Loaded and froze adversarial mask model with {args.num_masks} masks")
        print(f"  Crops per mask: {args.crops_per_mask}")
    else:
        print("Adversarial mask augmentation disabled (--use_adversarial_mask_augmentation=False)")
    
    # ============ Load pre-trained CellViT model (if enabled) ============
    cellvit_model_frozen = None
    if args.use_cellvit_augmentation:
        if args.cellvit_checkpoint is None:
            raise ValueError("--use_cellvit_augmentation is True but --cellvit_checkpoint not provided")
        
        cellvit_model_frozen = load_pretrained_cellvit_model(args.cellvit_checkpoint, device='cuda')
        cellvit_model_frozen.eval()
        
        for param in cellvit_model_frozen.parameters():
            param.requires_grad = False
        
        print(f"Loaded and froze CellViT model for nuclei/background segmentation")
        print(f"  CellViT crops per channel: {args.cellvit_crops_per_channel}")
    else:
        print("CellViT augmentation disabled (--use_cellvit_augmentation=False)")

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

    teacher.load_state_dict(student.state_dict())  
    for param in teacher.parameters():
        param.requires_grad = False
        
    # Apply FSDP2 wrapping (in-place modification)
    student, teacher = apply_fsdp_wrapping(student, teacher, args)

    # Prototype bank stays as regular DDP (replicated for KoLeo)
    if args.use_prototype_clustering:
        prototype_bank = nn.parallel.DistributedDataParallel(prototype_bank, device_ids=[args.gpu])

    print("FSDP2 wrapping complete")

    # Disable gradients for teacher
    for param in teacher.parameters():
        param.requires_grad = False

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
    optimizer_student = torch.optim.AdamW([
        *get_params_groups_fsdp2(student.backbone),
        *get_params_groups_fsdp2(student.classhead),
        *get_params_groups_fsdp2(student.patchhead),
    ])

    optimizer_prototypes = None
    if args.use_prototype_clustering:
        optimizer_prototypes = torch.optim.AdamW(
            prototype_bank.module.parameters(),
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        print(f"Created prototype optimizer")
    else:
        print("No prototype optimizer (clustering disabled)")

    print(f"Created optimizers")

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

    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_fsdp2")
    if os.path.exists(checkpoint_dir):
        current_iteration = load_checkpoint_fsdp2(
            checkpoint_dir,
            student,
            teacher,
            prototype_bank,
            optimizer_student,
            optimizer_prototypes,
            args,
            dino_class_loss=dino_class_loss,
            patch_prototype_loss=patch_prototype_loss,
            fp16_scaler=fp16_scaler,
        )
        to_restore["iteration"] = current_iteration
    else:
        current_iteration = 0

    dataset_position = to_restore.get("dataset_position", 0)

    # ============ Set resume position in dataset ============
    if current_iteration > 0:
        global_samples_processed = current_iteration * args.batch_size_per_gpu * dist.get_world_size()
        trainset.set_resume_position(global_samples_processed)
        print(f"Resuming from iteration {current_iteration}")

    # ============ Restore RNGs ============
    utils.fix_random_seeds(args.seed + utils.get_rank())
    if current_iteration > 0:
        print(f"Resuming from iteration {current_iteration} with reseeded RNGs")
    
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
        
        # ========== Organize Crops for Sequence Packing ==========
        idx = 0
        
        # 1. Teacher views (global crops only, no masking)
        teacher_global_crops = []
        for i in range(args.global_views):
            teacher_global_crops.append(batch_data[idx].cuda(non_blocking=True))
            idx += 1
        
        # 2. Student views - collect ALL crops in order
        student_all_crops = []
        
        # Add global crops (same as teacher)
        for crop in teacher_global_crops:
            student_all_crops.append(crop)
        
        # Add standard local crops
        student_local_crops = []
        for i in range(args.n_standard_local_crops):
            crop = batch_data[idx].cuda(non_blocking=True)
            student_local_crops.append(crop)
            student_all_crops.append(crop)
            idx += 1
        
        # Get original images for masking and iBOT
        original_images = batch_data[-1].cuda(non_blocking=True)
        
        # 3. Generate adversarial masked views if configured
        masked_global_crops = []
        masked_local_crops_all = []

        if args.use_adversarial_mask_augmentation and mask_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                    mask_output = mask_model_frozen(original_images)
                    masks = mask_output['masks'].float()
            
            # Apply masks to create masked global views
            masked_images = apply_masks_to_images(original_images, masks)
            masked_global_crops = masked_images
            
            # Add masked global crops to student views
            student_all_crops.extend(masked_global_crops)
            
            # Extract local crops from masked images
            if args.crops_per_mask > 0:
                for masked_img in masked_images:
                    crops = extract_local_crops_from_masked(
                        masked_img, 
                        n_crops=args.crops_per_mask,
                        crop_size=args.local_crop_size
                    )
                    masked_local_crops_all.extend(crops)
                
                # Add masked local crops to student views
                student_all_crops.extend(masked_local_crops_all)

        # 4. Generate CellViT (Nuclei/Background) masks and masked views if configured
        cellvit_nuclei_global = None
        cellvit_background_global = None
        cellvit_nuclei_crops = []
        cellvit_background_crops = []

        if args.use_cellvit_augmentation and cellvit_model_frozen is not None:
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                    cellvit_output = cellvit_model_frozen(original_images)
                    cellvit_masks = cellvit_output['masks'].float()
            
            del cellvit_output
            
            # Apply masks to create nuclei and background views
            nuclei_images, background_images = apply_cellvit_masks(original_images, cellvit_masks)
            del cellvit_masks
            
            # Global views (224x224)
            cellvit_nuclei_global = nuclei_images.clone()
            cellvit_background_global = background_images.clone()
            
            # Add global views to student crops
            student_all_crops.append(cellvit_nuclei_global)
            student_all_crops.append(cellvit_background_global)
            
            # Extract local crops (96x96) from each channel
            if args.cellvit_crops_per_channel > 0:
                nuclei_crops = extract_crops_from_cellvit_channel(
                    nuclei_images,
                    n_crops=args.cellvit_crops_per_channel,
                    crop_size=args.local_crop_size
                )
                background_crops = extract_crops_from_cellvit_channel(
                    background_images,
                    n_crops=args.cellvit_crops_per_channel,
                    crop_size=args.local_crop_size
                )
                
                # Add to student views
                cellvit_nuclei_crops = nuclei_crops
                cellvit_background_crops = background_crops
                student_all_crops.extend(nuclei_crops)
                student_all_crops.extend(background_crops)
                
                del nuclei_crops, background_crops
            
            del nuclei_images, background_images
        
        # 5. Generate Random Rectangular Masks if configured
        random_masked_global_crops = []
        random_masked_local_crops = []
        
        if args.use_random_mask_augmentation:
            # Generate random rectangular masks
            random_masks = generate_random_image_masks(
                batch_size=original_images.shape[0],
                num_masks=args.random_num_masks,
                height=224,
                width=224,
                device=original_images.device,
            )
            
            # Apply masks to create masked global views
            random_masked_images = apply_masks_to_images(original_images, random_masks)
            random_masked_global_crops = random_masked_images
            
            # Add masked global crops to student views
            student_all_crops.extend(random_masked_global_crops)
            
            # Extract local crops from masked images
            if args.random_crops_per_mask > 0:
                for masked_img in random_masked_images:
                    crops = extract_local_crops_from_masked(
                        masked_img,
                        n_crops=args.random_crops_per_mask,
                        crop_size=args.local_crop_size
                    )
                    random_masked_local_crops.extend(crops)
                
                # Add masked local crops to student views
                student_all_crops.extend(random_masked_local_crops)
        
        # ========== Debug: Print shapes on first iteration ==========
        if current_iteration == 0 and utils.is_main_process():
            print("\n=== Crop Organization (First Iteration) ===")
            print(f"Teacher global crops: {len(teacher_global_crops)} crops")
            for i, crop in enumerate(teacher_global_crops):
                print(f"  Teacher crop {i}: {crop.shape}")
            print(f"Student total crops: {len(student_all_crops)} crops")
            for i, crop in enumerate(student_all_crops):
                print(f"  Student crop {i}: {crop.shape}")
            print(f"Original images: {original_images.shape}")
            
            if args.use_adversarial_mask_augmentation:
                print(f"\nAdversarial Mask Augmentation:")
                print(f"  Masked global crops: {len(masked_global_crops)}")
                print(f"  Masked local crops: {len(masked_local_crops_all)}")
            
            if args.use_cellvit_augmentation:
                print(f"\nCellViT Augmentation:")
                print(f"  Nuclei global: {cellvit_nuclei_global.shape if cellvit_nuclei_global is not None else 'None'}")
                print(f"  Background global: {cellvit_background_global.shape if cellvit_background_global is not None else 'None'}")
                print(f"  Nuclei crops: {len(cellvit_nuclei_crops)} x {cellvit_nuclei_crops[0].shape if cellvit_nuclei_crops else 'None'}")
                print(f"  Background crops: {len(cellvit_background_crops)} x {cellvit_background_crops[0].shape if cellvit_background_crops else 'None'}")
            
            if args.use_random_mask_augmentation:
                print(f"\nRandom Mask Augmentation:")
                print(f"  Random masked global crops: {len(random_masked_global_crops)}")
                print(f"  Random masked local crops: {len(random_masked_local_crops)}")
            
            print("="*50 + "\n")
        
        # ========== Update learning rates ==========
        for i, param_group in enumerate(optimizer_student.param_groups):
            param_group["lr"] = student_lr_schedule[current_iteration]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[current_iteration]

        if args.use_prototype_clustering and optimizer_prototypes is not None:
            for param_group in optimizer_prototypes.param_groups:
                param_group["lr"] = proto_lr_schedule[current_iteration]

        optimizer_student.zero_grad()
        if args.use_prototype_clustering and optimizer_prototypes is not None:
            optimizer_prototypes.zero_grad()

        # ========== Forward passes and loss computation ==========
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
            # ========== DINO Loss with Sequence Packing ==========
            
            # Teacher forward (only global crops, no masking)
            with torch.no_grad():
                teacher_output = teacher(teacher_global_crops, token_masks=None, mode='dino')
                teacher_cls_outputs = teacher_output['cls_outputs']
            
            # Student forward (all crops, packed together)
            student_output = student(student_all_crops, token_masks=None, mode='dino')
            student_cls_outputs = student_output['cls_outputs']
            
            # Compute DINO loss
            dino_class_loss_val = dino_class_loss(
                student_cls_outputs,
                teacher_cls_outputs,
                current_iteration
            )
            
            # ========== KoLeo Loss on Global CLS Tokens ==========
            # Count global views including adversarial masks
            num_global_total = args.global_views
            if args.use_adversarial_mask_augmentation:
                num_global_total += args.num_masks
            
            global_features_list = student_output['features_list'][:num_global_total]
            global_cls_tokens = [feat_dict['clstoken'] for feat_dict in global_features_list]
            
            koleo_loss_val = torch.tensor(0.0).cuda()
            if len(global_cls_tokens) > 0:
                koleo_loss_val = sum(dino_koleo_loss(token) for token in global_cls_tokens) / len(global_cls_tokens)
            
            # ========== iBOT Loss (Canonical) ==========
            batch_size = original_images.shape[0]
            n_patches_h = n_patches_w = 224 // args.patch_size
            
            # Generate random token masks
            random_token_masks = generate_random_token_masks(
                batch_size, n_patches_h, n_patches_w,
                args.token_mask_ratio, original_images.device
            )
            
            # Teacher forward (no masking)
            with torch.no_grad():
                teacher_ibot_output = teacher(original_images, token_masks=None, mode='ibot')
                teacher_patch_outputs = teacher_ibot_output['patch_outputs']
            
            # Student forward (with masking)
            student_ibot_output = student(original_images, token_masks=random_token_masks, mode='ibot')
            student_patch_outputs = student_ibot_output['patch_outputs']
            
            # Compute iBOT loss
            ibot_loss_val = ibot_patch_loss.forward_masked(
                student_patch_outputs,
                teacher_patch_outputs,
                random_token_masks,
                n_masked_patches_tensor=random_token_masks.sum(),
                teacher_temp=dino_class_loss.teacher_temp_schedule(current_iteration)
            )

        # ========== Patch Prototype Clustering ==========
        if args.use_prototype_clustering:
            # Get current temperature (shared across all modes)
            current_teacher_temp = dino_class_loss.teacher_temp_schedule(current_iteration)
            
            if args.clustering_mode == 'visible':
                # Use visible patch tokens from ibot forward pass
                teacher_patch_features = teacher_ibot_output['features']['patchtokens']
                student_patch_features = student_ibot_output['features']['patchtokens']

                # random_token_masks: [B, N] where True = masked, False = visible
                visible_mask = ~random_token_masks  # [B, N] boolean
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                    clustering_loss, teacher_proto_loss, koleo_proto_loss = patch_prototype_loss(
                        teacher_patch_features,
                        student_patch_features,
                        visible_mask,
                        prototype_bank,
                        current_teacher_temp
                    )
                
                # Combine prototype losses
                prototype_loss = teacher_proto_loss + koleo_proto_loss
                
            elif args.clustering_mode == 'masked':
                # Use masked patch tokens from ibot forward pass
                teacher_patch_features = teacher_ibot_output['features']['patchtokens']
                student_patch_features = student_ibot_output['features']['patchtokens']
                
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                    clustering_loss, teacher_proto_loss, koleo_proto_loss = patch_prototype_loss(
                        teacher_patch_features,
                        student_patch_features,
                        random_token_masks,  # True = masked positions
                        prototype_bank,
                        current_teacher_temp
                    )
                
                # Combine prototype losses
                prototype_loss = teacher_proto_loss + koleo_proto_loss

            elif args.clustering_mode == 'separate':
                # Do a separate forward pass and use all patch tokens
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                    student_cluster = student(original_images, token_masks=None, mode='ibot')
                    with torch.no_grad():
                        teacher_cluster = teacher(original_images, token_masks=None, mode='ibot')
                    
                    all_mask = torch.ones_like(random_token_masks, dtype=torch.bool)  # Cluster all
                    clustering_loss, teacher_proto_loss, koleo_proto_loss = patch_prototype_loss(
                        teacher_cluster['features']['patchtokens'],
                        student_cluster['features']['patchtokens'],
                        all_mask,  # All positions
                        prototype_bank,
                        current_teacher_temp
                    )
                    
                    # Combine prototype losses
                    prototype_loss = teacher_proto_loss + koleo_proto_loss
                    
        else:
            clustering_loss = torch.tensor(0.0).cuda()
            teacher_proto_loss = torch.tensor(0.0).cuda()
            koleo_proto_loss = torch.tensor(0.0).cuda()
            prototype_loss = torch.tensor(0.0).cuda()

        # ========== Compute Total Losses ==========
        # Student loss: DINO + KoLeo + iBOT + Clustering prediction
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
            student_loss = (
                dino_class_loss_val +
                args.koleo_loss_weight * koleo_loss_val +
                args.ibot_loss_weight * ibot_loss_val +
                args.clustering_weight * clustering_loss
            )

        # ========== Backward and optimizer steps ==========
        if fp16_scaler is None:
            # ===== NON-MIXED PRECISION =====
            
            # 1. Prototype backward (if enabled)
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                prototype_loss.backward()
                optimizer_prototypes.step()
                optimizer_prototypes.zero_grad()
            
            # 2. Student backward
            student_loss.backward()
            
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.patchhead, args.freeze_last_layer_iters)
            
            optimizer_student.step()
            optimizer_student.zero_grad()
            
        else:
            # ===== MIXED PRECISION =====
            
            # 1. Prototype backward (NO scaler - uses bfloat16 directly)
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                prototype_loss.backward()
                optimizer_prototypes.step()
                optimizer_prototypes.zero_grad()
            
            # 2. Student backward (WITH scaler)
            fp16_scaler.scale(student_loss).backward()
            fp16_scaler.unscale_(optimizer_student)
            
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.patchhead, args.freeze_last_layer_iters)
            
            fp16_scaler.step(optimizer_student)
            fp16_scaler.update()
            optimizer_student.zero_grad()

        # ========== EMA update teacher ==========
        with torch.no_grad():
            m = momentum_schedule[current_iteration]
            
            # Build parameter lists once and cache them (like DINOv3)
            if not hasattr(train_dinov2, '_ema_param_lists'):
                student_param_list = []
                teacher_param_list = []
                
                # Collect parameters from FSDP2-wrapped modules
                for student_module, teacher_module in [
                    (student.backbone, teacher.backbone),
                    (student.classhead, teacher.classhead),
                    (student.patchhead, teacher.patchhead),
                ]:
                    for student_param, teacher_param in zip(
                        student_module.parameters(),
                        teacher_module.parameters()
                    ):
                        student_param_list.append(student_param)
                        teacher_param_list.append(teacher_param)
                
                train_dinov2._ema_param_lists = (student_param_list, teacher_param_list)
                if utils.is_main_process():
                    print(f"Cached {len(student_param_list)} parameters for EMA")
            else:
                student_param_list, teacher_param_list = train_dinov2._ema_param_lists
            
            # FSDP2-compatible vectorized operations
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1.0 - m)

        # ========== Clean cache periodically ==========
        if current_iteration % 100 == 0:
            torch.cuda.empty_cache()
        
        # ========== Visualize masks ==========
        if current_iteration % args.visualization_freq == 0 and current_iteration < 5000:
            # Visualize adversarial masks
            if args.use_adversarial_mask_augmentation and mask_model_frozen is not None:
                sample_image = original_images[:1]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                        vis_masks = mask_model_frozen(sample_image)['masks']
                    save_iteration_masks_efficient(
                        sample_image,
                        vis_masks,
                        current_iteration,
                        os.path.join(args.output_dir, 'adversarial_mask_visualizations'),
                        num_samples=1
                    )
            
            # Visualize CellViT masks
            if args.use_cellvit_augmentation and cellvit_model_frozen is not None:
                sample_image = original_images[:1]
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                        cellvit_vis = cellvit_model_frozen(sample_image)
                        cellvit_vis_masks = cellvit_vis['masks']

                    save_iteration_masks_efficient(
                        sample_image,
                        cellvit_vis_masks,
                        current_iteration,
                        os.path.join(args.output_dir, 'cellvit_mask_visualizations'),
                        num_samples=1
                    )
            
            # Visualize Random masks
            if args.use_random_mask_augmentation:
                sample_image = original_images[:1]
                random_vis_masks = generate_random_image_masks(
                    batch_size=1,
                    num_masks=args.random_num_masks,
                    height=224,
                    width=224,
                    device=sample_image.device,
                )
                save_iteration_masks_efficient(
                    sample_image,
                    random_vis_masks,
                    current_iteration,
                    os.path.join(args.output_dir, 'random_mask_visualizations'),
                    num_samples=1
                )
        
        # ========== Logging ==========
        metric_logger.update(student_loss=student_loss.item())
        metric_logger.update(dino_class_loss=dino_class_loss_val.item())
        metric_logger.update(koleo_loss=koleo_loss_val.item())
        metric_logger.update(ibot_loss=ibot_loss_val.item())
        if args.use_prototype_clustering:
            metric_logger.update(clustering_loss=patch_prototype_loss.last_prediction_loss)
            metric_logger.update(proto_koleo_loss=patch_prototype_loss.last_koleo_loss)
            metric_logger.update(teacher_proto_arrangement_loss=patch_prototype_loss.last_arrangement_loss)
            metric_logger.update(clustering_entropy=patch_prototype_loss.last_entropy)
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
                    'adversarial_mask_augmentation': args.use_adversarial_mask_augmentation,
                    'num_masks': args.num_masks if args.use_adversarial_mask_augmentation else 0,
                    'crops_per_mask': args.crops_per_mask if args.use_adversarial_mask_augmentation else 0,
                    'cellvit_augmentation': args.use_cellvit_augmentation,
                    'cellvit_crops_per_channel': args.cellvit_crops_per_channel if args.use_cellvit_augmentation else 0,
                    'random_mask_augmentation': args.use_random_mask_augmentation,
                    'random_num_masks': args.random_num_masks if args.use_random_mask_augmentation else 0,
                    'random_crops_per_mask': args.random_crops_per_mask if args.use_random_mask_augmentation else 0,
                    'total_student_views': total_student_views,
                }
            }
            
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # ========== Save checkpoints ==========
        if current_iteration % args.save_checkpoint_freq == 0:
            if utils.is_main_process():
                print(f"Preparing to save checkpoint at iteration {current_iteration}...")
            
            checkpoint_dir = os.path.join(args.output_dir, "checkpoint_fsdp2")
            save_checkpoint_fsdp2(
                checkpoint_dir,
                current_iteration,
                student,
                teacher,
                prototype_bank,
                optimizer_student,
                optimizer_prototypes,
                args,
                dino_class_loss=dino_class_loss,
                patch_prototype_loss=patch_prototype_loss,
                fp16_scaler=fp16_scaler,
            )
            
            if utils.is_main_process():
                print(f"Checkpoint saved successfully at iteration {current_iteration}")
        
        current_iteration += 1

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
