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
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import utils
from models import CombinedModelDINO, LinearPrototypeBank, ModernViT, DINOHead
from losses import DINOLoss, iBOTPatchLoss, KoLeoLoss, PatchPrototypeLoss
from data import DINOv2PathologyDataset
from .helpers import (
    load_pretrained_mask_model,
    apply_masks_to_images,
    extract_local_crops_from_masked,
    generate_random_token_masks,
    calculate_total_student_views,
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)


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
    print(f"Number of masks: {args.num_masks}")
    print(f"Crops per mask: {args.crops_per_mask}")
    total_student_views = calculate_total_student_views(args)
    print(f"Total student views: {total_student_views}")
    print("================================================\n")
    
    # ============ Load pre-trained mask model ============
    mask_model_frozen = None
    if args.num_masks > 0:
        mask_model_frozen = load_pretrained_mask_model(args.mask_checkpoint, args.num_masks)
        mask_model_frozen = mask_model_frozen.cuda()
        mask_model_frozen.eval()
        
        for param in mask_model_frozen.parameters():
            param.requires_grad = False
        
        print(f"Loaded and froze mask model with {args.num_masks} masks")
    else:
        print("No mask model loaded (num_masks=0)")

    # ============ Create dataset ============
    trainset = DINOv2PathologyDataset(
        base_dir=args.base_dir,
        index_file="dataset_index.pkl",
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
    
    # Wrap prototype bank with DDP
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
    optimizer_student = torch.optim.AdamW([
        *utils.get_params_groups(student.module.backbone),
        *utils.get_params_groups(student.module.classhead),
        *utils.get_params_groups(student.module.patchhead),
    ])
    
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
    
    checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth")
    loaded_checkpoint = None
    if os.path.exists(checkpoint_path):
        try:
            loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Pre-loaded checkpoint from iteration {loaded_checkpoint.get('iteration', 'N/A')}.")
        except Exception as e:
            print(f"Could not pre-load checkpoint. Starting fresh. Error: {e}")
            loaded_checkpoint = None

    # Build checkpoint loading kwargs conditionally
    checkpoint_kwargs = {
        'student': student,
        'teacher': teacher,
        'optimizer_student': optimizer_student,
        'fp16_scaler': fp16_scaler,
        'dino_class_loss': dino_class_loss,
    }

    # Add prototype-related modules only if enabled
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
        
        # 3. Generate masks and masked views if configured
        masked_global_crops = []
        masked_local_crops_all = []
        
        if args.num_masks > 0 and mask_model_frozen is not None:
            with torch.no_grad():
                mask_output = mask_model_frozen(original_images)
                masks = mask_output['masks']
            
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
            num_global_total = args.global_views + args.num_masks
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
                teacher_temp=dino_class_loss.teacher_temp_schedule(current_iteration)
            )

        # ========== Patch Prototype Clustering (Separate from student loss) ==========
        if args.use_prototype_clustering:
            # TEACHER PATH: Generate targets with gradients for prototype bank
            # This runs OUTSIDE the student's autocast context
            teacher_patch_features = teacher_ibot_output['features']['patchtokens']
            
            # Compute targets with grad enabled for prototype bank
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                teacher_patch_features_norm = F.normalize(teacher_patch_features, p=2, dim=-1)
                teacher_logits_all = prototype_bank(teacher_patch_features_norm)
                
                B, N, K = teacher_logits_all.shape
                teacher_logits_flat = teacher_logits_all.reshape(B * N, -1)
                
                # Sinkhorn-Knopp to get targets (no_grad inside the function)
                Q_tilde_flat = patch_prototype_loss.sinkhorn_knopp(
                    teacher_logits_flat, 
                    args.clustering_teacher_temp
                )
                Q_tilde_all = Q_tilde_flat.reshape(B, N, -1)
                
                # Arrangement Loss: KL(Q̃ || Q) - only trains prototype bank
                teacher_log_probs_all = F.log_softmax(
                    teacher_logits_all / args.clustering_teacher_temp, 
                    dim=-1
                )
                teacher_proto_loss = -torch.sum(Q_tilde_all.detach() * teacher_log_probs_all) / (B * N)
                
                # KoLeo on prototype weights
                weight_normalized = F.normalize(prototype_bank.module.proto_layer.weight, p=2, dim=1)
                koleo_proto_loss = patch_prototype_loss.koleo_loss(weight_normalized)
            
            # ========== CRITICAL: Detach targets before student uses them ==========
            Q_tilde_masked = Q_tilde_all[random_token_masks].detach()  # ← DETACH HERE
            
            # STUDENT PATH: Prediction loss (only trains student)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
                student_patch_features = student_ibot_output['features']['patchtokens']
                student_norm_masked = F.normalize(student_patch_features[random_token_masks], p=2, dim=-1)
                
                if student_norm_masked.shape[0] > 0:
                    student_logits_masked = prototype_bank(student_norm_masked)
                    student_log_probs_masked = F.log_softmax(
                        student_logits_masked / args.clustering_student_temp, 
                        dim=-1
                    )
                    clustering_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked) / student_norm_masked.shape[0]
                else:
                    clustering_loss = torch.tensor(0.0, device=original_images.device)
            
            # Update metrics
            with torch.no_grad():
                student_probs = torch.exp(student_log_probs_masked) if student_norm_masked.shape[0] > 0 else None
                if student_probs is not None:
                    entropy = -(student_probs * student_log_probs_masked).sum(dim=-1).mean()
                    patch_prototype_loss.last_entropy = (entropy / math.log(args.num_prototypes)).item()
                    
                    assignments = torch.argmax(Q_tilde_masked, dim=-1)
                    usage = torch.bincount(assignments, minlength=args.num_prototypes).float()
                    if dist.is_initialized():
                        dist.all_reduce(usage)
                    patch_prototype_loss.last_usage_std = (usage.std() / (usage.mean() + 1e-6)).item()
                
                patch_prototype_loss.last_prediction_loss = clustering_loss.item()
                patch_prototype_loss.last_arrangement_loss = teacher_proto_loss.item()
                patch_prototype_loss.last_koleo_loss = koleo_proto_loss.item()
        else:
            clustering_loss = torch.tensor(0.0).cuda()
            teacher_proto_loss = torch.tensor(0.0).cuda()
            koleo_proto_loss = torch.tensor(0.0).cuda()

        # ========== Compute Total Losses ==========
        # Student loss: DINO + KoLeo + iBOT + Clustering prediction
        student_loss = (
            dino_class_loss_val +
            args.koleo_loss_weight * koleo_loss_val +
            args.ibot_loss_weight * ibot_loss_val +
            args.clustering_weight * clustering_loss
        )

        # Prototype loss: Arrangement + KoLeo (separate backward)
        prototype_loss = teacher_proto_loss + koleo_proto_loss

        # ========== Backward and optimizer steps ==========
        if fp16_scaler is None:
            # ===== NON-MIXED PRECISION =====
            
            # 1. Prototype backward (if enabled)
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                optimizer_prototypes.step()
            
            # 2. Student backward
            optimizer_student.zero_grad()
            student_loss.backward()
            
            if args.clip_grad:
                utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(current_iteration, student.module.classhead, args.freeze_last_layer_iters)
            utils.cancel_gradients_last_layer(current_iteration, student.module.patchhead, args.freeze_last_layer_iters)
            
            optimizer_student.step()
            
        else:
            # ===== MIXED PRECISION =====
            
            # 1. Prototype backward (NO scaler - uses bfloat16 directly)
            if args.use_prototype_clustering and optimizer_prototypes is not None:
                optimizer_prototypes.zero_grad()
                prototype_loss.backward()
                optimizer_prototypes.step()
            
            # 2. Student backward (WITH scaler)
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
        if current_iteration % args.visualization_freq == 0 and current_iteration < 5000 and args.num_masks > 0:
            sample_image = original_images[:1]
            with torch.no_grad():
                vis_masks = mask_model_frozen(sample_image)['masks']
                save_iteration_masks_efficient(
                    sample_image,
                    vis_masks,
                    current_iteration,
                    os.path.join(args.output_dir, 'mask_visualizations'),
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
                    'num_masks': args.num_masks,
                    'crops_per_mask': args.crops_per_mask,
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

            # Add prototype-related state only if enabled
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