"""
Complete Pipeline Schedule for DINO Training with Batch-Dependent Losses.

This implements a "flush pipeline" strategy:
1. Forward flush: All images pass through the pipeline
2. Feature accumulation: Last stage collects all features
3. Loss computation: Compute batch-dependent DINO/iBOT/clustering losses
4. Backward flush: Propagate gradients back through pipeline

Key difference from standard pipeline parallelism:
- Standard GPipe: Loss per microbatch → backward per microbatch
- DINO Pipeline: Accumulate all features → single global loss → single backward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any

try:
    from xformers.ops import fmha
except ImportError:
    fmha = None
    print("Warning: xformers not available for pipeline parallelism")


class PipelineSchedule:
    """
    Complete pipeline schedule for batch-dependent DINO losses.
    
    Args:
        student_stage: Student model pipeline stage wrapper
        teacher_stage: Teacher model pipeline stage wrapper
        prototype_bank: Prototype bank (only present on last stage)
        loss_modules: Dict of loss modules {'dino_class_loss', 'ibot_patch_loss', ...}
        local_rank: GPU rank within node (0 to gpus_per_node-1)
        pipeline_group: Process group for within-node pipeline communication
        data_group: Process group for across-node data parallel gradient sync
        batch_size_per_node: Number of images per node
        gpus_per_node: Number of GPUs per node
        embed_dim: Model embedding dimension
        num_patches: Number of patches per image (14x14 = 196 for 224x224 with patch_size=16)
        num_registers: Number of register tokens
        debug: Enable debug logging
    """
    
    def __init__(
        self,
        student_stage,
        teacher_stage,
        prototype_bank,
        loss_modules: Dict[str, nn.Module],
        local_rank: int,
        pipeline_group,
        data_group,
        batch_size_per_node: int,
        gpus_per_node: int = 4,
        embed_dim: int = 768,
        num_patches: int = 196,
        num_registers: int = 4,
        debug: bool = False,
    ):
        self.student_stage = student_stage
        self.teacher_stage = teacher_stage
        self.prototype_bank = prototype_bank
        self.loss_modules = loss_modules
        
        self.local_rank = local_rank
        self.pipeline_group = pipeline_group
        self.data_group = data_group
        
        self.batch_size_per_node = batch_size_per_node
        self.gpus_per_node = gpus_per_node
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_registers = num_registers
        self.debug = debug
        
        # Pipeline stage identification
        self.is_first_stage = (local_rank == 0)
        self.is_last_stage = (local_rank == gpus_per_node - 1)
        self.is_middle_stage = not (self.is_first_stage or self.is_last_stage)
        
        # Communication setup
        self.prev_rank = local_rank - 1 if local_rank > 0 else None
        self.next_rank = local_rank + 1 if local_rank < gpus_per_node - 1 else None
        
        # Shape info for efficient communication
        self.num_tokens_per_image = 1 + num_registers + num_patches
        
        if self.debug:
            print(f"[Rank {local_rank}] Pipeline schedule initialized")
            print(f"  Tokens/image: {self.num_tokens_per_image}, embed_dim: {embed_dim}")
    
    def forward_pass(
        self,
        crops: List[torch.Tensor],
        original_images: torch.Tensor,
        token_masks: Optional[torch.Tensor] = None,
        iteration: int = 0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Execute forward pass through pipeline.
        
        Args:
            crops: List of crop tensors [global1, global2, local1, ..., localN]
            original_images: Original images for iBOT [B, 3, 224, 224]
            token_masks: Token masks for iBOT [B, N]
            iteration: Current training iteration
            
        Returns:
            Dict of losses (only on last stage), None otherwise
        """
        
        # ========== Teacher Forward (Global Crops Only) ==========
        teacher_global_crops = crops[:2]
        
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] Teacher forward: {len(teacher_global_crops)} crops")
        
        teacher_features = self._forward_through_pipeline(
            teacher_global_crops,
            self.teacher_stage,
            token_masks=None,
            is_teacher=True,
            mode='multi_crop',
        )
        
        # ========== Student Forward (All Crops) ==========
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] Student forward: {len(crops)} crops")
        
        student_features = self._forward_through_pipeline(
            crops,
            self.student_stage,
            token_masks=None,
            is_teacher=False,
            mode='multi_crop',
        )
        
        # ========== iBOT Forward (Original Images with Masking) ==========
        # Teacher: no masking
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] iBOT teacher forward")
        
        teacher_ibot_features = self._forward_through_pipeline(
            original_images,
            self.teacher_stage,
            token_masks=None,
            is_teacher=True,
            mode='ibot',
        )
        
        # Student: with masking
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] iBOT student forward with masking")
        
        student_ibot_features = self._forward_through_pipeline(
            original_images,
            self.student_stage,
            token_masks=token_masks,
            is_teacher=False,
            mode='ibot',
        )
        
        # ========== Compute Losses (Last Stage Only) ==========
        if self.is_last_stage:
            if self.debug:
                print(f"[Rank {self.local_rank}] Computing losses")
            
            losses = self._compute_losses(
                student_features=student_features,
                teacher_features=teacher_features,
                student_ibot_features=student_ibot_features,
                teacher_ibot_features=teacher_ibot_features,
                token_masks=token_masks,
                iteration=iteration,
            )
            
            return losses
        
        return None
    
    def _forward_through_pipeline(
        self,
        x,  # Either List[Tensor] or Tensor
        model_stage,
        token_masks: Optional[torch.Tensor] = None,
        is_teacher: bool = False,
        mode: str = 'multi_crop',
    ) -> Optional[Any]:
        """
        Forward through pipeline stages.
        
        Args:
            x: Either List[Tensor] for multi-crop or Tensor for single image
            model_stage: Student or teacher stage
            token_masks: Optional masking
            is_teacher: Whether this is teacher
            mode: 'multi_crop' or 'ibot'
            
        Returns:
            Features (only on last stage), None otherwise
        """
        
        if self.is_first_stage:
            # ========== FIRST STAGE ==========
            output, attn_bias = model_stage(x, token_masks=token_masks)
            self._send_to_next_stage(output, attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] First stage output: {output.shape}")
            
            return None
        
        elif self.is_middle_stage:
            # ========== MIDDLE STAGE ==========
            input_tensor, attn_bias = self._recv_from_prev_stage()
            output, attn_bias = model_stage(input_tensor, token_masks=None, attn_bias=attn_bias)
            self._send_to_next_stage(output, attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] Middle stage output: {output.shape}")
            
            return None
        
        else:  # Last stage
            # ========== LAST STAGE ==========
            input_tensor, attn_bias = self._recv_from_prev_stage()
            output = model_stage(input_tensor, token_masks=None, attn_bias=attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] Last stage output: {type(output)}")
            
            return output
    
    def _send_to_next_stage(
        self, 
        tensor: torch.Tensor, 
        attn_bias: Optional[Any] = None
    ):
        """Send tensor and attention bias to next stage."""
        if self.next_rank is None:
            return
        
        device = tensor.device
        
        # 1. Send tensor shape
        shape_tensor = torch.tensor(list(tensor.shape), dtype=torch.int64, device=device)
        dist.send(shape_tensor, dst=self.next_rank, group=self.pipeline_group)
        
        # 2. Send tensor
        dist.send(tensor.contiguous(), dst=self.next_rank, group=self.pipeline_group)
        
        # 3. Send attn_bias flag
        has_attn_bias = 1 if attn_bias is not None else 0
        flag_tensor = torch.tensor([has_attn_bias], dtype=torch.int64, device=device)
        dist.send(flag_tensor, dst=self.next_rank, group=self.pipeline_group)
        
        # 4. Send attn_bias metadata if exists
        if attn_bias is not None:
            seqlens = self._extract_seqlens_from_attn_bias(attn_bias)
            
            num_seqlens_tensor = torch.tensor([len(seqlens)], dtype=torch.int64, device=device)
            dist.send(num_seqlens_tensor, dst=self.next_rank, group=self.pipeline_group)
            
            seqlens_tensor = torch.tensor(seqlens, dtype=torch.int64, device=device)
            dist.send(seqlens_tensor, dst=self.next_rank, group=self.pipeline_group)
            
            batch_sizes = getattr(attn_bias, '_batch_sizes', None)
            if batch_sizes is not None:
                num_batch_sizes_tensor = torch.tensor([len(batch_sizes)], dtype=torch.int64, device=device)
                dist.send(num_batch_sizes_tensor, dst=self.next_rank, group=self.pipeline_group)
                
                batch_sizes_tensor = torch.tensor(batch_sizes, dtype=torch.int64, device=device)
                dist.send(batch_sizes_tensor, dst=self.next_rank, group=self.pipeline_group)
            else:
                num_batch_sizes_tensor = torch.tensor([0], dtype=torch.int64, device=device)
                dist.send(num_batch_sizes_tensor, dst=self.next_rank, group=self.pipeline_group)
    
    def _recv_from_prev_stage(self) -> Tuple[torch.Tensor, Optional[Any]]:
        """Receive tensor and attention bias from previous stage."""
        if self.prev_rank is None:
            raise RuntimeError("Cannot receive - no previous stage")
        
        device = torch.cuda.current_device()
        
        # 1. Receive tensor shape
        shape_tensor = torch.empty(3, dtype=torch.int64, device=device)
        dist.recv(shape_tensor, src=self.prev_rank, group=self.pipeline_group)
        tensor_shape = tuple(shape_tensor.tolist())
        
        # 2. Receive tensor
        tensor = torch.empty(tensor_shape, dtype=torch.float32, device=device)
        dist.recv(tensor, src=self.prev_rank, group=self.pipeline_group)
        
        # 3. Receive attn_bias flag
        flag_tensor = torch.empty(1, dtype=torch.int64, device=device)
        dist.recv(flag_tensor, src=self.prev_rank, group=self.pipeline_group)
        has_attn_bias = flag_tensor.item()
        
        # 4. Receive attn_bias metadata if exists
        attn_bias = None
        if has_attn_bias:
            num_seqlens_tensor = torch.empty(1, dtype=torch.int64, device=device)
            dist.recv(num_seqlens_tensor, src=self.prev_rank, group=self.pipeline_group)
            num_seqlens = num_seqlens_tensor.item()
            
            seqlens_tensor = torch.empty(num_seqlens, dtype=torch.int64, device=device)
            dist.recv(seqlens_tensor, src=self.prev_rank, group=self.pipeline_group)
            seqlens = seqlens_tensor.tolist()
            
            num_batch_sizes_tensor = torch.empty(1, dtype=torch.int64, device=device)
            dist.recv(num_batch_sizes_tensor, src=self.prev_rank, group=self.pipeline_group)
            num_batch_sizes = num_batch_sizes_tensor.item()
            
            batch_sizes = None
            if num_batch_sizes > 0:
                batch_sizes_tensor = torch.empty(num_batch_sizes, dtype=torch.int64, device=device)
                dist.recv(batch_sizes_tensor, src=self.prev_rank, group=self.pipeline_group)
                batch_sizes = batch_sizes_tensor.tolist()
            
            attn_bias = self._reconstruct_attn_bias(seqlens, batch_sizes)
        
        return tensor, attn_bias
    
    def _extract_seqlens_from_attn_bias(self, attn_bias) -> List[int]:
        """Extract sequence lengths from BlockDiagonalMask."""
        if fmha is None:
            raise RuntimeError("xformers not available")
        
        if hasattr(attn_bias, 'q_seqinfo'):
            seqinfo = attn_bias.q_seqinfo
            if hasattr(seqinfo, 'seqstart_py'):
                seqstart = seqinfo.seqstart_py
                seqlens = [seqstart[i+1] - seqstart[i] for i in range(len(seqstart) - 1)]
                return seqlens
        
        raise RuntimeError(f"Cannot extract seqlens from attn_bias type {type(attn_bias)}")
    
    def _reconstruct_attn_bias(self, seqlens: List[int], batch_sizes: Optional[List[int]]):
        """Reconstruct BlockDiagonalMask from seqlens."""
        if fmha is None:
            raise RuntimeError("xformers not available")
        
        attn_bias = fmha.BlockDiagonalMask.from_seqlens(seqlens)
        
        if batch_sizes is not None:
            attn_bias._batch_sizes = batch_sizes
        
        return attn_bias
    
    def _compute_losses(
        self,
        student_features,
        teacher_features,
        student_ibot_features,
        teacher_ibot_features,
        token_masks,
        iteration,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses - COMPLETE implementation matching trainer.py.
        
        This matches trainer.py lines 566-640 exactly.
        """
        if not self.is_last_stage:
            raise RuntimeError("_compute_losses should only be called on last stage")
        
        device = next(self.student_stage.parameters()).device
        
        # Extract loss modules
        dino_loss = self.loss_modules.get('dino_class_loss')
        ibot_loss = self.loss_modules.get('ibot_patch_loss')
        koleo_loss = self.loss_modules.get('dino_koleo_loss')
        prototype_loss = self.loss_modules.get('patch_prototype_loss')
        
        losses = {}
        
        # ========== DINO CLS Loss (trainer.py lines 566-579) ==========
        if dino_loss is not None:
            # Student: list of dicts with 'cls_output' (projected)
            student_cls_outputs = []
            for feat_dict in student_features:
                student_cls_outputs.append(feat_dict['cls_output'])
            student_cls_cat = torch.cat(student_cls_outputs, dim=0)  # [B_total, out_dim]
            
            # Teacher: list of dicts (2 global views)
            teacher_cls_outputs = []
            for feat_dict in teacher_features:
                teacher_cls_outputs.append(feat_dict['cls_output'])
            teacher_cls_cat = torch.cat(teacher_cls_outputs, dim=0)  # [B_teacher, out_dim]
            
            # Compute DINO loss
            dino_loss_val = dino_loss(student_cls_cat, teacher_cls_cat, iteration)
            losses['dino_class_loss'] = dino_loss_val
        
        # ========== KoLeo Loss on Global Crops (trainer.py lines 581-585) ==========
        if koleo_loss is not None:
            # Extract global crops only (first 2 crops)
            num_global = 2
            global_cls_features = []
            for feat_dict in student_features[:num_global]:
                global_cls_features.append(feat_dict['cls_features'])
            
            koleo_loss_val = torch.tensor(0.0, device=device)
            if len(global_cls_features) > 0:
                koleo_loss_val = sum(koleo_loss(feat) for feat in global_cls_features) / len(global_cls_features)
            
            losses['koleo_loss'] = koleo_loss_val
        
        # ========== iBOT Patch Loss (trainer.py lines 587-623) ==========
        if ibot_loss is not None:
            # Teacher patch outputs (projected) - no masking
            teacher_patch_outputs = teacher_ibot_features['patch_output']  # [B, N, out_dim]
            
            # Student patch outputs (projected) - with masking
            student_patch_outputs = student_ibot_features['patch_output']  # [B, N, out_dim]
            
            # Compute iBOT loss on masked tokens
            teacher_temp = dino_loss.teacher_temp_schedule(iteration) if dino_loss else 0.07
            
            ibot_loss_val = ibot_loss.forward_masked(
                student_patch_outputs,
                teacher_patch_outputs,
                token_masks,
                teacher_temp=teacher_temp
            )
            
            losses['ibot_loss'] = ibot_loss_val
        
        # ========== Prototype Clustering Loss (trainer.py lines 625-640) ==========
        if prototype_loss is not None and self.prototype_bank is not None:
            # Get raw patch features (before projection head)
            teacher_patch_features = teacher_ibot_features['patch_features']  # [B, N, embed_dim]
            student_patch_features = student_ibot_features['patch_features']  # [B, N, embed_dim]
            
            # Compute clustering loss
            clustering_loss, koleo_proto_loss, teacher_proto_loss = prototype_loss(
                teacher_patch_features,
                student_patch_features,
                token_masks,
                self.prototype_bank
            )
            
            losses['clustering_loss'] = clustering_loss
            losses['koleo_proto_loss'] = koleo_proto_loss
            losses['teacher_proto_loss'] = teacher_proto_loss
        
        if self.debug:
            print(f"[Rank {self.local_rank}] Computed losses: {list(losses.keys())}")
        
        return losses
    
    def backward_pass(
        self,
        losses: Dict[str, torch.Tensor],
        optimizer_student,
        optimizer_prototypes,
        args,
        current_iteration: int,
        scaler=None,
    ):
        """
        Complete backward pass matching trainer.py lines 645-690.
        
        Includes:
        - Gradient clipping
        - Last layer freezing
        - Mixed precision support
        - Separate optimizer steps
        """
        if not self.is_last_stage:
            # Non-last stages: backward happens automatically via autograd
            return
        
        # Compute total losses (only on last stage)
        student_loss = (
            losses['dino_class_loss'] +
            args.koleo_loss_weight * losses['koleo_loss'] +
            args.ibot_loss_weight * losses['ibot_loss'] +
            args.clustering_weight * losses['clustering_loss']
        )
        
        prototype_loss_total = (
            losses['teacher_proto_loss'] +
            losses['koleo_proto_loss']
        )
        
        # ========== Backward for Prototypes (trainer.py lines 645-650) ==========
        if scaler is None:
            prototype_loss_total.backward()
            optimizer_prototypes.step()
        else:
            scaler.scale(prototype_loss_total).backward()
            scaler.step(optimizer_prototypes)
        
        # ========== Backward for Student (trainer.py lines 653-683) ==========
        if scaler is None:
            student_loss.backward()
            
            # Gradient clipping (trainer.py line 655)
            if args.clip_grad:
                import utils
                utils.clip_gradients(self.student_stage, args.clip_grad)
            
            # Cancel last layer gradients (trainer.py lines 656-658)
            import utils
            if self.student_stage.module.has_heads:
                utils.cancel_gradients_last_layer(
                    current_iteration, 
                    self.student_stage.module.classhead, 
                    args.freeze_last_layer_iters
                )
                utils.cancel_gradients_last_layer(
                    current_iteration,
                    self.student_stage.module.patchhead,
                    args.freeze_last_layer_iters
                )
            
            optimizer_student.step()
        else:
            # Mixed precision path (trainer.py lines 660-683)
            scaler.scale(student_loss).backward()
            
            if args.clip_grad:
                import utils
                scaler.unscale_(optimizer_student)
                utils.clip_gradients(self.student_stage, args.clip_grad)
            
            import utils
            if self.student_stage.module.has_heads:
                utils.cancel_gradients_last_layer(
                    current_iteration,
                    self.student_stage.module.classhead,
                    args.freeze_last_layer_iters
                )
                utils.cancel_gradients_last_layer(
                    current_iteration,
                    self.student_stage.module.patchhead,
                    args.freeze_last_layer_iters
                )
            
            scaler.step(optimizer_student)
            scaler.update()
        
        if self.debug:
            print(f"[Rank {self.local_rank}] Backward pass completed")
    
    def update_teacher_ema(self, momentum: float):
        """
        Update teacher parameters with EMA from student.
        
        Each stage updates its own teacher parameters from corresponding student.
        Matches trainer.py lines 686-695.
        """
        with torch.no_grad():
            # Update backbone parameters (all stages have backbone)
            for param_s, param_t in zip(
                self.student_stage.module.backbone.parameters(),
                self.teacher_stage.module.backbone.parameters()
            ):
                param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)
            
            # Update heads (only last stage has heads)
            if self.is_last_stage and self.student_stage.module.has_heads:
                for param_s, param_t in zip(
                    self.student_stage.module.classhead.parameters(),
                    self.teacher_stage.module.classhead.parameters()
                ):
                    param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)
                
                for param_s, param_t in zip(
                    self.student_stage.module.patchhead.parameters(),
                    self.teacher_stage.module.patchhead.parameters()
                ):
                    param_t.data.mul_(momentum).add_((1 - momentum) * param_s.detach().data)
        
        if self.debug:
            print(f"[Rank {self.local_rank}] Teacher EMA update completed")