"""
Custom Pipeline Schedule for Batch-Dependent DINO Losses.

This implements a "flush pipeline" strategy where:
1. Forward flush: All images pass through the pipeline
2. Feature accumulation: Last stage collects all features
3. Loss computation: Compute batch-dependent DINO/iBOT/clustering losses
4. Backward flush: Propagate gradients back through pipeline

Key difference from standard pipeline parallelism:
- Standard GPipe: Loss per microbatch → backward per microbatch
- DINO Pipeline: Accumulate all features → single global loss → single backward

This is necessary because DINO losses require global operations:
- DINO CLS: Cross-view comparisons across all students/teachers
- iBOT: Sinkhorn-Knopp normalization over entire batch
- Clustering: Global Sinkhorn-Knopp over all patch tokens
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time


class PipelineSchedule:
    """
    Custom pipeline schedule for batch-dependent losses.
    
    Coordinates forward/backward passes across pipeline stages within a node,
    with data parallelism across nodes.
    
    Args:
        student_stage: Student model pipeline stage
        teacher_stage: Teacher model pipeline stage  
        prototype_bank: Prototype bank (only on last stage)
        loss_modules: Dictionary of loss modules (DINO, iBOT, KoLeo, Prototype)
        local_rank: GPU rank within node (0 to gpus_per_node-1)
        pipeline_group: Process group for within-node pipeline communication
        data_group: Process group for across-node data parallel gradient sync
        batch_size_per_node: Number of images per node (e.g., 768)
        gpus_per_node: Number of GPUs per node (default: 4)
        grad_checkpointing: Whether gradient checkpointing is enabled
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
        grad_checkpointing: bool = False,
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
        self.grad_checkpointing = grad_checkpointing
        self.debug = debug
        
        # Pipeline stage identification
        self.is_first_stage = (local_rank == 0)
        self.is_last_stage = (local_rank == gpus_per_node - 1)
        self.is_middle_stage = not (self.is_first_stage or self.is_last_stage)
        
        # Communication setup
        self.prev_rank = local_rank - 1 if local_rank > 0 else None
        self.next_rank = local_rank + 1 if local_rank < gpus_per_node - 1 else None
        
        # Feature accumulation buffers (only used on last stage)
        self.accumulated_features = {
            'student': [],
            'teacher': [],
        }
        
        # Gradient tracking for backward pass
        self.saved_tensors = []
        
        if self.debug:
            print(f"[Rank {local_rank}] Pipeline schedule initialized: "
                  f"first={self.is_first_stage}, middle={self.is_middle_stage}, "
                  f"last={self.is_last_stage}")
    
    def forward_pass(
        self,
        crops: List[torch.Tensor],
        token_masks: Optional[torch.Tensor] = None,
        iteration: int = 0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Execute forward pass through the pipeline.
        
        Args:
            crops: List of crop tensors [global_view1, global_view2, local1, ...]
            token_masks: Optional token masks for iBOT
            iteration: Current training iteration
            
        Returns:
            Dictionary of losses (only on last stage), None otherwise
        """
        
        # ========== STAGE 1: Teacher Forward (Global Crops Only) ==========
        teacher_global_crops = crops[:2]  # First 2 crops are global views
        
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] Teacher forward: {len(teacher_global_crops)} global crops")
        
        teacher_features = self._forward_through_pipeline(
            teacher_global_crops,
            self.teacher_stage,
            token_masks=None,  # Teacher doesn't use masking
            is_teacher=True,
        )
        
        # ========== STAGE 2: Student Forward (All Crops) ==========
        if self.debug and self.is_first_stage:
            print(f"[Rank {self.local_rank}] Student forward: {len(crops)} total crops")
        
        student_features = self._forward_through_pipeline(
            crops,
            self.student_stage,
            token_masks=None,  # Masking handled separately for iBOT
            is_teacher=False,
        )
        
        # ========== STAGE 3: Compute Losses (Last Stage Only) ==========
        if self.is_last_stage:
            if self.debug:
                print(f"[Rank {self.local_rank}] Computing losses")
            
            losses = self._compute_losses(
                student_features,
                teacher_features,
                crops,
                token_masks,
                iteration,
            )
            
            return losses
        
        return None
    
    def _forward_through_pipeline(
        self,
        crops: List[torch.Tensor],
        model_stage,
        token_masks: Optional[torch.Tensor] = None,
        is_teacher: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Forward crops through the pipeline stages.
        
        Strategy:
        - First stage: Process crops and send to next stage
        - Middle stages: Receive, process, send to next stage
        - Last stage: Receive, process, return features
        
        Args:
            crops: List of image tensors
            model_stage: Student or teacher stage
            token_masks: Optional masking for iBOT
            is_teacher: Whether this is teacher model (for debug)
            
        Returns:
            Features dict (only on last stage), None otherwise
        """
        
        if self.is_first_stage:
            # ========== FIRST STAGE: Patch Embed + Initial Blocks ==========
            # Forward through this stage
            output, attn_bias = model_stage(crops, token_masks=token_masks)
            
            # Send to next stage
            self._send_to_next_stage(output, attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] {'Teacher' if is_teacher else 'Student'} "
                      f"first stage output shape: {output.shape}")
            
            return None
        
        elif self.is_middle_stage:
            # ========== MIDDLE STAGE: Blocks Only ==========
            # Receive from previous stage
            input_tensor, attn_bias = self._recv_from_prev_stage()
            
            # Forward through this stage
            output, attn_bias = model_stage(input_tensor, token_masks=None, attn_bias=attn_bias)
            
            # Send to next stage
            self._send_to_next_stage(output, attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] {'Teacher' if is_teacher else 'Student'} "
                      f"middle stage output shape: {output.shape}")
            
            return None
        
        else:  # Last stage
            # ========== LAST STAGE: Final Blocks + Norm + Heads ==========
            # Receive from previous stage
            input_tensor, attn_bias = self._recv_from_prev_stage()
            
            # Forward through this stage
            output = model_stage(input_tensor, token_masks=None, attn_bias=attn_bias)
            
            if self.debug:
                print(f"[Rank {self.local_rank}] {'Teacher' if is_teacher else 'Student'} "
                      f"last stage output: {type(output)}")
            
            return output
    
    def _send_to_next_stage(self, tensor: torch.Tensor, attn_bias: Optional[Any] = None):
        """
        Send tensor to next pipeline stage.
        
        Uses point-to-point communication within pipeline group.
        """
        if self.next_rank is None:
            return
        
        # Send main tensor
        dist.send(tensor.contiguous(), dst=self.next_rank, group=self.pipeline_group)
        
        # Send attention bias metadata if present
        has_attn_bias = (attn_bias is not None)
        has_attn_bias_tensor = torch.tensor([1 if has_attn_bias else 0], 
                                            dtype=torch.int, device=tensor.device)
        dist.send(has_attn_bias_tensor, dst=self.next_rank, group=self.pipeline_group)
        
        if self.debug:
            print(f"[Rank {self.local_rank}] Sent tensor {tensor.shape} to rank {self.next_rank}")
    
    def _recv_from_prev_stage(self) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Receive tensor from previous pipeline stage.
        
        Returns:
            Tuple of (tensor, attn_bias)
        """
        if self.prev_rank is None:
            raise RuntimeError("Cannot receive - no previous stage")
        
        # We need to know the shape to receive - this is the