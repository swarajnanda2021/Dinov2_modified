"""
iBOT patch-level loss for masked token prediction.
Optimized: Vectorized masking eliminates per-sample loops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class iBOTPatchLoss(nn.Module):
    """
    Canonical iBOT patch loss implementation.
    
    Args:
        student_temp: Student temperature
        n_iterations: Sinkhorn-Knopp iterations
    """
    def __init__(self, student_temp=0.1, n_iterations=3):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        teacher_temp=0.07
    ):
        """
        Vectorized cross-entropy between teacher and student on masked patches.
        
        Key optimization: Single nonzero call instead of B calls.
        This eliminates the GPU-CPU sync bottleneck from the original loop.
        
        Args:
            student_patch_tokens_masked: [B, N, D] student patch tokens
            teacher_patch_tokens_masked: [B, N, D] teacher patch tokens
            student_masks_flat: [B, N] boolean mask (True = masked)
            n_masked_patches: Optional number of masked patches (unused, for compatibility)
            masks_weight: Optional per-token weights
            teacher_temp: Teacher temperature
            
        Returns:
            Loss value
        """
        B, N, D = student_patch_tokens_masked.shape
        device = student_patch_tokens_masked.device
        dtype = student_patch_tokens_masked.dtype
        
        # Flatten everything for vectorized operations
        student_flat = student_patch_tokens_masked.reshape(B * N, D)
        teacher_flat = teacher_patch_tokens_masked.reshape(B * N, D)
        mask_flat = student_masks_flat.reshape(-1)  # [B*N]
        
        # SINGLE nonzero call (was B calls in the loop)
        masked_indices = mask_flat.nonzero(as_tuple=True)[0]
        
        M = masked_indices.numel()
        if M == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        
        # Gather all masked tokens at once
        student_masked = student_flat[masked_indices]  # [M, D]
        teacher_masked = teacher_flat[masked_indices]  # [M, D]
        
        # Compute per-token weights
        if masks_weight is None:
            # Per-sample normalization: weight = 1 / num_masked_in_sample
            sample_idx = masked_indices // N  # Which sample each token belongs to [M]
            num_masked_per_sample = mask_flat.reshape(B, N).sum(dim=1).clamp(min=1.0)  # [B]
            weights = 1.0 / num_masked_per_sample[sample_idx]  # [M]
        else:
            weights_flat = masks_weight.reshape(-1)
            weights = weights_flat[masked_indices]  # [M]
        
        # Sinkhorn-Knopp normalization on teacher
        teacher_normalized = self.sinkhorn_knopp_normalization(teacher_masked, teacher_temp)
        
        # Student log probabilities
        student_log_probs = F.log_softmax(student_masked / self.student_temp, dim=-1)
        
        # Cross-entropy loss per token
        loss_per_token = -torch.sum(teacher_normalized.detach() * student_log_probs, dim=-1)  # [M]
        
        # Weighted sum
        weighted_loss = (loss_per_token * weights).mean()
        
        return weighted_loss

    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, teacher_output, teacher_temp, n_iterations=None):
        """Apply Sinkhorn-Knopp normalization to teacher outputs."""
        if n_iterations is None:
            n_iterations = self.n_iterations
            
        teacher_output = teacher_output.float()
        
        Q = torch.exp(teacher_output / teacher_temp).t()
        
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        B = Q.shape[1] * world_size
        K = Q.shape[0]
        
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        
        for it in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B
        return Q.t()
