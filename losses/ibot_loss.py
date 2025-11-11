"""
iBOT patch-level loss for masked token prediction.
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
        Args:
            student_patch_tokens_masked: [B, N, D]
            teacher_patch_tokens_masked: [B, N, D]
            student_masks_flat: [B, N] boolean mask
        """
        B, N, D = student_patch_tokens_masked.shape
        
        # Flatten everything - no loops!
        student_flat = student_patch_tokens_masked.reshape(B * N, D)  # [B*N, D]
        teacher_flat = teacher_patch_tokens_masked.reshape(B * N, D)  # [B*N, D]
        masks_flat = student_masks_flat.reshape(B * N)  # [B*N]
        
        # Vectorized extraction of masked tokens
        student_masked = student_flat[masks_flat]  # [M, D] where M = number of True values
        teacher_masked = teacher_flat[masks_flat]  # [M, D]
        
        if student_masked.shape[0] == 0:
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        # Compute weights
        if masks_weight is None:
            # Count masked tokens per sample
            n_masked_per_sample = student_masks_flat.sum(dim=-1).clamp(min=1.0)  # [B]
            per_sample_weight = 1.0 / n_masked_per_sample  # [B]
            # Expand to match masked tokens
            masks_weight = per_sample_weight.unsqueeze(-1).expand_as(student_masks_flat)  # [B, N]
            masks_weight = masks_weight.reshape(B * N)[masks_flat]  # [M]
        else:
            masks_weight = masks_weight.reshape(B * N)[masks_flat]  # [M]
        
        # Normalize teacher (happens inside, which is fine!)
        teacher_normalized = self.sinkhorn_knopp_normalization(teacher_masked, teacher_temp)
        
        # Compute loss
        student_log_probs = F.log_softmax(student_masked / self.student_temp, dim=-1)
        loss_per_token = -torch.sum(teacher_normalized * student_log_probs, dim=-1)
        weighted_loss = loss_per_token * masks_weight
        
        return weighted_loss.mean()

    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, teacher_output, teacher_temp, n_iterations=3):
        """
        Args:
            teacher_output: [M_local, K] where M_local = masked tokens on this rank
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, M_local]
        K = Q.shape[0]  # number of prototypes
        M_local = Q.shape[1]  # local masked tokens
        
        # Total masked tokens across all ranks
        M_total_tensor = torch.tensor(M_local, device=Q.device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(M_total_tensor)
        M_total = M_total_tensor.item()  # Total across all ranks
        
        # Normalize
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= (sum_Q + 1e-8)
        
        for _ in range(n_iterations):
            # Rows (prototypes)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= (sum_of_rows + 1e-8)
            Q /= K
            
            # Columns (samples)
            Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
            Q /= M_total
        
        Q *= M_total
        return Q.t()  # [M_local, K]