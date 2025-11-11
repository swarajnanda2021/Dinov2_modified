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
        Cross-entropy between teacher and student on masked patches.
        
        Args:
            student_patch_tokens_masked: [B, N, D] student patch tokens
            teacher_patch_tokens_masked: [B, N, D] teacher patch tokens
            student_masks_flat: [B, N] boolean mask (True = masked)
            n_masked_patches: Optional number of masked patches
            masks_weight: Optional per-token weights
            teacher_temp: Teacher temperature
            
        Returns:
            Loss value
        """
        B, N, D = student_patch_tokens_masked.shape
        
        all_student_tokens = []
        all_teacher_tokens = []
        all_weights = []
        
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)
            )
        
        for b in range(B):
            mask_indices = student_masks_flat[b]
            num_masked = mask_indices.sum().item()
            
            if num_masked > 0:
                student_tokens = student_patch_tokens_masked[b][mask_indices]
                teacher_tokens = teacher_patch_tokens_masked[b][mask_indices]
                weights = masks_weight[b][mask_indices]
                
                all_student_tokens.append(student_tokens)
                all_teacher_tokens.append(teacher_tokens)
                all_weights.append(weights)
        
        if len(all_student_tokens) == 0:
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        all_student_tokens = torch.cat(all_student_tokens, dim=0)
        all_teacher_tokens = torch.cat(all_teacher_tokens, dim=0)
        all_weights = torch.cat(all_weights, dim=0)
        
        teacher_normalized = self.sinkhorn_knopp_normalization(all_teacher_tokens, teacher_temp)
        
        student_log_probs = F.log_softmax(all_student_tokens / self.student_temp, dim=-1)
        
        loss_per_token = -torch.sum(teacher_normalized * student_log_probs, dim=-1)
        
        weighted_loss = loss_per_token * all_weights

        return weighted_loss.mean()

    @torch.no_grad()
    def sinkhorn_knopp_normalization(self, teacher_output, teacher_temp, n_iterations=3):
        """
        FSDP2-compatible Sinkhorn-Knopp for masked patches only.
        
        Args:
            teacher_output: [M_local, K] - masked patches from this rank
            teacher_temp: Temperature
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, M_local]
        K, M_local = Q.shape
        
        # Total number of masked patches across all ranks
        M_total = torch.tensor(M_local, device=Q.device, dtype=torch.int64)
        if dist.is_initialized():
            dist.all_reduce(M_total)
        M_total = M_total.item()
        
        # Global normalization
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= (sum_Q + 1e-8)
        
        for _ in range(n_iterations):
            # Normalize rows (prototypes)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= (sum_of_rows + 1e-8)
            Q /= K
            
            # Normalize columns (samples)
            Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
            Q /= M_total
        
        Q *= M_total
        return Q.t()  # [M_local, K]