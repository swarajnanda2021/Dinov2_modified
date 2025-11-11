"""
iBOT patch-level loss for masked token prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class iBOTPatchLoss(nn.Module):
    def __init__(self, student_temp=0.1, n_iterations=3):
        super().__init__()
        self.student_temp = student_temp
        self.n_iterations = n_iterations
        # Separate Sinkhorn-Knopp as a module
        self.sinkhorn_knopp_teacher = SinkhornKnoppTeacher()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches_tensor,  # ADD THIS PARAMETER
        masks_weight=None,
        teacher_temp=0.07
    ):
        """
        Args:
            n_masked_patches_tensor: Tensor containing count of masked patches (for distributed reduction)
        """
        B, N, D = student_patch_tokens_masked.shape

        # DEBUG: Check inputs
        if torch.isnan(student_patch_tokens_masked).any():
            print(f"[Rank {dist.get_rank()}] NaN in student_patch_tokens_masked INPUT")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        if torch.isnan(teacher_patch_tokens_masked).any():
            print(f"[Rank {dist.get_rank()}] NaN in teacher_patch_tokens_masked INPUT")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        
        student_flat = student_patch_tokens_masked.reshape(B * N, D)
        teacher_flat = teacher_patch_tokens_masked.reshape(B * N, D)
        masks_flat = student_masks_flat.reshape(B * N)
        
        student_masked = student_flat[masks_flat]
        teacher_masked = teacher_flat[masks_flat]

        # DEBUG: Check after masking
        if torch.isnan(student_masked).any():
            print(f"[Rank {dist.get_rank()}] NaN in student_masked after indexing")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        if torch.isnan(teacher_masked).any():
            print(f"[Rank {dist.get_rank()}] NaN in teacher_masked after indexing")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        
        if student_masked.shape[0] == 0:
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        if masks_weight is None:
            n_masked_per_sample = student_masks_flat.sum(dim=-1).clamp(min=1.0)
            per_sample_weight = 1.0 / n_masked_per_sample
            masks_weight = per_sample_weight.unsqueeze(-1).expand_as(student_masks_flat)
            masks_weight = masks_weight.reshape(B * N)[masks_flat]
        else:
            masks_weight = masks_weight.reshape(B * N)[masks_flat]

        # DEBUG: Check weights
        if torch.isnan(masks_weight).any():
            print(f"[Rank {dist.get_rank()}] NaN in masks_weight")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        
        # Use the module's Sinkhorn-Knopp with proper n_masked_patches_tensor
        teacher_normalized = self.sinkhorn_knopp_teacher(
            teacher_masked, 
            teacher_temp,
            n_masked_patches_tensor  # Pass as parameter
        )

        # DEBUG: Check after Sinkhorn-Knopp
        if torch.isnan(teacher_normalized).any():
            print(f"[Rank {dist.get_rank()}] NaN in teacher_normalized after Sinkhorn-Knopp")
            print(f"  teacher_masked stats: min={teacher_masked.min()}, max={teacher_masked.max()}, mean={teacher_masked.mean()}")
            print(f"  teacher_temp={teacher_temp}")
            print(f"  n_masked_patches_tensor={n_masked_patches_tensor}")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        
        student_log_probs = F.log_softmax(student_masked / self.student_temp, dim=-1)

        # DEBUG: Check log probs
        if torch.isnan(student_log_probs).any():
            print(f"[Rank {dist.get_rank()}] NaN in student_log_probs")
            print(f"  student_masked stats: min={student_masked.min()}, max={student_masked.max()}")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        loss_per_token = -torch.sum(teacher_normalized * student_log_probs, dim=-1)

        # DEBUG: Check per-token loss
        if torch.isnan(loss_per_token).any():
            print(f"[Rank {dist.get_rank()}] NaN in loss_per_token")
            return torch.tensor(0.0, device=student_patch_tokens_masked.device)
        
        weighted_loss = loss_per_token * masks_weight

        # DEBUG: Check final
        if torch.isnan(weighted_loss).any():
            print(f"[Rank {dist.get_rank()}] NaN in final_loss")
            print(f"  weighted_loss sum={weighted_loss.sum()}")
            print(f"  denominator={student_masks_flat.shape[0]}")
        
        return weighted_loss.mean() 


class SinkhornKnoppTeacher(nn.Module):
    """Separate module for Sinkhorn-Knopp (can be compiled)"""
    
    @torch.no_grad()
    def forward(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        """
        Args:
            teacher_output: [M_local, K] where M_local = masked tokens on this rank
            teacher_temp: Temperature
            n_masked_patches_tensor: TENSOR (not scalar) with local count
        """
        teacher_output = teacher_output.float()
        
        Q = torch.exp(teacher_output / teacher_temp).t()  # [K, M_local]
        K = Q.shape[0]
        
        # Get total masked patches by reducing the tensor directly
        B = n_masked_patches_tensor.clone()  # Clone to avoid modifying input
        if dist.is_initialized():
            dist.all_reduce(B)
        
        # Initial normalization
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q  # No epsilon here
        
        # Sinkhorn-Knopp iterations
        for _ in range(n_iterations):
            # Normalize rows (prototypes)
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            
            # Normalize columns (samples)
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        
        Q *= B
        return Q.t()  # [M_local, K]