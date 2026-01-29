"""
Patch prototype clustering loss with CAPI-inspired doubly stochastic Sinkhorn-Knopp.
Uses global processing (all tokens together) with doubly stochastic constraints
to eliminate positional bias while maintaining efficient computation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .koleo_loss import KoLeoLoss


class PatchPrototypeLoss(nn.Module):
    """
    CAPI-inspired patch clustering loss using doubly stochastic optimal transport.
    
    Key differences from original:
    - Doubly stochastic constraints (both samples and prototypes sum to 1)
    - Improved numerical stability across dtypes
    - Epsilon protection against division by zero
    - Eliminates spatial positional bias in prototype assignments
    
    Args:
        num_prototypes: Number of prototype vectors
        embed_dim: Embedding dimension
        teacher_temp: Teacher temperature
        student_temp: Student temperature
    """
    def __init__(
        self, 
        num_prototypes=8192, 
        embed_dim=768,
        teacher_temp=0.07, 
        student_temp=0.1
    ):
        super().__init__()
        
        self.k = num_prototypes
        self.embed_dim = embed_dim
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        
        # KoLeo for preventing collapse
        self.koleo_loss = KoLeoLoss()
        
        # Tracking metrics
        self.last_prediction_loss = 0.0
        self.last_arrangement_loss = 0.0
        self.last_koleo_loss = 0.0
        self.last_entropy = 0.0
        self.last_usage_std = 0.0
        self.last_num_masked = 0
    
    def forward(self, teacher_patch_tokens, student_patch_tokens, token_masks, 
                prototype_bank, current_iteration, teacher_temp, masks_weight=None):
        """
        Compute prototype clustering loss with optional per-sample weighting.
        
        Args:
            teacher_patch_tokens: [B, N, D] teacher patch tokens
            student_patch_tokens: [B, N, D] student patch tokens  
            token_masks: [B, N] boolean mask (True = masked positions)
            prototype_bank: LinearPrototypeBank module
            current_iteration: Current training iteration
            teacher_temp: Teacher temperature for Sinkhorn-Knopp
            masks_weight: Optional [B] per-sample weights (1/num_masked per sample)
            
        Returns:
            tuple: (student_clustering_loss, teacher_proto_loss, koleo_proto_loss)
        """
        B, N, D = teacher_patch_tokens.shape
        
        # Normalize features
        teacher_norm = F.normalize(teacher_patch_tokens, p=2, dim=-1)
        student_norm = F.normalize(student_patch_tokens, p=2, dim=-1)
        
        # ========== Teacher path: Generate targets and arrangement loss ==========
        teacher_logits_all = prototype_bank(teacher_norm)  # [B, N, K]
        teacher_logits_flat = teacher_logits_all.reshape(B * N, -1)
        
        # Get assignments (no grad)
        with torch.no_grad():
            Q_tilde_flat = self.sinkhorn_knopp(teacher_logits_flat, teacher_temp)
            Q_tilde_all = Q_tilde_flat.reshape(B, N, -1)
        
        # Arrangement loss (trains prototype bank)
        teacher_log_probs_all = F.log_softmax(teacher_logits_all / teacher_temp, dim=-1)
        teacher_proto_loss = -torch.sum(Q_tilde_all.detach() * teacher_log_probs_all) / (B * N)
        
        # KoLeo on prototypes
        weight_normalized = F.normalize(prototype_bank.module.proto_layer.weight, p=2, dim=1)
        koleo_proto_loss = self.koleo_loss(weight_normalized)
        
        # ========== Student path: Prediction loss ==========
        Q_tilde_masked = Q_tilde_all[token_masks].detach()  # [M, K]
        student_norm_masked = student_norm[token_masks]  # [M, D]
        
        M_total = student_norm_masked.shape[0]
        if M_total == 0:
            return torch.tensor(0.0, device=teacher_patch_tokens.device), \
                torch.tensor(0.0, device=teacher_patch_tokens.device), \
                torch.tensor(0.0, device=teacher_patch_tokens.device)
        
        # Student predictions
        student_logits_masked = prototype_bank(student_norm_masked)  # [M, K]
        student_log_probs_masked = F.log_softmax(student_logits_masked / self.student_temp, dim=-1)
        
        # Per-token cross-entropy
        per_token_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked, dim=-1)  # [M]
        
        # Compute clustering loss with per-sample weighting
        if masks_weight is not None:
            # Map sample weights to per-token weights
            # token_masks is [B, N], need to find which sample each masked token belongs to
            sample_indices = token_masks.nonzero(as_tuple=True)[0]  # [M] - sample index for each masked token
            per_token_weight = masks_weight[sample_indices]  # [M]
            
            # Weighted sum, normalized by batch size (not token count)
            clustering_loss = (per_token_loss * per_token_weight).sum() / B
        else:
            # Fallback: simple mean over tokens
            clustering_loss = per_token_loss.mean()
        
        # Update metrics
        with torch.no_grad():
            student_probs = torch.exp(student_log_probs_masked)
            entropy = -(student_probs * student_log_probs_masked).sum(dim=-1).mean()
            self.last_entropy = (entropy / math.log(self.k)).item()
            
            assignments = torch.argmax(Q_tilde_masked, dim=-1)
            usage = torch.bincount(assignments, minlength=self.k).float()
            if dist.is_initialized():
                dist.all_reduce(usage)
            self.last_usage_std = (usage.std() / (usage.mean() + 1e-6)).item()
            
            self.last_prediction_loss = clustering_loss.item()
            self.last_arrangement_loss = teacher_proto_loss.item()
            self.last_koleo_loss = koleo_proto_loss.item()
            self.last_num_masked = M_total
        
        return clustering_loss, teacher_proto_loss, koleo_proto_loss

    @torch.no_grad()
    def sinkhorn_knopp(self, teacher_output, teacher_temp, n_iterations=3, eps=1e-8):
        """
        CAPI-inspired doubly stochastic Sinkhorn-Knopp.
        Matches the official CAPI implementation exactly.
        
        Args:
            teacher_output: [M, K] logits where M = B*N tokens
            teacher_temp: Temperature for softmax
            n_iterations: Number of SK iterations
            eps: Epsilon for numerical stability
            
        Returns:
            Q: [M, K] doubly stochastic assignment matrix
        """
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        # Numerical stability: shift by global max before exp
        M = teacher_output / teacher_temp
        M_max = M.max()
        if dist.is_initialized():
            dist.all_reduce(M_max, op=dist.ReduceOp.MAX)
        M = M - M_max
        
        # Transpose for easier iteration: [K, M]
        Q = torch.exp(M).t()
        
        # Doubly stochastic iterations (matches CAPI exactly)
        for _ in range(n_iterations):
            # Normalize over samples (each prototype distribution sums to 1)
            # All-reduce needed because samples are distributed across GPUs
            sum_over_samples = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_over_samples)
            Q /= (sum_over_samples + eps)
            
            # Normalize over prototypes (each sample distribution sums to 1)
            # No all-reduce needed because prototypes are replicated
            sum_over_prototypes = torch.sum(Q, dim=0, keepdim=True)
            Q /= (sum_over_prototypes + eps)
        
        # Transpose back to [M, K]
        return Q.t()