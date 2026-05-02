"""
Per-sample DINO and iBOT helpers for the PonderNet mixture-weighted total used
by the looped DINOv2 backbone.

The looped student runs every crop through all T_max recursion steps in
lockstep, and the per-image halting marginal p_t weights each step's
contribution to the total loss:

    L[i] = sum_t p_t[i] * (L_DINO_t[i] + L_iBOT_t[i]) + beta * KL[i]

Computing this requires per-sample (rather than mean-reduced) DINO and iBOT
losses at each recursion step. The teacher's Sinkhorn-Knopp-normalized
targets are shared across all student steps (the teacher always runs at
T_max only) and are therefore computed once per batch, outside the t-loop.
"""

from typing import List

import torch
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def sinkhorn_knopp(
    teacher_output: torch.Tensor,
    teacher_temp: float,
    n_iterations: int = 3,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp normalization on teacher head outputs. Mirrors the
    implementation in `losses.dino_loss` and `losses.ibot_loss`, exposed as
    a standalone function so the looped trainer can run it once per batch
    rather than once per student recursion step (the teacher only runs at
    T_max, so its normalized targets are constant across student t).
    """
    teacher_output = teacher_output.float()
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    Q = torch.exp(teacher_output / teacher_temp).t()
    B = Q.shape[1] * world_size
    K = Q.shape[0]

    sum_Q = torch.sum(Q)
    if dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q

    for _ in range(n_iterations):
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


def dino_per_sample_loss(
    student_cls_outputs: torch.Tensor,
    teacher_cls_chunks_normalized: List[torch.Tensor],
    ncrops: int,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """
    Per-sample DINO CLS loss given pre-normalized teacher targets.

    Reduces over crop pairs (q, v) like the standard DINOLoss, but does not
    average over the batch dimension — the caller weights the [B] result by
    the per-image PonderNet marginal p_t before reducing.

    Args:
        student_cls_outputs: [B*ncrops, out_dim] student head outputs at
            ONE recursion step. Same layout as DINOLoss.forward expects.
        teacher_cls_chunks_normalized: list of `n_globals` tensors, each
            [B, out_dim], already Sinkhorn-Knopp normalized and detached.
        ncrops: total number of student crops (= len(student_cls_outputs.chunk(ncrops))).
        student_temp: student temperature (default 0.1, matching DINOLoss).

    Returns:
        [B] per-sample average cross-view DINO loss at this step.
    """
    student_out = student_cls_outputs / student_temp
    student_chunks = student_out.chunk(ncrops)
    B = student_chunks[0].shape[0]

    n_loss_terms = 0
    total = None
    for iq, q in enumerate(teacher_cls_chunks_normalized):
        for v, sv in enumerate(student_chunks):
            if v == iq:
                # Skip same-view pair to match DINO's cross-view-only objective.
                continue
            ce = -(q * F.log_softmax(sv, dim=-1)).sum(dim=-1)  # [B]
            total = ce if total is None else total + ce
            n_loss_terms += 1

    if total is None or n_loss_terms == 0:
        return torch.zeros(B, device=student_cls_outputs.device,
                           dtype=student_cls_outputs.dtype)
    return total / n_loss_terms


def ibot_per_sample_loss(
    student_proj: torch.Tensor,
    teacher_proj_normalized: torch.Tensor,
    weights: torch.Tensor,
    sample_idx: torch.Tensor,
    batch_size: int,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """
    Per-sample iBOT loss given pre-projected, pre-normalized teacher targets.

    Mathematically: per-sample contribution is
        L[i] = sum_{m in masked(i)} weights[m] * CE(student[m], teacher[m]).
    Mean over batch matches `iBOTPatchLoss.forward_gathered`'s scalar output:
        out.mean() == (loss_per_token * weights).sum() / B.

    Args:
        student_proj: [M, out_dim] student patchhead outputs for masked tokens
            at ONE recursion step.
        teacher_proj_normalized: [M, out_dim] Sinkhorn-Knopp-normalized teacher
            patchhead outputs (from the teacher's T_max forward, computed
            once outside the t-loop).
        weights: [M] per-token weights (typically 1 / num_masked_per_sample).
        sample_idx: [M] integer per-token sample index in [0, B).
        batch_size: B; used as the destination dim of the per-sample scatter.
        student_temp: student temperature (default 0.1).

    Returns:
        [B] per-sample iBOT loss. A sample with no masked tokens (e.g.
        because the iBOT mask gate dropped it) contributes 0.
    """
    if student_proj.numel() == 0:
        return torch.zeros(batch_size, device=student_proj.device,
                           dtype=student_proj.dtype)

    student_log_probs = F.log_softmax(student_proj / student_temp, dim=-1)
    ce = -(teacher_proj_normalized * student_log_probs).sum(dim=-1)  # [M]
    weighted = weights * ce  # [M]
    out = torch.zeros(batch_size, device=student_proj.device,
                      dtype=weighted.dtype)
    out.index_add_(0, sample_idx, weighted)
    return out


def gather_masked_tokens(
    patch_tokens: torch.Tensor,
    mask: torch.Tensor,
    masks_weight: torch.Tensor = None,
):
    """
    Gather masked tokens from [B, N, D] under boolean mask [B, N] and emit
    the per-token sample-index tensor and weights expected by
    `ibot_per_sample_loss`.

    Mirrors the gather block of `_gather_and_compute_weights` in trainer.py
    but additionally returns the per-token sample index, which is needed for
    per-sample reduction.

    Returns:
        gathered: [M, D] gathered tokens.
        weights: [M] per-token weights.
        sample_idx: [M] per-token sample index in [0, B).
        B: batch size (int).
    """
    B, N, D = patch_tokens.shape
    mask_flat = mask.reshape(-1)
    masked_indices = mask_flat.nonzero(as_tuple=True)[0]
    M = masked_indices.numel()

    if M == 0:
        return None, None, None, B

    gathered = patch_tokens.reshape(B * N, D)[masked_indices]
    sample_idx = (masked_indices // N).long()  # [M]
    if masks_weight is not None:
        weights = masks_weight[sample_idx]
    else:
        num_masked_per_sample = mask.sum(dim=1).float().clamp(min=1.0)
        weights = 1.0 / num_masked_per_sample[sample_idx]
    return gathered, weights, sample_idx, B
