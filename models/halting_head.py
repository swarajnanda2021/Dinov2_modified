"""
Image-level halting head and PonderNet utilities for the looped DINOv2 backbone.

Halting decisions are made per IMAGE (not per crop or per token), so iBOT's
requirement that masked patches reach the final layer is preserved: every
crop traverses every recursion step in lockstep during training, and the
halting distribution only weights the per-step loss mixture.

Mechanism:
  1. At each recursion step t, gather the CLS tokens of the image's crops
     (8 per image: 2 globals + 6 locals in the standard DINO setup) and
     mean-pool them into a single per-image vector.
  2. Apply a learned linear head + sigmoid to produce a halting probability
     h_t per image.
  3. Compute the PonderNet step-marginal p_t = h_t * prod_{s<t}(1 - h_s),
     with the final step absorbing remaining mass so sum_t p_t = 1.
  4. Regularize p toward a truncated geometric prior via KL.

References
----------
- Banino et al. 2021, "PonderNet: Learning to Ponder". arXiv:2107.05407.
- Yin et al. 2022, "A-ViT: Adaptive Tokens for Efficient Vision Transformer"
  (distributional-prior regularizer for vision halting). CVPR 2022.
- Graves 2016, "Adaptive Computation Time for Recurrent Neural Networks"
  (precursor; PonderNet's reformulation is more stable). arXiv:1603.08983.
"""

from typing import List

import torch
import torch.nn as nn


class HaltHead(nn.Module):
    """
    Per-image halting head: a single linear projection over the pooled CLS
    representation, applied at every recursion step.

    Negligible parameter cost relative to the shared backbone stack
    (D parameters for the weight, 1 for the bias).

    Args:
        embed_dim: backbone token dimension D.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 1)
        # Zero-init so the early-training halting distribution is approximately
        # uniform at sigmoid(0) = 0.5, which gives PonderNet a stable starting
        # point before the KL regularizer pulls it toward the geometric prior.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cls_pool: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_pool: [N, D] image-pooled CLS tokens (mean over an image's
                crops). N is the number of images in the batch.
        Returns:
            [N] halting probabilities in (0, 1).
        """
        logits = self.proj(cls_pool).squeeze(-1)
        return torch.sigmoid(logits)


def pool_cls_by_image(
    cls_per_crop: List[torch.Tensor],
) -> torch.Tensor:
    """
    Mean-pool CLS tokens by image across crops.

    Each entry of cls_per_crop is the CLS tokens for one crop type, shape
    [N, D], indexed in image order (so cls_per_crop[c][i] is image i's CLS
    for crop type c). The pool is therefore a simple stack-and-mean.

    Mean pooling is preferred over attention pooling because (a) it is
    parameter-free, and (b) DINO's CLS is explicitly trained to be view-
    invariant, so the pool approximates any single crop's CLS at convergence
    and the train-inference distribution gap (single-crop pool of one) stays
    small.

    Args:
        cls_per_crop: list of [N, D] tensors, one per crop type.
    Returns:
        [N, D] mean-pooled CLS per image.
    """
    if len(cls_per_crop) == 0:
        raise ValueError("pool_cls_by_image requires at least one crop tensor")
    stacked = torch.stack(cls_per_crop, dim=0)  # [C, N, D]
    return stacked.mean(dim=0)  # [N, D]


def pondernet_marginals(h_per_step: torch.Tensor) -> torch.Tensor:
    """
    Compute PonderNet step-marginals from per-step halting probabilities.

        p_t       = h_t * prod_{s<t} (1 - h_s)        for t = 1, ..., T_max-1
        p_{T_max} = prod_{s<T_max} (1 - h_s)          (absorbs remaining mass)

    so sum_t p_t = 1 per image, regardless of the values of h.

    Args:
        h_per_step: [T_max, N] halting probabilities (each in (0, 1)).

    Returns:
        [T_max, N] step-marginal probabilities.
    """
    T_max, N = h_per_step.shape
    not_halted = torch.ones(N, device=h_per_step.device, dtype=h_per_step.dtype)
    out = []
    for t in range(T_max - 1):
        out.append(h_per_step[t] * not_halted)
        not_halted = not_halted * (1.0 - h_per_step[t])
    out.append(not_halted)  # final-step absorption
    return torch.stack(out, dim=0)  # [T_max, N]


def geometric_prior(
    T_max: int,
    lambda_p: float,
    device,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Truncated geometric prior over recursion steps.

    Steps 1..T_max-1 follow the geometric distribution with parameter
    lambda_p; the last bin absorbs the truncation tail so the prior sums to 1.

        prior_t       = lambda_p * (1 - lambda_p)^(t-1)   for t = 1..T_max-1
        prior_{T_max} = (1 - lambda_p)^(T_max - 1)

    Smaller lambda_p concentrates mass at later steps (encourages deeper
    computation); larger lambda_p concentrates mass at earlier steps
    (encourages shallower computation). With lambda_p in [0.3, 0.9] the
    expected halted depth ranges roughly from ~3 to ~1.1 for T_max=4.

    Args:
        T_max: number of recursion steps.
        lambda_p: prior parameter in (0, 1).
        device: target device.
        dtype: target dtype.

    Returns:
        [T_max] prior probabilities summing to 1.
    """
    if not (0.0 < lambda_p < 1.0):
        raise ValueError(f"lambda_p must be in (0, 1), got {lambda_p}")
    if T_max < 1:
        raise ValueError(f"T_max must be >= 1, got {T_max}")
    vals = []
    not_halted = 1.0
    for _ in range(T_max - 1):
        vals.append(lambda_p * not_halted)
        not_halted = not_halted * (1.0 - lambda_p)
    vals.append(not_halted)
    return torch.tensor(vals, device=device, dtype=dtype)


def pondernet_kl_to_geometric(
    p: torch.Tensor,
    lambda_p: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL( p || Geom(lambda_p) ) per image.

    Args:
        p: [T_max, N] PonderNet step-marginals.
        lambda_p: prior parameter in (0, 1).
        eps: clamp floor for log stability.

    Returns:
        [N] per-image KL divergence.
    """
    T_max = p.shape[0]
    prior = geometric_prior(T_max, lambda_p, device=p.device, dtype=p.dtype)
    log_p = torch.log(p.clamp_min(eps))
    log_prior = torch.log(prior.clamp_min(eps)).unsqueeze(1)  # [T_max, 1]
    return (p * (log_p - log_prior)).sum(dim=0)  # [N]


def lambda_p_anneal(
    iteration: int,
    total_iterations: int,
    lambda_start: float,
    lambda_end: float,
    anneal_frac: float,
) -> float:
    """
    Linearly anneal lambda_p from `lambda_start` to `lambda_end` over the first
    `anneal_frac` of training, then hold at `lambda_end`. Mirrors Huginn's
    variable-depth curriculum: a high lambda_p (shallow prior) at the start
    keeps the shared stack from being asked for deep compute before the blocks
    have learned anything useful, and tapering to a lower lambda_p later lets
    harder images earn more compute as the representations mature.
    """
    if total_iterations <= 0 or anneal_frac <= 0:
        return lambda_end
    anneal_iters = max(1, int(round(anneal_frac * total_iterations)))
    frac = min(1.0, max(0.0, iteration / anneal_iters))
    return lambda_start + (lambda_end - lambda_start) * frac


def expected_halt_step(p: torch.Tensor) -> torch.Tensor:
    """
    Expected halt step under PonderNet marginals: E[t] = sum_t t * p_t.
    Steps are 1-indexed for consistency with the design document.

    Args:
        p: [T_max, N] step-marginals.
    Returns:
        [N] expected halt step per image.
    """
    T_max, N = p.shape
    t_axis = torch.arange(1, T_max + 1, device=p.device, dtype=p.dtype).unsqueeze(1)  # [T_max, 1]
    return (p * t_axis).sum(dim=0)
