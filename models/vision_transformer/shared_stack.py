"""
Weight-tied shared transformer stack for the looped DINOv2 backbone.

The shared stack contains L unique transformer blocks that are applied T_max
times per forward pass. Each pass through the stack is bracketed by sandwich
LayerNorm, receives a per-step time embedding tau_t added to the previous
state, and re-injects the patch-embedded input z0 at the end (input injection).
These three ingredients are the standard stability package from the looped /
recurrent-depth transformer line of work (Huginn, Ouro/LoopLM, time-modulated
looped transformers).

References
----------
- Geiping et al. 2025, "Scaling up Test-Time Compute with Latent Reasoning"
  (Huginn). arXiv:2502.05171.
- Zhu et al. 2025, "Ouro / LoopLM: Scaling Language Models with Recursive
  Computation". arXiv:2510.25741.
- Yang et al. 2024, "Looped Transformers are Better at Learning Learning
  Algorithms" (input injection). ICLR 2024.
- Xu and Sato 2024, "On the Expressive Power of Looped Transformers"
  (time-modulated looped transformers). arXiv:2410.01405.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class SharedStack(nn.Module):
    """
    Run a weight-tied stack of L transformer blocks T_max times.

    At step t (0-indexed) the recurrence is:

        z_pre = pre_norm(z_prev + tau_t)
        z_blk = block_L o ... o block_1 (z_pre)
        z_t   = post_norm(z_blk) + z0

    where tau_t is a learnable per-step embedding (initialized to zero so the
    initial pass behaves like a plain weight-tied stack), and z0 is the patch-
    embedded input re-added at every step to keep the loop anchored to the
    input signal.

    Compatible with xformers BlockDiagonalMask packed forward via the
    `attn_bias` argument forwarded to each transformer block. Per-recursion-
    step gradient checkpointing is supported so activation memory scales with
    T_max instead of T_max * L.

    Args:
        blocks: nn.ModuleList of L transformer blocks. The blocks are stored
            here by reference; they are the only learnable substructure of
            the stack (besides tau and the sandwich LNs).
        embed_dim: token dimension D.
        T_max: maximum recursion depth (>= 1).
        norm_layer: norm constructor for the sandwich pre/post LNs.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        embed_dim: int,
        T_max: int,
        norm_layer=None,
    ):
        super().__init__()
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        if not isinstance(blocks, nn.ModuleList):
            blocks = nn.ModuleList(list(blocks))

        self.blocks = blocks
        self.L = len(blocks)
        self.T_max = T_max
        self.embed_dim = embed_dim

        norm_layer = norm_layer or nn.LayerNorm
        self.pre_norm = norm_layer(embed_dim)
        self.post_norm = norm_layer(embed_dim)

        # Per-step time embedding. Zero-initialized so the very first forward
        # behaves like a weight-tied stack with no time conditioning, which
        # makes it easy to unit-test parity against a non-looped reference at
        # T_max=1.
        self.tau = nn.Parameter(torch.zeros(T_max, embed_dim))

    def step(
        self,
        z_prev: torch.Tensor,
        z0: torch.Tensor,
        tau_t: torch.Tensor,
        attn_bias=None,
    ) -> torch.Tensor:
        """
        One recursion step. Kept as a separate method so it can be wrapped
        with `torch.utils.checkpoint.checkpoint` per-step.
        """
        z = z_prev + tau_t
        z = self.pre_norm(z)
        for blk in self.blocks:
            z = blk(z, attn_bias=attn_bias)
        z = self.post_norm(z)
        z = z + z0
        return z

    def forward(
        self,
        z0: torch.Tensor,
        attn_bias=None,
        grad_checkpoint: bool = False,
        return_per_step: bool = False,
    ):
        """
        Run all T_max recursion steps starting from z0.

        Args:
            z0: input-embedded tensor (CLS + register + patch tokens), used
                both as the initial state and as the input-injection target.
            attn_bias: optional BlockDiagonalMask for packed forward. Reused
                identically at every step.
            grad_checkpoint: if True, checkpoint each recursion step.
            return_per_step: if True, return a list [z_1, ..., z_T_max];
                otherwise return only z_T_max.

        Returns:
            Tensor (final step) or list of tensors (one per step).
        """
        outs: List[torch.Tensor] = []
        z_prev = z0
        for t in range(self.T_max):
            tau_t = self.tau[t].to(z_prev.dtype)
            if grad_checkpoint and torch.is_grad_enabled():
                z_t = torch.utils.checkpoint.checkpoint(
                    self.step, z_prev, z0, tau_t, attn_bias,
                    use_reentrant=False,
                )
            else:
                z_t = self.step(z_prev, z0, tau_t, attn_bias=attn_bias)
            if return_per_step:
                outs.append(z_t)
            z_prev = z_t

        if return_per_step:
            return outs
        return z_prev
