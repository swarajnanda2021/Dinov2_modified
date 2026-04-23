"""
KDE regularization loss with von Mises-Fisher kernel.
Replaces KoLeo for pathology data where near-duplicate minibatch tiles
cause KoLeo's nearest-neighbor distance to collapse and its gradient
to explode.

Reference: Zimmermann et al., "Virchow 2: Scaling Self-Supervised Mixed
Magnification Models in Pathology", arXiv:2408.00738 Section 5.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class KDELoss(nn.Module):
    """
    Kernel density estimation regularizer with von Mises-Fisher kernel.
    Features are gathered across all GPUs before density estimation to
    give a more accurate uniformity signal when world_size > 1.

    Args:
        kappa: vMF concentration parameter (Virchow2 uses 5.0)
    """
    def __init__(self, kappa=5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, student_output, eps=1e-8):
        """
        Compute KDE loss with cross-GPU feature pooling.

        Args:
            student_output: Feature vectors [B_local, D]
            eps: Small constant for numerical stability inside the log

        Returns:
            Scalar loss value encouraging uniform feature distribution
        """
        with torch.cuda.amp.autocast(enabled=False):
            x_local = F.normalize(student_output.float(), p=2, dim=-1)

            # Gather features from all GPUs for global density estimation
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                x_list = [torch.zeros_like(x_local) for _ in range(world_size)]
                dist.all_gather(x_list, x_local)
                # Replace this GPU's slot with the local tensor so the graph
                # stays connected for local samples. Remote slots are detached
                # by design (all_gather does not backprop across GPUs).
                rank = dist.get_rank()
                x_list[rank] = x_local
                x_all = torch.cat(x_list, dim=0)
            else:
                x_all = x_local

            N = x_all.shape[0]

            # Pairwise cosine similarities [N, N]
            sim = x_all @ x_all.t()

            # vMF kernel with log-sum-exp numerical stability
            logits = self.kappa * sim
            max_per_row = logits.max(dim=1, keepdim=True).values
            kernel = torch.exp(logits - max_per_row)

            # Mask diagonal (exclude self-similarity)
            mask = ~torch.eye(N, dtype=torch.bool, device=x_all.device)
            density = (kernel * mask.float()).sum(dim=1) / (N - 1)

            # Log-density per sample with stability offset recovered
            log_density = torch.log(density + eps) + max_per_row.squeeze(1)

            # Gradient only through local slice, so each GPU contributes
            # gradient over its own batch samples
            if dist.is_available() and dist.is_initialized():
                B_local = x_local.shape[0]
                rank = dist.get_rank()
                start = rank * B_local
                end = start + B_local
                loss = log_density[start:end].mean()
            else:
                loss = log_density.mean()

            return loss
