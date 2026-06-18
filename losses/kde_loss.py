"""
KDE regularization loss with von Mises-Fisher kernel.
Replaces KoLeo for pathology data where near-duplicate minibatch tiles
cause KoLeo's nearest-neighbor distance to collapse and its gradient
to explode.

Reference: Zimmermann et al., "Virchow 2: Scaling Self-Supervised Mixed
Magnification Models in Pathology", arXiv:2408.00738 Section 3.2 / 5.2;
open implementation: MedARC-AI/OpenMidnight dinov2/loss/kde_loss.py.

Canonical behavior (matches Virchow2 / OpenMidnight):
  - density is estimated on the LOCAL per-GPU batch (no cross-GPU all-gather),
  - the self-comparison term (the diagonal, exp(kappa)) is INCLUDED in the
    per-sample sum; that bounded self term is what keeps the gradient bounded.
Deliberate deviation, kept on purpose: the reference returns the entropy
-mean(log density); we return +mean(log density) so that MINIMIZING the loss
lowers density and spreads features (the correct repulsion sign).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDELoss(nn.Module):
    """
    Kernel density estimation regularizer with von Mises-Fisher kernel,
    estimated over the local per-GPU batch (canonical Virchow2 / OpenMidnight).

    Args:
        kappa: vMF concentration parameter (Virchow2 / OpenMidnight use 5.0)
    """
    def __init__(self, kappa=5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, student_output, eps=1e-8):
        """
        Compute the KDE regularizer on the local batch.

        Args:
            student_output: Feature vectors [B, D]
            eps: Small constant for numerical stability inside the log

        Returns:
            Scalar; minimizing it lowers per-sample feature density and thus
            encourages a more uniform feature distribution.
        """
        with torch.cuda.amp.autocast(enabled=False):
            x = F.normalize(student_output.float(), p=2, dim=-1)

            # Pairwise cosine similarities [B, B].
            sim = x @ x.t()

            # Unnormalized vMF kernel exp(kappa * cos_sim). The diagonal equals
            # exp(kappa) (the self-comparison) and is INCLUDED in the row sum:
            # it floors the density away from zero and bounds the gradient.
            density = torch.exp(self.kappa * sim).sum(dim=1)

            # Fork sign: +mean(log density) (NOT the reference's -entropy), so
            # minimizing the loss pushes density down -> spreads features.
            loss = torch.log(density + eps).mean()
            return loss
