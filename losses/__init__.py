"""
Loss functions for DINOv2 training.
Exports all loss components.
"""

from .dino_loss import DINOLoss
from .ibot_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss
from .kde_loss import KDELoss
from .prototype_loss import PatchPrototypeLoss
from .looped_loss import (
    sinkhorn_knopp,
    dino_per_sample_loss,
    ibot_per_sample_loss,
    gather_masked_tokens,
)

__all__ = [
    'DINOLoss',
    'iBOTPatchLoss',
    'KoLeoLoss',
    'KDELoss',
    'PatchPrototypeLoss',
    'sinkhorn_knopp',
    'dino_per_sample_loss',
    'ibot_per_sample_loss',
    'gather_masked_tokens',
]
