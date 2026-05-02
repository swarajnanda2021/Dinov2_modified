"""
Models package for DINOv2 training.
Exports all model components for easy importing.
"""

from .dinov2_model import CombinedModelDINO
from .prototype_bank import LinearPrototypeBank
from .vision_transformer.modern_vit import VisionTransformer as ModernViT
from .vision_transformer.shared_stack import SharedStack
from .vision_transformer.auxiliary_models import (
    DINOHead,
    ADIOSMaskModel,
    MaskModel,
    CellViT,
)
from .halting_head import (
    HaltHead,
    pool_cls_by_image,
    pondernet_marginals,
    pondernet_kl_to_geometric,
    geometric_prior,
    lambda_p_anneal,
    expected_halt_step,
)

__all__ = [
    'CombinedModelDINO',
    'LinearPrototypeBank',
    'ModernViT',
    'SharedStack',
    'DINOHead',
    'ADIOSMaskModel',
    'MaskModel',
    'CellViT',
    'HaltHead',
    'pool_cls_by_image',
    'pondernet_marginals',
    'pondernet_kl_to_geometric',
    'geometric_prior',
    'lambda_p_anneal',
    'expected_halt_step',
]