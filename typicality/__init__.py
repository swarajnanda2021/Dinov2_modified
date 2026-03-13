"""
Adaptive Redundancy Dampening for Rare Morphology Preservation.
Operates in the student's DINO head bottleneck space (256-dim).
Fully detached from DINO gradients — only R has its own optimizer.
"""

from .representative_prototypes import RepresentativePrototypes
from .typicality_bank import TypicalityBank
from .typicality_scorer import TypicalityScorer

__all__ = [
    'RepresentativePrototypes',
    'TypicalityBank',
    'TypicalityScorer',
]