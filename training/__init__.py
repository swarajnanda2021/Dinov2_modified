"""
Training utilities and orchestration for DINOv2.
"""

from .trainer import train_dinov2
from .helpers import (
    load_pretrained_mask_model,
    load_pretrained_cellvit_model,
    apply_masks_to_images,
    apply_cellvit_masks,
    extract_local_crops_from_masked,
    extract_crops_from_cellvit_channel,
    generate_random_token_masks,
    generate_random_image_masks,
    calculate_total_student_views,
    save_iteration_masks_efficient,
    worker_init_fn,
    setup_ddp_model,
)

from .fsdp_setup import apply_fsdp_wrapping, get_mixed_precision_policy
from .checkpoint_fsdp2 import save_checkpoint_fsdp2, load_checkpoint_fsdp2
from .param_groups_fsdp2 import get_params_groups_fsdp2, remove_fsdp_compile_names


__all__ = [
    'train_dinov2',
    'load_pretrained_mask_model',
    'load_pretrained_cellvit_model',
    'apply_masks_to_images',
    'apply_cellvit_masks',
    'extract_local_crops_from_masked',
    'extract_crops_from_cellvit_channel',
    'generate_random_token_masks',
    'generate_random_image_masks',
    'calculate_total_student_views',
    'save_iteration_masks_efficient',
    'worker_init_fn',
    'setup_ddp_model',
    'apply_fsdp_wrapping',
    'get_mixed_precision_policy',
    'save_checkpoint_fsdp2',
    'load_checkpoint_fsdp2',
    'get_params_groups_fsdp2',
    'remove_fsdp_compile_names',
]