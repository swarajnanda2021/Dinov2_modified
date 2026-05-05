"""
Argument parser and configuration for DINOv2 training.
"""

import argparse
import utils


def get_args_parser():
    """
    Create argument parser with all training configuration options.

    Returns:
        ArgumentParser with all training arguments
    """
    parser = argparse.ArgumentParser('Semantic-DINOv2 with Sequence Packing', add_help=False)

    # ========== Model parameters ==========
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for vision transformer')
    parser.add_argument('--embeddingdim', default=768, type=int,
                        help='Embedding dimension')
    parser.add_argument('--vitheads', default=12, type=int,
                        help='Number of attention heads')
    parser.add_argument('--vitdepth', default=12, type=int,
                        help='Number of transformer blocks')
    parser.add_argument('--out_dim', default=65536, type=int,
                        help='Output dimension of projection heads')
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help='Normalize last layer of projection head')
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help='Use batch normalization in projection head')

    # ========== Flexible augmentation parameters ==========
    parser.add_argument('--global_views', default=2, type=int,
                        help='Number of global views')
    parser.add_argument('--n_standard_local_crops', default=3, type=int,
                        help='Number of standard local crops')
    parser.add_argument('--local_crop_size', default=96, type=int,
                        help='Size of local crops')

    # ========== Mask model parameters ==========
    parser.add_argument('--mask_model_arch', default='unet', type=str,
                    choices=['unet', 'vit_unet'],
                    help='Mask model architecture: unet (ADIOS) or vit_unet')
    parser.add_argument('--mask_checkpoint', type=str,
                        help='Path to pre-trained mask model checkpoint')
    parser.add_argument('--num_masks', default=3, type=int,
                        help='Number of semantic mask channels')
    parser.add_argument('--mask_encoder_dim', default=192, type=int,
                        help='Encoder dimension for vit_unet mask model architecture')

    # ========== Semantic iBOT parameters ==========
    parser.add_argument('--use_semantic_ibot', default=False, type=utils.bool_flag,
                    help='Enable semantic masking in iBOT loss. When True, loads the adversarial '
                         'mask model specified by --mask_checkpoint and --mask_model_arch, converts '
                         'its soft masks to token-level binary masks, and adds a semantic iBOT loss '
                         'term on global crop 1.')
    parser.add_argument('--use_semantic_prototypes', default=False, type=utils.bool_flag,
                    help='Enable semantic masking in prototype clustering loss. Requires '
                         '--use_semantic_ibot=True (semantic masks must be generated). Adds a '
                         'semantic prototype prediction term using the same semantic token masks.')
    parser.add_argument('--semantic_ibot_weight', default=1.0, type=float,
                        help='Weight multiplier for the semantic iBOT loss term')
    parser.add_argument('--semantic_clustering_weight', default=1.0, type=float,
                        help='Weight multiplier for the semantic prototype clustering loss term')
    parser.add_argument('--semantic_masks_per_iteration', default=1, type=int,
                        help='How many of the num_masks semantic channels to use per iteration. '
                             '1 = randomly sample one channel each iteration (cheapest). '
                             'num_masks = use all channels every iteration (most expensive).')

    # ========== Loss parameters ==========
    parser.add_argument('--momentum_teacher', default=0.996, type=float,
                        help='EMA momentum for teacher update')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help='Initial teacher temperature')
    parser.add_argument('--teacher_temp', default=0.07, type=float,
                        help='Final teacher temperature')
    parser.add_argument('--teacher_temp_warmup_iters', default=30000, type=int,
                        help='Teacher temperature warmup iterations')
    parser.add_argument('--koleo_loss_weight', default=0.1, type=float,
                        help='Weight for KoLeo regularization loss')
    parser.add_argument('--ibot_loss_weight', default=1.0, type=float,
                        help='Weight for iBOT patch loss')
    parser.add_argument('--mask_ratio_min', default=0.1, type=float,
                        help='Minimum mask ratio for iBOT block masking')
    parser.add_argument('--mask_ratio_max', default=0.5, type=float,
                        help='Maximum mask ratio for iBOT block masking')
    parser.add_argument('--mask_sample_probability', default=0.5, type=float,
                        help='Fraction of samples in batch to apply masking')

    # ========== Patch Prototype Clustering parameters ==========
    parser.add_argument('--use_prototype_clustering', default=True, type=utils.bool_flag,
                    help='Enable patch prototype clustering loss')
    parser.add_argument('--num_prototypes', default=8192, type=int,
                        help='Number of prototypes for clustering')
    parser.add_argument('--clustering_weight', default=1.0, type=float,
                        help='Weight for prototype clustering loss')
    parser.add_argument('--clustering_teacher_temp', default=0.07, type=float,
                        help='Teacher temperature for clustering')
    parser.add_argument('--clustering_student_temp', default=0.1, type=float,
                        help='Student temperature for clustering')

    # ========== Typicality Dampening parameters ==========
    parser.add_argument('--use_typicality_dampening', default=False, type=utils.bool_flag,
                        help='Enable adaptive redundancy dampening for rare morphology preservation')
    parser.add_argument('--typicality_K_prime', default=256, type=int,
                        help='Number of representative prototypes (128=undercomplete, 256=complete, 512=overcomplete)')
    parser.add_argument('--typicality_bank_size', default=8192, type=int,
                        help='Bank capacity M (number of stored signatures)')
    parser.add_argument('--typicality_modulation', default='adaptive_temp', type=str,
                        choices=['adaptive_temp', 'weighted_loss'],
                        help='Gradient modulation variant')
    parser.add_argument('--typicality_alpha', default=1.0, type=float,
                        help='Dampening strength for adaptive temperature variant')
    parser.add_argument('--typicality_beta', default=0.5, type=float,
                        help='Dampening strength for weighted loss variant')
    parser.add_argument('--typicality_warmup_iters', default=15000, type=int,
                        help='Iterations before typicality scores modulate the loss')
    parser.add_argument('--typicality_repr_lr', default=1e-3, type=float,
                        help='Fixed learning rate for representative prototype optimizer')
    parser.add_argument('--typicality_replace_fraction', default=0.1, type=float,
                        help='Fraction of bank to refresh per step')

    # ========== Adversarial mask-as-student-view augmentation parameters ==========
    # Note: --num_masks, --mask_model_arch, --mask_checkpoint are already declared
    # under the Mask model parameters section (shared with semantic iBOT).
    parser.add_argument('--use_adversarial_mask_augmentation', default=False, type=utils.bool_flag,
                        help='Apply the adversarial mask model output as IMAGE-LEVEL masks to create '
                             'additional student views (distinct from semantic iBOT which uses the same '
                             'model at the TOKEN level inside the iBOT loss).')
    parser.add_argument('--crops_per_mask', default=1, type=int,
                        help='Number of local crops to extract per adversarial-masked global view')

    # ========== CellViT augmentation parameters ==========
    parser.add_argument('--use_cellvit_augmentation', default=False, type=utils.bool_flag,
                        help='Enable CellViT-B based nuclei/background augmentation')
    parser.add_argument('--cellvit_checkpoint', type=str, default=None,
                        help='Path to trained CellViT model checkpoint')
    parser.add_argument('--cellvit_crops_per_channel', default=1, type=int,
                        help='Number of crops per channel (nuclei/background)')

    # ========== Random rectangular mask augmentation parameters ==========
    parser.add_argument('--use_random_mask_augmentation', default=False, type=utils.bool_flag,
                        help='Enable random rectangular mask-based augmentation')
    parser.add_argument('--random_num_masks', default=2, type=int,
                        help='Number of random rectangular masks to generate')
    parser.add_argument('--random_crops_per_mask', default=1, type=int,
                        help='Number of local crops per random mask')

    # ========== Training parameters ==========
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--total_iterations', default=300000, type=int,
                        help='Total number of training iterations')
    parser.add_argument('--warmup_iterations', default=10000, type=int,
                        help='Number of warmup iterations')
    parser.add_argument('--freeze_last_layer_iters', default=5000, type=int,
                        help='Freeze last layer for this many iterations')
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
                        help='Use mixed precision training')
    parser.add_argument('--clip_grad', type=float, default=3.0,
                        help='Gradient clipping value')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Base learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04,
                        help='Initial weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.4,
                        help='Final weight decay')
    parser.add_argument('--lr_decay_rate', default=0.9, type=float,
                        help='Layer-wise LR decay rate (1.0 = no decay, 0.9 = typical)')
    parser.add_argument('--grad_checkpointing', default=False, type=utils.bool_flag,
                    help='Enable gradient checkpointing to reduce memory at cost of ~40% speed')

    # ========== Dataset and I/O ==========
    parser.add_argument('--dataset_sources', type=str, nargs='+',
                        help='Dataset sources in format NAME:BASE_DIR:INDEX_FILE')
    parser.add_argument('--output_dir', default=".", type=str,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--save_checkpoint_freq', default=2000, type=int,
                        help='Checkpoint saving frequency')
    parser.add_argument('--visualization_freq', default=100, type=int,
                        help='Mask visualization frequency')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--num_workers', default=10, type=int,
                        help='Number of data loading workers')

    # ========== Distributed training ==========
    parser.add_argument("--dist_url", default="env://", type=str,
                        help='URL for distributed training setup')
    parser.add_argument("--local_rank", default=0, type=int,
                        help='Local rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')

    # ========== Pathology FM Recipe ==========
    parser.add_argument('--use_pathology_recipe', default=False, type=utils.bool_flag,
                        help='Enable pathology-FM recipe bundle. Sources: KDE regularizer, '
                             'ECT augmentation, teacher_temp=0.04 '
                             '[Virchow/Virchow2, Paige/MSKCC/MSR, arXiv:2309.07778 and '
                             'arXiv:2408.00738]; solarization off, V-flip, 90-deg rotations '
                             '[Virchow2 + RudolfV + Hibou convergence]; patch_size=14 '
                             '[community standard across Virchow family, Midnight, RudolfV, '
                             'H-optimus]; bf16 end-to-end [scaling-regime choice, flagged '
                             'retroactively by Virchow2G]. Auto-enables qk_norm, 8+ register '
                             'tokens, out_dim=131072 (Virchow v1 Methods, Paige standard), '
                             'and StableAdamW beta2=0.95 when embeddingdim >= 1280 '
                             '[Virchow2G scaling package, arXiv:2408.00738 Section 6].')

    parser.add_argument('--ect_probability', default=0.4, type=float,
                        help='Probability of applying ECT branch on 40x tiles (source size '
                             '>= 448). Default 0.4 means 40%% ECT, 60%% standard crop-and-'
                             'resize. ECT itself is from Virchow2 arXiv:2408.00738 Section '
                             '5.1. The probabilistic per-tile-size framing is a user '
                             'adaptation for mixed-magnification (40x + 20x) training data. '
                             'Only active when --use_pathology_recipe=True.')

    parser.add_argument('--kde_kappa', default=5.0, type=float,
                        help='vMF kernel concentration for KDE regularizer. Value 5.0 is the '
                             'Virchow2 default (arXiv:2408.00738 Section 5.2 and ablation). '
                             'Only used when --use_pathology_recipe=True.')

    parser.add_argument('--qk_norm', default=None, type=utils.bool_flag,
                        help='Enable QK normalization in attention [Virchow2G scaling '
                             'package, arXiv:2408.00738 Section 6]. If None (default), '
                             'auto-enables when embeddingdim >= 1280 AND '
                             '--use_pathology_recipe=True. Explicit True/False overrides '
                             'the auto-gate.')

    parser.add_argument('--num_register_tokens', default=4, type=int,
                        help='Number of register tokens [Darcet et al. 2023, adopted by '
                             'Virchow2 (4), H-optimus-1 (4), Midnight (4), Virchow2G (8), '
                             'UNI2-h (8)]. Auto-bumped to 8 if --use_pathology_recipe=True '
                             'and embeddingdim >= 1280.')

    # ========== Looped backbone (weight-tied recurrent depth + adaptive halting) ==========
    # When --use_looped_backbone=True, the standard depth-`vitdepth` stack of
    # unique transformer blocks is replaced by a shared stack of `shared_stack_L`
    # blocks applied `recursion_T_max` times, with sandwich LayerNorm, per-step
    # time embeddings, and input injection. Halting decisions are made per IMAGE
    # via a small linear head + PonderNet KL-to-geometric prior. The student
    # always runs all T_max steps in lockstep; halting only weights the per-
    # step loss mixture during training, and is consumed for real early exit
    # at inference.
    #
    # This feature is incompatible with --use_typicality_dampening,
    # --use_prototype_clustering, --use_semantic_ibot, --use_semantic_prototypes,
    # --use_adversarial_mask_augmentation, --use_cellvit_augmentation, and
    # --use_random_mask_augmentation in this revision (the looped trainer path
    # only supports the standard DINO + iBOT + KoLeo/KDE objective). Combinations
    # are gated by an explicit error in the trainer.
    #
    # References:
    #   - Geiping et al. 2025 "Huginn" (recurrent-depth LMs).      arXiv:2502.05171
    #   - Zhu et al. 2025 "Ouro / LoopLM".                          arXiv:2510.25741
    #   - Banino et al. 2021 "PonderNet" (halting + KL prior).      arXiv:2107.05407
    #   - Yin et al. 2022 "A-ViT" (vision halting + dist. prior).   CVPR 2022
    #   - Yang et al. 2024 (input injection).                       ICLR 2024
    #   - Saunshi et al. 2025 (looped inductive bias).              arXiv:2502.17416
    #   - Schwethelm et al. 2026 (iso-depth scaling).               arXiv:2604.21106
    parser.add_argument('--use_looped_backbone', default=False, type=utils.bool_flag,
                        help='Enable weight-tied looped backbone with image-level PonderNet '
                             'adaptive halting. Default False reduces to vanilla DINOv2.')
    parser.add_argument('--shared_stack_L', default=3, type=int,
                        help='Number of unique transformer blocks in the shared stack. '
                             'The stack is applied `recursion_T_max` times per forward, so '
                             'effective depth is shared_stack_L * recursion_T_max. Ouro and '
                             'Huginn report L=3, T_max=4 as a reasonable default at LM scale; '
                             'pathology pilots should ablate this.')
    parser.add_argument('--recursion_T_max', default=4, type=int,
                        help='Number of recursion steps T_max. Both teacher and student run '
                             'all T_max steps during training; the halting head only weights '
                             'the per-step loss mixture.')
    parser.add_argument('--ponder_kl_beta', default=0.01, type=float,
                        help='Weight of the KL-to-geometric-prior regularizer in the '
                             'PonderNet halting term (Banino et al. 2021 default).')
    parser.add_argument('--ponder_lambda_p_start', default=0.9, type=float,
                        help='Starting value of the geometric-prior parameter lambda_p. '
                             'Larger lambda_p concentrates mass at earlier steps (encourages '
                             'shallower halting). Default 0.9 -> E[t] approx 1.1.')
    parser.add_argument('--ponder_lambda_p_end', default=0.3, type=float,
                        help='Ending value of lambda_p after annealing. Smaller value '
                             'concentrates mass at later steps (encourages deeper halting). '
                             'Default 0.3 -> E[t] approx 3.0 for T_max=4.')
    parser.add_argument('--ponder_lambda_p_anneal_frac', default=0.3, type=float,
                        help='Fraction of total iterations over which to linearly anneal '
                             'lambda_p from start -> end, then hold at end. 0.0 disables '
                             'annealing.')
    parser.add_argument('--ponder_inference_threshold', default=0.99, type=float,
                        help='Cumulative halt probability threshold for inference-time early '
                             'exit (1 - epsilon). Halts at step t when cumulative h reaches '
                             'this value. Used by inference drivers; not consumed at training.')

    return parser
