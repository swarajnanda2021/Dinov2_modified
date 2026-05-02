"""
PonderNet mixture-weighted training step for the looped DINOv2 backbone.

This module factors out the looped student/teacher forward + loss flow so the
main `trainer.py` only needs a small dispatching branch. It implements the
loss equation from `LOOPED_DINOV2.md`:

    L[i] = sum_t p_t[i] * (L_DINO_t[i] + ibot_w * L_iBOT_t[i])
           + koleo_w * L_KoLeo[i] (final step only)
           + beta * KL( p[i] || Geom(lambda_p) )

where p_t is the PonderNet step-marginal, computed from the per-image halting
probabilities h_t = sigma(W_halt . pool_image_t(CLS)).

Only the standard DINO + iBOT + KoLeo / KDE objective is supported. Semantic
iBOT, prototype clustering, typicality dampening, adversarial-mask-as-view,
CellViT, and random-rectangle augmentations are intentionally excluded in this
revision (the trainer rejects those combinations up front).
"""

from typing import Dict

import torch
import torch.distributed as dist

from losses.looped_loss import (
    sinkhorn_knopp,
    dino_per_sample_loss,
    ibot_per_sample_loss,
    gather_masked_tokens,
)
from models.halting_head import (
    pool_cls_by_image,
    pondernet_marginals,
    pondernet_kl_to_geometric,
    geometric_prior,
    lambda_p_anneal,
    expected_halt_step,
)


def _module(m):
    """Return the inner module if `m` is a DDP wrapper, else `m`."""
    return m.module if hasattr(m, 'module') else m


def compute_looped_step(
    student,
    teacher,
    student_all_crops,
    student_masks,
    teacher_global_crops,
    block_masks_1,
    masks_weight_1,
    block_masks_2,
    masks_weight_2,
    dino_class_loss,
    dino_koleo_loss,
    args,
    current_iteration,
    bf16_mode: bool,
) -> Dict[str, torch.Tensor]:
    """
    Run one looped training step. The student runs T_max recursion steps in
    lockstep on `student_all_crops`; the teacher runs once at T_max on
    `teacher_global_crops`. Returns a dict with the total loss and the
    component metrics used by the metric logger.

    Args:
        student: DDP-wrapped CombinedModelDINO with a `halt_head` submodule.
        teacher: DDP-wrapped CombinedModelDINO at EMA of student. No halt head.
        student_all_crops: list of [B, C, H, W] per-crop tensors (8 per image
            in the canonical recipe: 2 globals + 6 locals).
        student_masks: list aligned with student_all_crops; iBOT block masks
            on the two globals, None on locals.
        teacher_global_crops: list of [B, C, H, W] global crops (length 2).
        block_masks_1, block_masks_2: [B, N] iBOT block masks for the two
            globals.
        masks_weight_1, masks_weight_2: [B] per-sample iBOT weights
            (1 / num_masked_per_sample).
        dino_class_loss: DINOLoss instance, used here only for its
            teacher-temp schedule (we compute per-sample CE inline).
        dino_koleo_loss: KoLeoLoss or KDELoss instance, applied to the final
            step's CLS tokens.
        args: training args.
        current_iteration: int.
        bf16_mode: whether bf16 autocast is active.

    Returns:
        Dict with: 'student_loss', 'dino_class_loss_val', 'ibot_loss_val',
        'koleo_loss_val', 'kl_loss_val', 'mean_halt_step', 'h_mean_per_step'.
    """
    student_module = _module(student)
    teacher_module = _module(teacher)
    backbone = student_module.backbone
    halt_head = student_module.halt_head
    if halt_head is None:
        raise RuntimeError(
            "compute_looped_step requires student.halt_head to be set "
            "(use_looped_backbone=True branch)."
        )

    T_max = backbone.looped_T_max
    if T_max <= 0:
        raise RuntimeError("compute_looped_step requires looped backbone (T_max > 0).")

    ncrops = len(student_all_crops)
    n_globals = args.global_views
    B = teacher_global_crops[0].shape[0]
    teacher_temp = dino_class_loss.teacher_temp_schedule(current_iteration)

    # -- Teacher forward (single-step, at T_max) -------------------------------
    with torch.no_grad():
        teacher_output = teacher(
            teacher_global_crops, token_masks=[None] * n_globals, mode='dino',
        )
        teacher_cls_outputs = teacher_output['cls_outputs']  # [n_globals*B, out_dim]
        # iBOT teacher patch tokens (only the two iBOT-masked globals)
        teacher_patch_tokens_g1 = teacher_output['features_list'][0]['patchtokens']
        teacher_patch_tokens_g2 = teacher_output['features_list'][1]['patchtokens']

        # Sinkhorn-Knopp on teacher CLS, computed once for all student steps.
        teacher_cls_normalized = sinkhorn_knopp(
            teacher_cls_outputs, teacher_temp, n_iterations=dino_class_loss.n_iterations
        ).chunk(n_globals)
        teacher_cls_normalized = [q.detach() for q in teacher_cls_normalized]

        # iBOT teacher: gather masked positions, project via teacher patchhead,
        # Sinkhorn-normalize. One pass per global crop, reused across all
        # student steps.
        def _teacher_ibot_targets(t_patch_tokens, mask):
            mask_flat = mask.reshape(-1)
            idx = mask_flat.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                return None
            t_gathered = t_patch_tokens.reshape(-1, t_patch_tokens.shape[-1])[idx]
            t_proj = teacher_module.patchhead(t_gathered)
            return sinkhorn_knopp(t_proj, teacher_temp,
                                  n_iterations=dino_class_loss.n_iterations).detach()

        teacher_ibot_g1 = _teacher_ibot_targets(teacher_patch_tokens_g1, block_masks_1)
        teacher_ibot_g2 = _teacher_ibot_targets(teacher_patch_tokens_g2, block_masks_2)

    # -- Student forward (per-step) -------------------------------------------
    student_output = student(
        student_all_crops,
        token_masks=student_masks,
        mode='dino',
        return_per_step=True,
    )
    cls_outputs_per_step = student_output['cls_outputs_per_step']      # list[T_max] of [ncrops*B, out_dim]
    features_list_per_step = student_output['features_list_per_step']  # list[T_max] of list[ncrops] of dict

    # -- Image-level CLS pooling per step + halting head ----------------------
    h_per_step_list = []
    for t in range(T_max):
        cls_per_crop = [d['clstoken'] for d in features_list_per_step[t]]  # list[ncrops] of [B, D]
        cls_pool = pool_cls_by_image(cls_per_crop)                          # [B, D]
        h_t = halt_head(cls_pool)                                           # [B]
        h_per_step_list.append(h_t)
    h_per_step = torch.stack(h_per_step_list, dim=0)  # [T_max, B]

    p_per_step = pondernet_marginals(h_per_step)      # [T_max, B], sum=1 per image

    # -- KL regularizer toward truncated geometric prior ----------------------
    lambda_p = lambda_p_anneal(
        iteration=current_iteration,
        total_iterations=args.total_iterations,
        lambda_start=args.ponder_lambda_p_start,
        lambda_end=args.ponder_lambda_p_end,
        anneal_frac=args.ponder_lambda_p_anneal_frac,
    )
    kl_per_image = pondernet_kl_to_geometric(p_per_step, lambda_p)  # [B]
    kl_loss = kl_per_image.mean()

    # -- Per-step DINO + iBOT losses, weighted by p_t -------------------------
    student_temp = dino_class_loss.student_temp
    total_dino_per_image = torch.zeros(B, device=h_per_step.device, dtype=h_per_step.dtype)
    total_ibot_per_image = torch.zeros(B, device=h_per_step.device, dtype=h_per_step.dtype)

    for t in range(T_max):
        cls_out_t = cls_outputs_per_step[t]
        feats_t = features_list_per_step[t]

        # ---- DINO at step t ----
        dino_t = dino_per_sample_loss(
            student_cls_outputs=cls_out_t,
            teacher_cls_chunks_normalized=teacher_cls_normalized,
            ncrops=ncrops,
            student_temp=student_temp,
        )  # [B]

        # ---- iBOT at step t (global crop 1 + global crop 2) ----
        s_patch_g1 = feats_t[0]['patchtokens']
        s_patch_g2 = feats_t[1]['patchtokens']

        ibot_g1 = torch.zeros(B, device=s_patch_g1.device, dtype=s_patch_g1.dtype)
        ibot_g2 = torch.zeros(B, device=s_patch_g2.device, dtype=s_patch_g2.dtype)

        if teacher_ibot_g1 is not None:
            s_g1, w_g1, samp_g1, _ = gather_masked_tokens(
                s_patch_g1, block_masks_1, masks_weight_1
            )
            if s_g1 is not None:
                s_proj_g1 = student_module.patchhead(s_g1)
                ibot_g1 = ibot_per_sample_loss(
                    student_proj=s_proj_g1,
                    teacher_proj_normalized=teacher_ibot_g1,
                    weights=w_g1,
                    sample_idx=samp_g1,
                    batch_size=B,
                    student_temp=student_temp,
                )

        if teacher_ibot_g2 is not None:
            s_g2, w_g2, samp_g2, _ = gather_masked_tokens(
                s_patch_g2, block_masks_2, masks_weight_2
            )
            if s_g2 is not None:
                s_proj_g2 = student_module.patchhead(s_g2)
                ibot_g2 = ibot_per_sample_loss(
                    student_proj=s_proj_g2,
                    teacher_proj_normalized=teacher_ibot_g2,
                    weights=w_g2,
                    sample_idx=samp_g2,
                    batch_size=B,
                    student_temp=student_temp,
                )

        ibot_t = (ibot_g1 + ibot_g2) / 2.0  # [B]

        p_t = p_per_step[t]  # [B]
        total_dino_per_image = total_dino_per_image + p_t * dino_t
        total_ibot_per_image = total_ibot_per_image + p_t * ibot_t

    dino_class_loss_val = total_dino_per_image.mean()
    ibot_loss_val = total_ibot_per_image.mean()

    # -- KoLeo / KDE on FINAL-step CLS tokens of the global crops -------------
    final_features = features_list_per_step[T_max - 1]
    global_cls_tokens = [final_features[i]['clstoken'] for i in range(n_globals)]
    if len(global_cls_tokens) > 0:
        koleo_loss_val = sum(dino_koleo_loss(t) for t in global_cls_tokens) / len(global_cls_tokens)
    else:
        koleo_loss_val = torch.tensor(0.0, device=h_per_step.device)

    # -- Total loss -----------------------------------------------------------
    student_loss = (
        dino_class_loss_val
        + args.koleo_loss_weight * koleo_loss_val
        + args.ibot_loss_weight * ibot_loss_val
        + args.ponder_kl_beta * kl_loss
    )

    # -- Diagnostic: expected halt step ---------------------------------------
    with torch.no_grad():
        mean_halt_step = expected_halt_step(p_per_step).mean()
        h_mean_per_step = h_per_step.mean(dim=1).detach()  # [T_max]

    return {
        'student_loss': student_loss,
        'dino_class_loss_val': dino_class_loss_val.detach(),
        'ibot_loss_val': ibot_loss_val.detach(),
        'koleo_loss_val': koleo_loss_val.detach() if torch.is_tensor(koleo_loss_val) else koleo_loss_val,
        'kl_loss_val': kl_loss.detach(),
        'mean_halt_step': mean_halt_step.detach(),
        'h_mean_per_step': h_mean_per_step,
        'lambda_p': lambda_p,
    }
