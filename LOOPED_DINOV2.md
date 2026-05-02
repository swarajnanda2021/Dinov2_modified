# Looped DINOv2 with Image-Level Adaptive Halting — Implementation Notes

This branch (`loop_pathology-fm-recipe`) layers a *looped* (weight-tied
recurrent-depth) backbone with PonderNet-style image-level adaptive halting on
top of the `pathology-fm-recipe` DINOv2 trainer. Every change is opt-in: with
`--use_looped_backbone=False` (default) the code paths are byte-for-byte
identical to `pathology-fm-recipe`.

The full design and references are in the executive summary that motivated this
work; this document is the **engineer-facing companion** that maps that design
to the files in this repo. A schematic of the training pipeline (with explicit
residual pathway for the input injection) is at
[`figures/looped_dinov2_method_residual.svg`](figures/looped_dinov2_method_residual.svg)
(also `.pdf` and `.png`; source:
[`figures/generate_looped_dinov2_method_residual.py`](figures/generate_looped_dinov2_method_residual.py)).
The launch plan for the `(L, T_max)` sweep that evaluates this method, plus a
detailed walkthrough of the schematic, is in
[`LOOPED_DINOV2_SWEEP_PLAN.md`](LOOPED_DINOV2_SWEEP_PLAN.md).

---

## 1. What changed

### 1.1 Architectural substrate

The standard DINOv2 stack of `vitdepth` unique transformer blocks is replaced
by a **shared stack** of `shared_stack_L` blocks applied `recursion_T_max`
times. Each pass through the stack is bracketed by sandwich LayerNorm,
receives a learnable per-step time embedding `tau_t`, and re-injects the
patch-embedded input `z_0`:

```
z_t = post_norm( block_L o ... o block_1 ( pre_norm( z_{t-1} + tau_t ) ) ) + z_0
```

Default: `L = 3`, `T_max = 4`, giving 12 effective block applications per
crop at ~4× parameter reduction vs. a 12-block dense ViT.

Compatible with xformers `BlockDiagonalMask` packed multi-crop forward;
compatible with FSDP2 (fewer unique parameters → smaller shards); compatible
with iBOT masking (every masked token traverses every recursion step);
compatible with register tokens; compatible with the pathology-FM recipe.

### 1.2 Image-level adaptive halting

Halting decisions are made **per image**, not per crop or per token. At each
recursion step `t`:

1. **Pool CLS by image.** Mean-pool the per-crop CLS tokens for an image
   (8 in the canonical recipe: 2 globals + 6 locals) into a single per-image
   vector.
2. **Halt head.** A single linear projection + sigmoid emits the halting
   probability `h_t` per image.
3. **PonderNet marginals.** `p_t = h_t * prod_{s<t}(1 - h_s)`, with the
   final step absorbing remaining mass so `sum_t p_t = 1`.
4. **KL regularizer.** `beta * KL( p || Geom(lambda_p) )`, with `lambda_p`
   linearly annealed from `0.9 -> 0.3` over the first 30% of training.

**Training does not halt early.** Every crop runs all `T_max` steps in
lockstep; the halting distribution only weights the per-step DINO + iBOT
loss mixture. Inference (added separately) uses `h_t` for real early exit.

### 1.3 Loss

```
L[i] = sum_t p_t[i] * (L_DINO_t[i] + ibot_w * L_iBOT_t[i])
       + koleo_w * L_KoLeo[i]            (final step only)
       + beta * KL( p[i] || Geom(lambda_p) )
```

Teacher always runs at `T_max` only (no halting head); its outputs are
shared across all student steps. Sinkhorn-Knopp on teacher CLS and on the
teacher-projected iBOT-masked patch tokens is computed **once per batch**
outside the t-loop.

The DINO and iBOT losses needed for the mixture are computed in **per-sample
form** (`[B]` rather than scalar) so the per-image PonderNet marginals can
weight each image's contribution before reduction.

---

## 2. Files added

| File | Purpose |
| --- | --- |
| [`models/vision_transformer/shared_stack.py`](models/vision_transformer/shared_stack.py) | `SharedStack`: weight-tied L-block stack applied T_max times with sandwich LN, time embeddings, input injection, per-step gradient checkpointing. |
| [`models/halting_head.py`](models/halting_head.py) | `HaltHead` (linear) and PonderNet utilities: `pool_cls_by_image`, `pondernet_marginals`, `geometric_prior`, `pondernet_kl_to_geometric`, `lambda_p_anneal`, `expected_halt_step`. |
| [`losses/looped_loss.py`](losses/looped_loss.py) | Per-sample DINO/iBOT loss helpers + standalone `sinkhorn_knopp` so the teacher's targets can be computed once per batch and reused across all `T_max` student steps. |
| [`training/looped_step.py`](training/looped_step.py) | The PonderNet mixture-weighted training step. Does the per-step student forward, image-level pooling, halt-head application, marginal computation, KL term, per-step DINO + iBOT, and KoLeo on the final step. |
| [`LOOPED_DINOV2.md`](LOOPED_DINOV2.md) | This document. |

## 3. Files modified

| File | Change |
| --- | --- |
| [`models/vision_transformer/modern_vit.py`](models/vision_transformer/modern_vit.py) | New constructor args `looped_T_max`, `looped_L`. When enabled, blocks are an `nn.ModuleList` driven by `SharedStack`; new `forward_features_list_per_step` returns per-recursion-step outputs. Single-step forward paths (`forward_features`, `forward_features_list`, single-tensor `forward`) all dispatch through `SharedStack(return_per_step=False)` when looped, returning the final-step state for back-compat. `get_intermediate_layers` raises in looped mode. |
| [`models/dinov2_model.py`](models/dinov2_model.py) | `CombinedModelDINO` accepts an optional `halt_head` submodule (so DDP covers it) and a `return_per_step` kwarg that routes the multi-crop forward through `forward_features_list_per_step` and applies the projection heads to all `T_max` step outputs. |
| [`models/__init__.py`](models/__init__.py) | Export `HaltHead`, `SharedStack`, and the PonderNet utilities. |
| [`losses/__init__.py`](losses/__init__.py) | Export the new looped-loss helpers. |
| [`models/vision_transformer/__init__.py`](models/vision_transformer/__init__.py) | Export `SharedStack`. |
| [`configs/config.py`](configs/config.py) | New CLI flags: `--use_looped_backbone`, `--shared_stack_L`, `--recursion_T_max`, `--ponder_kl_beta`, `--ponder_lambda_p_start`, `--ponder_lambda_p_end`, `--ponder_lambda_p_anneal_frac`, `--ponder_inference_threshold`. All defaults preserve baseline behavior. |
| [`training/trainer.py`](training/trainer.py) | (a) Validates that `--use_looped_backbone=True` is not combined with semantic iBOT, prototype clustering, typicality dampening, or any of the augmentation-as-view extensions (this revision intentionally supports only the standard DINO + iBOT + KoLeo / KDE objective in the looped path). (b) Threads `looped_T_max` and `looped_L` into the student/teacher `ModernViT`. (c) Constructs the `HaltHead` and attaches it to the student's `CombinedModelDINO` so the DDP wrapper covers it. (d) Adds the halt head's parameters to the student optimizer with no weight decay. (e) Branches the forward+loss section: when looped, calls `compute_looped_step`; otherwise the existing standard path runs unchanged. (f) Logs `ponder_kl_loss`, `ponder_mean_halt_step`, `ponder_lambda_p`, and `h_step_{t}` per step. |

---

## 4. Default hyperparameters

| Flag | Default | Source / rationale |
| --- | --- | --- |
| `--use_looped_backbone` | `False` | Off → baseline behavior. |
| `--shared_stack_L` | 3 | Ouro / LoopLM at LM scale; also Huginn. Pathology pilots should ablate. |
| `--recursion_T_max` | 4 | Ouro / LoopLM. |
| `--ponder_kl_beta` | 0.01 | PonderNet's published default. |
| `--ponder_lambda_p_start` | 0.9 | Shallow prior at start so the stack doesn't get penalized for not running deep before its blocks have learned anything useful. |
| `--ponder_lambda_p_end` | 0.3 | Deeper prior later → expected halted depth ~ 2.5–3 at convergence. |
| `--ponder_lambda_p_anneal_frac` | 0.3 | First 30% of training. Mirrors Huginn's variable-depth curriculum. |
| `--ponder_inference_threshold` | 0.99 | `1 - epsilon` for inference-time early exit. Not consumed during training. |

**Optimizer and LR.** No looped-specific LR rescaling. Layer-wise LR decay
treats `shared_stack_L` (not `vitdepth`) as `num_layers` because the backbone
exposes `blocks.0 ... blocks.{L-1}` in `named_parameters`. The empirical
literature on shared-stack LMs (Ouro, Huginn, Schwethelm et al. iso-depth
scaling) reports that LR transfers across recurrence count at matched
effective depth. If loss spikes appear during pilots, the canonical
intervention order is (i) increase global batch size, (ii) lengthen warmup,
(iii) reduce `T_max` by one, (iv) only as a last resort scale LR down by a
small constant factor (0.5–0.7), not by `1/sqrt(T_max)`.

---

## 5. Compatibility matrix

| Component | Status when `--use_looped_backbone=True` |
| --- | --- |
| xformers `BlockDiagonalMask` packed multi-crop | Unchanged. The mask is built once per packed forward and reused across all `T_max` steps. |
| Multi-crop (2 globals + N locals) | Unchanged. All crops traverse all `T_max` steps. |
| Standard (block) iBOT masking | Unchanged. Masked tokens reach the final layer at every step; per-step iBOT losses are weighted by `p_t`. |
| Register tokens | Unchanged. Traverse every step alongside CLS and patches. |
| FSDP2 sharding | Net benefit (fewer unique parameters → smaller shards / smaller optimizer state / less gradient traffic). |
| Gradient checkpointing | Per recursion step. Activation memory scales with `T_max`, not `T_max * L`. |
| EMA teacher | Unchanged momentum schedule. Teacher runs at `T_max` only with no halt head. |
| Stochastic depth | Per-block rate inherited from `drop_path_rate=0.4`; reused at every recursion step. |
| Pathology FM recipe (`--use_pathology_recipe`) | Compatible. |
| Semantic iBOT | **Disabled** — `--use_semantic_ibot` must be False. The trainer raises if combined. |
| Patch prototype clustering | **Disabled** — `--use_prototype_clustering` must be False. |
| Typicality dampening | **Disabled** — `--use_typicality_dampening` must be False. |
| Adversarial / CellViT / random-rectangle augmentation | **Disabled** in this revision. |

The disabled features can be re-enabled in future revisions; each requires
explicit interleaving with per-recursion-step state (semantic iBOT calls the
backbone separately per channel; prototype clustering operates on patch
tokens; typicality reads a bottleneck signature). The current scope keeps the
new code path small and auditable.

---

## 6. Inference

This branch implements the **training** half of the design. Inference-time
early exit is straightforward to add later: at deployment, run `t = 1, 2, ...`
on a single tile, compute `h_t` from the (pool-of-one) CLS, accumulate
`H_t = sum_{s<=t} h_s`, halt when `H_t >= 1 - epsilon` (default
`epsilon = 0.01`). For batched inference, halted samples can be removed from
the active batch via continuous depth-wise batching (Bae et al. 2024).

For single-tile pathology inference (the common case), there is no batching
friction — the halt head + cumulative threshold gives a clean anytime
deployment with expected mean depth ~ 2.5 at convergence.

---

## 7. Open questions (pilot before ViT-L commitment)

These are reproduced from the design doc's §8. A ViT-S, 10–20M-patch,
1–2-epoch pilot is sufficient to answer them.

1. **SSL inductive bias.** Does weight-tying under DINOv2's SSL objective
   reproduce the Saunshi-style inductive bias — worse pretraining loss at
   matched parameters, but better linear / kNN probes and patch-quality
   diagnostics?
2. **Train–inference calibration.** Does image-level pooled halting calibrate
   well enough for single-crop inference? The train-inference distribution
   gap should be small at convergence but could be noticeable early.
3. **`(L, T_max)` allocation.** `L = 3`, `T_max = 4` (Ouro) at LM scale;
   pathology tiles may prefer a different split.
4. **PonderNet stability in SSL.** Does the KL regularizer stabilize halting
   in the SSL regime as reliably as in supervised settings? Fallback:
   Ouro's entropy-regularized depth objective.
5. **Consistency regularizer.** Does per-image pooled halting need an
   auxiliary consistency term (KL between per-crop halting distributions),
   or does DINO's view-invariance alone suffice?
6. **Gradient stability under weight tying.** The shared-stack literature
   reports flat LR landscapes. To be confirmed empirically under DINOv2's
   SSL objective and the pathology data distribution.

---

## 8. References

- Bae et al. 2024, *Relaxed Recursive Transformers: Effective Parameter
  Sharing with Layer-wise LoRA*. [arXiv:2410.20672](https://arxiv.org/abs/2410.20672).
- Bae et al. 2025, *Mixture-of-Recursions* (MoR). NeurIPS 2025.
  [arXiv:2507.10524](https://arxiv.org/abs/2507.10524).
- Banino, Balaguer, Blundell 2021, *PonderNet: Learning to Ponder*.
  [arXiv:2107.05407](https://arxiv.org/abs/2107.05407).
- Caron et al. 2021, *Emerging Properties in Self-Supervised Vision
  Transformers* (DINO). [arXiv:2104.14294](https://arxiv.org/abs/2104.14294).
- Darcet et al. 2024, *Vision Transformers Need Registers*. ICLR 2024.
  [arXiv:2309.16588](https://arxiv.org/abs/2309.16588).
- Dehghani et al. 2019, *Universal Transformers*. ICLR 2019.
  [arXiv:1807.03819](https://arxiv.org/abs/1807.03819).
- Geiping et al. 2025, *Scaling up Test-Time Compute with Latent Reasoning*
  (Huginn). NeurIPS 2025 Spotlight.
  [arXiv:2502.05171](https://arxiv.org/abs/2502.05171).
- Giannou et al. 2023, *Looped Transformers as Programmable Computers*.
  ICML 2023. [arXiv:2301.13196](https://arxiv.org/abs/2301.13196).
- Goyal et al. 2026, *Elastic Looped Transformers for Visual Generation*
  (ELT, intra-loop self-distillation).
  [arXiv:2604.09168](https://arxiv.org/abs/2604.09168).
- Graves 2016, *Adaptive Computation Time for Recurrent Neural Networks*
  (ACT). [arXiv:1603.08983](https://arxiv.org/abs/1603.08983).
- Lefaudeux et al. 2022, *xFormers: A modular and hackable Transformer
  modelling library*. [github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers).
- Li 2025, *MoR-ViT: Efficient Vision Transformer with Mixture-of-Recursions*.
  [arXiv:2507.21761](https://arxiv.org/abs/2507.21761).
- Oquab et al. 2024, *DINOv2: Learning Robust Visual Features without
  Supervision*. TMLR. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193).
- Raposo et al. 2024, *Mixture-of-Depths*.
  [arXiv:2404.02258](https://arxiv.org/abs/2404.02258).
- Sablayrolles et al. 2019, *Spreading vectors for similarity search* (KoLeo
  regularizer). ICLR 2019. [arXiv:1806.03198](https://arxiv.org/abs/1806.03198).
- Saunshi et al. 2025, *Reasoning with Latent Thoughts: On the Power of
  Looped Transformers*. ICLR 2025.
  [arXiv:2502.17416](https://arxiv.org/abs/2502.17416).
- Schwethelm, Rückert, Kaissis 2026, *How Much Is One Recurrence Worth?
  Iso-Depth Scaling Laws for Looped Language Models*.
  [arXiv:2604.21106](https://arxiv.org/abs/2604.21106).
- Xu and Sato 2024, *On the Expressive Power of Looped Transformers*
  (time-modulated looped transformers).
  [arXiv:2410.01405](https://arxiv.org/abs/2410.01405).
- Yang et al. 2024, *Looped Transformers are Better at Learning Learning
  Algorithms* (input injection). ICLR 2024.
  [arXiv:2311.12424](https://arxiv.org/abs/2311.12424).
- Yin et al. 2022, *A-ViT: Adaptive Tokens for Efficient Vision Transformer*
  (vision halting + distributional prior). CVPR 2022.
  [openaccess.thecvf.com](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf).
- Zhou et al. 2022, *iBOT: Image BERT Pre-Training with Online Tokenizer*.
  ICLR 2022. [arXiv:2111.07832](https://arxiv.org/abs/2111.07832).
- Zhu et al. 2025, *Ouro / LoopLM: Scaling Language Models with Recursive
  Computation*. [arXiv:2510.25741](https://arxiv.org/abs/2510.25741).

**Reference implementation for MoR (used as the starting point for routing
mechanics, though this design ultimately uses PonderNet halting instead):**
[github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions).
