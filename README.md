# Pathology Foundation Model Training — Methodological Extensions to DINOv2

This repository trains Vision Transformer foundation models on whole-slide pathology tiles via self-supervised learning. It starts from the DINOv2 training recipe (Meta AI, 2023) — a joint DINO CLS-token objective + iBOT masked-patch objective + KoLeo regularizer on a student/EMA-teacher pair — and extends it in four directions that target pathology-specific failure modes of the vanilla recipe: semantic masking in iBOT, an auxiliary patch-prototype clustering loss, adaptive per-sample gradient dampening on morphologically redundant tiles ("typicality dampening" / "batch concurrence"), and three optional augmentation modes that produce additional student views from frozen segmentation models.

This branch (`loop_pathology-fm-recipe`) additionally introduces a **looped (weight-tied recurrent-depth) backbone with image-level adaptive halting** on top of the pathology-FM recipe. Section 7 below describes it; full design and implementation notes are in [`LOOPED_DINOV2.md`](LOOPED_DINOV2.md).

Every extension is opt-in through a command-line flag defaulting to `False`. With every flag off, the code reduces to a clean DINOv2 implementation with the minor alignments noted at the bottom of this document.

---

## 1. Semantic iBOT — `--use_semantic_ibot`

Vanilla iBOT masks tokens by a Bernoulli draw or by rectangular block, so the positions the student has to reconstruct are spatially arbitrary. On H&E pathology tiles, most of the image is either stroma, fat, or empty background — unsupervised masks drawn uniformly land on semantically uninformative regions the majority of the time, and the iBOT signal degrades to "reconstruct the texture of lymphocyte-free whitespace."

This repository replaces the random mask with a **semantic mask** produced by a frozen ADIOS- or ViT-UNet-based mask model (`models.vision_transformer.auxiliary_models.ADIOSMaskModel`, `MaskModel`). The frozen model was pretrained separately (ADIOS / adversarial inpainting) and outputs a `[B, num_masks, H, W]` soft segmentation: per-sample probability maps over `num_masks` semantic classes (e.g. nuclei, stroma, glands, etc. — depending on how the mask model was trained).

At each training step:

1. The first teacher global crop is forwarded through the frozen mask model (no grad, bfloat16), yielding soft masks.
2. `training.helpers.convert_semantic_masks_to_token_masks` avg-pools the soft masks to patch resolution and thresholds to produce `[B, num_masks, N]` binary token masks plus per-mask weights (1 / num-masked so every mask channel contributes equally regardless of how many tokens it selects).
3. A subset of `num_masks` channels is sampled per iteration (`--semantic_masks_per_iteration`, typically 1 — randomly chosen every step) to limit compute.
4. The student backbone is called again on the same global crop with each selected semantic mask, and the iBOT loss is computed on the masked patch tokens against the teacher's unmasked targets. The result is added to the total loss with weight `--semantic_ibot_weight`.
5. When `--use_semantic_prototypes` is also true, one of the already-computed semantic backbone outputs is routed into the patch prototype clustering loss (§2) so that clustering also receives semantically-gated supervision.

The iBOT loss in this codebase uses a **gather-then-project** path (`_gather_and_compute_weights` + `iBOTPatchLoss.forward_gathered`) that collects only the masked tokens before projection through the `patchhead`, avoiding the full `[B, N, 65536]` activation that naive implementations materialize.

## 2. Patch Prototype Clustering — `--use_prototype_clustering`

On top of DINO/iBOT, the student patch tokens are clustered against a learnable soft-prototype bank (`models.prototype_bank.LinearPrototypeBank`, `--num_prototypes` prototypes). `losses.prototype_loss.PatchPrototypeLoss` emits three terms:

- **Clustering loss** — cross-entropy of student patch-prototype assignments against the teacher's Sinkhorn-Knopp-normalized assignments (temperatures: `--clustering_teacher_temp`, `--clustering_student_temp`).
- **Teacher-prototype arrangement loss** — a weight-norm regularizer on the prototype vectors themselves.
- **KoLeo on prototypes** — repulsive regularization over the prototype bank, driving it to spread out uniformly on the sphere.

The prototype bank is its own `DistributedDataParallel` module with its own `AdamW` optimizer (`optimizer_prototypes`), stepped independently from the student optimizer — which lets the prototypes train at a different LR and weight decay from the backbone. Default in `run_with_submitit.py`: `use_prototype_clustering = False`, but `num_prototypes = 16384` when turned on.

## 3. Typicality Dampening / Batch Concurrence — `--use_typicality_dampening`

![Typicality dampening — schematic of the data, loss, and R-training pathways alongside the standalone DINOv2 + iBOT pipeline (with semantic iBOT)](figures/typicality_dampening.png)

*Schematic above:* end-to-end view of how the typicality module sits next to a standalone DINOv2 + iBOT student/teacher pair (vector originals at [`figures/typicality_dampening.svg`](figures/typicality_dampening.svg) / [`.pdf`](figures/typicality_dampening.pdf); source: [`figures/generate_typicality_dampening.py`](figures/generate_typicality_dampening.py)). The top two lanes are the standard DINOv2 EMA-teacher / trainable-student arms, fully drawn out (patch embed → ViT backbone → CLS / patches → DINO and iBOT heads → projector `W^{S/T}` → projected `z'`). The pink "semantic iBOT" lane on the left shows the frozen mask model fan-out: the first global crop drives the model, all `num_masks=3` soft channels are used per step (no sampling), and these 3 semantic masks `M_sem` together with the always-on random block mask `M_block` are inserted as `[MASK]` tokens at the student PE→backbone wire — producing 3 sem-masked patch-token columns feeding `CE_iBOT^{sem}` alongside 1 block-masked column feeding the standard `CE_iBOT^{block}`. The green "typicality dampening" lane at the bottom taps the student's projected CLS state `z'^S(x)` (stop-grad, dashed red), feeds the representative-prototype matrix `R` (256×256, unit-sphere ≈ orthonormal), produces the typicality signature `s(x)`, queries the bank by `‖·‖₁`, computes `d(x)`, normalizes via `1−Φ((d−μ)/σ)` to get `t(x)`, and runs a return bus along the figure floor carrying `w(x) = 1 − β · t(x)` up to the `⊗` node that multiplicatively gates `CE_DINO` (and only `CE_DINO` — `iBOT^{block}` and `iBOT^{sem}` are not modulated). The R-training sub-island on the same green lane shows `L_nn` (vs. `W^S` rows; dashed-red stop-grad arrow drops in from the student's `W^S`) and `L_cov` (Gram-glyph showing penalized off-diagonals of `RR^⊤`) merging into `L_repr`, which loops back up to update `R`. Total loss in the right-hand mix box is `L_total = w(x) · CE_DINO + CE_iBOT^{block} + λ_sem · CE_iBOT^{sem}`. Purple dashed EMA arrows tie every paired student↔teacher submodule (patch embed, ViT backbone, heads, `W^S → W^T`, `W^S_{p,i} → W^T_{p,i}`).

*Motivation.* Pathology datasets are long-tailed. At ViT-L scale a training batch can be dominated by morphologically similar tiles (several thousand slices of stroma/fat/epithelium look interchangeable at the embedding level). Under the standard DINO temperature, these common tiles contribute as much per-sample gradient as rare ones, so the rare signal is drowned; but they are also highly redundant, so the gradient they contribute is largely wasted.

*Mechanism* (`typicality/`, three small modules):

- **`RepresentativePrototypes`** (`typicality.representative_prototypes`) holds `K'` unit-norm anchors on the sphere of the student's 256-dim DINO-head bottleneck (before the final `out_dim`-wide prototype layer). Each step, a representative-prototype loss `L = L_nn + L_cov` pushes every anchor toward its nearest DINO prototype (`L_nn`) and spreads the anchors out (`L_cov`); the anchors are re-normalized to the sphere after each step. They are optimized by a dedicated `R_optimizer` (AdamW, `--typicality_repr_lr`, weight decay 0).
- **`TypicalityBank`** keeps a running reservoir of `--typicality_bank_size` softmax signatures — one signature per sample, each of length `K'` — computed from the student's bottleneck representation of the first global crop against the anchors. The bank refreshes a fraction (`--typicality_replace_fraction`) of its entries each step.
- **`TypicalityScorer`** compares each new signature's L1-distance to the bank's current mean/std distance distribution, squashes that into a per-sample typicality `t ∈ [0, 1]` (1 = very typical, 0 = very rare), and exposes two modulations of the DINO CLS loss:
  - `adaptive_temp` — per-sample student temperature τ_i = τ_base × (1 + α × t_i). Typical samples get a higher τ, producing flatter student distributions and therefore smaller gradients.
  - `weighted_loss` — per-sample loss weights w_i = 1 − β × t_i. Typical samples are weighted down directly.

`DINOLoss.forward` accepts both `sample_temperatures` and `sample_weights` kwargs; whichever is non-`None` is applied. A `--typicality_warmup_iters`-step warmup lets the bank fill before any modulation kicks in.

The feature adds three pieces of state (`RepresentativePrototypes`, `R_optimizer`, `TypicalityBank`) to the checkpoint, gated by the flag so disabled runs still produce lean checkpoints.

## 4. Augmentation Extensions

Three optional augmentation modes each append extra student views to the DINO CLS forward pass (and, where noted, to the KoLeo regularizer). They are independent — any combination of the three can be enabled. Each creates its crops from the *first* teacher global crop via a frozen upstream model or random masks, and `training.helpers.calculate_total_student_views` accounts for the expanded view count when constructing `DINOLoss`.

### 4a. Adversarial-mask-as-student-view — `--use_adversarial_mask_augmentation`

This re-uses the same frozen mask model as semantic iBOT (§1) — `--mask_checkpoint`, `--mask_model_arch`, `--num_masks`. The difference is the **scope of consumption**: semantic iBOT uses the model's output as *token-level* masks inside the iBOT loss; this feature uses the same model's output as *image-level* masks to create additional student views. Soft masks from the frozen model are element-wise multiplied onto the global crop (`training.helpers.apply_masks_to_images`), producing `num_masks` masked global views that the student sees as additional crops; `--crops_per_mask` local crops are optionally extracted from each. These masked globals also enter the KoLeo regularizer (matching the morphological_rarity reference behavior).

Both features can be enabled simultaneously — the mask model is loaded once and shared.

### 4b. CellViT augmentation — `--use_cellvit_augmentation`

A frozen CellViT model (`--cellvit_checkpoint`) produces a 2-channel nuclei/background segmentation of the first global crop. The student then sees two masked global views (nuclei-only, background-only) plus `--cellvit_crops_per_channel` random local crops extracted from each. The CellViT architecture (`models.vision_transformer.auxiliary_models.CellViT`) is a ViT encoder with a convolutional U-Net-style decoder using `Conv2DBlockBN` / `Deconv2DBlockBN` blocks, loaded in bfloat16 and frozen. The helpers live in `training.helpers`: `load_pretrained_cellvit_model`, `apply_cellvit_masks`, `extract_crops_from_cellvit_channel`.

### 4c. Random rectangular mask augmentation — `--use_random_mask_augmentation`

For ablation and reproducibility of prior experiments. `training.helpers.generate_random_image_masks` emits `--random_num_masks` independent rectangular masks of random position and size per image; these are multiplied onto the first global crop to produce additional masked global student views, with `--random_crops_per_mask` locals extracted from each. No learned model involved.

---

## 5. Pathology FM Recipe — `--use_pathology_recipe`

*Motivation.* Independent of the four feature toggles above, this branch bundles a set of training-recipe changes that the open pathology-FM literature has converged on between 2024 and 2026. They are individually small but jointly change training behavior enough that mixing them with the `consolidated` defaults would produce a config that is neither the old recipe nor the new one. A single `--use_pathology_recipe` flag pulls the whole bundle in; leaving it `False` preserves the `consolidated` behavior bit-for-bit. Primary references: Virchow (arXiv:2309.07778), Virchow2 / Virchow2G (arXiv:2408.00738), Midnight (MICCAI 2025), RudolfV (arXiv:2401.04079), Hibou (arXiv:2406.05074).

*Cross-preset changes applied when the flag is on.*

- `patch_size` 16 → 14 (community standard across the Virchow family, Midnight, RudolfV, H-optimus; only Phikon-v2 and PathOrchestra stay at 16).
- bf16 autocast end-to-end with `fp16_scaler=None` (Virchow2G retrospectively flagged fp16 as the cause of late-training NaN; H100 supports bf16 natively).
- Solarization off on global crop 2 (Virchow2 §5.2 ablation; also in Virchow2G, RudolfV, Hibou).
- Vertical flip on and 90-degree discrete rotations on in the color-jitter chain (Virchow2, RudolfV, Hibou, Lunit all adopt this — pathology tiles have no canonical orientation).
- Teacher temperature fixed at 0.04 (Virchow2G §5.1).
- `KoLeoLoss` → `KDELoss` with a von-Mises–Fisher kernel (κ from `--kde_kappa`, default 5.0), all-gather pooled across GPUs. Replaces KoLeo because pathology batches contain near-duplicate tiles; KoLeo's nearest-neighbor distance collapses and its gradient explodes on those. Virchow2 §5.2.
- `--koleo_loss_weight` nudged from 0.1 → 0.05 only when the user hasn't overridden it (Virchow2 KDE default λ).

*Probabilistic ECT augmentation.* ECT = Extended-Context Translation from Virchow2 §5.1. The **probabilistic per-tile-size framing** in this repo is the user's adaptation for mixed-magnification (40× + 20×) training data. At transform-call time, `TMEDinoTransforms.__call__` inspects `img.size[0]`:

- **40× tiles** (source ≥ 448 pixels, native 40× at MPP 0.25) route to the ECT branch with probability `--ect_probability` (default 0.4) and to a standard crop-and-resize branch otherwise. The ECT branch crops at native 40× ± 10% (global `scale=(0.203, 0.303)`, `ratio=(0.95, 1.05)`; local `scale=(0.037, 0.056)`, `ratio=(0.95, 1.05)`), so the model sees cells at correct physical scale with no aggressive resize.
- **20× tiles** (source < 448) always route to the standard branch.
- The standard branch uses canonical DINOv2 ranges per Virchow2 §5.1 (global `scale=(0.32, 1.0)`, `ratio=(0.75, 1.33)`; local `scale=(0.05, 0.32)`, `ratio=(0.75, 1.33)`) and spans apparent magnification 20×–35× globally and 15×–38× locally — so the downstream 20× evaluation magnification is always covered regardless of source tile. A small gap at 35–36× where neither branch covers densely is an accepted trade-off: widening standard would defeat ECT's morphology guarantee. The full magnification table is documented as a module-level comment block in `run_with_submitit.py`.

Under the recipe, the pre-resize to (global_size, global_size) is skipped — `RandomResizedCrop` handles the final resize to output size, so ECT actually operates on the native resolution it was designed for.

*ViT-G auto-gate.* When the recipe is on **and** `args.embeddingdim >= 1280`, four additional scaling-regime fixes kick in automatically and are logged at startup under `[pathology recipe auto-gate]`:

- `qk_norm=True` inside attention (the feature was already wired through `ModernViT`/`TransformerBlock`; the auto-gate just flips the default). Explicit `--qk_norm=True/False` on the CLI overrides the auto-gate. (Virchow2G §6.)
- `num_register_tokens` bumped to `max(current, 8)` (Virchow2G + UNI2-h).
- `out_dim` 65,536 → 131,072 (Virchow v1 Methods; Paige standard). This is a scaled-regime change — Virchow v1 chose 131,072 at ViT-H scale and Virchow2 / Virchow2G carried it at ViT-H/G — not a universal pathology-recipe component, so ViT-B and ViT-L runs with `--use_pathology_recipe=True` keep their CLI/default `out_dim` (typically 65,536).
- Optimizer swapped from `torch.optim.AdamW` to `utils.StableAdamW(betas=(0.9, 0.95))`. StableAdamW uses decoupled weight decay and per-step RMS-clipped updates; Virchow2G reports it prevented late-training NaN at ViT-G scale. The implementation lives alongside LARS in `utils.py`. (Virchow2G §6.)

*Control surface.* Four new CLI args, all defaulting off / neutral so the branch is a no-op unless opted in: `--use_pathology_recipe`, `--ect_probability`, `--kde_kappa`, `--qk_norm`, plus `--num_register_tokens` (was previously hardcoded to 4 at the `ModernViT` call site). `run_with_submitit.py` ships with a commented-out toggle stanza and the full magnification table inline; uncomment three lines to enable. Runtime verification checklist (no-op parity, ECT routing frequency, KDE stability, auto-gate log, bf16 end-to-end, ViT-L smoke test) is tracked in [`pathology_fm_recipe_verification.md`](pathology_fm_recipe_verification.md).

---

## 6. Alignments with Official DINOv2

A handful of smaller corrections relative to common in-the-wild DINOv2 forks, already present on this branch (not features per se — implementation hygiene):

- **Per-sample mask-ratio normalization in iBOT.** Block masks have variable ratio; the iBOT loss divides per sample by the number of masked tokens in that sample (`masks_weight = 1 / num_masked`) so samples with fewer masked tokens don't receive a smaller gradient contribution.
- **Layer-wise LR decay** over the backbone (`--lr_decay_rate`) via `utils.get_params_groups_with_layer_decay`; heads train at the base LR.
- **Cosine weight-decay schedule** from `--weight_decay` to `--weight_decay_end` (official ramps up over training rather than the constant value some forks use).
- **Sinkhorn-Knopp normalization** in both `DINOLoss` and `PatchPrototypeLoss` with `dist.all_reduce` over the whole world, matching the official global-denominator semantics.
- **Sequence-packed multi-crop forward pass** inside the backbone (`CombinedModelDINO.forward` → `backbone.forward_features_list`): all crops are processed as a single packed sequence instead of per-crop loops.
- **Typicality's per-sample DINO-temperature path** exposes `sample_temperatures` and `sample_weights` as optional kwargs on `DINOLoss.forward`; when both are `None` the loss reduces exactly to the scalar-temperature form.

---

## 7. Looped DINOv2 with Image-Level Adaptive Halting — `--use_looped_backbone`

![Looped DINOv2 — training schematic with explicit residual pathway for the input injection](figures/looped_dinov2_method_residual.png)

*Schematic above:* one full forward pass of the looped student backbone for `T_max = 4` recursion steps. The recursion lane at the top draws the recurrence `z_0 → f_θ → ⊕ → z_1 → f_θ → ⊕ → z_2 → ...`; the same `f_θ` block is repeated four times under the brace `shared parameters f_θ (applied T_max times)` to make the weight-tying explicit, with per-step time embeddings `τ_t` fanned in from above. The purple bypass highway taps `z_0` and feeds it back into a `⊕` summing node between every `f_θ` and its `z_t`, drawing the residual structure `z_t = f_θ(...) + z_0` the same way ResNet / Transformer figures draw skip connections. Each `z_t` is tapped to (a) the green halt-head row producing `h_t = σ(W_halt · z̄_t)` (image-pooled CLS, hence the per-image PonderNet step-marginals `{p_t}` exiting right) and (b) the orange per-step loss row `L_t` (student vs. teacher targets at step `t`). The two buses `{p_t}` and `{L_t}` flow into the total-loss box at the right: `L = Σ_t p_t · L_t + β · KL(p ∥ Geom(λ_p))`. Vector originals at [`figures/looped_dinov2_method_residual.svg`](figures/looped_dinov2_method_residual.svg) / [`.pdf`](figures/looped_dinov2_method_residual.pdf); source: [`figures/generate_looped_dinov2_method_residual.py`](figures/generate_looped_dinov2_method_residual.py); a longer walkthrough lives in [`LOOPED_DINOV2_SWEEP_PLAN.md`](LOOPED_DINOV2_SWEEP_PLAN.md).

*Motivation.* Two observations about the DINOv2 baseline are worth revisiting at pathology-FM scale: (a) much of the representational work in a `vitdepth=24` ViT-L is *iterative refinement* of the same operation, so a much smaller stack applied repeatedly should match downstream quality at a fraction of the parameters [Dehghani et al. 2019; Bae et al. 2024; Geiping et al. 2025; Zhu et al. 2025]; and (b) tiles vary enormously in content complexity (pure stroma vs. tumor–stroma interface), so allocating uniform compute per image leaves obvious efficiency on the table [Graves 2016; Banino et al. 2021; Raposo et al. 2024]. This feature combines both: a **shared (weight-tied) stack** of `--shared_stack_L` blocks applied `--recursion_T_max` times for the architectural substrate, plus a **PonderNet-style image-level halting head** for adaptive depth.

*Architectural substrate (shared stack).* The standard DINOv2 stack of `vitdepth` unique transformer blocks is replaced (when the flag is on) by an `nn.ModuleList` of `--shared_stack_L` blocks driven by a [`SharedStack`](models/vision_transformer/shared_stack.py) module. The recurrence at step `t` is

```
z_t = post_norm( block_L o ... o block_1 ( pre_norm( z_{t-1} + tau_t ) ) ) + z_0
```

with sandwich LayerNorm bracketing each pass, a learnable per-step time embedding `tau_t` (zero-initialized) added before entry, and the patch-embedded input `z_0` re-injected at the end of every step. These three ingredients are the standard stability package from the looped / recurrent-depth transformer line [Geiping et al. 2025 (Huginn); Zhu et al. 2025 (Ouro / LoopLM); Yang et al. 2024 (input injection); Xu & Sato 2024 (time modulation)]. Default `L = 3, T_max = 4` gives 12 effective block applications at ~4× parameter reduction vs. a 12-block dense ViT. xformers `BlockDiagonalMask` packed multi-crop and FSDP2 sharding both work unchanged; gradient checkpointing is per recursion step, so activation memory scales with `T_max`, not `T_max * L`.

*Image-level adaptive halting.* Halting is decided per **image**, not per crop or per token — per-token / per-crop halting was rejected because iBOT supervision requires masked patches to reach the final layer, which conflicts with per-token / per-crop early exit. At each recursion step `t`:

1. The image's per-crop CLS tokens (e.g., 8 in the canonical recipe: 2 globals + 6 locals) are mean-pooled into a single per-image vector. Mean pooling is parameter-free and DINO's CLS is explicitly view-invariant, so the train–inference distribution gap (single-crop pool of one at deployment) stays small.
2. A linear head + sigmoid emits the halting probability `h_t` per image (see [`HaltHead`](models/halting_head.py)).
3. PonderNet step-marginals are computed: `p_t = h_t * prod_{s<t}(1 - h_s)`, with the final step absorbing remaining mass so `sum_t p_t = 1` per image [Banino et al. 2021].
4. A KL term `beta * KL( p || Geom(lambda_p) )` (β default 0.01) regularizes the halting distribution toward a truncated geometric prior, with `lambda_p` linearly annealed from `0.9 -> 0.3` over the first 30% of training (Huginn-style variable-depth curriculum: `lambda_p` near 0.9 keeps the stack from being asked for deep compute too early; near 0.3 lets harder images earn more compute as the representations mature).

*Critical training-time invariant.* All crops traverse all `T_max` recursion steps in lockstep during training. The halting distribution **only weights the per-step loss mixture**; it does **not** truncate the forward. This preserves byte-for-byte compatibility with packed multi-crop, register tokens, iBOT block masking, and the EMA teacher. Adaptive halting is an inference-time deployment feature, not a training-time speedup.

*Loss.* The teacher always runs at `T_max` only and produces fixed targets (Sinkhorn-Knopp on teacher CLS and on teacher-projected iBOT-masked patches is computed once per batch, outside the t-loop, and reused across all student steps). This asymmetric arrangement — teacher at maximum depth, student at adaptive depth — mirrors the Intra-Loop Self-Distillation pattern of ELT [Goyal et al. 2026]. The total per-image loss is

```
L[i] = sum_t p_t[i] * (L_DINO_t[i] + ibot_w * L_iBOT_t[i])
       + koleo_w * L_KoLeo[i]   (final step only)
       + beta * KL( p[i] || Geom(lambda_p) )
```

implemented as **per-sample** (`[B]`-shaped) DINO and iBOT helpers in [`losses/looped_loss.py`](losses/looped_loss.py) so the per-image marginals can weight each image's contribution before reduction. The per-step student forward is orchestrated by [`training/looped_step.py`](training/looped_step.py), invoked from `trainer.py` behind an `if use_looped_backbone:` branch.

*Anytime supervision property.* iBOT supervision is evaluated at every step but always against the teacher's step-`T_max` patch output. An image whose halting mass concentrates at small `t` trains its shared stack to produce iBOT-valid patches early; one whose mass concentrates at large `t` trains it to refine further. Because the shared stack is one set of parameters, these signals cooperate rather than compete — the block learns to be useful at any intermediate depth. This is the inductive bias argued by Saunshi et al. [2025] for looped models.

*Inference.* Implemented separately. The intended pattern: at `t = 1, 2, ...` compute `h_t` from the (pool-of-one) CLS, accumulate `H_t = sum_{s<=t} h_s`, halt when `H_t >= 1 - epsilon` (default `epsilon = 0.01`, exposed as `--ponder_inference_threshold`). At convergence with `lambda_p = 0.3` the expected halted depth is ~ 2.5 for `T_max = 4`, giving roughly 1.6× wall-clock inference speedup at 4× parameter reduction. For batched inference, halted samples can be removed from the active batch via continuous depth-wise batching [Bae et al. 2024]; for single-tile pathology inference the common case has no batching friction.

*Compatibility.* Compatible with `--use_pathology_recipe` (KDE regularizer, ECT augmentation, teacher-temp 0.04, patch_size 14, bf16 end-to-end, ViT-G auto-gate). **Incompatible** in this revision with `--use_semantic_ibot`, `--use_semantic_prototypes`, `--use_prototype_clustering`, `--use_typicality_dampening`, `--use_adversarial_mask_augmentation`, `--use_cellvit_augmentation`, `--use_random_mask_augmentation` — the trainer raises a clear error if any of those is combined with the looped backbone (each requires interleaving per-recursion-step state with feature-specific forward passes, out of scope for this revision).

*Control surface.* Eight new CLI args, all defaulting off / neutral: `--use_looped_backbone`, `--shared_stack_L` (3), `--recursion_T_max` (4), `--ponder_kl_beta` (0.01), `--ponder_lambda_p_start` (0.9), `--ponder_lambda_p_end` (0.3), `--ponder_lambda_p_anneal_frac` (0.3), `--ponder_inference_threshold` (0.99). Engineering details (file boundaries, hyperparameter rationale, optimizer treatment, open empirical questions for the ViT-S pilot) are in [`LOOPED_DINOV2.md`](LOOPED_DINOV2.md). The training schematic (with explicit residual pathway for the input injection) is at [`figures/looped_dinov2_method_residual.svg`](figures/looped_dinov2_method_residual.svg) and is walked through in [`LOOPED_DINOV2_SWEEP_PLAN.md`](LOOPED_DINOV2_SWEEP_PLAN.md), which also carries the `(L, T_max)` sweep launch plan and per-mutation post-processing investigations.

*Citations (looped / adaptive-compute / vision halting).* Bae et al. 2024 [arXiv:2410.20672](https://arxiv.org/abs/2410.20672); Bae et al. 2025 (MoR) [arXiv:2507.10524](https://arxiv.org/abs/2507.10524); Banino, Balaguer, Blundell 2021 (PonderNet) [arXiv:2107.05407](https://arxiv.org/abs/2107.05407); Dehghani et al. 2019 (Universal Transformers) [arXiv:1807.03819](https://arxiv.org/abs/1807.03819); Geiping et al. 2025 (Huginn) [arXiv:2502.05171](https://arxiv.org/abs/2502.05171); Giannou et al. 2023 (Looped Transformers as Programmable Computers) [arXiv:2301.13196](https://arxiv.org/abs/2301.13196); Goyal et al. 2026 (ELT) [arXiv:2604.09168](https://arxiv.org/abs/2604.09168); Graves 2016 (ACT) [arXiv:1603.08983](https://arxiv.org/abs/1603.08983); Li 2025 (MoR-ViT) [arXiv:2507.21761](https://arxiv.org/abs/2507.21761); Raposo et al. 2024 (Mixture-of-Depths) [arXiv:2404.02258](https://arxiv.org/abs/2404.02258); Saunshi et al. 2025 [arXiv:2502.17416](https://arxiv.org/abs/2502.17416); Schwethelm, Rückert, Kaissis 2026 [arXiv:2604.21106](https://arxiv.org/abs/2604.21106); Xu & Sato 2024 [arXiv:2410.01405](https://arxiv.org/abs/2410.01405); Yang et al. 2024 [arXiv:2311.12424](https://arxiv.org/abs/2311.12424); Yin et al. 2022 (A-ViT) [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.pdf); Zhu et al. 2025 (Ouro / LoopLM) [arXiv:2510.25741](https://arxiv.org/abs/2510.25741). Reference implementation for MoR (used as a starting point for routing mechanics, though this design ultimately uses PonderNet halting): [github.com/raymin0223/mixture_of_recursions](https://github.com/raymin0223/mixture_of_recursions).

---

## 8. Repo layout

- `configs/` — `argparse`-based config builder (`configs.config.get_args_parser`); includes the looped-backbone CLI flags (§7).
- `data/` — pathology tile dataset + proportional multi-source wrapper (`DINOv2PathologyDataset`, `ProportionalMultiDatasetWrapper`, `TMEDinoTransforms`; `TMEDinoTransforms` also owns the `Random90Rotation` + probabilistic ECT routing used by §5).
- `losses/` — `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`, `KDELoss` (used by §5), `PatchPrototypeLoss`, plus `looped_loss.py` (per-sample DINO/iBOT helpers + standalone Sinkhorn-Knopp used by §7).
- `models/` — `CombinedModelDINO` (the packed-sequence student/teacher wrapper, optionally hosting a `halt_head` for §7), `LinearPrototypeBank`, vision-transformer bits (`ModernViT`, `DINOHead`, `MaskModel`, `ADIOSMaskModel`, `CellViT`), `SharedStack` (weight-tied recurrent-depth stack for §7), and `halting_head.py` (`HaltHead` + PonderNet utilities).
- `typicality/` — the three modules implementing the typicality dampening feature (§3).
- `training/` — `train_dinov2` orchestration (`trainer.py`) and `helpers.py` with all the mask/crop utilities; `looped_step.py` factors out the PonderNet mixture-weighted forward+loss for §7; `training/__init__.py` re-exports the common helpers.
- `visualizations/` — standalone plotting scripts for loss curves, clustering entropy, PCA, prototype dendrograms, and prototype heatmaps.
- `run_with_submitit.py` — SLURM launcher that sets defaults and submits via `submitit`; also carries the pathology-FM recipe toggle stanza and magnification table (§5).
- `main_train.py` — single-process entry point (used locally for debugging; cluster training goes through `run_with_submitit.py`).
- `utils.py` — distributed setup, layer-wise LR helpers, schedulers, serialization utilities, and the `LARS` / `StableAdamW` optimizers.
- `LOOPED_DINOV2.md` — engineer-facing companion to §7: file-by-file map of the looped-backbone changes, default hyperparameters, compatibility matrix, open empirical questions for the ViT-S pilot.

## 9. Running

Cluster training recipes (job layouts, monitoring, checkpoint recovery) live in internal runbooks rather than this repo. For local debugging, `main_train.py` takes the same arguments as the SLURM launcher and runs without submitit — useful for verifying config validity and doing a smoke test on a single node.
