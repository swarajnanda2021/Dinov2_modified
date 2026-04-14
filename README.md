# Pathology Foundation Model Training — Methodological Extensions to DINOv2

This repository trains Vision Transformer foundation models on whole-slide pathology tiles via self-supervised learning. It starts from the DINOv2 training recipe (Meta AI, 2023) — a joint DINO CLS-token objective + iBOT masked-patch objective + KoLeo regularizer on a student/EMA-teacher pair — and extends it in four directions that target pathology-specific failure modes of the vanilla recipe: semantic masking in iBOT, an auxiliary patch-prototype clustering loss, adaptive per-sample gradient dampening on morphologically redundant tiles ("typicality dampening" / "batch concurrence"), and three optional augmentation modes that produce additional student views from frozen segmentation models.

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

## 5. Alignments with Official DINOv2

A handful of smaller corrections relative to common in-the-wild DINOv2 forks, already present on this branch (not features per se — implementation hygiene):

- **Per-sample mask-ratio normalization in iBOT.** Block masks have variable ratio; the iBOT loss divides per sample by the number of masked tokens in that sample (`masks_weight = 1 / num_masked`) so samples with fewer masked tokens don't receive a smaller gradient contribution.
- **Layer-wise LR decay** over the backbone (`--lr_decay_rate`) via `utils.get_params_groups_with_layer_decay`; heads train at the base LR.
- **Cosine weight-decay schedule** from `--weight_decay` to `--weight_decay_end` (official ramps up over training rather than the constant value some forks use).
- **Sinkhorn-Knopp normalization** in both `DINOLoss` and `PatchPrototypeLoss` with `dist.all_reduce` over the whole world, matching the official global-denominator semantics.
- **Sequence-packed multi-crop forward pass** inside the backbone (`CombinedModelDINO.forward` → `backbone.forward_features_list`): all crops are processed as a single packed sequence instead of per-crop loops.
- **Typicality's per-sample DINO-temperature path** exposes `sample_temperatures` and `sample_weights` as optional kwargs on `DINOLoss.forward`; when both are `None` the loss reduces exactly to the scalar-temperature form.

## 6. Repo layout

- `configs/` — `argparse`-based config builder (`configs.config.get_args_parser`).
- `data/` — pathology tile dataset + proportional multi-source wrapper (`DINOv2PathologyDataset`, `ProportionalMultiDatasetWrapper`, `TMEDinoTransforms`).
- `losses/` — `DINOLoss`, `iBOTPatchLoss`, `KoLeoLoss`, `PatchPrototypeLoss`.
- `models/` — `CombinedModelDINO` (the packed-sequence student/teacher wrapper), `LinearPrototypeBank`, and vision-transformer bits (`ModernViT`, `DINOHead`, `MaskModel`, `ADIOSMaskModel`, `CellViT`).
- `typicality/` — the three modules implementing the typicality dampening feature (§3).
- `training/` — `train_dinov2` orchestration (`trainer.py`) and `helpers.py` with all the mask/crop utilities; `training/__init__.py` re-exports the common helpers.
- `visualizations/` — standalone plotting scripts for loss curves, clustering entropy, PCA, prototype dendrograms, and prototype heatmaps.
- `run_with_submitit.py` — SLURM launcher that sets defaults and submits via `submitit`.
- `main_train.py` — single-process entry point (used locally for debugging; cluster training goes through `run_with_submitit.py`).
- `utils.py` — distributed setup, layer-wise LR helpers, schedulers, serialization utilities.

## 7. Running

Cluster training recipes (job layouts, monitoring, checkpoint recovery) live in internal runbooks rather than this repo. For local debugging, `main_train.py` takes the same arguments as the SLURM launcher and runs without submitit — useful for verifying config validity and doing a smoke test on a single node.
