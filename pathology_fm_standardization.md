# Implementation Spec: Pathology Foundation Model Recipe Branch

## Purpose

This document specifies a set of modifications to add to the `consolidated` branch of this DINOv2-based pathology foundation model training codebase. The modifications are derived from a literature survey of vision-only self-supervised pathology foundation models released between January 2024 and April 2026, filtered to those with strong evidence and no dependency on pretrained checkpoints or data-pipeline changes.

The modifications are grouped into two toggles that can be enabled independently:

1. `--use_pathology_recipe`: a bundle of recipe modifications (mostly Virchow2-derived, some community-converged)
2. Auto-gate at `args.embeddingdim >= 1280`: ViT-H/G scaling-regime fixes (Virchow2G)

Three existing codebase-native modifications remain as orthogonal toggles and are **not** altered by this spec:

- `--use_patch_prototype_clustering`
- `--use_semantic_ibot`
- `--use_typicality_dampening`

## Branch Setup

Create a new branch from `consolidated`:

```bash
git checkout consolidated
git pull
git checkout -b pathology-fm-recipe
```

All work described in this document happens on the `pathology-fm-recipe` branch. Nothing in this spec modifies the `consolidated` branch directly.

## Primary References

- **Virchow2 paper**: Zimmermann et al. (2024). *Virchow 2: Scaling Self-Supervised Mixed Magnification Models in Pathology.* arXiv:2408.00738.
- **Virchow paper**: Vorontsov et al. (2023). *Virchow: A Million-Slide Digital Pathology Foundation Model.* arXiv:2309.07778.
- **Midnight paper**: Karasikov et al. (2025). *Training state-of-the-art pathology foundation models with orders of magnitude less data.* MICCAI 2025.
- **RudolfV paper**: Dippel et al. (2024). *RudolfV: A Foundation Model by Pathologists for Pathologists.* arXiv:2401.04079.
- **Hibou paper**: Nechaev et al. (2024). *Hibou: A Family of Foundational Vision Transformers for Pathology.* arXiv:2406.05074.

## Source Attribution Per Modification

Each modification below cites its originating source(s). This matters because not all modifications come from a single paper, and the CLI flag comments must be accurate.

| Modification | Primary Source | Also Adopted By | Notes |
|---|---|---|---|
| patch_size 14 | Community standard | Virchow, Virchow2, Virchow2G, Midnight, RudolfV, H-optimus, H0-mini, Atlas | Only Phikon-v2 and PathOrchestra stay at 16 |
| out_dim 131,072 | Virchow v1 §Methods (Vorontsov 2023) | Virchow2, Virchow2G | Paige standard |
| bf16 end-to-end | Virchow2G retrospective note | — | Authors flagged fp16 as cause of NaN; H100 supports bf16 natively |
| Solarization OFF | Virchow2 §5.2 ablation | Virchow2G, RudolfV, Hibou | Distorts H&E chromophores |
| V-flip + 90° rotations | Virchow2, RudolfV, Hibou | Lunit DINO | Pathology has no canonical orientation |
| Teacher temp fixed 0.04 | Virchow2G §5.1 | — | Virchow2 uses decreasing 0.07→0.04 |
| KDE regularizer (κ=5) | Virchow2 §5.2 | Midnight (cites Virchow2) | vMF kernel, replaces KoLeo |
| ECT augmentation | Virchow2 §5.1 | — | Probabilistic framing is user's variation |
| qk_norm at ViT-G | Virchow2G §6 | — | Training stability at scale |
| StableAdamW β₂=0.95 | Virchow2G §6 | — | Prevents late-training NaN at ViT-G |
| 8 register tokens at ≥ViT-H | Virchow2G, UNI2-h | — | Darcet et al. register tokens |

## Modifications, Grouped

### Group A: Cross-preset changes (applied by `--use_pathology_recipe=True`)

- **A1. Patch size 16 → 14** — Community standard (Virchow family, Midnight, RudolfV, H-optimus)
- **A2. DINO prototype count (`out_dim`) 65,536 → 131,072** — Virchow v1 Methods; Paige standard
- **A3. Mixed precision fp16+GradScaler → bf16 end-to-end, no scaler** — H100-appropriate; avoids fp16 NaN issues flagged by Virchow2G team
- **A4. Solarization OFF** — Virchow2 §5.2 ablation; also in Virchow2G, RudolfV, Hibou
- **A5. Vertical flip ON + 90° rotations ON** — Virchow2, RudolfV, Hibou, Lunit
- **A6. Teacher temperature fixed at 0.04** — Virchow2G §5.1
- **A7. KoLeo regularizer → KDE regularizer** — Virchow2 §5.2; vMF kernel κ=5; all-gather across GPUs
- **A8. Probabilistic ECT augmentation** — ECT from Virchow2 §5.1; probabilistic framing is user's variation (see Group C)

### Group B: ViT-G auto-gate (triggers when `args.embeddingdim >= 1280`)

- **B1. `qk_norm`: False → True** — Virchow2G §6
- **B2. `num_register_tokens`: max(current, 8)** — Virchow2G, UNI2-h (monotonic override)
- **B3. Optimizer: AdamW → StableAdamW with β₂=0.95** — Virchow2G §6

### Group C: Probabilistic ECT Specification

ECT = Extended-Context Translation (Virchow2 paper §5.1). The probabilistic per-tile-size framing below is the user's adaptation of Virchow2's deterministic ECT to their specific mixed-magnification dataset (40× at 448² pixels, 20× at 224² pixels).

For each sample, inspect `img.size[0]` at transform-call time:

**If source ≥ 448 (40× magnification tile):**
- Draw `p ~ Uniform(0, 1)`
- If `p < 0.4`: apply ECT branch
  - Global crop: `RandomResizedCrop(size=224, scale=(0.203, 0.303), ratio=(0.95, 1.05))`
  - Local crop:  `RandomResizedCrop(size=96,  scale=(0.037, 0.056), ratio=(0.95, 1.05))`
- Else: apply standard crop-and-resize
  - Global crop: `RandomResizedCrop(size=224, scale=(0.32, 1.0), ratio=(0.75, 1.33))`
  - Local crop:  `RandomResizedCrop(size=96,  scale=(0.05, 0.32), ratio=(0.75, 1.33))`

**If source < 448 (20× magnification tile):**
- Always apply standard crop-and-resize
  - Global crop: `RandomResizedCrop(size=224, scale=(0.32, 1.0), ratio=(0.75, 1.33))`
  - Local crop:  `RandomResizedCrop(size=96,  scale=(0.05, 0.32), ratio=(0.75, 1.33))`

**Magnification table (must be present as a comment block in `run_with_submitit.py`):**

```text
# PROBABILISTIC ECT — MAGNIFICATION TABLE
#
# Formula: apparent_mag = source_mag * output_side / (sqrt(scale) * source_side)
#
# 40x tiles (source >= 448, native 40x at MPP 0.25):
#   ECT branch (p=0.4) — preserves cellular morphology:
#     Global (224 out): scale=(0.203, 0.303), ratio=(0.95, 1.05)
#     Local  (96  out): scale=(0.037, 0.056), ratio=(0.95, 1.05)
#     Apparent mag: globals 36.3-44.4x, locals 36.2-44.5x
#   Standard branch (p=0.6):
#     Global (224 out): scale=(0.32, 1.0),    ratio=(0.75, 1.33)
#     Local  (96  out): scale=(0.05, 0.32),   ratio=(0.75, 1.33)
#     Apparent mag: globals 20.0-35.4x, locals 15.2-38.3x
#
# 20x tiles (source == 224, native 20x at MPP 0.50):
#   Standard branch always:
#     Global (224 out): scale=(0.32, 1.0),    ratio=(0.75, 1.33)
#     Local  (96  out): scale=(0.05, 0.32),   ratio=(0.75, 1.33)
#     Apparent mag: globals 20.0-35.4x, locals 15.2-38.3x
#
# Design notes:
#  - ECT branch (Virchow2 recipe) lives at native 40x +/- 10%. Model sees
#    cells at correct physical scale; no aggressive resize.
#  - Standard branch spans 20x-35x globally, 15x-38x locally on both tile
#    types. This includes the downstream evaluation magnification (20x).
#  - Small gap at 35-36x where neither branch covers densely. Acceptable
#    trade-off: widening standard would defeat ECT's morphology guarantee.
```

## File-by-File Changes

### File 1: `configs/config.py`

Add four new CLI arguments. The `help` text **must** include source attribution as specified below, verbatim.

```python
# ========== Pathology FM Recipe ==========
parser.add_argument('--use_pathology_recipe', default=False, type=utils.bool_flag,
                    help='Enable pathology-FM recipe bundle. Sources: KDE regularizer, '
                         'ECT augmentation, teacher_temp=0.04, out_dim=131072 '
                         '[Virchow/Virchow2, Paige/MSKCC/MSR, arXiv:2309.07778 and '
                         'arXiv:2408.00738]; solarization off, V-flip, 90-deg rotations '
                         '[Virchow2 + RudolfV + Hibou convergence]; patch_size=14 '
                         '[community standard across Virchow family, Midnight, RudolfV, '
                         'H-optimus]; bf16 end-to-end [scaling-regime choice, flagged '
                         'retroactively by Virchow2G]. Auto-enables qk_norm, 8+ register '
                         'tokens, StableAdamW beta2=0.95 when embeddingdim >= 1280 '
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
```

Also ensure `num_register_tokens` exists as a CLI arg if it isn't already (the VisionTransformer currently may be instantiated with a hardcoded `num_register_tokens=4` in `trainer.py`). Add:

```python
parser.add_argument('--num_register_tokens', default=4, type=int,
                    help='Number of register tokens [Darcet et al. 2023, adopted by '
                         'Virchow2 (4), H-optimus-1 (4), Midnight (4), Virchow2G (8), '
                         'UNI2-h (8)]. Auto-bumped to 8 if --use_pathology_recipe=True '
                         'and embeddingdim >= 1280.')
```

### File 2: `losses/kde_loss.py` (new file)

```python
"""
KDE regularization loss with von Mises-Fisher kernel.
Replaces KoLeo for pathology data where near-duplicate minibatch tiles
cause KoLeo's nearest-neighbor distance to collapse and its gradient
to explode.

Reference: Zimmermann et al., "Virchow 2: Scaling Self-Supervised Mixed
Magnification Models in Pathology", arXiv:2408.00738 Section 5.2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class KDELoss(nn.Module):
    """
    Kernel density estimation regularizer with von Mises-Fisher kernel.
    Features are gathered across all GPUs before density estimation to
    give a more accurate uniformity signal when world_size > 1.

    Args:
        kappa: vMF concentration parameter (Virchow2 uses 5.0)
    """
    def __init__(self, kappa=5.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, student_output, eps=1e-8):
        """
        Compute KDE loss with cross-GPU feature pooling.

        Args:
            student_output: Feature vectors [B_local, D]
            eps: Small constant for numerical stability inside the log

        Returns:
            Scalar loss value encouraging uniform feature distribution
        """
        with torch.cuda.amp.autocast(enabled=False):
            x_local = F.normalize(student_output.float(), p=2, dim=-1)

            # Gather features from all GPUs for global density estimation
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                x_list = [torch.zeros_like(x_local) for _ in range(world_size)]
                dist.all_gather(x_list, x_local)
                # Replace this GPU's slot with the local tensor so the graph
                # stays connected for local samples. Remote slots are detached
                # by design (all_gather does not backprop across GPUs).
                rank = dist.get_rank()
                x_list[rank] = x_local
                x_all = torch.cat(x_list, dim=0)
            else:
                x_all = x_local

            N = x_all.shape[0]

            # Pairwise cosine similarities [N, N]
            sim = x_all @ x_all.t()

            # vMF kernel with log-sum-exp numerical stability
            logits = self.kappa * sim
            max_per_row = logits.max(dim=1, keepdim=True).values
            kernel = torch.exp(logits - max_per_row)

            # Mask diagonal (exclude self-similarity)
            mask = ~torch.eye(N, dtype=torch.bool, device=x_all.device)
            density = (kernel * mask.float()).sum(dim=1) / (N - 1)

            # Log-density per sample with stability offset recovered
            log_density = torch.log(density + eps) + max_per_row.squeeze(1)

            # Gradient only through local slice, so each GPU contributes
            # gradient over its own batch samples
            if dist.is_available() and dist.is_initialized():
                B_local = x_local.shape[0]
                rank = dist.get_rank()
                start = rank * B_local
                end = start + B_local
                loss = log_density[start:end].mean()
            else:
                loss = log_density.mean()

            return loss
```

### File 3: `losses/__init__.py`

Update the existing file to export `KDELoss`:

```python
from .dino_loss import DINOLoss
from .ibot_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss
from .kde_loss import KDELoss
from .prototype_loss import PatchPrototypeLoss

__all__ = [
    'DINOLoss',
    'iBOTPatchLoss',
    'KoLeoLoss',
    'KDELoss',
    'PatchPrototypeLoss',
]
```

### File 4: `data/transforms.py`

Rewrite `TMEDinoTransforms` to support magnification-aware routing.

**Constructor changes:**
- Accept `use_pathology_recipe: bool = False`
- Accept `ect_probability: float = 0.4`
- When `use_pathology_recipe=True`, build two sets of transform primitives (ECT and standard) at construction time
- When `use_pathology_recipe=True`, modify `flip_and_color_jitter` chain to add vertical flip and 90-degree rotations; remove solarization from global_2

**Augmentation chain when pathology recipe is on:**

```python
# flip_and_color_jitter with V-flip and 90-degree rotations
# Sources: V-flip and 90-deg rotations from Virchow2, RudolfV, Hibou, Lunit
self.flip_and_color_jitter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # 90-degree discrete rotations. If torchvision's RandomChoice is available,
    # prefer it. Otherwise write a small custom Random90Rotation transform.
    Random90Rotation(p=0.75),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                saturation=0.2, hue=0.1)],
        p=0.8),
    transforms.RandomGrayscale(p=0.01),
])
```

Implement `Random90Rotation` as a small custom class (use `torchvision.transforms.functional.rotate` with angle sampled uniformly from {0, 90, 180, 270} when triggered). Place it at the top of `data/transforms.py`.

**Global crop 2 when pathology recipe is on: no `RandomSolarize`.** Virchow2 §5.2 ablation.

**ECT transform primitives (when source tile is 40× / size ≥ 448):**

```python
self.ect_global_1 = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224, scale=(0.203, 0.303), ratio=(0.95, 1.05),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])

self.ect_global_2 = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224, scale=(0.203, 0.303), ratio=(0.95, 1.05),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    # No RandomSolarize — Virchow2 §5.2 ablation
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])

self.ect_local = transforms.Compose([
    transforms.RandomResizedCrop(
        size=96, scale=(0.037, 0.056), ratio=(0.95, 1.05),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])
```

**Standard transform primitives (scale values from canonical DINOv2 per Virchow2 §5.1):**

```python
self.std_global_1 = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224, scale=(0.32, 1.0), ratio=(0.75, 1.33),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])

self.std_global_2 = transforms.Compose([
    transforms.RandomResizedCrop(
        size=224, scale=(0.32, 1.0), ratio=(0.75, 1.33),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])

self.std_local = transforms.Compose([
    transforms.RandomResizedCrop(
        size=96, scale=(0.05, 0.32), ratio=(0.75, 1.33),
        interpolation=Image.BICUBIC),
    self.flip_and_color_jitter,
    transforms.GaussianBlur(3, (0.1, 0.15)),
    self.to_tensor,
    transforms.Normalize(mean=mean, std=std),
])
```

Note: the original `TMEDinoTransforms` calls `transforms.Resize((224, 224))` as the first step of every transform chain. Under the pathology recipe, **do NOT pre-resize** — the whole point of ECT is operating on source at native resolution. `RandomResizedCrop` handles the final resize to output size.

**`__call__` method when pathology recipe is on:**

```python
def __call__(self, x):
    """
    Apply augmentation transforms with optional probabilistic ECT routing.

    When use_pathology_recipe is True:
    - 40x tiles (img size >= 448) get ECT with probability ect_probability,
      standard crop-and-resize otherwise.
    - 20x tiles (img size < 448) always get standard crop-and-resize.

    When use_pathology_recipe is False, falls back to the original
    pre-resize-to-224 + RandomResizedCrop behavior for all tiles.
    """
    crops = []

    if self.use_pathology_recipe:
        source_size = x.size[0]  # PIL image size[0] is width

        if source_size >= 448:
            # 40x tile — probabilistic ECT routing
            use_ect = random.random() < self.ect_probability
            if use_ect:
                g1, g2, lc = self.ect_global_1, self.ect_global_2, self.ect_local
            else:
                g1, g2, lc = self.std_global_1, self.std_global_2, self.std_local
        else:
            # 20x tile — standard only (no extended context available)
            g1, g2, lc = self.std_global_1, self.std_global_2, self.std_local

        crops.append(g1(x))
        crops.append(g2(x))
        for _ in range(self.n_local_crops):
            crops.append(lc(x))
    else:
        # Original behavior, preserved for when recipe is off
        crops.append(self.global_1(x))
        crops.append(self.global_2(x))
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))

    return crops
```

### File 5: `data/datasets.py`

Thread `use_pathology_recipe` and `ect_probability` as new parameters through three classes:

- `MemoryEfficientShardedPathologyDataset.__init__`
- `DINOv2PathologyDataset.__init__`
- `ProportionalMultiDatasetWrapper.__init__`

Each class passes both parameters into its child `TMEDinoTransforms` constructor. Defaults should be `use_pathology_recipe=False` and `ect_probability=0.4` to preserve current behavior when recipe is off.

No logic changes — purely parameter plumbing.

### File 6: `models/vision_transformer/modern_vit.py`

The `qk_norm` argument already exists in `VisionTransformer.__init__` and is wired into `TransformerBlock`. Verify the code path activates when `qk_norm=True`. No code changes expected unless verification reveals a bug.

### File 7: `utils.py`

Add a `StableAdamW` implementation. Prefer importing from `timm.optim` or `torch_optimizer` if the installed version provides a vetted implementation; fall back to an inline implementation only if necessary.

**Preferred (if available):**

```python
try:
    from timm.optim import Lamb as _StableAdamW_timm_fallback
    # timm doesn't ship StableAdamW under that exact name — check
    # installed version. If unavailable, use inline implementation below.
except ImportError:
    pass
```

**Inline fallback implementation** (add to `utils.py`):

```python
class StableAdamW(torch.optim.Optimizer):
    """
    StableAdamW with update clipping to prevent late-training NaN.
    
    Reference: Virchow2G (Zimmermann et al. 2024, arXiv:2408.00738 Section 6)
    reports using StableAdamW with beta2=0.95 to avoid NaN incidents observed
    with standard AdamW at ViT-G scale. The paper does not release a reference
    implementation; this follows the common StableAdamW formulation where the
    update is clipped by the RMS of the recent gradient.
    
    Args:
        params: iterable of parameters to optimize
        lr: learning rate
        betas: (beta1, beta2) — Virchow2G uses (0.9, 0.95)
        eps: epsilon for numerical stability
        weight_decay: decoupled weight decay coefficient
        clip_threshold: RMS bound for update clipping (default 1.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.01, clip_threshold=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        clip_threshold=clip_threshold)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                step = state['step']

                # Decoupled weight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() /
                         (bias_correction2 ** 0.5)).add_(group['eps'])

                # Stability clip on update magnitude
                rms = grad.pow(2).mean().sqrt()
                clip_value = rms.clamp(min=group['clip_threshold'])
                clip_ratio = group['clip_threshold'] / clip_value

                step_size = group['lr'] / bias_correction1 * clip_ratio
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
```

### File 8: `training/trainer.py`

This is the most involved change. Listed by location within `train_dinov2`:

**8a. Resolve auto-gate at function entry** (after `utils.init_distributed_mode(args)`):

```python
# Resolve auto-gate based on embedding dimension
auto_gate_active = (args.embeddingdim >= 1280) and args.use_pathology_recipe

if auto_gate_active:
    # qk_norm: auto-enable if not explicitly set
    if args.qk_norm is None:
        args.qk_norm = True
    # register tokens: monotonic max with 8
    if args.num_register_tokens < 8:
        args.num_register_tokens = 8
        print(f"[pathology recipe auto-gate] Bumped num_register_tokens to 8 "
              f"(Virchow2G scaling package)")
    print(f"[pathology recipe auto-gate] qk_norm={args.qk_norm}, "
          f"register_tokens={args.num_register_tokens}, StableAdamW active")
elif args.qk_norm is None:
    args.qk_norm = False  # safe default when auto-gate doesn't fire
```

**8b. Replace KoLeoLoss instantiation with conditional KDE/KoLeo:**

```python
if args.use_pathology_recipe:
    from losses import KDELoss
    dino_koleo_loss = KDELoss(kappa=args.kde_kappa).cuda()
    print(f"Using KDE regularizer (kappa={args.kde_kappa}, all-gather across "
          f"{dist.get_world_size()} GPUs). Source: Virchow2 Section 5.2.")
else:
    dino_koleo_loss = KoLeoLoss().cuda()
    print("Using KoLeo regularizer (DINOv2 default)")
```

**8c. Replace fp16 GradScaler with bf16 path when pathology recipe is on:**

```python
if args.use_pathology_recipe:
    fp16_scaler = None
    bf16_mode = True
    print("Using bf16 end-to-end (no GradScaler). H100-appropriate; "
          "avoids fp16 NaN issues flagged by Virchow2G.")
else:
    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None
    bf16_mode = False
```

Then in each `torch.cuda.amp.autocast` site in the training loop, change:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
```

to:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16,
                              enabled=args.use_fp16 or bf16_mode):
```

**8d. Swap AdamW for StableAdamW at ViT-G scale:**

```python
if auto_gate_active:
    from utils import StableAdamW
    optimizer_student = StableAdamW(all_param_groups, betas=(0.9, 0.95))
    print(f"Using StableAdamW (betas=(0.9, 0.95)). Source: Virchow2G Section 6.")
else:
    optimizer_student = torch.optim.AdamW(all_param_groups)
```

**8e. Override teacher temperature when pathology recipe is on:**

```python
if args.use_pathology_recipe:
    teacher_temp_effective = 0.04
    warmup_teacher_temp_effective = 0.04
    print(f"Teacher temperature fixed at 0.04. Source: Virchow2G Section 5.1.")
else:
    teacher_temp_effective = args.teacher_temp
    warmup_teacher_temp_effective = args.warmup_teacher_temp

dino_class_loss = DINOLoss(
    ncrops=total_student_views,
    warmup_teacher_temp=warmup_teacher_temp_effective,
    teacher_temp=teacher_temp_effective,
    warmup_teacher_temp_iters=args.teacher_temp_warmup_iters,
    n_iterations=5,
    student_temp=0.1,
).cuda()
```

**8f. Adjust `koleo_loss_weight` default when pathology recipe is on:**

```python
if args.use_pathology_recipe and abs(args.koleo_loss_weight - 0.1) < 1e-6:
    args.koleo_loss_weight = 0.05
    print(f"Set koleo_loss_weight=0.05 (Virchow2 KDE default lambda)")
```

This preserves user override: if the user explicitly passed a different weight, it's respected.

**8g. Wire `num_register_tokens` and `qk_norm` into VisionTransformer instantiation:**

Find the `student_encoder = ModernViT(...)` and `teacher_encoder = deepcopy(student_encoder)` lines and ensure `num_register_tokens=args.num_register_tokens` and `qk_norm=args.qk_norm` are passed. Currently `num_register_tokens=4` may be hardcoded at the call site.

**8h. Wire `patch_size=14` through when recipe is on:**

Find where `args.patch_size` is used for ModernViT instantiation. Add near the top of `train_dinov2`:

```python
if args.use_pathology_recipe:
    if args.patch_size != 14:
        print(f"[pathology recipe] Overriding patch_size {args.patch_size} -> 14 "
              f"(community standard for pathology FMs)")
        args.patch_size = 14
```

**8i. Override `out_dim` when recipe is on:**

```python
if args.use_pathology_recipe:
    if args.out_dim != 131072:
        print(f"[pathology recipe] Overriding out_dim {args.out_dim} -> 131072 "
              f"(Virchow v1 Methods, Paige standard)")
        args.out_dim = 131072
```

### File 9: `run_with_submitit.py`

Add a documentation block near the top of `main()` explaining the recipe sources and magnification table (the full Group C block).

Add the toggle as a commented-out block the user can uncomment:

```python
# ================================================================
# PATHOLOGY FM RECIPE — toggle
# ================================================================
# Uncomment to enable the Virchow2-derived pathology recipe bundle.
#
# Recipe includes:
#   - KDE regularizer replaces KoLeo        [Virchow2 Sec 5.2]
#   - Probabilistic ECT augmentation         [Virchow2 Sec 5.1 + user variation]
#   - Teacher temp fixed at 0.04             [Virchow2G Sec 5.1]
#   - out_dim=131,072                        [Virchow v1 Methods]
#   - patch_size=14                          [pathology FM community standard]
#   - bf16 end-to-end                        [Virchow2G retrospective]
#   - Solarization off, V-flip, 90-deg rot   [Virchow2/RudolfV/Hibou convergence]
#
# args.use_pathology_recipe = True
# args.ect_probability = 0.4
# args.kde_kappa = 5.0
#
# When embeddingdim >= 1280, the auto-gate additionally enables:
#   - qk_norm=True                           [Virchow2G Sec 6]
#   - num_register_tokens >= 8               [Virchow2G + UNI2-h]
#   - StableAdamW with beta2=0.95            [Virchow2G Sec 6]
#
# [Full probabilistic ECT magnification table — see Group C comment above]
# ================================================================
```

Leave `args.use_pathology_recipe = False` as the default. Existing behavior is preserved unless explicitly opted in.

## Testing and Verification

The implementer must verify the following before marking the branch complete:

1. **No-op check.** Running with `--use_pathology_recipe=False` produces a training run that is bit-for-bit identical to the `consolidated` branch's behavior. Verify by checking the first 10 iterations' loss values match.

2. **ECT branching.** Add temporary print statements in `TMEDinoTransforms.__call__` confirming that 40× tiles route to ECT with roughly 40% frequency and 20× tiles always route to standard. Remove prints after verification.

3. **KDE numerical stability.** Run with a toy batch where all features are identical (clone one feature vector to fill the batch). Verify `KDELoss` returns a finite value; compare against `KoLeoLoss` which should return either `inf` or an extremely large value.

4. **Auto-gate trigger.** Launch with `embeddingdim=1280` and confirm the log output includes `[pathology recipe auto-gate]` messages showing `qk_norm=True` and StableAdamW active.

5. **bf16 end-to-end.** With pathology recipe on, confirm no `GradScaler` operations appear in the training loop — simply check that `fp16_scaler is None` throughout.

6. **Single-epoch smoke test.** Full training launch for 100 iterations with `--use_pathology_recipe=True` on the ViT-L config. Must complete without NaN or shape errors. Loss values should be reasonable (not orders of magnitude different from `consolidated` baseline).

## Out of Scope

This spec deliberately does **not** include:

- DINOv2-LVD-142M natural-image initialization (checkpoint-dependent)
- Multi-magnification sampling beyond what the data already provides
- Macenko normalization or stain-transfer augmentation (data pipeline)
- GPFM-style multi-teacher distillation (checkpoint-dependent)
- PLUTO's MAE + Fourier losses (weak evidence, ambitious engineering)
- HED color augmentation from Midnight (data-pipeline complication not justified by evidence delta vs. simpler changes)
- Changes to the three codebase-native modifications (`--use_patch_prototype_clustering`, `--use_semantic_ibot`, `--use_typicality_dampening`) — these remain orthogonal toggles

## Commit Structure

Suggested commit sequence on `pathology-fm-recipe` branch:

1. `Add CLI args for pathology recipe (configs/config.py)`
2. `Add KDELoss with vMF kernel and all-gather pooling (losses/kde_loss.py)`
3. `Add Random90Rotation and probabilistic ECT to TMEDinoTransforms`
4. `Thread pathology recipe args through dataset classes`
5. `Add StableAdamW to utils.py`
6. `Wire pathology recipe into trainer.py with auto-gate`
7. `Document recipe and magnification table in run_with_submitit.py`
8. `Testing and verification notes`
