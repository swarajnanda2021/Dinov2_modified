# DINOv2 Implementation Differences

This document outlines the differences between this implementation and the [official Meta DINOv2](https://github.com/facebookresearch/dinov2) codebase, and specifies the required changes to align with the official approach.


---

## Table of Contents

1. [iBOT Architecture: Separate vs Unified Forward Pass](#1-ibot-architecture-separate-vs-unified-forward-pass)
2. [Augmentation Pipeline: Extra Global View](#2-augmentation-pipeline-extra-global-view)
3. [Masking Strategy: Bernoulli vs Block Masking](#3-masking-strategy-bernoulli-vs-block-masking)
4. [Patch Prototype Loss Simplification](#4-patch-prototype-loss-simplification)
5. [Per-Sample Normalization for Variable Mask Ratios](#5-per-sample-normalization-for-variable-mask-ratios)
6. [Weight Decay Schedule Bug](#6-weight-decay-schedule-bug)
7. [Layer-wise Learning Rate Decay](#7-layer-wise-learning-rate-decay)

---

## 1. iBOT Architecture: Separate vs Unified Forward Pass

### The Core Difference

| Aspect | Official DINOv2 | This Implementation |
|--------|-----------------|---------------------|
| **iBOT input** | Same global crops as DINO | Separate `global_3` view |
| **Forward passes** | Single unified pass | Separate DINO + iBOT passes |
| **Mask application** | On global crops during unified forward | On separate view in separate forward |

### Official Architecture

In official DINOv2, the **same two global crops** are used for both DINO CLS loss and iBOT patch loss. The training flow is:
```
Image → Augmentation → [global_1, global_2, local_1, ..., local_N]
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
              Teacher Forward      Student Forward
              (no masking)         (block masks on globals)
                    ↓                   ↓
              ┌─────┴─────┐       ┌─────┴─────┐
              ↓           ↓       ↓           ↓
           CLS tokens  Patch    CLS tokens  Patch
                      tokens               tokens
                    ↓                   ↓
              DINO CLS Loss ←──────→ (all views)
              iBOT Patch Loss ←────→ (global views only, masked positions)
```

**Key insight**: Official uses `forward_features_list()` which processes all crops in a single packed forward pass. Masks are applied **inside** the backbone via `prepare_tokens_with_masks()`, replacing masked patch embeddings with a learnable `mask_token`.

### This Implementation

Currently, this implementation uses a **separate third global view** (`global_3`) for iBOT, requiring an extra forward pass:
```
Image → Augmentation → [global_1, global_2, local_1, ..., local_N, global_3]
                              ↓                                        ↓
                    ┌─────────┴─────────┐                    ┌─────────┴─────────┐
                    ↓                   ↓                    ↓                   ↓
              Teacher Forward      Student Forward      Teacher iBOT       Student iBOT
              (globals only)       (all views)          (global_3)         (global_3 + masks)
                    ↓                   ↓                    ↓                   ↓
              DINO CLS Loss ←──────────→               iBOT Patch Loss ←────────→
```

### Current Code (`training/trainer.py`, lines ~330-400)
```python
# Current: Extract global_3 as separate iBOT input
original_images = batch_data[-1].cuda(non_blocking=True)  # This is global_3

# ... later in the loop ...

# Current: Separate iBOT forward pass
with torch.no_grad():
    teacher_ibot_output = teacher(original_images, token_masks=None, mode='ibot')
    teacher_patch_outputs = teacher_ibot_output['patch_outputs']

student_ibot_output = student(original_images, token_masks=random_token_masks, mode='ibot')
student_patch_outputs = student_ibot_output['patch_outputs']
```

### Required Changes

**Goal**: Eliminate the separate iBOT forward pass by applying masks to global crops during the unified student forward.

#### Change 1: Remove `original_images` extraction
```python
# REMOVE:
original_images = batch_data[-1].cuda(non_blocking=True)
```

#### Change 2: Generate masks for global crops
```python
# ADD: Generate block masks for global crops (student only)
batch_size = teacher_global_crops[0].shape[0]
n_patches_h = n_patches_w = 224 // args.patch_size

block_masks_1, masks_weight_1 = generate_block_masks(
    batch_size, n_patches_h, n_patches_w,
    mask_ratio_min=args.mask_ratio_min,
    mask_ratio_max=args.mask_ratio_max,
    mask_sample_probability=args.mask_sample_probability,
    device=teacher_global_crops[0].device
)
block_masks_2, masks_weight_2 = generate_block_masks(...)
```

#### Change 3: Unified forward with masks
```python
# Teacher: global crops, NO masking
with torch.no_grad():
    teacher_output = teacher(
        teacher_global_crops,
        token_masks=[None, None]
    )

# Student: all crops, masks on global crops only
student_masks = [block_masks_1, block_masks_2] + [None] * len(student_local_crops)
student_output = student(
    student_all_crops,
    token_masks=student_masks
)
```

#### Change 4: Extract features for losses
```python
# DINO CLS loss: uses all views
teacher_cls = torch.cat([out['clstoken'] for out in teacher_output])
student_cls = torch.cat([out['clstoken'] for out in student_output])

# iBOT patch loss: uses global views only (indices 0, 1)
# Teacher patches are unmasked, student patches have mask_token at masked positions
teacher_patches = [teacher_output[i]['patchtokens'] for i in range(2)]
student_patches = [student_output[i]['patchtokens'] for i in range(2)]

# Prototype loss: piggybacks on same student patch features
```

#### Change 5: Remove separate iBOT forward
```python
# DELETE entire section:
# teacher_ibot_output = teacher(original_images, ...)
# student_ibot_output = student(original_images, ...)
```

### Backbone Support

The backbone (`models/vision_transformer/modern_vit.py`) already supports this architecture:
```python
# VisionTransformer.prepare_tokens_with_masks() - applies mask_token
def prepare_tokens_with_masks(self, x, token_masks=None):
    x = self.patch_embed(x)
    if token_masks is not None:
        x = torch.where(token_masks.unsqueeze(-1), 
                        self.mask_token.to(x.dtype).unsqueeze(0), 
                        x)
    # ... add cls, pos_embed, registers ...

# VisionTransformer.forward_features_list() - sequence packing
def forward_features_list(self, x, masks_list):
    # Processes multiple crops with different masks in single forward
```

---

## 2. Augmentation Pipeline: Extra Global View

### Current Implementation (`data/transforms.py`)

The `TMEDinoTransforms` class currently produces **three distinct global augmentations**:
```python
class TMEDinoTransforms(object):
    def __init__(self, ...):
        # Global view 1: blur only
        self.global_1 = transforms.Compose([
            transforms.RandomResizedCrop(...),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(3, (0.1, 0.15)),
            self.normalize,
        ])

        # Global view 2: blur + solarize
        self.global_2 = transforms.Compose([
            transforms.RandomResizedCrop(...),
            self.flip_and_color_jitter,
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.15)),
            transforms.RandomSolarize(threshold=64, p=0.5),
            self.normalize,
        ])

        # Global view 3: separate augmentation for iBOT (REMOVE THIS)
        self.global_3 = transforms.Compose([
            transforms.RandomResizedCrop(...),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomApply([transforms.ColorJitter(...)], p=0.8),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
            self.normalize,
        ])

    def __call__(self, x):
        crops = []
        crops.append(self.global_1(x))
        crops.append(self.global_2(x))
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))
        crops.append(self.global_3(x))  # REMOVE THIS
        return crops
```

### Official Implementation (`dinov2/data/augmentations.py`)

Official DINOv2 produces only **two global views**, and they are reused for both DINO and iBOT:
```python
class DataAugmentationDINO(object):
    def __call__(self, image):
        output = {}
        
        # Two global crops with different augmentations
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)  # blur p=1.0
        
        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)  # blur p=0.1, solarize p=0.2
        
        output["global_crops"] = [global_crop_1, global_crop_2]
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]  # SAME crops
        output["local_crops"] = [self.local_transfo(...) for _ in range(n_local)]
        
        return output
```

### Required Changes

#### In `data/transforms.py`:
```python
class TMEDinoTransforms(object):
    def __init__(self, ...):
        # Keep global_1 and global_2
        # REMOVE: self.global_3 definition
        pass

    def __call__(self, x):
        crops = []
        crops.append(self.global_1(x))
        crops.append(self.global_2(x))
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))
        # REMOVE: crops.append(self.global_3(x))
        return crops
```

#### In `training/trainer.py`:
```python
# REMOVE: This line and all references to original_images
original_images = batch_data[-1].cuda(non_blocking=True)
```

### Output Format Comparison

| | Current | Target |
|---|---------|--------|
| **Return** | `[g1, g2, l1, ..., lN, g3]` | `[g1, g2, l1, ..., lN]` |
| **Length** | `2 + N + 1` | `2 + N` |
| **iBOT input** | `g3` (index -1) | `g1, g2` (indices 0, 1) |

---

## 3. Masking Strategy: Bernoulli vs Block Masking

### Comparison

| Aspect | Official | This Implementation |
|--------|----------|---------------------|
| **Mask type** | Block masking (spatially coherent rectangles) | Independent Bernoulli per token |
| **Mask ratio** | Variable per sample (0.1–0.5) | Fixed ratio (e.g., 0.3) |
| **Mask probability** | `mask_sample_probability` (e.g., 0.5) | 100% of samples masked |
| **Spatial structure** | Rectangular blocks, aspect ratio 0.3–3.3 | Random scattered tokens |

### Current Implementation (`training/helpers.py`)
```python
def generate_random_token_masks(batch_size, n_patches_h, n_patches_w, mask_ratio, device):
    """Independent Bernoulli sampling - each token masked with probability mask_ratio."""
    n_patches = n_patches_h * n_patches_w
    token_masks = torch.bernoulli(
        torch.ones(batch_size, n_patches) * mask_ratio
    ).bool().to(device)
    return token_masks
```

**Problems with Bernoulli masking**:
1. No spatial coherence - scattered random tokens
2. Fixed ratio - no curriculum or variation
3. All samples masked - no unmasked samples in batch

### Official Block Masking (`dinov2/data/masking.py`)
```python
class MaskingGenerator:
    def __init__(self, input_size, min_num_patches=4, min_aspect=0.3, max_aspect=None):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        """Try to place a single rectangular block."""
        delta = 0
        for _ in range(10):  # 10 attempts
            # Sample target area
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            # Sample aspect ratio (log-uniform for symmetric distribution)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            # Compute dimensions
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                # Check overlap with existing mask
                num_masked = mask[top:top+h, left:left+w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top:top+h, left:left+w] = 1
                    delta += h * w - num_masked
                    if delta >= max_mask_patches:
                        break
        return delta

    def __call__(self, num_masking_patches):
        """Generate mask with exactly num_masking_patches masked."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask.flatten()
```

### Official Batch Masking (`dinov2/data/collate.py`)
```python
def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, ...):
    B = len(samples_list)
    N = n_tokens  # e.g., 196 for 14x14 patches
    
    # Only mask_probability fraction of samples get masks
    n_samples_masked = int(B * mask_probability)
    
    # Variable ratio: linspace creates smooth gradient across masked samples
    # e.g., mask_ratio_tuple=(0.1, 0.5), n_samples_masked=16
    # → probs = [0.1, 0.125, 0.15, ..., 0.475, 0.5]
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    
    masks_list = []
    for i in range(n_samples_masked):
        prob_min, prob_max = probs[i], probs[i + 1]
        num_patches_to_mask = int(N * random.uniform(prob_min, prob_max))
        masks_list.append(torch.BoolTensor(mask_generator(num_patches_to_mask)))
    
    # Remaining samples get empty masks (no masking)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))
    
    # Shuffle to distribute masked/unmasked randomly
    random.shuffle(masks_list)
    
    collated_masks = torch.stack(masks_list).flatten(1)
    
    # Compute inverse weights for loss normalization
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1)
    masks_weight = masks_weight.expand_as(collated_masks)[collated_masks]
    
    return {
        "collated_masks": collated_masks,
        "masks_weight": masks_weight,
        ...
    }
```

### Required Changes

#### Add to `training/helpers.py`:
```python
class BlockMaskGenerator:
    """
    Generate spatially coherent block masks for iBOT training.
    Based on official DINOv2 MaskingGenerator.
    """
    def __init__(self, input_size, min_num_patches=4, min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        """Attempt to place a rectangular block on the mask."""
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top:top+h, left:left+w].sum()
                
                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top:top+h, left:left+w] = 1
                    delta += h * w - num_masked
                    if delta >= max_mask_patches:
                        break
        return delta

    def __call__(self, num_masking_patches):
        """Generate a single mask with num_masking_patches masked tokens."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            delta = self._mask(mask, num_masking_patches - mask_count)
            if delta == 0:
                break
            mask_count += delta
        return mask.flatten()


def generate_block_masks(
    batch_size: int,
    n_patches_h: int,
    n_patches_w: int,
    mask_ratio_min: float = 0.1,
    mask_ratio_max: float = 0.5,
    mask_sample_probability: float = 0.5,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate block masks for a batch with variable ratios.
    
    Args:
        batch_size: Number of samples
        n_patches_h: Number of patches in height
        n_patches_w: Number of patches in width
        mask_ratio_min: Minimum mask ratio (e.g., 0.1 = 10%)
        mask_ratio_max: Maximum mask ratio (e.g., 0.5 = 50%)
        mask_sample_probability: Fraction of samples to mask (e.g., 0.5)
        device: Device for output tensors
        
    Returns:
        masks: [B, N] boolean tensor where True = masked
        masks_weight: [B] tensor with 1/num_masked for loss weighting
    """
    n_patches = n_patches_h * n_patches_w
    n_masked_samples = int(batch_size * mask_sample_probability)
    
    mask_generator = BlockMaskGenerator(
        input_size=(n_patches_h, n_patches_w),
        min_num_patches=4,
        min_aspect=0.3
    )
    
    # Generate variable-ratio masks for masked samples
    masks = []
    probs = np.linspace(mask_ratio_min, mask_ratio_max, n_masked_samples + 1)
    
    for i in range(n_masked_samples):
        prob_min, prob_max = probs[i], probs[i + 1]
        ratio = random.uniform(prob_min, prob_max)
        num_to_mask = int(n_patches * ratio)
        masks.append(mask_generator(num_to_mask))
    
    # Empty masks for unmasked samples
    for _ in range(n_masked_samples, batch_size):
        masks.append(np.zeros(n_patches, dtype=bool))
    
    # Shuffle to distribute randomly
    random.shuffle(masks)
    
    masks_tensor = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)
    
    # Compute weights: 1 / num_masked_per_sample (0 for unmasked samples)
    num_masked = masks_tensor.sum(dim=1).float()
    masks_weight = torch.where(
        num_masked > 0,
        1.0 / num_masked,
        torch.zeros_like(num_masked)
    )
    
    return masks_tensor, masks_weight
```

#### Update arguments:
```python
# REMOVE:
parser.add_argument('--token_mask_ratio', type=float, default=0.3)

# ADD:
parser.add_argument('--mask_ratio_min', type=float, default=0.1,
                    help='Minimum mask ratio for iBOT block masking')
parser.add_argument('--mask_ratio_max', type=float, default=0.5,
                    help='Maximum mask ratio for iBOT block masking')
parser.add_argument('--mask_sample_probability', type=float, default=0.5,
                    help='Fraction of samples in batch to apply masking')
```

### Visual Comparison

**Bernoulli masking** (current):
```
□ ■ □ ■ □ □ ■ □ □ ■ □ □ ■ □
■ □ □ □ ■ □ □ ■ □ □ □ ■ □ □
□ □ ■ □ □ □ ■ □ □ ■ □ □ □ ■
□ ■ □ □ ■ □ □ □ ■ □ □ ■ □ □
```

**Block masking** (target):
```
□ □ □ □ □ □ □ □ □ □ □ □ □ □
□ □ ■ ■ ■ ■ ■ □ □ □ □ □ □ □
□ □ ■ ■ ■ ■ ■ □ □ ■ ■ ■ □ □
□ □ ■ ■ ■ ■ ■ □ □ ■ ■ ■ □ □
```

---

## 4. Patch Prototype Loss Simplification

### Current Implementation (`losses/prototype_loss.py`)

The current implementation supports three clustering modes:
```python
if args.clustering_mode == 'visible':
    # Use visible (unmasked) patch tokens
    visible_mask = ~random_token_masks
    clustering_loss = patch_prototype_loss(..., visible_mask, ...)

elif args.clustering_mode == 'masked':
    # Use masked patch tokens
    clustering_loss = patch_prototype_loss(..., random_token_masks, ...)

elif args.clustering_mode == 'separate':
    # Separate forward pass on original_images with all tokens
    student_cluster = student(original_images, token_masks=None, mode='ibot')
    teacher_cluster = teacher(original_images, token_masks=None, mode='ibot')
    all_mask = torch.ones_like(random_token_masks, dtype=torch.bool)
    clustering_loss = patch_prototype_loss(..., all_mask, ...)
```

### Problems

1. **`visible` mode**: Using unmasked tokens for prototype clustering defeats the purpose. The iBOT objective is specifically about predicting masked content from visible context. Clustering on visible tokens provides no additional learning signal beyond what the encoder already computes.

2. **`separate` mode**: Requires an extra forward pass on `original_images`, which is exactly what we're trying to eliminate.

3. **Complexity**: Three modes add unnecessary code complexity and configuration burden.

### Required Changes

**Remove `visible` and `separate` modes. Keep only `masked` mode.**

The prototype loss should operate on the **same masked patch features** used for iBOT loss. Since iBOT and prototype clustering both work on masked positions, they share the same features and the prototype loss effectively "piggybacks" on the iBOT forward pass.

#### In `training/trainer.py`:
```python
# REMOVE: clustering_mode argument and conditional logic

# REPLACE with single masked-only implementation:
if args.use_prototype_clustering:
    # Get masked patch features from unified forward (same as iBOT)
    teacher_patch_features = [teacher_output[i]['patchtokens'] for i in range(2)]
    student_patch_features = [student_output[i]['patchtokens'] for i in range(2)]
    
    # Concatenate global crops
    teacher_patches = torch.cat(teacher_patch_features, dim=0)  # [2B, N, D]
    student_patches = torch.cat(student_patch_features, dim=0)  # [2B, N, D]
    combined_masks = torch.cat([block_masks_1, block_masks_2], dim=0)  # [2B, N]
    combined_weights = torch.cat([masks_weight_1, masks_weight_2], dim=0)  # [2B]
    
    clustering_loss, teacher_proto_loss, koleo_proto_loss = patch_prototype_loss(
        teacher_patches,
        student_patches,
        combined_masks,  # True = masked positions
        prototype_bank,
        current_teacher_temp,
        masks_weight=combined_weights  # For per-sample normalization
    )
```

#### Remove arguments:
```python
# REMOVE:
parser.add_argument('--clustering_mode', type=str, default='masked',
                    choices=['visible', 'masked', 'separate'])
```

---

## 5. Per-Sample Normalization for Variable Mask Ratios

### The Problem

With variable mask ratios (0.1–0.5), different samples contribute different numbers of tokens to the loss:

| Sample | Mask Ratio | Tokens Masked (of 196) |
|--------|------------|------------------------|
| A | 50% | 98 tokens |
| B | 30% | 59 tokens |
| C | 10% | 20 tokens |

**Current normalization** divides by total masked tokens:
```python
M_total = student_norm_masked.shape[0]  # e.g., 98 + 59 + 20 = 177
clustering_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked) / M_total
```

**Problem**: Sample A contributes 98/177 ≈ 55% of the gradient, while Sample C contributes only 20/177 ≈ 11%. This creates:
- Inconsistent gradients dominated by high-ratio samples
- Loss magnitude fluctuation across batches (depends on ratio distribution)
- Bias toward learning from samples with more masked tokens

### Official iBOT Loss Weighting

Official DINOv2 computes per-token weights so each **sample** contributes equally:
```python
# From collate.py:
masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1)
masks_weight = masks_weight.expand_as(collated_masks)[collated_masks]

# Usage in loss:
# Each token's loss is weighted by 1/(num_masked_in_its_sample)
# Then sum is divided by batch_size, not total_tokens
```

### Required Changes

#### In `losses/prototype_loss.py`:
```python
def forward(self, teacher_features, student_features, token_masks, 
            prototype_bank, teacher_temp, masks_weight=None):
    """
    Args:
        teacher_features: [B, N, D] teacher patch tokens
        student_features: [B, N, D] student patch tokens
        token_masks: [B, N] boolean, True = masked positions
        prototype_bank: LinearPrototypeBank module
        teacher_temp: Current temperature
        masks_weight: [B] optional, 1/num_masked for each sample
    """
    B, N, D = student_features.shape
    
    # Gather masked tokens
    student_masked = student_features[token_masks]  # [M, D]
    teacher_masked = teacher_features[token_masks]  # [M, D]
    
    # Compute prototype assignments and loss...
    # ... (existing logic for Q_tilde, student_log_probs, etc.)
    
    # Per-token cross-entropy
    per_token_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked, dim=-1)  # [M]
    
    if masks_weight is not None:
        # Map sample weights to per-token weights
        sample_indices = token_masks.nonzero(as_tuple=True)[0]  # [M] - which sample each token belongs to
        per_token_weight = masks_weight[sample_indices]  # [M]
        
        # Weighted sum, normalized by batch size (not token count)
        clustering_loss = (per_token_loss * per_token_weight).sum() / B
    else:
        # Fallback: simple mean over tokens
        clustering_loss = per_token_loss.mean()
    
    return clustering_loss, ...
```

### Mathematical Equivalence

With per-sample weighting:
```
Loss = (1/B) * Σ_samples [ (1/M_i) * Σ_tokens_in_sample_i [ loss_token ] ]
     = (1/B) * Σ_samples [ mean_loss_per_sample ]
```

This ensures each sample contributes `1/B` to the total gradient, regardless of how many tokens it has masked.

---

## 6. Weight Decay Schedule Bug

### The Bug

**Official** applies the weight decay schedule to ALL regularized parameter groups:
```python
def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
```

**This implementation** only updates group 0:
```python
# In training/trainer.py:
for i, param_group in enumerate(optimizer_student.param_groups):
    param_group["lr"] = student_lr_schedule[current_iteration]
    if i == 0:  # BUG: Only backbone regularized group!
        param_group["weight_decay"] = wd_schedule[current_iteration]
```

### Parameter Group Structure
```python
optimizer = AdamW([
    *get_params_groups(student.module.backbone),   # groups 0 (reg), 1 (no reg)
    *get_params_groups(student.module.classhead),  # groups 2 (reg), 3 (no reg)
    *get_params_groups(student.module.patchhead),  # groups 4 (reg), 5 (no reg)
])
```

### Impact

With WD schedule 0.04 → 0.4:

| Group | Module | WD Applied |
|-------|--------|------------|
| 0 | backbone (regularized) | schedule ✅ |
| 1 | backbone (not regularized) | 0.0 ✅ |
| 2 | classhead (regularized) | ~0.01 (AdamW default) ❌ |
| 3 | classhead (not regularized) | 0.0 ✅ |
| 4 | patchhead (regularized) | ~0.01 (AdamW default) ❌ |
| 5 | patchhead (not regularized) | 0.0 ✅ |

The DINO and iBOT projection heads receive **~25× less regularization** than intended at peak schedule.

### Fix
```python
for i, param_group in enumerate(optimizer_student.param_groups):
    param_group["lr"] = student_lr_schedule[current_iteration]
    if i % 2 == 0:  # Even indices are regularized groups
        param_group["weight_decay"] = wd_schedule[current_iteration]
```

---

## 7. Layer-wise Learning Rate Decay

### Official Implementation

Applies exponential LR decay based on layer depth:
```python
def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    layer_id = 0 for embeddings
    layer_id = 1..N for transformer blocks
    layer_id = N+1 for head
    """
    # ... parse layer_id from name ...
    return lr_decay_rate ** (num_layers + 1 - layer_id)
```

With `lr_decay_rate=0.9` and 24 layers (ViT-L):

| Layer | LR Multiplier |
|-------|---------------|
| Patch embed (layer 0) | 0.9^24 ≈ 0.08 |
| Block 0 (layer 1) | 0.9^23 ≈ 0.09 |
| Block 12 (layer 13) | 0.9^12 ≈ 0.28 |
| Block 23 (layer 24) | 0.9^1 ≈ 0.90 |
| Head (layer 25) | 1.0 |

### This Implementation

Uniform LR across all layers.

### Impact

Layer-wise decay:
- **Stabilizes early layers** (patch embedding, early blocks) that learn general features
- **Allows later layers** to adapt more aggressively to the SSL objective
- Particularly important for fine-tuning and transfer learning

### Required Changes (Optional Enhancement)
```python
def get_params_groups_with_decay(model, lr_decay_rate=0.9, patch_embed_lr_mult=0.2):
    """Create parameter groups with layer-wise LR decay."""
    num_layers = len(model.blocks)
    params_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Determine layer ID
        if "patch_embed" in name or "cls_token" in name or "pos_embed" in name:
            layer_id = 0
        elif "blocks." in name:
            layer_id = int(name.split("blocks.")[1].split(".")[0]) + 1
        else:
            layer_id = num_layers + 1
        
        # Compute LR multiplier
        lr_mult = lr_decay_rate ** (num_layers + 1 - layer_id)
        if "patch_embed" in name:
            lr_mult *= patch_embed_lr_mult
        
        # WD multiplier (0 for bias/norm)
        wd_mult = 0.0 if name.endswith(".bias") or "norm" in name else 1.0
        
        params_groups.append({
            "params": param,
            "lr_multiplier": lr_mult,
            "wd_multiplier": wd_mult,
        })
    
    return params_groups
```

---

## Summary of Required Changes

| Component | File | Change |
|-----------|------|--------|
| Transforms | `data/transforms.py` | Remove `global_3` |
| Masking | `training/helpers.py` | Add `BlockMaskGenerator` and `generate_block_masks()` |
| Trainer | `training/trainer.py` | Remove separate iBOT forward; unify masks on global crops |
| Prototype Loss | `losses/prototype_loss.py` | Remove `visible`/`separate` modes; add per-sample weighting |
| Prototype Loss | `training/trainer.py` | Remove `clustering_mode` logic |
| WD Bug | `training/trainer.py` | Fix to update all even-indexed param groups |
| Arguments | `main.py` | Add `mask_ratio_min/max`, `mask_sample_probability`; remove `token_mask_ratio`, `clustering_mode` |

---

## References

- [Official DINOv2 Repository](https://github.com/facebookresearch/dinov2)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832)
