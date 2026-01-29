# DINOv2 Implementation Differences

This document outlines the differences between this implementation and the [official Meta DINOv2](https://github.com/facebookresearch/dinov2) codebase.

---

## 1. iBOT Masking Strategy

| Aspect | Official | This Implementation |
|--------|----------|---------------------|
| **Mask type** | Block masking via `MaskingGenerator` | Independent Bernoulli per token |
| **Mask ratio** | Variable per sample (e.g., 0.1–0.5) | Fixed ratio |
| **Mask probability** | `mask_sample_probability` — not all images masked | All images always masked |

### Official Implementation (`dinov2/data/masking.py`)

```python
class MaskingGenerator:
    def __init__(
        self,
        input_size,  # (H_patches, W_patches), e.g. (14, 14) for 224px with patch_size=16
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,  # defaults to 1/min_aspect
    ):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches or num_masking_patches
        self.log_aspect_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):  # 10 attempts to place a block
            # Sample block area
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            # Sample aspect ratio (log-uniform)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            # Compute block dimensions
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            # Clamp to grid
            if h < self.height and w < self.width:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                # Count newly masked patches
                num_masked = mask[top:top+h, left:left+w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    # Place the block
                    mask[top:top+h, left:left+w] = 1
                    delta += h * w - num_masked
                    if delta >= max_mask_patches:
                        break
        return delta

    def __call__(self, num_masking_patches=None):
        mask = np.zeros((self.height, self.width), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask.flatten()  # Shape: (num_patches,)
```

### Official Collate Function (`dinov2/data/collate.py`)

```python
def collate_data_and_cast(...):
    B = len(samples_list)
    n_samples_masked = int(B * mask_sample_probability)  # e.g., 0.5 → half get masks
    
    # Variable mask ratio per sample
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    # e.g., mask_ratio_tuple = (0.1, 0.5) → probs = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    masks_list = []
    for i in range(n_samples_masked):
        prob_min, prob_max = probs[i], probs[i + 1]
        num_patches_to_mask = int(N * random.uniform(prob_min, prob_max))
        masks_list.append(mask_generator(num_patches_to_mask))
    
    for i in range(n_samples_masked, B):
        masks_list.append(np.zeros(N, dtype=bool))  # No mask
    
    random.shuffle(masks_list)
    collated_masks = torch.stack([torch.from_numpy(m) for m in masks_list])
    return collated_masks  # Shape: (B, num_patches)
```

### This Implementation (`training/helpers.py`)

```python
def generate_random_token_masks(batch_size, n_patches_h, n_patches_w, mask_ratio, device):
    n_patches = n_patches_h * n_patches_w
    token_masks = torch.bernoulli(
        torch.ones(batch_size, n_patches) * mask_ratio
    ).bool().to(device)
    return token_masks
```

### What to Change

Replace Bernoulli sampling with block masking:

```python
# New file: data/masking.py (copy MaskingGenerator from official)

# In collate or training loop:
mask_generator = MaskingGenerator(
    input_size=(n_patches_h, n_patches_w),
    min_num_patches=4,
    max_num_patches=int(0.5 * n_patches_h * n_patches_w),
    min_aspect=0.3,
)

def generate_block_masks(batch_size, n_patches_h, n_patches_w, 
                         mask_ratio_min=0.1, mask_ratio_max=0.5,
                         mask_sample_probability=0.5, device='cuda'):
    n_patches = n_patches_h * n_patches_w
    n_masked_samples = int(batch_size * mask_sample_probability)
    
    masks = []
    for i in range(n_masked_samples):
        # Variable ratio per sample
        ratio = random.uniform(mask_ratio_min, mask_ratio_max)
        num_to_mask = int(n_patches * ratio)
        masks.append(mask_generator(num_to_mask))
    
    for i in range(n_masked_samples, batch_size):
        masks.append(np.zeros(n_patches, dtype=bool))
    
    random.shuffle(masks)
    return torch.tensor(np.stack(masks), dtype=torch.bool, device=device)
```

### Impact

Block masking creates spatially coherent rectangular regions (varying aspect ratios from 0.3 to 3.3). This forces the model to reconstruct contiguous spatial areas rather than isolated scattered patches, encouraging learning of larger-scale spatial structure.

---

## 2. Layer-wise Learning Rate Decay

**Official** applies exponential LR decay based on layer depth:

```python
def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    # layer_id = 0 for embeddings, 1..N for transformer blocks
    return lr_decay_rate ** (num_layers + 1 - layer_id)
```

With `lr_decay_rate=0.9` and 24 layers (ViT-L):
| Layer | LR Multiplier |
|-------|---------------|
| Embeddings (layer 0) | 0.9^24 ≈ 0.08 |
| Block 12 | 0.9^12 ≈ 0.28 |
| Block 23 | 0.9^1 ≈ 0.90 |
| Head | 1.0 |

**This implementation**: Uniform LR across all layers.

**Impact**: Layer-wise decay stabilizes early layers (general features) while allowing later layers to adapt more aggressively.

---

## 3. Patch Embedding LR Multiplier

**Official**:
```python
if "patch_embed" in name:
    d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})
```

Typically `patch_embed_lr_mult < 1.0` to stabilize the input projection layer.

**This implementation**: Patch embedding uses same LR as other layers.

---

## 4. Weight Decay Schedule Application

### Param Group Structure

**Official** creates per-parameter groups with explicit `wd_multiplier`:
```python
for name, param in model.named_parameters():
    d = {"params": param, "wd_multiplier": 1.0, "name": name}
    if name.endswith(".bias") or "norm" in name or "gamma" in name:
        d.update({"wd_multiplier": 0.0})
```

**This implementation** groups by regularization status:
```python
# utils.get_params_groups returns 2 groups per module:
# [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]

optimizer = AdamW([
    *get_params_groups(student.backbone),   # groups 0, 1
    *get_params_groups(student.classhead),  # groups 2, 3
    *get_params_groups(student.patchhead),  # groups 4, 5
])
```

### The Bug

**Official** applies WD schedule to ALL groups:
```python
def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
```

**This implementation** only updates group 0:
```python
for i, param_group in enumerate(optimizer_student.param_groups):
    param_group["lr"] = student_lr_schedule[current_iteration]
    if i == 0:  # BUG: Only backbone regularized group!
        param_group["weight_decay"] = wd_schedule[current_iteration]
```

**Result** (with WD schedule 0.04 → 0.4):
| Group | Module | WD Applied |
|-------|--------|------------|
| 0 | backbone (regularized) | schedule ✅ |
| 1 | backbone (not regularized) | 0.0 ✅ |
| 2 | classhead (regularized) | ~0.01 (AdamW default) ❌ |
| 3 | classhead (not regularized) | 0.0 ✅ |
| 4 | patchhead (regularized) | ~0.01 (AdamW default) ❌ |
| 5 | patchhead (not regularized) | 0.0 ✅ |

**Impact**: DINO/iBOT projection heads receive ~25× less regularization than intended at peak schedule.

### Fix
```python
for i, param_group in enumerate(optimizer_student.param_groups):
    param_group["lr"] = student_lr_schedule[current_iteration]
    if i % 2 == 0:  # Even indices are regularized groups
        param_group["weight_decay"] = wd_schedule[current_iteration]
```

---

## 5. Patch Prototype Loss: Per-Sample Normalization

**Required when using variable mask ratios (0.1–0.5) with block masking.**

### Problem

With `clustering_mode='masked'` and variable mask ratios:
- Sample A (50% masked): contributes ~98 tokens to loss
- Sample B (10% masked): contributes ~20 tokens to loss

Current implementation uses per-token normalization (`/ M_total`), causing samples with higher mask ratios to dominate the gradient. This creates inconsistent training signal and loss magnitude fluctuation across batches.

### Current Implementation (`losses/prototype_loss.py`)

```python
M_total = student_norm_masked.shape[0]
clustering_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked) / M_total
```

### Required Change

Add inverse weighting so each sample contributes equally, matching iBOT loss behavior:

```python
# Compute per-token weights based on how many tokens each sample contributes
sample_idx = token_masks.nonzero(as_tuple=True)[0]  # Which sample each masked token belongs to
num_masked_per_sample = token_masks.sum(dim=1).clamp(min=1.0)  # [B]
weights = 1.0 / num_masked_per_sample[sample_idx]  # [M]

# Per-token cross-entropy
per_token_loss = -torch.sum(Q_tilde_masked * student_log_probs_masked, dim=-1)  # [M]

# Weighted sum, normalized by batch size
B = token_masks.shape[0]
clustering_loss = (per_token_loss * weights).sum() / B
```

### Impact

Ensures each sample in the batch contributes equally to the prototype clustering gradient regardless of its individual mask ratio, consistent with how iBOT patch loss handles variable masking.


## 6. iBOT Crop Selection Strategy

| Aspect | Official | This Implementation |
|--------|----------|---------------------|
| **Selection method** | Random at crop level (~50% of all crops) | Random per-image (exactly 1 crop per image) |
| **Semantic meaning** | ~50% of crops masked, ~25% images get both/neither masked | One randomly selected crop per image masked |
| **Index tracking** | Implicit via shuffle | Explicit `ibot_crop_indices` |
| **Forward passes** | All crops through teacher/student | Same (no extra passes) |
| **Patch proto reuse** | N/A | Same indices used for consistency |

### Official Implementation

The official DINOv2 randomly masks crops at the batch level:
```python
# dinov2/data/collate.py
B = len(collated_global_crops)  # e.g., 128 crops (64 images × 2)
n_samples_masked = int(B * mask_sample_probability)  # e.g., 64 crops
# ... generate masks for 64 randomly selected crops
# ... give empty masks to remaining 64 crops
random.shuffle(masks_list)
```

This results in a binomial distribution:
- ~25% of images: both crops masked
- ~50% of images: one crop masked
- ~25% of images: neither crop masked

### This Implementation

Select exactly one global crop per image for iBOT and patch prototype loss:
```python
def collate_data_and_cast(...):
    B = batch_size  # Number of images (not crops)
    
    # Randomly choose which global crop (0 or 1) to use per image
    ibot_crop_indices = torch.randint(0, 2, (B,))  # [1, 0, 1, 1, 0, ...]
    
    # Convert to indices in flattened crop batch
    # If we have 2 global crops per image: [crop0_img0, crop1_img0, crop0_img1, crop1_img1, ...]
    crop_indices_in_batch = torch.arange(B) * 2 + ibot_crop_indices
    
    # Generate masks ONLY for the selected crops
    masks = torch.zeros(2 * B, num_patches, dtype=torch.bool)
    for i, crop_idx in enumerate(crop_indices_in_batch):
        num_to_mask = random.randint(
            int(num_patches * mask_ratio_min),
            int(num_patches * mask_ratio_max)
        )
        masks[crop_idx] = mask_generator(num_to_mask)
    
    return {
        'global_crops': collated_global_crops,
        'masks': masks,
        'ibot_crop_indices': crop_indices_in_batch,
    }
```

### Usage in Training
```python
# Teacher forward (both crops for DINO)
teacher_out = teacher(global_crops)

# Student forward (masked selected crops)
student_out = student(global_crops, masks=masks)

# iBOT loss: use only selected crops
teacher_ibot_patches = teacher_out['x_norm_patchtokens'].view(B, 2, N, D)
teacher_ibot_patches = teacher_ibot_patches[torch.arange(B), ibot_crop_indices // 2]

student_ibot_patches = student_out['x_norm_patchtokens'].view(B, 2, N, D)
student_ibot_patches = student_ibot_patches[torch.arange(B), ibot_crop_indices // 2]

ibot_loss = compute_ibot(student_ibot_patches, teacher_ibot_patches, masks[ibot_crop_indices])

# Patch prototype loss: reuse same selection
patch_proto_loss = compute_patch_proto(
    student_ibot_patches,
    masks[ibot_crop_indices]
)
```

### Benefits

1. **Clearer semantics**: "One random crop per image" vs "~50% of crops randomly"
2. **Guaranteed coverage**: Every image participates in iBOT (not ~75%)
3. **Explicit tracking**: `ibot_crop_indices` makes selection transparent
4. **Code simplicity**: No shuffling, direct indexing
5. **Consistency**: Same crop used for both iBOT and patch prototype loss
6. **Same regularization**: Random crop choice per image provides equivalent dropout effect

### Impact

This approach provides equivalent regularization to the official implementation (50% dropout at crop level) while being more interpretable and easier to implement. The explicit indexing also enables reusing the same crop selection for multiple auxiliary losses without additional forward passes.


## References

- [Official DINOv2 Repository](https://github.com/facebookresearch/dinov2)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
