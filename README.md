# DINOv2 Implementation Differences

This document outlines the differences between this implementation and the [official Meta DINOv2](https://github.com/facebookresearch/dinov2) codebase.

---

## 1. iBOT Masking Strategy

| Aspect | Official | This Implementation |
|--------|----------|---------------------|
| **Mask type** | Block masking via `MaskingGenerator` | Independent Bernoulli per token |
| **Mask ratio** | Variable per sample (`mask_ratio_min_max` tuple, e.g., 0.1–0.5) | Fixed ratio |
| **Mask probability** | `mask_sample_probability` — not all images masked | All images always masked |

**Official approach**:
```python
mask_generator = MaskingGenerator(
    input_size=(img_size // patch_size, img_size // patch_size),
    max_num_patches=0.5 * n_patches,
)
```

**This implementation**:
```python
token_masks = torch.bernoulli(
    torch.ones(batch_size, n_patches) * mask_ratio
).bool()
```

**Impact**: Block masking creates spatially coherent masked regions, forcing the model to learn larger spatial context. Bernoulli masking creates independent salt-and-pepper noise patterns.

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

## 5. Teacher Normalization

### DINO CLS Loss
Both implementations use **Sinkhorn-Knopp** normalization for teacher CLS tokens. ✅

(The official code supports "centering" mode for backward compatibility with DINO v1, but DINOv2 defaults to `centering="sinkhorn_knopp"`.)

### iBOT Patch Loss
**Official** uses centering (subtracting a running mean from teacher patch outputs):
```python
def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
    self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))

def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
    self.apply_center_update()
    return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)
```

**This implementation**: Sinkhorn-Knopp normalization for iBOT as well (no running mean centering).

---

## 6. Last Layer Freezing

**Official** sets last layer LR to exactly 0 during warmup:
```python
last_layer_lr_schedule.schedule[:freeze_epochs * epoch_length] = 0

# Applied via:
param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier
```

**This implementation** zeros gradients instead:
```python
def cancel_gradients_last_layer(iteration, module, freeze_iters):
    if iteration < freeze_iters:
        for n, p in module.named_parameters():
            if "last_layer" in n:
                p.grad = None
```

Both achieve the same effect.

---

## Summary Table

| Feature | Official | This Impl | Priority to Fix |
|---------|:--------:|:---------:|:---------------:|
| Block masking for iBOT | ✅ | ❌ | Medium |
| Variable mask ratio | ✅ | ❌ | Low |
| Mask sample probability | ✅ | ❌ | Low |
| Layer-wise LR decay | ✅ | ❌ | **High** |
| Patch embed LR multiplier | ✅ | ❌ | Medium |
| WD schedule all groups | ✅ | ❌ | **High** |
| DINO CLS: Sinkhorn-Knopp | ✅ | ✅ | — |
| iBOT centering | ✅ | ❌ | Low |

---

## References

- [Official DINOv2 Repository](https://github.com/facebookresearch/dinov2)
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
