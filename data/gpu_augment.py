"""GPU-side DINOv2 multi-crop augmentation via kornia -- STREAM-THINNING ONLY.

Matches data/transforms.py::TMEDinoTransforms (the non-recipe branch, which every bc_* arm uses)
op-for-op, but runs batched on the GPU so it can be applied to ONLY the thinned survivors. In
thinned mode the loader emits the raw Resize(G) uint8 tile (the same tensor the scout is normalized
from); 5/6 of the over-drawn pool is discarded by density admission, so augmenting in the CPU
workers would waste ~6x the augmentation. Here we augment just the committed N tiles, on-GPU.

Input : raw uint8 [N, 3, G, G] (0-255), the Resize(G) output, any device.
Output: [g1, g2, l1, ..., lL] -- two global crops + L local crops, normalized float on the input
        device, in the exact [global1, global2, local...] order the trainer already extracts.

The weighted / off arms NEVER use this -- they keep the exact torchvision CPU pipeline unchanged
(byte-identical). This path is a deliberately DIFFERENT augmentation implementation (kornia RNG,
not torchvision), so thinned runs are not byte-comparable to weighted at the pixel level; they
target the same augmentation DISTRIBUTION (same crops/scales/jitter/blur/solarize probabilities).

Mirrors, from transforms.py (use_pathology_recipe=False):
  flip+color: RandomHorizontalFlip(0.5), RandomApply([ColorJitter(0.4,0.4,0.2,0.1)],0.8), Grayscale(0.2)
  global_1  : RRC(G, scale=(0.32,1.0))       + flip+color + GaussianBlur(9,(0.1,2.0)) p=1.0    + norm
  global_2  : RRC(G, scale=(0.32,1.0))       + flip+color + GaussianBlur p=0.1 + Solarize(0.5) p=0.2 + norm
  local     : RRC(L, scale=(0.05,0.32))      + flip+color + GaussianBlur p=0.5                  + norm
RandomResizedCrop default ratio (3/4, 4/3) matches torchvision's default (transforms.py passes no ratio).
"""

import torch
import torch.nn as nn
import kornia.augmentation as K


def _flip_color_gray():
    # RandomHorizontalFlip(0.5) -> RandomApply([ColorJitter(0.4,0.4,0.2,0.1)], 0.8) -> RandomGrayscale(0.2)
    return [
        K.RandomHorizontalFlip(p=0.5),
        K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        K.RandomGrayscale(p=0.2),
    ]


class GPUCropAugment(nn.Module):
    def __init__(self, global_size=224, local_size=96, n_local_crops=8,
                 global_scale=(0.32, 1.0), local_scale=(0.05, 0.32),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 # MUST match the dataloader normalization -- these are the PATHOLOGY mean/std that
                 # ProportionalMultiDatasetWrapper / MemoryEfficientShardedPathologyDataset default to
                 # (NOT ImageNet). The scout is normalized with the same constants (trainer
                 # _normalize_raw), so signatures stay consistent with the bank.
                 mean=(0.6816, 0.5640, 0.7232), std=(0.1617, 0.1714, 0.1389)):
        super().__init__()
        self.n_local_crops = n_local_crops
        norm = K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))

        # global_1: blur ALWAYS (torchvision GaussianBlur, not RandomApply -> p=1.0)
        self.global_1 = K.AugmentationSequential(
            K.RandomResizedCrop((global_size, global_size), scale=global_scale, ratio=ratio,
                                resample='bicubic'),
            *_flip_color_gray(),
            K.RandomGaussianBlur((9, 9), (0.1, 2.0), p=1.0),
            norm,
        )
        # global_2: blur p=0.1, then Solarize p=0.2 (threshold 128/255 ~ 0.5; thresholds=0.0 -> fixed 0.5)
        self.global_2 = K.AugmentationSequential(
            K.RandomResizedCrop((global_size, global_size), scale=global_scale, ratio=ratio,
                                resample='bicubic'),
            *_flip_color_gray(),
            K.RandomGaussianBlur((9, 9), (0.1, 2.0), p=0.1),
            K.RandomSolarize(thresholds=0.0, additions=0.0, p=0.2),
            norm,
        )
        # local: RRC to local_size, blur p=0.5
        self.local = K.AugmentationSequential(
            K.RandomResizedCrop((local_size, local_size), scale=local_scale, ratio=ratio,
                                resample='bicubic'),
            *_flip_color_gray(),
            K.RandomGaussianBlur((9, 9), (0.1, 2.0), p=0.5),
            norm,
        )

    @torch.no_grad()
    def forward(self, raw_u8):
        """raw_u8: uint8 [N,3,G,G] -> [g1, g2, l1..lL] normalized float on raw_u8.device."""
        x = raw_u8.float().div(255.0)                 # [0,1], same device
        crops = [self.global_1(x), self.global_2(x)]  # each re-samples independent per-sample params
        for _ in range(self.n_local_crops):
            crops.append(self.local(x))
        return crops
