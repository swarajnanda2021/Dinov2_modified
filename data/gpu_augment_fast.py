"""Hand-rolled GPU DINOv2 multi-crop augmentation -- faster drop-in for data/gpu_augment.py.

STATUS: staged but NOT wired in. training/trainer.py still imports GPUCropAugment from
data/gpu_augment.py (the kornia path). To adopt this, either import FastCropAugment here or
replace gpu_augment.py's class. It changes the thinned arm's augmentation RNG stream (own RNG,
not kornia), so it is a science-arm change to an in-flight experiment -- adopt deliberately.

v3 = v1 + EXACT torchvision HSV hue (v1 used a YIQ rotation). Everything else is v1:
same RNG draw order and count (DDP-safe), no in-place mutation of caller tensors,
unchanged constructor signature.

Same interface and same augmentation DISTRIBUTION as GPUCropAugment (kornia), but built from
raw tensor ops:

  * RandomResizedCrop  -> torchvision's get_params sampling (10 vectorized attempts, log-uniform
    aspect in (3/4,4/3), area U(scale)), realized as ONE batched F.grid_sample per crop group.
    All 8 local crops are sampled in a single call by stacking their grids along the output H
    axis (output [N,3,8*96,96]) -- so the 224^2 source is never duplicated 8x in memory.
  * RandomHorizontalFlip(0.5) -> folded into the sampling grid (free).
  * ColorJitter(0.4,0.4,0.2,0.1)@0.8 -> per-sample brightness/contrast/saturation as tensor ops;
    hue as an exact torchvision HSV shift. Op order is permuted once per call (tv permutes per sample).
  * RandomGrayscale(0.2) -> luma (0.299,0.587,0.114) blend.
  * GaussianBlur(k=9, sigma U(0.1,2.0)) -> separable depthwise conv with PER-SAMPLE kernels via
    groups=B*3 (one conv per axis, not per sample).
  * RandomSolarize(0.5)@0.2 -> where(x>=0.5, 1-x, x).
  * Normalize with the PATHOLOGY mean/std (must match _normalize_raw in the trainer).

Deviations from torchvision/kornia, deliberate and flagged:
  (a) hue is an exact torchvision HSV shift (0.0 max abs err vs tv adjust_hue).
  (b) ColorJitter op order is permuted per CALL, not per SAMPLE.
  (c) resampling is grid_sample bicubic without antialias -- exactly what kornia's RRC does too
      (torchvision's PIL path does antialias; the CPU/GPU arms were already non-identical).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rrc_params(B, H, W, scale, ratio, device, tries=10):
    """torchvision RandomResizedCrop.get_params, vectorized. Returns (i, j, h, w) float tensors."""
    area = float(H * W)
    log_lo, log_hi = torch.log(torch.tensor(ratio[0], device=device)), torch.log(torch.tensor(ratio[1], device=device))

    ta = area * (torch.rand(tries, B, device=device) * (scale[1] - scale[0]) + scale[0])
    ar = torch.exp(torch.rand(tries, B, device=device) * (log_hi - log_lo) + log_lo)
    w = torch.sqrt(ta * ar).round()
    h = torch.sqrt(ta / ar).round()
    ok = (w > 0) & (w <= W) & (h > 0) & (h <= H)

    # first successful attempt per sample
    idx = torch.where(ok.any(0), ok.float().argmax(0), torch.zeros(B, dtype=torch.long, device=device))
    ar_b = torch.arange(B, device=device)
    w_s, h_s, ok_s = w[idx, ar_b], h[idx, ar_b], ok.any(0)

    # torchvision fallback: clamp aspect to the range and center-crop
    in_ratio = W / H
    if in_ratio < ratio[0]:
        fw, fh = torch.tensor(float(W), device=device), torch.tensor(round(W / ratio[0]), device=device).float()
    elif in_ratio > ratio[1]:
        fh, fw = torch.tensor(float(H), device=device), torch.tensor(round(H * ratio[1]), device=device).float()
    else:
        fw, fh = torch.tensor(float(W), device=device), torch.tensor(float(H), device=device)
    w_s = torch.where(ok_s, w_s, fw.expand(B))
    h_s = torch.where(ok_s, h_s, fh.expand(B))

    i = (torch.rand(B, device=device) * (H - h_s + 1).clamp(min=1)).floor()
    j = (torch.rand(B, device=device) * (W - w_s + 1).clamp(min=1)).floor()
    i = torch.where(ok_s, i, ((H - h_s) / 2).round())
    j = torch.where(ok_s, j, ((W - w_s) / 2).round())
    return i, j, h_s, w_s


def _crop_grid(i, j, h, w, H, W, out, flip):
    """Affine sampling grid in normalized [-1,1] coords for crop (i,j,h,w) resized to out x out."""
    B = i.shape[0]
    device = i.device
    lin = (torch.arange(out, device=device, dtype=torch.float32) + 0.5) / out    # (0,1) pixel centers
    # source pixel coords, then to normalized align_corners=False coords
    ys = (i.view(B, 1) + lin.view(1, out) * h.view(B, 1))
    xs = (j.view(B, 1) + lin.view(1, out) * w.view(B, 1))
    ysn = ys / H * 2 - 1
    xsn = xs / W * 2 - 1
    xsn = torch.where(flip.view(B, 1), -xsn, xsn)
    grid = torch.empty(B, out, out, 2, device=device)
    grid[..., 0] = xsn.view(B, 1, out).expand(B, out, out)
    grid[..., 1] = ysn.view(B, out, 1).expand(B, out, out)
    return grid


def _multi_crop(x, n_crops, out, scale, ratio):
    """n_crops RRC crops of every image in x, in ONE grid_sample.
    x: [N,3,H,W] -> [n_crops*N, 3, out, out] (crop-major: crop0 for all N, then crop1, ...)."""
    N, _, H, W = x.shape
    dev = x.device
    grids = []
    for _ in range(n_crops):
        i, j, h, w = _rrc_params(N, H, W, scale, ratio, dev)
        flip = torch.rand(N, device=dev) < 0.5
        grids.append(_crop_grid(i, j, h, w, H, W, out, flip))
    # stack along the output H axis -> [N, n*out, out, 2]; one sample call, no input duplication
    grid = torch.cat(grids, dim=1)
    o = F.grid_sample(x, grid, mode='bicubic', padding_mode='reflection', align_corners=False)
    o = o.view(N, 3, n_crops, out, out).permute(2, 0, 1, 3, 4).reshape(n_crops * N, 3, out, out)
    return o.clamp_(0.0, 1.0)


def _rgb2hsv(img):
    """Exact port of torchvision.transforms._functional_tensor._rgb2hsv (batched)."""
    r, g, b = img.unbind(dim=-3)
    maxc = torch.max(img, dim=-3).values
    minc = torch.min(img, dim=-3).values

    eqc = maxc == minc
    cr = maxc - minc
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & ~(maxc == r)) * (2.0 + rc - bc)
    hb = (~(maxc == g) & ~(maxc == r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return h, s, maxc

def _hsv2rgb(h, s, v):
    """Exact port of torchvision _hsv2rgb, but the one-hot 6-way einsum is replaced by
    where-chains.  The einsum is sum_k onehot_k * a_k = 0 + ... + a_sel + ... + 0, which in IEEE
    float is exactly a_sel, so the selection is bit-identical while never allocating [.,3,6,H,W]."""
    h6 = h * 6.0
    i = torch.floor(h6)
    f = h6 - i
    i = i.to(torch.int32) % 6

    p = torch.clamp(v * (1.0 - s), 0.0, 1.0)
    q = torch.clamp(v * (1.0 - f * s), 0.0, 1.0)
    t = torch.clamp(v * (1.0 - (1.0 - f) * s), 0.0, 1.0)

    # r = [v,q,p,p,t,v][i]; g = [t,v,v,q,p,p][i]; b = [p,p,t,v,v,q][i]
    r = torch.where(i == 0, v, torch.where(i == 1, q, torch.where(i < 4, p, torch.where(i == 4, t, v))))
    g = torch.where(i == 0, t, torch.where(i < 3, v, torch.where(i == 3, q, p)))
    b = torch.where(i < 2, p, torch.where(i == 2, t, torch.where(i < 5, v, q)))
    return torch.stack((r, g, b), dim=-3)

def _adj_hue(x, f):
    """torchvision.transforms.functional.adjust_hue, float path, per-sample factor."""
    h, s, v = _rgb2hsv(x)
    h = (h + f.view(-1, 1, 1)) % 1.0
    return _hsv2rgb(h, s, v)




_LUMA = (0.299, 0.587, 0.114)


def _color_jitter(x, b=0.4, c=0.4, s=0.2, hue=0.1, p=0.8, order=None):
    B = x.shape[0]
    dev = x.device
    on = (torch.rand(B, 1, 1, 1, device=dev) < p).float()

    def _b(y):
        f = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * b
        f = 1 + (f - 1) * on
        return (y * f).clamp_(0, 1)

    def _c(y):
        f = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * c
        f = 1 + (f - 1) * on
        g = (y[:, 0:1] * _LUMA[0] + y[:, 1:2] * _LUMA[1] + y[:, 2:3] * _LUMA[2]).mean((2, 3), keepdim=True)
        return (g + (y - g) * f).clamp_(0, 1)

    def _s(y):
        f = 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * s
        f = 1 + (f - 1) * on
        g = y[:, 0:1] * _LUMA[0] + y[:, 1:2] * _LUMA[1] + y[:, 2:3] * _LUMA[2]
        return (g + (y - g) * f).clamp_(0, 1)

    def _h(y):
        # exact torchvision adjust_hue semantics (RGB->HSV, wrap H, HSV->RGB), verified to
        # 0.0 max abs error vs torchvision.transforms.functional.adjust_hue including the
        # achromatic case. Replaces v1's YIQ-rotation approximation.
        f = (torch.rand(B, device=dev) * 2 - 1) * hue * on.view(B)
        return _adj_hue(y, f)

    ops = [_b, _c, _s, _h]
    for k in (order if order is not None else torch.randperm(4).tolist()):
        x = ops[k](x)
    return x


def _grayscale(x, p=0.2):
    B = x.shape[0]
    g = x[:, 0:1] * _LUMA[0] + x[:, 1:2] * _LUMA[1] + x[:, 2:3] * _LUMA[2]
    m = (torch.rand(B, 1, 1, 1, device=x.device) < p).float()
    return x * (1 - m) + g * m


def _blur(x, k=9, smin=0.1, smax=2.0, p=1.0):
    """Separable depthwise 9x9 with PER-SAMPLE sigma, one grouped conv per axis."""
    B, C, H, W = x.shape
    dev = x.device
    sig = torch.rand(B, device=dev) * (smax - smin) + smin
    if p < 1.0:
        apply = torch.rand(B, device=dev) < p
        sig = torch.where(apply, sig, torch.full_like(sig, 1e-6))   # ~identity kernel
    r = k // 2
    t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    ker = torch.exp(-(t.view(1, k) ** 2) / (2 * sig.view(B, 1) ** 2))
    ker = ker / ker.sum(1, keepdim=True)                             # [B,k]
    ker = ker.view(B, 1, 1, k).repeat_interleave(C, dim=0)           # [B*C,1,1,k]
    y = x.reshape(1, B * C, H, W)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode='reflect'), ker, groups=B * C)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), ker.transpose(2, 3), groups=B * C)
    return y.reshape(B, C, H, W)


def _solarize(x, thr=0.5, p=0.2):
    B = x.shape[0]
    m = (torch.rand(B, 1, 1, 1, device=x.device) < p)
    return torch.where(m & (x >= thr), 1.0 - x, x)


class FastCropAugment(nn.Module):
    """Drop-in replacement for data/gpu_augment.py::GPUCropAugment."""

    def __init__(self, global_size=224, local_size=96, n_local_crops=8,
                 global_scale=(0.32, 1.0), local_scale=(0.05, 0.32),
                 ratio=(3.0 / 4.0, 4.0 / 3.0),
                 mean=(0.6816, 0.5640, 0.7232), std=(0.1617, 0.1714, 0.1389)):
        super().__init__()
        self.G, self.L, self.n_local = global_size, local_size, n_local_crops
        self.gs, self.ls, self.ratio = global_scale, local_scale, ratio
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def _norm(self, x):
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, raw_u8):
        x = raw_u8.float().div(255.0)
        N = x.shape[0]

        # ---- 2 global crops, one grid_sample ----
        g = _multi_crop(x, 2, self.G, self.gs, self.ratio)           # [2N,3,G,G]
        g = _color_jitter(g)
        g = _grayscale(g)
        g1, g2 = g[:N], g[N:]
        g1 = _blur(g1, p=1.0)                                        # global_1: blur ALWAYS
        g2 = _blur(g2, p=0.1)                                        # global_2: p=0.1
        g2 = _solarize(g2, p=0.2)                                    # then solarize p=0.2
        out = [self._norm(g1), self._norm(g2)]

        # ---- 8 local crops, one grid_sample ----
        l = _multi_crop(x, self.n_local, self.L, self.ls, self.ratio)   # [8N,3,L,L]
        l = _color_jitter(l)
        l = _grayscale(l)
        l = _blur(l, p=0.5)
        l = self._norm(l)
        for i in range(self.n_local):
            out.append(l[i * N:(i + 1) * N])
        return out
