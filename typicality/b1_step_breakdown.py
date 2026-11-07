# B1: step-time breakdown of the thinned v2 curation step (4 GPU benchmark, torchrun).
# Cluster command (4 H100, about 1-2 h for the full grid):
#   cd /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base \
#   && export TORCHINDUCTOR_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/inductor_cache \
#             TRITON_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/triton_cache \
#   && torchrun --standalone --nproc_per_node=4 typicality/b1_step_breakdown.py \
#        --checkpoint /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base/logs/checkpoint_iter_00050000.pth \
#        --out_dir /data1/vanderbc/test_dinov2_swaraj/bench
# Smoke (about 10 min): add  --pools 1,4 --precisions bf16,fp8_rowwise --steps 50
# Outputs: <out_dir>/b1_step_breakdown.csv and .json. Nothing in the home directory.
"""B1: step-time breakdown of the thinned v2 curation step, per pool size and per scout precision.

Why: find which part of the curation cost fp8 reduces, and measure the loader rate so that
--keep_f_max can be set from it.

One benchmark step mirrors training/trainer.py (thinned mode, --keep_rule v2): draw `pool`
loader batches -> normalize -> chunked eval-mode scout forward (chunks of N) -> signatures ->
all_gather -> bank.score_and_update -> v2 keep step -> GPU augment of the N survivors -> the full
training step (teacher forward, student forward on the 10 crops with iBOT block masks, DINO +
KoLeo/KDE + iBOT loss, backward, sync_grads, clip, AdamW step, R update, EMA update). Every rank
runs the same collectives in the same order every step, like the trainer.

Timing: one torch.cuda.Event pair per phase (summed over chunks), read after a device sync at the
end of the step. loader_wait is host wall time around next(iterator). step_ms is the wall time of
the whole step and also covers host glue (pool cat, survivor gather) that no phase owns. The bank's
s sweep (every 500 steps) lands in some windows; bank_sweeps counts them, p50 is unaffected.
"""
import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from copy import deepcopy

import numpy as np
import torch
import torch._dynamo
import torch.backends.cudnn as cudnn
import torch.distributed as dist

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import utils  # noqa: E402
import models.fp8_scout as fp8_mod  # noqa: E402
from models import CombinedModelDINO, DINOHead, ModernViT  # noqa: E402
from models.fp8_scout import check_fp8_capability, fp8_scout, swap_student_linears  # noqa: E402
from models.vision_transformer.modern_vit import Mlp, SwiGLUFFNFused  # noqa: E402
from losses import DINOLoss, KDELoss, KoLeoLoss, iBOTPatchLoss  # noqa: E402
from data import ProportionalMultiDatasetWrapper  # noqa: E402
from data.gpu_augment_fast import FastCropAugment  # noqa: E402
from training.hand_rolled_dp import HandRolledDP  # noqa: E402
from training.helpers import generate_block_masks, worker_init_fn  # noqa: E402
from training.scout_mode import scout_eval_mode  # noqa: E402
from typicality import RepresentativePrototypes  # noqa: E402
from typicality.counted_coverage_bank_fast import FastCountedCoverageBank  # noqa: E402
from typicality.keep_rule_v2 import f_update_v2, keep_chance_v2  # noqa: E402

PHASES = ('loader_wait', 'normalize', 'scout_forward', 'all_gather', 'bank', 'keep',
          'augment', 'train', 'step')
PRECISIONS = ('fp32', 'bf16', 'fp8_rowwise', 'fp8_tensorwise')
COMPILE_MODES = ('eager', 'compiled')
# The rev13 thinned-arm bank settings (run_with_submitit.py; s/j knobs at their defaults).
ARM_BANK = dict(M=8192, K_prime=256, pool_j=64, halflife_steps=250, reserve_residency=300,
                reserve_size=550, s_buffer_size=60000, s_sweep_interval=500, s_grid_points=9,
                grid_span=(0.3, 2.0), s_ema_alpha=0.2, s_min_buffer=60000, s_headroom=0.0,
                graduation_hits=2, pool_selftune=True, pool_rse_target=0.05, pool_max=256,
                pool_ema=0.2, radius_mult=1.5)
KEEP_M_EMA, KEEP_F_MARGIN = 0.99, 1.1          # configs/config.py defaults for f_update_v2


def parse_args():
    p = argparse.ArgumentParser(description="B1 step-time breakdown (4 GPU, torchrun)")
    p.add_argument('--checkpoint', required=True, help="trainer checkpoint (.pth) with saved args")
    p.add_argument('--pools', default='1,2,4,8,16', help="pool sizes in loader batches")
    p.add_argument('--precisions', default=','.join(PRECISIONS))
    p.add_argument('--compile', dest='compile_modes', default=','.join(COMPILE_MODES))
    p.add_argument('--steps', type=int, default=200, help="timed steps per configuration")
    p.add_argument('--warmup', type=int, default=20, help="untimed steps per configuration")
    p.add_argument('--out_dir', default='/data1/vanderbc/test_dinov2_swaraj/bench')
    p.add_argument('--keep_u0', type=float, default=0.05)
    p.add_argument('--typicality_a', type=float, default=1.0)
    p.add_argument('--batch_size', type=int, default=256, help="overrides batch_size_per_gpu")
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--prefill_cap', type=int, default=2000, help="max scout steps to fill a fresh bank")
    p.add_argument('--dist_url', default='env://')
    a = p.parse_args()
    a.pools = [int(x) for x in a.pools.split(',') if x]
    a.precisions = [x.strip() for x in a.precisions.split(',') if x.strip()]
    a.compile_modes = [x.strip() for x in a.compile_modes.split(',') if x.strip()]
    assert set(a.precisions) <= set(PRECISIONS), "precisions must be among %s" % (PRECISIONS,)
    assert set(a.compile_modes) <= set(COMPILE_MODES), "compile modes must be among %s" % (COMPILE_MODES,)
    assert all(q >= 1 for q in a.pools), "pool sizes must be >= 1"
    return a


# ----------------------------------------------------------------- distributed helpers
def _dist_on():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_gather_signatures(s_local):
    """[B_local, K'] -> [world*B_local, K'] in rank order (identical on every rank)."""
    if not _dist_on():
        return s_local
    gathered = [torch.empty_like(s_local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, s_local.contiguous())
    return torch.cat(gathered, dim=0)


def _local_rows(x_global, b_local):
    if not _dist_on():
        return x_global
    r = dist.get_rank()
    return x_global[r * b_local:(r + 1) * b_local]


def _reduce(value, op):
    """One float, reduced over ranks: 'mean' or 'min'. Identity on one rank."""
    if not _dist_on():
        return float(value)
    t = torch.tensor([float(value)], dtype=torch.float64, device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MIN if op == 'min' else dist.ReduceOp.SUM)
    return float(t.item()) / (1 if op == 'min' else dist.get_world_size())


class PhaseTimer:
    """CUDA event pairs. A phase may be entered several times per step; collect() sums them."""

    def __init__(self):
        self.pairs = []

    @contextlib.contextmanager
    def phase(self, name):
        a = torch.cuda.Event(enable_timing=True)
        a.record()
        try:
            yield
        finally:
            b = torch.cuda.Event(enable_timing=True)
            b.record()
            self.pairs.append((name, a, b))

    def collect(self):
        torch.cuda.synchronize()
        out = {}
        for name, a, b in self.pairs:
            out[name] = out.get(name, 0.0) + a.elapsed_time(b)
        self.pairs = []
        return out


# ----------------------------------------------------------------- construction (as the trainer)
def resolve_recipe(a):
    """The trainer's in-place recipe resolution. The saved args already carry it; this is idempotent."""
    recipe = bool(getattr(a, 'use_pathology_recipe', False))
    auto_gate = recipe and a.embeddingdim >= 1280
    if recipe:
        a.patch_size = 14
        if abs(a.koleo_loss_weight - 0.1) < 1e-6:
            a.koleo_loss_weight = 0.05
    if auto_gate:
        a.num_register_tokens = max(a.num_register_tokens, 8)
        a.out_dim = 131072
    if getattr(a, 'qk_norm', None) is None:
        a.qk_norm = True
    return auto_gate


def build_models(a):
    ls_init = None if getattr(a, 'layerscale_schedule', 'uniform') == 'cait' else a.layerscale_init
    mlp_cls = Mlp if getattr(a, 'ffn_type', 'swiglu') == 'mlp' else SwiGLUFFNFused
    enc = ModernViT(img_size=224, patch_size=a.patch_size, embed_dim=a.embeddingdim,
                    depth=a.vitdepth, num_heads=a.vitheads, mlp_ratio=4.0, qkv_bias=True,
                    qk_norm=bool(a.qk_norm), dual_norm=False, drop_path_rate=a.drop_path_rate,
                    drop_path_uniform=a.drop_path_uniform, pre_norm=False,
                    num_register_tokens=a.num_register_tokens, layerscale_init=ls_init,
                    mlp_layer=mlp_cls)
    t_enc = deepcopy(enc)

    def head(student_side):
        # The teacher heads keep DINOHead's default norm_last_layer=True, as in the trainer.
        kw = {'norm_last_layer': a.norm_last_layer} if student_side else {}
        return DINOHead(a.embeddingdim, a.out_dim, use_bn=a.use_bn_in_head, **kw)

    student = CombinedModelDINO(backbone=enc, classhead=head(True), patchhead=head(True),
                                num_masks=a.num_masks, patch_size=a.patch_size)
    teacher = CombinedModelDINO(backbone=t_enc, classhead=head(False), patchhead=head(False),
                                num_masks=a.num_masks, patch_size=a.patch_size)
    return student, teacher


def load_wrapped(model, sd, name):
    """Load a state dict into a HandRolledDP model (its keys carry the 'module.' prefix)."""
    fixed = {}
    for k, v in sd.items():
        fixed[k if k.startswith('module.') else 'module.' + k] = v
    msg = model.load_state_dict(fixed, strict=False)
    print("loaded %s: missing=%d unexpected=%d %s"
          % (name, len(msg.missing_keys), len(msg.unexpected_keys), msg.missing_keys[:5]))


def schedules_at(a, it, world, n):
    """(lr, weight decay, teacher momentum) the trainer would use at iteration `it`."""
    scale = math.sqrt(n * world / 1024.0)
    total = int(a.total_iterations)
    i = min(max(int(it), 0), total - 1)
    if bool(getattr(a, 'continual', False)):
        lr_plateau = (a.lr if a.lr_continual is None else a.lr_continual) * scale
        warm = a.warmup_iterations if a.warmup_iterations_continual is None else a.warmup_iterations_continual
        lr = lr_plateau * min(1.0, (i + 1) / float(max(warm, 1)))
        return lr, float(a.weight_decay_continual), float(a.momentum_teacher_continual)
    lr = utils.cosine_scheduler(a.lr * scale, a.min_lr, total, a.warmup_iterations, 0)[i]
    wd = utils.cosine_scheduler(a.weight_decay, a.weight_decay_end, total, a.warmup_iterations,
                                a.weight_decay)[i]
    m = utils.cosine_scheduler(a.momentum_teacher, 1.0, total, 0, a.momentum_teacher)[i]
    return float(lr), float(wd), float(m)


def build_loader(a, rank, world):
    cfgs = []
    for src in a.dataset_sources:
        name, base_dir, index_file = src.split(':')
        cfgs.append({'name': name, 'base_dir': base_dir, 'index_file': index_file})
    ds = ProportionalMultiDatasetWrapper(
        dataset_configs=cfgs, batch_size_per_gpu=a.batch_size_per_gpu,
        n_standard_local_crops=a.n_standard_local_crops, global_views=a.global_views,
        local_crop_size=a.local_crop_size, worker_id=0, num_workers=a.num_workers,
        rank=rank, world_size=world, seed=a.seed, global_size=224,
        use_pathology_recipe=getattr(a, 'use_pathology_recipe', False),
        ect_probability=getattr(a, 'ect_probability', 0.4),
        emit_scout=False, scout_pool_mode=True)
    return torch.utils.data.DataLoader(
        ds, batch_size=a.batch_size_per_gpu, num_workers=a.num_workers, drop_last=True,
        pin_memory=True, persistent_workers=a.num_workers > 0, worker_init_fn=worker_init_fn)


def set_compile(backbones, mode):
    """'compiled': per-block torch.compile as the trainer. 'eager': drop the compiled wrappers."""
    for bb in backbones:
        for blk in bb.blocks:
            if mode == 'compiled':
                if 'forward' not in blk.__dict__:
                    blk.forward = torch.compile(blk.forward, dynamic=False)
            elif 'forward' in blk.__dict__:
                del blk.forward


# ----------------------------------------------------------------- the benchmark
class Bench(object):
    def __init__(self, args):
        self.args = args
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        a = ck['args']
        a.batch_size_per_gpu = args.batch_size
        a.num_workers = args.num_workers
        a.dist_url = args.dist_url
        self.auto_gate = resolve_recipe(a)
        self.ck_args = a
        self.N = int(a.batch_size_per_gpu)
        self.it0 = int(ck.get('iteration', 0))
        self.step = self.it0
        self.world = utils.get_world_size()
        self.rank = utils.get_rank()
        for flag in ('use_adversarial_mask_augmentation', 'use_cellvit_augmentation',
                     'use_random_mask_augmentation', 'use_semantic_ibot'):
            assert not getattr(a, flag, False), "%s is on; the benchmark builds only the 2+L views" % flag
        self.timer = PhaseTimer()
        self.m_ema = None

        student, teacher = build_models(a)
        student = student.cuda()
        teacher = teacher.cuda()
        # One model serves every precision: the swap is inert unless inside fp8_scout().
        self.linears = swap_student_linears(student.backbone)
        if getattr(a, 'grad_checkpointing', False):
            student.set_grad_checkpointing(True)
        self.student = HandRolledDP(student)
        self.teacher = HandRolledDP(teacher, broadcast=False)
        load_wrapped(self.student, ck['student'], 'student')
        load_wrapped(self.teacher, ck['teacher'], 'teacher')
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.backbones = [self.student.module.backbone, self.teacher.module.backbone]

        groups = utils.get_params_groups_with_layer_decay(
            self.student.module.backbone, lr_decay_rate=a.lr_decay_rate, num_layers=a.vitdepth,
            patch_embed_lr_mult=a.patch_embed_lr_mult)
        groups += utils.get_params_groups_with_decay_for_heads(self.student.module.classhead)
        groups += utils.get_params_groups_with_decay_for_heads(self.student.module.patchhead)
        self.optimizer = (utils.StableAdamW(groups, betas=(0.9, 0.95)) if self.auto_gate
                          else torch.optim.AdamW(groups))
        lr, wd, self.momentum = schedules_at(a, self.it0, self.world, self.N)
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_multiplier', 1.0)
            if pg.get('wd_multiplier', 1.0) > 0:
                pg['weight_decay'] = wd * pg.get('wd_multiplier', 1.0)
        print("schedules at iteration %d: lr=%.3e wd=%.4f momentum=%.5f" % (self.it0, lr, wd, self.momentum))

        self.repr_protos = RepresentativePrototypes(K_prime=ARM_BANK['K_prime'], bottleneck_dim=256).cuda()
        if 'repr_protos' in ck:
            self.repr_protos.load_state_dict(ck['repr_protos'])
            print("loaded repr_protos from the checkpoint")
        self.r_opt = torch.optim.AdamW(self.repr_protos.parameters(), lr=a.typicality_repr_lr, weight_decay=0.0)

        self.bank = FastCountedCoverageBank(**ARM_BANK).cuda()
        sd = ck.get('typicality_bank')
        self.bank_loaded = sd is not None and tuple(sd['est_b'].shape) == tuple(self.bank.est_b.shape)
        if self.bank_loaded:
            self.bank.load_state_dict(sd)
            print("loaded typicality_bank: n_est=%d/%d reserve=%d s=%.4f"
                  % (self.bank._ne(), self.bank.M, self.bank._nr(), float(self.bank.s.item())))
        elif sd is not None:
            print("checkpoint bank est_b %s differs from the arm bank; starting fresh" % (tuple(sd['est_b'].shape),))
        del ck

        self.loader = build_loader(a, self.rank, self.world)
        self.it = iter(self.loader)
        self.gpu_augment = FastCropAugment(global_size=224, local_size=a.local_crop_size,
                                           n_local_crops=a.n_standard_local_crops).cuda()
        self.raw_mean = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1).cuda()
        self.raw_std = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1).cuda()

        n_views = a.global_views + a.n_standard_local_crops
        t_warm, t_final = (0.04, 0.04) if self.auto_gate else (a.warmup_teacher_temp, a.teacher_temp)
        self.dino_loss = DINOLoss(ncrops=n_views, warmup_teacher_temp=t_warm, teacher_temp=t_final,
                                  warmup_teacher_temp_iters=a.teacher_temp_warmup_iters,
                                  n_iterations=5, student_temp=0.1).cuda()
        self.ibot_loss = iBOTPatchLoss(student_temp=0.1, n_iterations=3).cuda()
        recipe = bool(getattr(a, 'use_pathology_recipe', False))
        self.reg_loss = (KDELoss(kappa=a.kde_kappa) if recipe else KoLeoLoss()).cuda()
        self.amp = bool(getattr(a, 'use_fp16', True) or recipe)

    # ------------------------------------------------------------- pieces of one step
    def draw(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)

    def normalize_raw(self, raw_u8):
        x = raw_u8.to('cuda', non_blocking=True).float().div_(255.0)
        return (x - self.raw_mean) / self.raw_std

    def scout_forward(self, pool_raw, precision):
        """Raw uint8 pool (CPU) -> [pool_n, K'] fp32 signatures. Chunks of N, eval mode,
        autocast bf16 unless fp32, fp8 linears inside fp8_scout() for the two fp8 precisions."""
        N = self.N
        pool_n = int(pool_raw.shape[0])
        use_bf16 = precision != 'fp32'
        use_fp8 = precision.startswith('fp8_')
        old_kernel = fp8_mod.FP8_KERNEL
        if use_fp8:
            fp8_mod.FP8_KERNEL = precision[len('fp8_'):]
        fp8_ctx = fp8_scout(self.linears) if use_fp8 else contextlib.nullcontext()
        sigs = []
        try:
            with torch.no_grad(), scout_eval_mode(self.student), fp8_ctx:
                for c0 in range(0, pool_n, N):
                    with self.timer.phase('normalize'):
                        x = self.normalize_raw(pool_raw[c0:c0 + N])
                    with self.timer.phase('scout_forward'):
                        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                            so = self.student([x], token_masks=[None], mode='dino', bottleneck_only=True)
                        # fp32 signature matmul, outside autocast, as in the trainer.
                        z = so['bottleneck'].detach().float()
                        sigs.append(self.repr_protos.compute_signatures(z).detach())
                    del so, x
        finally:
            fp8_mod.FP8_KERNEL = old_kernel
        return torch.cat(sigs, dim=0)

    def keep_step(self, p_gathered, pool_n):
        """The trainer's v2 keep step. Returns the CPU index of the N survivors."""
        N = self.N
        if p_gathered is None:
            return torch.arange(min(N, pool_n))
        q_g, _c, _guard, _sure, _zero, m = keep_chance_v2(p_gathered, self.args.typicality_a, self.args.keep_u0)
        q = _local_rows(q_g, pool_n)
        accepted = torch.nonzero(torch.rand_like(q) < q, as_tuple=False).flatten()
        if accepted.numel() >= N:
            keep_idx = accepted[:N]
        else:
            rej = torch.ones(pool_n, dtype=torch.bool, device=q.device)
            rej[accepted] = False
            rej_idx = torch.nonzero(rej, as_tuple=False).flatten()
            order = torch.argsort(q[rej_idx], descending=True)
            keep_idx = torch.cat([accepted, rej_idx[order[:N - accepted.numel()]]])
        # The trainer's pool-size update and its cross-rank MAX. The pool is fixed here; the value is unused.
        self.m_ema, _raw, _next, n_next, _capped = f_update_v2(
            m, self.m_ema, KEEP_M_EMA, KEEP_F_MARGIN, self.args.keep_u0, 0.0)
        if _dist_on():
            t = torch.tensor([int(n_next)], dtype=torch.int64, device='cuda')
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return keep_idx.cpu()

    def ibot_term(self, s_tokens, t_tokens, mask, mask_weight, temp):
        Bn, Nn, D = s_tokens.shape
        idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return torch.zeros((), device=s_tokens.device)
        weights = mask_weight[idx // Nn]
        with torch.no_grad():
            t_proj = self.teacher.module.patchhead(t_tokens.reshape(-1, D)[idx])
        s_proj = self.student.module.patchhead(s_tokens.reshape(-1, D)[idx])
        return self.ibot_loss.forward_gathered(s_proj, t_proj, weights, Bn, temp)

    def train_step(self, crops, step):
        a = self.ck_args
        g = [crops[0], crops[1]]
        locs = list(crops[2:])
        B = int(g[0].shape[0])
        nph = 224 // a.patch_size

        def masks():
            return generate_block_masks(B, nph, nph, mask_ratio_min=a.mask_ratio_min,
                                        mask_ratio_max=a.mask_ratio_max,
                                        mask_sample_probability=a.mask_sample_probability,
                                        device=g[0].device)

        bm1, w1 = masks()
        bm2, w2 = masks()
        self.optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.amp):
            with torch.no_grad():
                to = self.teacher(g, token_masks=[None, None], mode='dino')
            so = self.student(g + locs, token_masks=[bm1, bm2] + [None] * len(locs), mode='dino',
                              return_bottleneck=True)
            dino = self.dino_loss(so['cls_outputs'], to['cls_outputs'], step)
            reg = sum(self.reg_loss(f['clstoken']) for f in so['features_list'][:2])
            temp = self.dino_loss.teacher_temp_schedule(step)
            ibot = (self.ibot_term(so['features_list'][0]['patchtokens'], to['features_list'][0]['patchtokens'], bm1, w1, temp)
                    + self.ibot_term(so['features_list'][1]['patchtokens'], to['features_list'][1]['patchtokens'], bm2, w2, temp)) / 2.0
        loss = dino + a.koleo_loss_weight * reg + a.ibot_loss_weight * ibot
        loss.backward()
        self.student.sync_grads()
        if a.clip_grad:
            utils.clip_gradients(self.student, a.clip_grad)
        self.optimizer.step()
        # Representative prototypes (R) update, as the trainer does every step.
        P = self.student.module.classhead.last_layer.weight.detach()
        r_loss, _nn, _cov = self.repr_protos.compute_loss(P)
        self.r_opt.zero_grad()
        r_loss.backward()
        self.r_opt.step()
        self.repr_protos.project_to_sphere()
        with torch.no_grad():
            m = self.momentum
            for pq, pk in zip(self.student.module.parameters(), self.teacher.module.parameters()):
                pk.data.mul_(m).add_((1 - m) * pq.detach().data)

    def run_step(self, pool, precision):
        """One full step. Returns the phase times (ms), the loader rate and the sweep flag."""
        step = self.step
        self.step += 1
        sweep_before = int(self.bank.last_sweep_step.item())
        t_step = time.perf_counter()
        t0 = time.perf_counter()
        batches = [self.draw() for _ in range(pool)]
        loader_wait = time.perf_counter() - t0
        pool_raw = batches[0][0] if pool == 1 else torch.cat([b[0] for b in batches], dim=0)
        pool_n = int(pool_raw.shape[0])
        sig = self.scout_forward(pool_raw, precision)
        with self.timer.phase('all_gather'):
            sg = _all_gather_signatures(sig)
        with self.timer.phase('bank'):
            out = self.bank.score_and_update(sg, step)
        with self.timer.phase('keep'):
            keep_idx = self.keep_step(out.get('p_hat'), pool_n)
            survivor_raw = pool_raw[keep_idx]
        with self.timer.phase('augment'):
            crops = self.gpu_augment(survivor_raw.to('cuda', non_blocking=True))
        with self.timer.phase('train'):
            self.train_step(crops, step)
        rec = self.timer.collect()
        rec['loader_wait'] = loader_wait * 1000.0
        rec['step'] = (time.perf_counter() - t_step) * 1000.0
        rec['loader_rate'] = pool * self.N / max(loader_wait, 1e-9)
        rec['sweep'] = 1.0 if int(self.bank.last_sweep_step.item()) != sweep_before else 0.0
        for ph in PHASES:
            rec.setdefault(ph, 0.0)
        return rec

    def prefill_bank(self, cap):
        """Scout steps (pool 1, bf16, eval) until the bank is full and matured, so score_and_update
        and the keep step run in their steady-state regime. Returns the number of steps."""
        n = 0
        while n < cap and not (self.bank._ne() == self.bank.M and self.bank._matured()):
            b = self.draw()
            with torch.no_grad():
                sg = _all_gather_signatures(self.scout_forward(b[0], 'bf16'))
                self.bank.score_and_update(sg, self.step)
            self.timer.collect()
            self.step += 1
            n += 1
            if n % 100 == 0:
                print("  prefill %d: n_est=%d/%d matured=%s" % (n, self.bank._ne(), self.bank.M, self.bank._matured()))
        print("bank prefill: %d steps, n_est=%d/%d reserve=%d matured=%s s=%.4f" % (
            n, self.bank._ne(), self.bank.M, self.bank._nr(), self.bank._matured(), float(self.bank.s.item())))
        return n

    def run_config(self, pool, precision, cmode):
        set_compile(self.backbones, cmode)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        recs = []
        for k in range(self.args.warmup + self.args.steps):
            rec = self.run_step(pool, precision)
            if k >= self.args.warmup:
                recs.append(rec)
        peak_mb = torch.cuda.max_memory_allocated() / float(2 ** 20)
        return self.summarize(recs, pool, precision, cmode, peak_mb)

    def summarize(self, recs, pool, precision, cmode, peak_mb):
        """Rank-0 statistics of this rank's samples plus the cross-rank reductions (every rank calls this)."""
        row = {'pool': pool, 'pool_tiles': pool * self.N, 'precision': precision, 'compile': cmode}
        for ph in PHASES:
            v = np.array([r[ph] for r in recs], dtype=np.float64)
            row[ph + '_mean_ms'] = float(v.mean())
            row[ph + '_p50_ms'] = float(np.percentile(v, 50))
            row[ph + '_p90_ms'] = float(np.percentile(v, 90))
        rate = np.array([r['loader_rate'] for r in recs], dtype=np.float64)
        row['loader_rate_tiles_per_s_mean'] = float(rate.mean())
        row['loader_wait_mean_over_ranks_ms'] = _reduce(row['loader_wait_mean_ms'], 'mean')
        row['scout_forward_mean_over_ranks_ms'] = _reduce(row['scout_forward_mean_ms'], 'mean')
        row['loader_rate_tiles_per_s_mean_over_ranks'] = _reduce(rate.mean(), 'mean')
        row['loader_rate_tiles_per_s_min_over_ranks'] = _reduce(rate.mean(), 'min')
        row['bank_sweeps'] = int(sum(r['sweep'] for r in recs))
        row['peak_mem_mb'] = float(peak_mb)
        row['n_steps'] = len(recs)
        return row


def print_table(rows):
    print("\nB1 step breakdown (p50 ms per step, rank 0; tiles/s = loader rate, mean over ranks)")
    print("%5s %15s %9s " % ('pool', 'precision', 'compile') + ' '.join('%11s' % c[:11] for c in PHASES)
          + " %10s %9s" % ('tiles/s', 'peak MB'))
    for r in rows:
        print("%5d %15s %9s " % (r['pool'], r['precision'], r['compile'])
              + ' '.join('%11.1f' % r[c + '_p50_ms'] for c in PHASES)
              + " %10.0f %9.0f" % (r['loader_rate_tiles_per_s_mean_over_ranks'], r['peak_mem_mb']))


def main():
    args = parse_args()
    utils.init_distributed_mode(args)
    cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 16        # torch._dynamo is imported at module scope

    bench = Bench(args)
    utils.fix_random_seeds(bench.ck_args.seed + bench.rank)
    kernels = sorted(set(p[len('fp8_'):] for p in args.precisions if p.startswith('fp8_')))
    for k in kernels:
        name, cap = check_fp8_capability(kernel=k)
        print("fp8 kernel '%s' ok on %s (capability %s)" % (k, name, cap))

    prefill_steps = bench.prefill_bank(args.prefill_cap)
    if not (bench.bank._ne() == bench.bank.M and bench.bank._matured()):
        print("WARNING: bank not in steady state after the prefill cap; bank/keep times are fill-regime times")

    rows = []
    t_all = time.perf_counter()
    for cmode in args.compile_modes:
        for precision in args.precisions:
            for pool in args.pools:
                t_cfg = time.perf_counter()
                row = bench.run_config(pool, precision, cmode)
                rows.append(row)
                print("[%s %s pool=%d] step p50 %.1f ms (scout %.1f bank %.1f keep %.1f train %.1f) "
                      "loader %.0f tiles/s peak %.0f MB [%.0f s]"
                      % (cmode, precision, pool, row['step_p50_ms'], row['scout_forward_p50_ms'],
                         row['bank_p50_ms'], row['keep_p50_ms'], row['train_p50_ms'],
                         row['loader_rate_tiles_per_s_mean_over_ranks'], row['peak_mem_mb'],
                         time.perf_counter() - t_cfg))

    if bench.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        csv_path = os.path.join(args.out_dir, 'b1_step_breakdown.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        env = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
               'world_size': bench.world, 'iteration': bench.it0, 'batch_size_per_gpu': bench.N,
               'bank_loaded': bench.bank_loaded, 'prefill_steps': prefill_steps, 'bank': dict(ARM_BANK),
               'args': vars(args), 'total_seconds': time.perf_counter() - t_all}
        with open(os.path.join(args.out_dir, 'b1_step_breakdown.json'), 'w') as f:
            json.dump({'env': env, 'rows': rows}, f, indent=1)
        print_table(rows)
        print("\nwrote %s and .json (%.0f s total)" % (csv_path, env['total_seconds']))
    if _dist_on():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
