# Cluster command (1 H100, about 15-40 min):
#   cd /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base \
#   && export TORCHINDUCTOR_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/inductor_cache \
#             TRITON_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/triton_cache \
#   && python typicality/b2_fp8_signature_agreement.py \
#        --checkpoint /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base/logs/checkpoint_iter_00050000.pth \
#        --out_dir /data1/vanderbc/test_dinov2_swaraj/bench/b2
# Outputs: <out_dir>/b2_fp8_signature_agreement.json and the printed tables. Nothing in the home directory.
"""B2: does the fp8 scout see the same tiles as the bf16 scout? One GPU, no training.

Rebuild the student from a checkpoint, fit or load R, fill a fresh cache from eval-mode bf16
scout signatures until the trainer's gate condition holds, then score the same eval pools five
ways: eval bf16, eval fp8 (rowwise and tensorwise), and train-mode bf16 twice (the v1 scout with
stochastic depth). Per pair: signature distance in units of the hit radius s, Spearman of p_hat,
and the share of keep decisions that flip under the same uniforms. Plus the v2 pool size f each
arm would start with. No pass or fail threshold. The user decides from the numbers.
"""
import argparse
import contextlib
import copy
import json
import math
import os
import platform
import socket
import sys
import time

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import models.fp8_scout as fp8_mod                                            # noqa: E402
from models import CombinedModelDINO, DINOHead, ModernViT                      # noqa: E402
from models.fp8_scout import check_fp8_capability, fp8_scout, swap_student_linears  # noqa: E402
from models.vision_transformer.modern_vit import Mlp, SwiGLUFFNFused           # noqa: E402
from data import ProportionalMultiDatasetWrapper                              # noqa: E402
from training.helpers import worker_init_fn                                   # noqa: E402
from training.scout_mode import scout_eval_mode                               # noqa: E402
from typicality.representative_prototypes import RepresentativePrototypes     # noqa: E402
from typicality.counted_coverage_bank_fast import FastCountedCoverageBank     # noqa: E402
from typicality.keep_rule_v2 import f_update_v2, keep_chance_v2               # noqa: E402

# The trainer's pathology normalization (training/trainer.py _normalize_raw).
RAW_MEAN = (0.6816, 0.5640, 0.7232)
RAW_STD = (0.1617, 0.1714, 0.1389)
# The arm's cache settings (the trainer's --typicality_* values for the curator v2 arm).
BANK_ARM = dict(M=8192, pool_j=64, halflife_steps=250, reserve_residency=300, reserve_size=550,
                s_buffer_size=60000, s_sweep_interval=500, s_grid_points=9, grid_span=(0.3, 2.0),
                s_ema_alpha=0.2, s_min_buffer=60000, s_headroom=0.0, graduation_hits=2,
                pool_selftune=True, pool_rse_target=0.05, pool_max=256, pool_ema=0.2)
GATE_K = 5                           # the trainer's THIN_GATE_K
A_GRID = (1.0, 1.5)
U0_GRID = (0.01, 0.05, 0.10)
M_EMA, F_MARGIN = 0.99, 1.1          # --keep_m_ema and --keep_f_margin defaults
WAYS = ('eval_bf16', 'eval_fp8_rowwise', 'eval_fp8_tensorwise', 'train_bf16_a', 'train_bf16_b')
PAIRS = (('eval_fp8_rowwise', 'eval_bf16'), ('eval_fp8_tensorwise', 'eval_bf16'),
         ('train_bf16_a', 'train_bf16_b'), ('eval_bf16', 'train_bf16_a'))
FORECAST_WAYS = ('eval_bf16', 'eval_fp8_rowwise', 'eval_fp8_tensorwise')


def parse_args():
    p = argparse.ArgumentParser('B2: fp8 vs bf16 scout signature agreement')
    p.add_argument('--checkpoint', required=True, help='any checkpoint with a student state dict')
    p.add_argument('--out_dir', default='/data1/vanderbc/test_dinov2_swaraj/bench/b2')
    p.add_argument('--r_steps', default=2000, type=int, help='R fit steps when the checkpoint has no R')
    p.add_argument('--eval_pools', default=20, type=int, help='pools scored after the fill')
    p.add_argument('--world', default=4, type=int, help='loader batches per pool (pool = world*N tiles)')
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--max_fill_steps', default=3000, type=int)
    return p.parse_args()


# ----------------------------------------------------------------------------- model
def build_student(args):
    """The trainer's student construction from the saved args (training/trainer.py)."""
    ls_init = None if getattr(args, 'layerscale_schedule', 'uniform') == 'cait' else args.layerscale_init
    mlp_cls = Mlp if getattr(args, 'ffn_type', 'swiglu') == 'mlp' else SwiGLUFFNFused
    qk_norm = getattr(args, 'qk_norm', None)
    qk_norm = True if qk_norm is None else bool(qk_norm)      # the trainer's rule when unset
    backbone = ModernViT(
        img_size=224, patch_size=args.patch_size, embed_dim=args.embeddingdim, depth=args.vitdepth,
        num_heads=args.vitheads, mlp_ratio=4.0, qkv_bias=True, qk_norm=qk_norm, dual_norm=False,
        drop_path_rate=args.drop_path_rate, drop_path_uniform=args.drop_path_uniform, pre_norm=False,
        num_register_tokens=args.num_register_tokens, layerscale_init=ls_init, mlp_layer=mlp_cls)
    classhead = DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head,
                         norm_last_layer=args.norm_last_layer)
    patchhead = DINOHead(args.embeddingdim, args.out_dim, use_bn=args.use_bn_in_head,
                         norm_last_layer=args.norm_last_layer)
    return CombinedModelDINO(backbone=backbone, classhead=classhead, patchhead=patchhead,
                             num_masks=args.num_masks, patch_size=args.patch_size)


def load_student(student, state):
    """Strict load. The trainer saves the DDP-wrapped student, so keys carry a module. prefix.
    A compiled checkpoint also carries ._orig_mod segments. Both are removed."""
    clean = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        clean[k.replace('._orig_mod', '')] = v
    student.load_state_dict(clean, strict=True)


def student_prototypes(student):
    """P = classhead.last_layer.weight, as the trainer reads it. last_layer is weight-normed and its
    .weight is only recomputed by a forward pre-hook, so one tiny forward refreshes it after the load."""
    ll = student.classhead.last_layer
    if hasattr(ll, 'weight_v'):
        with torch.no_grad():
            ll(torch.zeros(1, ll.weight_v.shape[1], device=ll.weight_v.device))
    return ll.weight.detach()


def build_or_fit_R(args, ckpt, student, r_steps):
    """Load R from the checkpoint when present. Otherwise fit a fresh R to the student prototypes
    with the trainer's loss and optimizer (training/trainer.py, representative prototype update)."""
    k_prime = int(getattr(args, 'typicality_K_prime', 256))
    repr_protos = RepresentativePrototypes(K_prime=k_prime, bottleneck_dim=256).cuda()
    P = student_prototypes(student)
    log = {'K_prime': k_prime, 'steps': []}

    def _rec(step, loss, l_nn, l_cov):
        rec = {'step': step, 'loss': float(loss), 'l_nn': float(l_nn), 'l_cov': float(l_cov)}
        log['steps'].append(rec)
        print('[R] step %5d  loss %.6f  l_nn %.6f  l_cov %.6f' % (step, rec['loss'], rec['l_nn'], rec['l_cov']))

    if 'repr_protos' in ckpt:
        repr_protos.load_state_dict(ckpt['repr_protos'])
        log['source'] = 'checkpoint'
        with torch.no_grad():
            _rec(-1, *repr_protos.compute_loss(P))
        return repr_protos, log
    log['source'] = 'fit'
    lr = float(getattr(args, 'typicality_repr_lr', 1e-3))
    opt = torch.optim.AdamW(repr_protos.parameters(), lr=lr, weight_decay=0.0)
    for step in range(r_steps):
        loss, l_nn, l_cov = repr_protos.compute_loss(P)
        opt.zero_grad()
        loss.backward()
        opt.step()
        repr_protos.project_to_sphere()
        if step == 0 or step == r_steps - 1 or step % 500 == 0:
            _rec(step, loss.detach(), l_nn.detach(), l_cov.detach())
    return repr_protos, log


def build_bank(args):
    k_prime = int(getattr(args, 'typicality_K_prime', 256))
    radius_mult = float(getattr(args, 'typicality_radius_mult', 1.5))
    return FastCountedCoverageBank(K_prime=k_prime, radius_mult=radius_mult, **BANK_ARM).cuda()


# ----------------------------------------------------------------------------- data
def build_loader(args, n_batch, seed):
    """The trainer's loader with scout_pool_mode on: one raw Resize(224) uint8 tile per sample."""
    configs = []
    for source in args.dataset_sources:
        name, base_dir, index_file = source.split(':')
        configs.append({'name': name, 'base_dir': base_dir, 'index_file': index_file})
    num_workers = int(args.num_workers)
    dataset = ProportionalMultiDatasetWrapper(
        dataset_configs=configs, batch_size_per_gpu=n_batch,
        n_standard_local_crops=args.n_standard_local_crops, global_views=args.global_views,
        local_crop_size=args.local_crop_size, worker_id=0, num_workers=num_workers, rank=0,
        world_size=1, seed=seed, global_size=224,
        use_pathology_recipe=getattr(args, 'use_pathology_recipe', False),
        ect_probability=getattr(args, 'ect_probability', 0.4),
        emit_scout=False, scout_pool_mode=True)
    return torch.utils.data.DataLoader(
        dataset, batch_size=n_batch, num_workers=num_workers, drop_last=True, pin_memory=True,
        persistent_workers=(num_workers > 0), worker_init_fn=worker_init_fn)


class PoolDrawer(object):
    """Draws world loader batches and concatenates them into one pool, as one trainer process
    would see after the all-gather. Restarts the iterator on StopIteration."""

    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)
        self.batches = 0

    def batch(self):
        try:
            b = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            b = next(self.it)
        self.batches += 1
        return b[0]                                          # [N,3,224,224] uint8, CPU

    def pool(self, world):
        return torch.cat([self.batch() for _ in range(world)], dim=0)


_MEAN = torch.tensor(RAW_MEAN).view(1, 3, 1, 1)
_STD = torch.tensor(RAW_STD).view(1, 3, 1, 1)


def normalize_raw(raw_u8):
    """Raw uint8 [B,3,224,224] -> normalized fp32 on the GPU (trainer _normalize_raw)."""
    x = raw_u8.to('cuda', non_blocking=True).float().div_(255.0)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


# ----------------------------------------------------------------------------- scout
@contextlib.contextmanager
def fp8_kernel(kernel):
    """Set the module constant models.fp8_scout.FP8_KERNEL for the block, restore after."""
    old = fp8_mod.FP8_KERNEL
    fp8_mod.FP8_KERNEL = kernel
    try:
        yield
    finally:
        fp8_mod.FP8_KERNEL = old


def pool_signatures(student, repr_protos, pool_u8, n_chunk, way, linears, seed):
    """One pool -> [pool_n, K'] fp32 signatures on the GPU, in pool order, in chunks of N.
    eval_*: eval mode (no stochastic depth). eval_fp8_*: also the fp8 block linears with the
    named kernel. train_bf16_a/b: train mode (the v1 scout), seeded before the pool.
    All ways run the forward under bf16 autocast. compute_signatures runs outside autocast on
    the fp32 bottleneck."""
    pool_n = int(pool_u8.shape[0])
    assert pool_n % n_chunk == 0, 'pool %d is not a multiple of N=%d' % (pool_n, n_chunk)
    chunks = []
    with contextlib.ExitStack() as st:
        st.enter_context(torch.no_grad())
        if way.startswith('eval'):
            st.enter_context(scout_eval_mode(student))
        else:
            student.train()
            torch.manual_seed(seed if way == 'train_bf16_a' else seed + 1)
        if way.startswith('eval_fp8_'):
            st.enter_context(fp8_kernel(way[len('eval_fp8_'):]))
            st.enter_context(fp8_scout(linears))
        for c0 in range(0, pool_n, n_chunk):
            x = normalize_raw(pool_u8[c0:c0 + n_chunk])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                so = student([x], token_masks=[None], mode='dino', bottleneck_only=True)
            z = so['bottleneck'].detach().float()
            chunks.append(repr_protos.compute_signatures(z).detach())
            del so, x
    return torch.cat(chunks, dim=0)


def gate_ready(bank):
    """The trainer's bank_ready condition (training/trainer.py, gate OPEN)."""
    return (float(bank.p_ref.item()) > 0.0 and bank._ne() == bank.M
            and bank._nr() == bank.reserve_cap)


def fill_cache(bank, drawer, sig_fn, world, max_steps):
    """Fill the cache from eval bf16 signatures until bank_ready holds GATE_K steps in a row."""
    log = []
    streak = 0
    step = 0
    t0 = time.time()
    while step < max_steps:
        sig = sig_fn(drawer.pool(world))
        out = bank.score_and_update(sig, step)
        streak = streak + 1 if gate_ready(bank) else 0
        rec = {'step': step, 'ready': bool(out['ready']), 'p_hat': out['p_hat'] is not None,
               'n_est': bank._ne(), 'n_reserve': bank._nr(), 's': float(bank.s.item()),
               'p_ref': float(bank.p_ref.item()), 'gate_streak': streak,
               'elapsed_s': round(time.time() - t0, 1)}
        log.append(rec)
        if step % 25 == 0 or streak in (1, GATE_K):
            print('[fill] step %4d  n_est %5d/%d  reserve %3d/%d  s %.4f  p_ref %.4g  streak %d  %.0fs'
                  % (step, rec['n_est'], bank.M, rec['n_reserve'], bank.reserve_cap, rec['s'],
                     rec['p_ref'], streak, rec['elapsed_s']))
        step += 1
        if streak >= GATE_K:
            break
    return log, step, streak >= GATE_K


def read_p_hat(bank, S):
    """p_hat without touching the cache: the readout runs on a deep copy (it updates p_ref
    on the copy only)."""
    return copy.deepcopy(bank)._counted_readout(S.float().to(bank.est_b.device)).detach().cpu()


# ----------------------------------------------------------------------------- metrics
def _ranks(x):
    """Rank by double argsort. Ties are broken by index (stable sort)."""
    return torch.argsort(torch.argsort(x, stable=True)).double()


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum()) / den if den > 0.0 else float('nan')


def keep_q(p, pool_n, a, u0):
    """Keep chance per tile. c is the u0-quantile of each pool of world*N tiles, as in one
    trainer step (the trainer applies keep_chance_v2 to the gathered pool of a step)."""
    qs = [keep_chance_v2(p[i:i + pool_n], a, u0)[0] for i in range(0, int(p.numel()), pool_n)]
    return torch.cat(qs)


def grid_key(a, u0):
    return 'a=%g,u0=%g' % (a, u0)


def p_hat_summary(p):
    return {'n': int(p.numel()), 'zero_share': float((p == 0).float().mean()),
            'median': float(p.median()), 'mean': float(p.mean()),
            'p05': float(torch.quantile(p, 0.05)), 'p95': float(torch.quantile(p, 0.95))}


def pair_report(SA, SB, pA, pB, s, uniforms, pool_n):
    d = (SA - SB).abs().sum(dim=1) / s
    rep = {'sig_l1_over_s': {'mean': float(d.mean()), 'p95': float(torch.quantile(d, 0.95)),
                             'max': float(d.max())},
           'spearman_p_hat': spearman(pA, pB), 'keep_changed': {}}
    for (a, u0), u in uniforms.items():
        qA = keep_q(pA, pool_n, a, u0)
        qB = keep_q(pB, pool_n, a, u0)
        rep['keep_changed'][grid_key(a, u0)] = float(((u < qA) != (u < qB)).float().mean())
    return rep


def forecast(p_by_way, pool_n, n_batch):
    """The v2 pool size each arm would start curation with: f = min(max(1.1/m, 1), 1/u0)."""
    rows = []
    for way in FORECAST_WAYS:
        p = p_by_way[way]
        zero = float((p == 0).float().mean())
        for a in A_GRID:
            for u0 in U0_GRID:
                m = float(keep_q(p, pool_n, a, u0).mean())
                _m_ema, f_raw, f, n_draws, capped = f_update_v2(m, None, M_EMA, F_MARGIN, u0)
                rows.append({'way': way, 'a': a, 'u0': u0, 'm': m, 'zero_share': zero,
                             'f_raw': f_raw, 'f': f, 'capped': bool(capped), 'n_draws': n_draws,
                             'pool_tiles_per_rank': n_draws * n_batch})
    return rows


# ----------------------------------------------------------------------------- printing
def print_pairs(pairs):
    keys = [grid_key(a, u0) for a in A_GRID for u0 in U0_GRID]
    print('\n== pair agreement (signature L1 / s; Spearman of p_hat; keep decisions changed) ==')
    print('%-38s %8s %8s %8s %9s | %s' % ('pair', 'L1/s mu', 'p95', 'max', 'spearman',
                                          ' '.join('%10s' % k for k in keys)))
    for name, rep in pairs.items():
        l1 = rep['sig_l1_over_s']
        print('%-38s %8.4f %8.4f %8.4f %9.4f | %s' % (
            name, l1['mean'], l1['p95'], l1['max'], rep['spearman_p_hat'],
            ' '.join('%10.4f' % rep['keep_changed'][k] for k in keys)))


def print_forecast(rows):
    print('\n== f forecast at the start of curation (f = min(max(1.1/m, 1), 1/u0)) ==')
    print('%-22s %5s %5s %8s %8s %8s %8s %6s %8s %12s' % (
        'way', 'a', 'u0', 'm', 'zero', 'f_raw', 'f', 'capped', 'n_draws', 'tiles/rank'))
    for r in rows:
        print('%-22s %5.2f %5.2f %8.4f %8.4f %8.3f %8.3f %6s %8d %12d' % (
            r['way'], r['a'], r['u0'], r['m'], r['zero_share'], r['f_raw'], r['f'],
            'yes' if r['capped'] else 'no', r['n_draws'], r['pool_tiles_per_rank']))


def environment(ckpt_path, bench, args, caps):
    sha = 'N/A'
    try:
        import utils
        sha = utils.get_sha()
    except Exception:
        pass
    keep = ('batch_size_per_gpu', 'num_workers', 'typicality_K_prime', 'typicality_radius_mult',
            'ffn_type', 'out_dim', 'embeddingdim', 'vitdepth', 'vitheads', 'patch_size',
            'drop_path_rate', 'drop_path_uniform', 'num_register_tokens', 'qk_norm',
            'scout_amp_bf16', 'keep_rule', 'balance_mode', 'fp8_scout', 'use_pathology_recipe')
    return {'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
            'host': socket.gethostname(), 'git': sha,
            'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
            'fp8_capability_check': {k: [v[0], list(v[1])] for k, v in caps.items()},
            'fp8_kernel_default': fp8_mod.FP8_KERNEL, 'checkpoint': ckpt_path,
            'bench_args': vars(bench),
            'saved_args': {k: getattr(args, k, None) for k in keep}}


# ----------------------------------------------------------------------------- main
def main():
    bench = parse_args()
    seed = int(bench.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(bench.out_dir, exist_ok=True)
    t_start = time.time()
    timing = {}

    caps = {}
    for kernel in ('rowwise', 'tensorwise'):
        caps[kernel] = check_fp8_capability(kernel)
        print('[fp8] kernel %-10s ok on %s (capability %s)' % (kernel, caps[kernel][0], caps[kernel][1]))

    ckpt = torch.load(bench.checkpoint, map_location='cpu', weights_only=False)
    args = ckpt['args']
    n_batch = int(args.batch_size_per_gpu)
    pool_n = int(bench.world) * n_batch
    print('[ckpt] %s  iteration %s  N=%d  pool=%d tiles  workers=%s'
          % (bench.checkpoint, ckpt.get('iteration'), n_batch, pool_n, args.num_workers))

    student = build_student(args).cuda()
    load_student(student, ckpt['student'])
    linears = swap_student_linears(student.backbone)
    print('[student] loaded strict; %d block linears swapped to ScoutFP8Linear' % len(linears))

    t0 = time.time()
    repr_protos, r_log = build_or_fit_R(args, ckpt, student, int(bench.r_steps))
    timing['r_fit_s'] = round(time.time() - t0, 1)
    print('[R] source=%s  (%.0fs)' % (r_log['source'], timing['r_fit_s']))

    bank = build_bank(args)
    loader = build_loader(args, n_batch, seed)
    drawer = PoolDrawer(loader)

    def sig_fill(pool):
        return pool_signatures(student, repr_protos, pool, n_batch, 'eval_bf16', linears, seed)

    t0 = time.time()
    fill_log, n_fill, gate_open = fill_cache(bank, drawer, sig_fill, int(bench.world), int(bench.max_fill_steps))
    timing['fill_s'] = round(time.time() - t0, 1)
    s = float(bank.s.item())
    fill_summary = {'steps': n_fill, 'gate_open': gate_open, 's': s, 'p_ref': float(bank.p_ref.item()),
                    'n_est': bank._ne(), 'n_reserve': bank._nr(), 'tiles_seen': n_fill * pool_n,
                    'graduations': int(bank.graduations.item()), 'j': int(bank.j.item())}
    print('[fill] done: %d steps (%d tiles), gate_open=%s, s=%.4f, p_ref=%.4g, n_est=%d, n_reserve=%d (%.0fs)'
          % (n_fill, fill_summary['tiles_seen'], gate_open, s, fill_summary['p_ref'],
             fill_summary['n_est'], fill_summary['n_reserve'], timing['fill_s']))
    if not gate_open:
        print('[fill] WARNING: the gate did not open within --max_fill_steps; the readout below uses the cache as is')
    if bank._ne() == 0 or s <= 0.0:
        raise RuntimeError('the cache has no established signatures or no hit radius; raise --max_fill_steps')

    # Eval pools: drawn once, kept on the CPU as uint8.
    t0 = time.time()
    pools = [drawer.pool(int(bench.world)) for _ in range(int(bench.eval_pools))]
    timing['eval_draw_s'] = round(time.time() - t0, 1)
    print('[eval] %d pools of %d tiles drawn (%.0fs)' % (len(pools), pool_n, timing['eval_draw_s']))

    sigs = {}
    p_hat = {}
    way_time = {}
    for way in WAYS:
        t0 = time.time()
        S_list, p_list = [], []
        for pool in pools:
            S = pool_signatures(student, repr_protos, pool, n_batch, way, linears, seed)
            p_list.append(read_p_hat(bank, S))
            S_list.append(S.cpu())
        sigs[way] = torch.cat(S_list, dim=0)
        p_hat[way] = torch.cat(p_list, dim=0)
        torch.cuda.synchronize()
        way_time[way] = round(time.time() - t0, 1)
        sm = p_hat_summary(p_hat[way])
        print('[way] %-20s zero %.4f  median %.4g  mean %.4g  (%.0fs)'
              % (way, sm['zero_share'], sm['median'], sm['mean'], way_time[way]))
    timing['ways_s'] = way_time

    n_total = int(p_hat['eval_bf16'].numel())
    uniforms = {}
    for i, (a, u0) in enumerate([(a, u0) for a in A_GRID for u0 in U0_GRID]):
        g = torch.Generator().manual_seed(seed * 1000 + i)
        uniforms[(a, u0)] = torch.rand(n_total, generator=g)

    pairs = {}
    for wa, wb in PAIRS:
        pairs['%s vs %s' % (wa, wb)] = pair_report(sigs[wa], sigs[wb], p_hat[wa], p_hat[wb], s, uniforms, pool_n)
    print_pairs(pairs)
    rows = forecast(p_hat, pool_n, n_batch)
    print_forecast(rows)

    timing['total_s'] = round(time.time() - t_start, 1)
    result = {
        'environment': environment(bench.checkpoint, bench, args, caps),
        'settings': {'N': n_batch, 'world': int(bench.world), 'pool_tiles': pool_n,
                     'eval_pools': int(bench.eval_pools), 'eval_tiles': n_total, 'seed': seed,
                     'bank': dict(BANK_ARM, grid_span=list(BANK_ARM['grid_span'])),
                     'gate_k': GATE_K, 'a_grid': list(A_GRID), 'u0_grid': list(U0_GRID),
                     'm_ema': M_EMA, 'f_margin': F_MARGIN, 'hit_radius_s': s},
        'r_fit': r_log,
        'fill': {'summary': fill_summary, 'log': fill_log},
        'p_hat_summary': {w: p_hat_summary(p_hat[w]) for w in WAYS},
        'p_hat': {w: [round(float(v), 7) for v in p_hat[w].tolist()] for w in WAYS},
        'pairs': pairs,
        'f_forecast': rows,
        'timing_s': timing,
    }
    out_path = os.path.join(bench.out_dir, 'b2_fp8_signature_agreement.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=1)
    print('\n[done] wrote %s  (%.0fs total)' % (out_path, timing['total_s']))


if __name__ == '__main__':
    main()
