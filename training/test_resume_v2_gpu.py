#!/usr/bin/env python
# Cluster command (1 to 4 GPUs; about 5-10 min):
#   cd /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base \
#   && export TORCHINDUCTOR_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/inductor_cache \
#             TRITON_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/triton_cache \
#   && python training/test_resume_v2_gpu.py --gpus 2 --work_dir /data1/vanderbc/test_dinov2_swaraj/bench/t3_resume
# Outputs: <work_dir>/t3_resume.json, the two run directories, the printed PASS/FAIL table. Nothing in the home directory.
"""GPU test of the keep_rule v2 checkpoint resume in training/trainer.py.

The real trainer runs twice on a small synthetic corpus. Run A runs K + n steps without a stop.
Run B stops after the checkpoint of iteration K and resumes from checkpoint.pth in a fresh
process. Each phase is one torchrun subprocess (train_dinov2 initialises the process group).
Pass --fp8 True for the fp8 scout (H100 only); the default runs the v2 scout in bf16.

The resume contract the checks are written against:
  1. Under --keep_rule v2 the checkpoint carries save_dict['thin_state'] with the gate, the
     ready streak, m_ema, f_next, f_raw, f_capped, n_draws_cur (pool of the next step),
     n_draws_used (pool of the saved step; None while the gate is closed) and iteration. The
     checkpoint of step i is written at the end of step i. On restart step i runs again.
  2. On resume the loader position is dataset_position - used, used = n_draws_used or 1, passed
     as set_resume_position(position * N * W, replay_passes=True). The re-run step draws `used`
     batches, later steps draw n_draws_cur. Gate, streak, m_ema and f_next continue as saved.
  3. RNG on resume: each rank seeds (seed + 1_000_003 * iteration + 7_919 * rank) % 2**32 and
     prints "[resume] rank r: seed s at iteration i". At iteration 0 the seed is seed + rank.
  4. The loader skip is per worker in samples while the DataLoader takes whole batches from the
     workers round-robin. Batch composition after a resume matches an uninterrupted run only
     when the draw count is a multiple of num_workers, so the stream check compares sorted
     multisets of tile hashes over a window, with one loader worker.

Run A and run B share the loader stream after K but not the RNG (contract 3), and the re-run
of step K updates m_ema once more than run A did. The m_ema / f_next tolerance (--tol) is
therefore a drift bound, not a bit-equality claim. Both runs use --continual True so that the
stop at K + 1 does not change run B's schedule (the cosine schedules depend on
--total_iterations; the constant ones do not).
"""
import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image

DEFAULT_WORK = '/data1/vanderbc/test_dinov2_swaraj/bench/t3_resume'
SAVE_FREQ = 10
TILE = 224
# (name, zips, tiles per zip). Zip counts are multiples of 8, so every worker of a 1, 2 or 4 GPU
# run with 2 workers holds the same number of zips. The loader's per-worker pass length,
# total_images // (world_size * num_workers), is then exact and the resume skip wraps exactly.
SOURCES = (('srcA', 8, 48), ('srcB', 16, 48), ('srcC', 24, 48))
THIN_KEYS = ('thin_gate_open', 'thin_ready_streak', 'thin_m_ema', 'thin_f_next', 'thin_f_raw',
             'thin_f_capped', 'thin_n_draws_cur', 'thin_n_draws_used', 'iteration')
GATE_RE = re.compile(r'\[thinned\] gate OPEN at iter (\d+)')
RESUME_RE = re.compile(r'\[resume\] rank (\d+): seed (\d+) at iteration (\d+)')


# ----------------------------------------------------------------------------- (1) seeds
def resume_seed(seed, iteration, rank):
    """Contract 3: the per-rank seed of a resumed run (iteration > 0)."""
    return (seed + 1_000_003 * iteration + 7_919 * rank) % (2 ** 32)


def rank_seed(seed, iteration, rank):
    """The seed a rank uses: seed + rank at iteration 0, resume_seed after a resume."""
    return seed + rank if iteration == 0 else resume_seed(seed, iteration, rank)


def test_resume_seeds():
    """CPU check of the seed formula. Returns the seeds of ranks 0..3 at K = 100."""
    seed, K = 42, 100
    seeds = [resume_seed(seed, K, r) for r in range(4)]
    assert len(set(seeds)) == 4, 'ranks 0..3 must get different seeds: %r' % seeds
    assert seeds == [resume_seed(seed, K, r) for r in range(4)], 'the formula must be repeatable'
    assert resume_seed(seed, K, 0) != resume_seed(seed, K + 1, 0), 'iterations K and K+1 must differ'
    assert resume_seed(seed, K, 0) != resume_seed(seed, K + 10, 0), 'iterations K and K+10 must differ'
    for it, r in ((10 ** 9, 0), (10 ** 9, 1023), (K, 7)):
        assert 0 <= resume_seed(seed, it, r) < 2 ** 32, 'seed out of the 32-bit range'
    assert rank_seed(seed, 0, 3) == seed + 3, 'iteration 0 must use seed + rank'
    assert rank_seed(seed, K, 3) == resume_seed(seed, K, 3), 'iteration K must use the resume seed'
    return {'seeds_rank0_3_at_K100': seeds}


# ----------------------------------------------------------------------------- (2) corpus
def make_tile(tile_id):
    """One 224x224 RGB tile. A smooth pattern from one of 12 families with a per-tile phase, and
    the tile id written as 24 pixels of the first row, so every tile has distinct pixels."""
    frng = np.random.RandomState(tile_id % 12)
    rng = np.random.RandomState(1000 + tile_id)
    y, x = np.mgrid[0:TILE, 0:TILE].astype(np.float32) / TILE
    img = np.zeros((TILE, TILE, 3), np.float32)
    for c in range(3):
        fx, fy = frng.uniform(1.0, 6.0, 2)
        ph = rng.uniform(0.0, 2.0 * np.pi)
        img[..., c] = 0.5 + 0.35 * np.sin(2 * np.pi * (fx * x + fy * y) + ph) * np.cos(np.pi * fy * x)
    arr = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bits = np.array([(tile_id >> b) & 1 for b in range(24)], np.uint8) * 255
    arr[0, :24, :] = bits[:, None]
    return Image.fromarray(arr, 'RGB')


def build_corpus(data_dir, rebuild):
    """Write the zips, the index pickles and the metadata pickles in the loader's format.
    Reused when the manifest matches. Returns the number of tiles."""
    manifest = os.path.join(data_dir, 'manifest.json')
    want = {'sources': [list(s) for s in SOURCES], 'tile': TILE, 'version': 1}
    if not rebuild and os.path.exists(manifest):
        with open(manifest) as f:
            if json.load(f) == want:
                return sum(z * p for _, z, p in SOURCES)
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir)
    tile_id = 0
    for name, n_zips, per_zip in SOURCES:
        base = os.path.join(data_dir, name)
        os.makedirs(base)
        index = []
        for z in range(n_zips):
            zpath = os.path.join(base, '%s_%03d.zip' % (name, z))
            names = []
            with zipfile.ZipFile(zpath, 'w') as zf:
                for _ in range(per_zip):
                    buf = io.BytesIO()
                    make_tile(tile_id).save(buf, format='WEBP', lossless=True, method=0)
                    nm = 'tile_%06d.webp' % tile_id
                    zf.writestr(nm, buf.getvalue())
                    names.append(nm)
                    tile_id += 1
            index.append((zpath, names))
        with open(os.path.join(base, name + '_index.pkl'), 'wb') as f:
            pickle.dump(index, f)
        meta = {'total_images': n_zips * per_zip, 'zip_files': [p for p, _ in index],
                'images_per_zip': [per_zip] * n_zips}
        with open(os.path.join(base, name + '_index_metadata.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    with open(manifest, 'w') as f:
        json.dump(want, f)
    return tile_id


def data_dir_of(cli):
    return os.path.join(cli.work_dir, 'data')


def dataset_sources(data_dir):
    return ['%s:%s:%s_index.pkl' % (n, os.path.join(data_dir, n), n) for n, _, _ in SOURCES]


def dataset_configs(data_dir):
    return [{'name': n, 'base_dir': os.path.join(data_dir, n), 'index_file': n + '_index.pkl'}
            for n, _, _ in SOURCES]


# ----------------------------------------------------------------------------- (3) training
def train_args(cli):
    """The trainer args of both runs: a tiny ViT, keep_rule v2, a small cache so the gate opens
    within a few dozen steps, every optional feature off."""
    from configs import get_args_parser
    a = get_args_parser().parse_args([])
    a.dataset_sources = dataset_sources(data_dir_of(cli))
    a.output_dir = cli.output_dir
    a.total_iterations = int(cli.total_iterations)
    a.seed = cli.seed
    a.num_workers = cli.num_workers
    a.batch_size_per_gpu = cli.batch_size
    a.save_checkpoint_freq = SAVE_FREQ
    a.save_iter_checkpoint_freq = 0          # checkpoint_iter_*.pth at every save (the comparison needs them)
    a.warmup_iterations = 2
    a.continual = True
    a.freeze_last_layer_iters = 0
    a.visualization_freq = 10 ** 9
    a.use_fp16 = True
    a.dist_url = 'env://'
    a.patch_size, a.embeddingdim, a.vitdepth, a.vitheads, a.out_dim = 16, 192, 4, 3, 1024
    a.ffn_type, a.drop_path_rate, a.drop_path_uniform, a.qk_norm = 'mlp', 0.1, True, True
    a.global_views, a.n_standard_local_crops, a.local_crop_size = 2, 2, 96
    for k in ('use_prototype_clustering', 'use_semantic_ibot', 'use_semantic_prototypes',
              'use_adversarial_mask_augmentation', 'use_cellvit_augmentation',
              'use_random_mask_augmentation', 'use_pathology_recipe', 'compile_blocks',
              'grad_checkpointing', 'use_gpu_augmentation', 'thin_richardson_correct'):
        setattr(a, k, False)
    a.use_typicality_dampening = True
    a.balance_mode, a.keep_rule = 'thinned', 'v2'
    a.keep_u0, a.typicality_a = 0.05, 1.0
    a.fp8_scout = bool(cli.fp8)
    a.scout_amp_bf16 = True
    a.thin_oversample_factor = 2.0
    a.typicality_bank_size, a.typicality_reserve_size = 128, 32
    a.typicality_s_buffer_size = a.typicality_s_min_buffer = 512
    a.typicality_halflife_steps, a.typicality_reserve_residency = 8, 16
    a.typicality_s_sweep_interval = 20
    a.typicality_warmup_iters = 0
    return a


def phase_train(cli):
    """One torchrun rank: build the args and run the trainer. Resumes from checkpoint.pth when
    the output dir has one."""
    from training.trainer import train_dinov2
    train_dinov2(train_args(cli))


def run_phase(cli, label, output_dir, total_iterations, log_path):
    """Launch one training phase under torchrun. stdout and stderr go to log_path."""
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes', '1',
           '--nproc_per_node', str(cli.gpus), os.path.abspath(__file__), '--phase', 'train',
           '--work_dir', cli.work_dir, '--fp8', str(cli.fp8), '--seed', str(cli.seed),
           '--batch_size', str(cli.batch_size), '--num_workers', str(cli.num_workers),
           '--output_dir', output_dir, '--total_iterations', str(total_iterations)]
    env = dict(os.environ)
    env.setdefault('OMP_NUM_THREADS', '4')
    env['PYTHONUNBUFFERED'] = '1'
    print('[%s] %d GPU(s), total_iterations %d, log %s' % (label, cli.gpus, total_iterations, log_path),
          flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    print('[%s] exit %d after %.0f s' % (label, rc, time.time() - t0), flush=True)
    if rc != 0:
        with open(log_path, errors='replace') as f:
            tail = f.read().splitlines()[-40:]
        print('\n'.join('    ' + line for line in tail), flush=True)
    return rc


# ----------------------------------------------------------------------------- helpers
class Report(object):
    """Collects PASS/FAIL rows and prints them as they come."""

    def __init__(self):
        self.rows = []

    def add(self, name, ok, **info):
        self.rows.append({'name': name, 'pass': bool(ok), 'info': info})
        detail = ', '.join('%s=%s' % (k, v) for k, v in info.items())
        print('  %s  %-34s %s' % ('PASS' if ok else 'FAIL', name, detail), flush=True)

    def all_pass(self):
        return all(r['pass'] for r in self.rows)


def read_log(path):
    with open(path, errors='replace') as f:
        return f.read()


def ckpt_path(run_dir, it):
    return os.path.join(run_dir, 'checkpoint_iter_%08d.pth' % it)


def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def thin_state(ck):
    ts = ck.get('thin_state')
    return ts if isinstance(ts, dict) else None


def make_dataset(cli, num_workers, resume_batches=None):
    """The trainer's loader dataset (scout_pool_mode) for rank 0 of a world of 1."""
    from data import ProportionalMultiDatasetWrapper
    ds = ProportionalMultiDatasetWrapper(
        dataset_configs=dataset_configs(data_dir_of(cli)), batch_size_per_gpu=cli.batch_size,
        n_standard_local_crops=2, global_views=2, local_crop_size=96, worker_id=0,
        num_workers=num_workers, rank=0, world_size=1, seed=cli.seed, global_size=224,
        use_pathology_recipe=False, ect_probability=0.4, emit_scout=False, scout_pool_mode=True)
    if resume_batches is not None:
        ds.set_resume_position(resume_batches * cli.batch_size * 1, replay_passes=True)
    return ds


def tile_hashes(batch):
    """sha256 of every uint8 tile of one loader batch ([N,3,224,224])."""
    return [hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for t in batch[0]]


# ----------------------------------------------------------------------------- (4) checkpoints
def check_checkpoints(cli, rep, run_a, run_b, K, n):
    """Compare the checkpoints of the two runs. Returns B's thin_state and position at K."""
    pa, pb = ckpt_path(run_a, K), ckpt_path(run_b, K)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        rep.add('checkpoint_at_K_exists', False, A=os.path.exists(pa), B=os.path.exists(pb))
        return None, None
    ck_a, ck_b = load_ckpt(pa), load_ckpt(pb)
    ts = thin_state(ck_b)
    missing = [k for k in THIN_KEYS if ts is None or k not in ts]
    ok = (not missing and ts['thin_gate_open'] is True and ts['iteration'] == K
          and ts['thin_n_draws_used'] is not None and int(ts['thin_n_draws_cur']) >= 1)
    rep.add('thin_state_at_K', ok, missing=missing,
            **{k: ts.get(k) for k in THIN_KEYS if ts is not None and k in ts})
    if not ok:
        return None, None
    ts_a = thin_state(ck_a)
    floor = abs(float(ts_a['thin_m_ema']) - float(ts['thin_m_ema'])) if ts_a else None
    rep.add('fresh_runs_agree_at_K',
            ts_a is not None and ck_a['dataset_position'] == ck_b['dataset_position']
            and ts_a['thin_n_draws_cur'] == ts['thin_n_draws_cur'],
            dataset_position=(ck_a['dataset_position'], ck_b['dataset_position']),
            n_draws_cur=(ts_a['thin_n_draws_cur'] if ts_a else None, ts['thin_n_draws_cur']),
            d_m_ema_floor=floor)
    for it in range(K + SAVE_FREQ, K + n, SAVE_FREQ):
        pa, pb = ckpt_path(run_a, it), ckpt_path(run_b, it)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            rep.add('checkpoint_exists_%d' % it, False, A=os.path.exists(pa), B=os.path.exists(pb))
            continue
        a, b = load_ckpt(pa), load_ckpt(pb)
        ta, tb = thin_state(a), thin_state(b)
        if ta is None or tb is None:
            rep.add('thin_state_match_%d' % it, False, A=ta is not None, B=tb is not None)
            continue
        dm = abs(float(ta['thin_m_ema']) - float(tb['thin_m_ema']))
        df = abs(float(ta['thin_f_next']) - float(tb['thin_f_next']))
        rep.add('thin_state_match_%d' % it,
                ta['thin_gate_open'] is True and tb['thin_gate_open'] is True
                and ta['thin_n_draws_cur'] == tb['thin_n_draws_cur']
                and ta['iteration'] == tb['iteration'] == it and dm <= cli.tol and df <= cli.tol,
                gate=(ta['thin_gate_open'], tb['thin_gate_open']),
                n_draws_cur=(ta['thin_n_draws_cur'], tb['thin_n_draws_cur']),
                m_ema=(ta['thin_m_ema'], tb['thin_m_ema']), d_m_ema=dm, d_f_next=df,
                streak=(ta['thin_ready_streak'], tb['thin_ready_streak']), tol=cli.tol,
                d_m_ema_floor_at_K=floor)
        rep.add('position_match_%d' % it,
                a['dataset_position'] == b['dataset_position'] and a['iteration'] == b['iteration'] == it,
                dataset_position=(a['dataset_position'], b['dataset_position']),
                iteration=(a['iteration'], b['iteration']))
    return ts, int(ck_b['dataset_position'])


# ----------------------------------------------------------------------------- (5) stream
def check_stream(cli, rep, ts_K, pos_K):
    """Uninterrupted stream advanced by P batches vs a fresh loader resumed at P, where P is the
    position at the start of step K. The next used + n_draws_cur batches must hold the same
    tiles (sorted multisets of hashes)."""
    used = int(ts_K['thin_n_draws_used'] or 1)
    n_cur = int(ts_K['thin_n_draws_cur'])
    P, window = pos_K - used, used + n_cur
    nw = max(1, cli.stream_workers)
    if P % nw:
        print('  stream check: P=%d is not a multiple of %d workers; using 1 worker' % (P, nw))
        nw = 1
    ds1 = make_dataset(cli, nw)
    it1 = iter(torch.utils.data.DataLoader(ds1, batch_size=cli.batch_size, num_workers=nw, drop_last=True))
    for _ in range(P):
        next(it1)
    h1 = [h for _ in range(window) for h in tile_hashes(next(it1))]
    del it1
    ds2 = make_dataset(cli, nw, resume_batches=P)
    it2 = iter(torch.utils.data.DataLoader(ds2, batch_size=cli.batch_size, num_workers=nw, drop_last=True))
    h2 = [h for _ in range(window) for h in tile_hashes(next(it2))]
    del it2
    passes = {}
    for name, spd, size in zip(ds2.dataset_names, ds2.samples_per_dataset, ds2.dataset_sizes):
        passes[name] = ((P * spd) // nw) // max(size // nw, 1)
    rep.add('stream_multiset_at_resume', sorted(h1) == sorted(h2), P=P, used=used, n_draws_cur=n_cur,
            window_batches=window, workers=nw, tiles=len(h1), distinct=len(set(h1)),
            only_uninterrupted=len(set(h1) - set(h2)), only_resumed=len(set(h2) - set(h1)),
            passes_done=passes)


# ----------------------------------------------------------------------------- (6) replay
def check_replay_passes(cli, rep):
    """A resume position past one pass of every sub-dataset: after the first __iter__ each
    sub-dataset's _epoch must equal passes_done + 1 (world_size 1, one worker)."""
    ds = make_dataset(cli, 1)
    N = cli.batch_size
    P = max(-(-2 * size // spd) for spd, size in zip(ds.samples_per_dataset, ds.dataset_sizes) if spd > 0)
    ds.set_resume_position(P * N, replay_passes=True)    # the wrapper hands each source its share
    got, want = {}, {}
    for name, sub, spd, size in zip(ds.dataset_names, ds.datasets, ds.samples_per_dataset, ds.dataset_sizes):
        next(iter(sub))                      # starts this source's generator: the pass counter is set
        want[name] = (int(P * N * spd / N) // size) + 1
        got[name] = int(sub._epoch)
    rep.add('replay_passes_epoch', got == want and min(want.values()) >= 2,
            resume_batches=P, epoch_got=got, epoch_want=want)


# ----------------------------------------------------------------------------- driver
def phase_drive(cli):
    rep = Report()
    K, n = cli.K, cli.n_after
    if K % SAVE_FREQ or n < SAVE_FREQ + 1:
        print('--K must be a multiple of %d and --n_after >= %d' % (SAVE_FREQ, SAVE_FREQ + 1))
        return 1
    os.makedirs(cli.work_dir, exist_ok=True)
    print('== (1) seed formula (CPU)')
    try:
        rep.add('resume_seed_formula', True, **test_resume_seeds())
    except AssertionError as e:
        rep.add('resume_seed_formula', False, error=str(e))

    print('== (2) synthetic corpus in %s' % data_dir_of(cli))
    t0 = time.time()
    n_tiles = build_corpus(data_dir_of(cli), cli.rebuild_data)
    print('  %d tiles in %d sources (%.0f s)' % (n_tiles, len(SOURCES), time.time() - t0))

    print('== (3) training runs: A = %d steps; B = %d steps, then resume to %d' % (K + n, K + 1, K + n))
    run_a, run_b = os.path.join(cli.work_dir, 'run_A'), os.path.join(cli.work_dir, 'run_B')
    for d in (run_a, run_b):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    log_a, log_b1, log_b2 = (os.path.join(run_a, 'train.log'), os.path.join(run_b, 'train_phase1.log'),
                             os.path.join(run_b, 'train_phase2.log'))
    rc_a = run_phase(cli, 'run A', run_a, K + n, log_a)
    rc_b1 = run_phase(cli, 'run B phase 1', run_b, K + 1, log_b1)
    rc_b2 = run_phase(cli, 'run B phase 2', run_b, K + n, log_b2) if rc_b1 == 0 else -1
    rep.add('runs_exit_0', rc_a == 0 and rc_b1 == 0 and rc_b2 == 0, A=rc_a, B1=rc_b1, B2=rc_b2)
    if not (rc_a == 0 and rc_b1 == 0 and rc_b2 == 0):
        return finish(cli, rep)

    print('== (4) logs and checkpoints')
    text_a, text_b1, text_b2 = read_log(log_a), read_log(log_b1), read_log(log_b2)
    m = GATE_RE.search(text_a)
    gate_a = int(m.group(1)) if m else None
    if m:
        print('  A: ' + text_a[m.start():text_a.find('\n', m.start())])
    m = GATE_RE.search(text_b1)
    gate_b1 = int(m.group(1)) if m else None
    if m:
        print('  B1: ' + text_b1[m.start():text_b1.find('\n', m.start())])
    ok_gate = gate_a is not None and gate_a < K and gate_b1 is not None and gate_b1 < K
    rep.add('gate_open_before_K', ok_gate, gate_iter_A=gate_a, gate_iter_B1=gate_b1, K=K,
            hint='' if ok_gate else 'raise --K')
    rep.add('fresh_run_has_no_resume_line', RESUME_RE.search(text_a) is None)
    rep.add('no_gate_reopen_after_resume', GATE_RE.search(text_b2) is None)
    lines = [(int(r), int(s), int(i)) for r, s, i in RESUME_RE.findall(text_b2)]
    want = resume_seed(cli.seed, K, 0)
    rep.add('resume_seed_line_rank0', (0, want, K) in lines, found=lines[:4], want_rank0=want)
    ts_K, pos_K = check_checkpoints(cli, rep, run_a, run_b, K, n)

    if ts_K is not None:
        print('== (5) loader stream at the resume point (world_size 1)')
        check_stream(cli, rep, ts_K, pos_K)
    print('== (6) replay_passes pass counter (world_size 1)')
    check_replay_passes(cli, rep)
    return finish(cli, rep)


def finish(cli, rep):
    out = os.path.join(cli.work_dir, 't3_resume.json')
    summary = {'config': {k: v for k, v in vars(cli).items()}, 'checks': rep.rows,
               'all_pass': rep.all_pass()}
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    n_fail = sum(1 for r in rep.rows if not r['pass'])
    print('\n%s: %d checks, %d failed. Summary: %s' % ('ALL PASS' if n_fail == 0 else 'FAIL',
                                                       len(rep.rows), n_fail, out))
    return 0 if n_fail == 0 else 1


def parse_cli():
    import utils
    p = argparse.ArgumentParser(description='keep_rule v2 checkpoint resume test (GPU)')
    p.add_argument('--phase', default='drive', choices=['drive', 'train'],
                   help="'train' is the torchrun entry the driver launches")
    p.add_argument('--gpus', default=2, type=int)
    p.add_argument('--work_dir', default=DEFAULT_WORK)
    p.add_argument('--fp8', default=False, type=utils.bool_flag,
                   help='--fp8_scout of both runs. True needs an H100; False also runs on an A100')
    p.add_argument('--K', default=100, type=int,
                   help='resume iteration, a multiple of %d; the gate must open before it' % SAVE_FREQ)
    p.add_argument('--n_after', default=21, type=int, help='steps after K in both runs, >= %d' % (SAVE_FREQ + 1))
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--num_workers', default=2, type=int, help='loader workers of the training runs')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--stream_workers', default=1, type=int, help='loader workers of the stream check')
    p.add_argument('--tol', default=1e-2, type=float, help='m_ema / f_next tolerance between the runs (the re-run of step K applies one more EMA update by design, so 1e-6 cannot hold)')
    p.add_argument('--rebuild_data', action='store_true', help='rebuild the corpus even if present')
    p.add_argument('--output_dir', default=None, help='train phase only')
    p.add_argument('--total_iterations', default=None, type=int, help='train phase only')
    return p.parse_args()


if __name__ == '__main__':
    cli = parse_cli()
    if cli.phase == 'train':
        phase_train(cli)
    else:
        sys.exit(phase_drive(cli))
