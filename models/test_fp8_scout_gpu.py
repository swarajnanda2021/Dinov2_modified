# Cluster command (1 H100, about 10-15 min):
#   cd /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base \
#   && export TORCHINDUCTOR_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/inductor_cache \
#             TRITON_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/triton_cache \
#   && TORCH_LOGS=recompiles python models/test_fp8_scout_gpu.py --compile \
#        --ckpt /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base/logs/checkpoint_iter_00050000.pth \
#        --out_dir /data1/vanderbc/test_dinov2_swaraj/bench 2> /data1/vanderbc/test_dinov2_swaraj/bench/t2_fp8_recompiles.log
# Outputs: <out_dir>/t2_fp8.json and the printed PASS/FAIL table. Nothing is written to the home directory.
#
# GPUs: 1. Run time: about 10-15 min, most of it torch.compile warm-ups in checks (b) and (f).
# Keep TORCH_LOGS=recompiles on: it writes the guard behind every dynamo recompile to stderr, which
# is the only way to read WHY check (f) saw a recompile.
"""GPU test for the fp8 scout (models/fp8_scout.py). No pytest. Four checks:
(a) fp8 vs bf16 error per linear and end to end, both kernels (rowwise, tensorwise).
(b) Training forward bit-identical with the switch off; grads within the run-to-run floor.
(c) state_dict keys and shapes unchanged by the swap; checkpoint round trip; flag logic.
(f) Dynamo graph count per stage; the fp8 scout must not recompile across optimizer steps.
"""
import argparse
import contextlib
import copy
import gc
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.fp8_scout as fp8mod  # noqa: E402
from models.fp8_scout import (ScoutFP8Linear, check_fp8_capability, fp8_active_for,  # noqa: E402
                              fp8_linear, fp8_scout, quantize_rowwise, swap_student_linears)
from models import CombinedModelDINO, DINOHead, ModernViT  # noqa: E402
from models.vision_transformer.modern_vit import Mlp, SwiGLUFFNFused  # noqa: E402
from training.hand_rolled_dp import HandRolledDP  # noqa: E402
from training.scout_mode import scout_eval_mode  # noqa: E402
from typicality.representative_prototypes import RepresentativePrototypes  # noqa: E402

KERNELS = ('rowwise', 'tensorwise')
SEED = 0
N_SWAPPED = 48          # 4 linears per block, 12 blocks
PROTOS = None           # RepresentativePrototypes, set in main()


# ----------------------------------------------------------------- helpers
def reset_rng(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_core(ffn, out_dim, seed=SEED):
    """One CombinedModelDINO on CPU with the rev13 ViT-B settings (trainer L393-453)."""
    torch.manual_seed(seed)
    enc = ModernViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                    mlp_ratio=4.0, qkv_bias=True, qk_norm=True, dual_norm=False,
                    drop_path_rate=0.4, drop_path_uniform=True, pre_norm=False,
                    num_register_tokens=4, layerscale_init=1e-5,
                    mlp_layer=Mlp if ffn == 'mlp' else SwiGLUFFNFused)
    heads = [DINOHead(768, out_dim, use_bn=False, norm_last_layer=False) for _ in range(2)]
    model = CombinedModelDINO(enc, heads[0], heads[1], num_masks=3, patch_size=16)
    model.set_grad_checkpointing(True)
    return model


def build_pair(ffn, out_dim):
    """Student and teacher wrapped like the trainer. The teacher is a deepcopy taken BEFORE the
    swap; the swap runs after .cuda() and before HandRolledDP. Returns (student, teacher, swapped)."""
    core = build_core(ffn, out_dim)
    teacher_core = build_core(ffn, out_dim)  # fresh, same seed = same weights; deepcopy of the
    # weight_norm heads raises on torch 2.6 (non-leaf weight). The trainer also deepcopies only
    # the backbone and rebuilds the heads.
    core = core.cuda()
    swapped = swap_student_linears(core.backbone)
    student = HandRolledDP(core, broadcast=False)
    teacher = HandRolledDP(teacher_core.cuda(), broadcast=False)
    teacher.eval()
    teacher.requires_grad_(False)
    return student, teacher, swapped


def make_tiles(n, seed=SEED):
    """n fixed random normalized tiles [n, 3, 224, 224] on the GPU."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 3, 224, 224, generator=g).cuda()


def make_crops(n, seed=SEED + 1):
    """The trainer's packed input: 2 global crops [n, 3, 224, 224] + 8 local crops [n, 3, 96, 96]."""
    g = torch.Generator().manual_seed(seed)
    crops = [torch.randn(n, 3, 224, 224, generator=g) for _ in range(2)]
    crops += [torch.randn(n, 3, 96, 96, generator=g) for _ in range(8)]
    return [c.cuda() for c in crops]


def scout_sigs(student, x_norm, chunk, fp8_linears=None, eval_mode=True):
    """The trainer's v2 scout: no_grad, eval mode (scout_eval_mode), chunks of `chunk` tiles,
    bf16 autocast around the forward only, compute_signatures OUTSIDE autocast (fp32 matmul).
    With fp8_linears the whole chunk loop sits inside fp8_scout(): one weight quantization per call.
    eval_mode=False is the v1 train-mode scout. Returns (signatures [n, K'], bottleneck dtype)."""
    sigs, z_dtype = [], None
    ev = scout_eval_mode(student) if eval_mode else contextlib.nullcontext()
    q8 = fp8_scout(fp8_linears) if fp8_linears is not None else contextlib.nullcontext()
    with torch.no_grad(), ev, q8:
        for i in range(0, x_norm.shape[0], chunk):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                so = student([x_norm[i:i + chunk]], token_masks=[None], mode='dino', bottleneck_only=True)
            z = so['bottleneck'].detach()
            z_dtype = z.dtype
            sigs.append(PROTOS.compute_signatures(z.float()).detach())   # .float() as in the trainer
    return torch.cat(sigs, 0), z_dtype


def train_forward(model, crops, seed=SEED):
    """The trainer's packed student forward: train mode, grad on, bf16 autocast. The RNG is reset
    right before the forward so the DropPath draws of two models match."""
    model.train()
    reset_rng(seed)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(crops, token_masks=[None] * len(crops), mode='dino')


def train_step_grads(model, crops):
    """Forward + backward on cls_outputs.float().sum(). Returns a copy of every parameter grad."""
    model.zero_grad(set_to_none=True)
    out = train_forward(model, crops)
    out['cls_outputs'].float().sum().backward()
    del out
    return [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]


def max_grad_diff(ga, gb):
    worst = 0.0
    for a, b in zip(ga, gb):
        if a is None and b is None:
            continue
        assert a is not None and b is not None, "a grad is present in one model and None in the other"
        worst = max(worst, float((a.float() - b.float()).abs().max().item()))
    return worst


def forward_equal(o1, o2):
    """torch.equal on cls_outputs and on clstoken / patchtokens of every features_list entry.
    Returns (all equal, max abs difference)."""
    pairs = [(o1['cls_outputs'], o2['cls_outputs'])]
    for f1, f2 in zip(o1['features_list'], o2['features_list']):
        pairs += [(f1['clstoken'], f2['clstoken']), (f1['patchtokens'], f2['patchtokens'])]
    eq = all(torch.equal(a, b) for a, b in pairs)
    return eq, max(float((a.float() - b.float()).abs().max().item()) for a, b in pairs)


def compile_blocks(models, backend=None):
    """The trainer's per-block compile (trainer L550-555) on every model in `models`."""
    for m in models:
        for blk in m.module.backbone.blocks:
            kw = {} if backend is None else {'backend': backend}
            blk.forward = torch.compile(blk.forward, dynamic=False, **kw)


def snapshot(model):
    sd = model.state_dict()
    return sorted(sd.keys()), {k: (tuple(v.shape), str(v.dtype)) for k, v in sd.items()}


# ----------------------------------------------------------------- (a)
def test_a_fp8_vs_bf16(args):
    """fp8 vs bf16: per-layer relative error and end-to-end signature cosine, both kernels."""
    student, _, swapped = build_pair(args.ffn_type, args.out_dim)
    names = {id(m): n for n, m in student.module.backbone.named_modules()}
    x = make_tiles(args.n)
    n_hook = min(args.n_hook, args.n)
    res = {'n': args.n, 'chunk': args.chunk, 'n_hook': n_hook, 'kernels': {}}

    # Reference pass with hooks: switch off, eval mode, bf16 autocast. Stores (input, output).
    store = {}
    hooks = [m.register_forward_hook(
        lambda mod, inp, out: store.__setitem__(id(mod), (inp[0].detach(), out.detach())))
        for m in swapped]
    assert fp8mod._FP8_ON is False, "the fp8 switch must be off outside fp8_scout()"
    with torch.no_grad(), scout_eval_mode(student), torch.autocast('cuda', dtype=torch.bfloat16):
        so = student([x[:n_hook]], token_masks=[None], mode='dino', bottleneck_only=True)
    for h in hooks:
        h.remove()
    assert len(store) == len(swapped) == N_SWAPPED, "hooks fired on %d of %d" % (len(store), N_SWAPPED)
    res['bottleneck_dtype'] = str(so['bottleneck'].dtype)
    print("  bottleneck dtype under autocast: %s (expect torch.float32)" % so['bottleneck'].dtype)
    del so
    sig16, _ = scout_sigs(student, x, args.chunk)
    res['signature_dtype'] = str(sig16.dtype)
    print("  signature dtype: %s" % sig16.dtype)

    problems = []
    for kernel in KERNELS:
        rows = []
        print("\n  [a] kernel '%s': row-wise ||y8 - y16|| / ||y16|| per linear, %d tiles" % (kernel, n_hook))
        print("  %-20s %-14s %10s %10s" % ('linear', 'weight', 'mean_rel', 'max_rel'))
        for m in swapped:
            xin, y16 = store[id(m)]
            w8, s = quantize_rowwise(m.weight.detach())
            with torch.no_grad():
                y8 = fp8_linear(xin, w8, s.t().contiguous(), m.bias, kernel)
            d = (y8.float() - y16.float()).reshape(-1, y8.shape[-1]).norm(dim=1)
            rel = d / y16.float().reshape(-1, y8.shape[-1]).norm(dim=1).clamp_min(1e-12)
            rows.append([names[id(m)], list(m.weight.shape), float(rel.mean()), float(rel.max())])
            print("  %-20s %-14s %10.5f %10.5f" % (rows[-1][0], tuple(rows[-1][1]), rows[-1][2], rows[-1][3]))
            del y8, d, rel
        worst_mean, worst_max = max(r[2] for r in rows), max(r[3] for r in rows)
        bad = [r[0] for r in rows if not (r[2] < 0.1)]
        if bad:
            problems.append("kernel %s: mean rel error >= 0.1 on %s" % (kernel, bad))

        # End to end: the fp8 scout with this kernel vs the bf16 scout, cosine per tile.
        fp8mod.FP8_KERNEL = kernel
        try:
            sig8, z_dtype = scout_sigs(student, x, args.chunk, fp8_linears=swapped)
        finally:
            fp8mod.FP8_KERNEL = 'rowwise'
        cos = F.cosine_similarity(sig8.float(), sig16.float(), dim=1)
        st = {'min': float(cos.min()), 'mean': float(cos.mean()), 'p05': float(torch.quantile(cos, 0.05))}
        print("  kernel %-10s worst layer mean_rel %.5f, worst max_rel %.5f | signature cosine min %.5f "
              "mean %.5f p05 %.5f | fp8 bottleneck dtype %s"
              % (kernel, worst_mean, worst_max, st['min'], st['mean'], st['p05'], z_dtype))
        if not (st['mean'] > 0.99):
            problems.append("kernel %s: mean signature cosine %.5f <= 0.99" % (kernel, st['mean']))
        res['kernels'][kernel] = {'layers': rows, 'worst_mean_rel': worst_mean, 'worst_max_rel': worst_max,
                                  'cosine': st, 'fp8_bottleneck_dtype': str(z_dtype)}
    res['key'] = "; ".join("%s: worst mean_rel %.4f, cos mean %.5f" % (k, v['worst_mean_rel'], v['cosine']['mean'])
                           for k, v in res['kernels'].items())
    assert not problems, "; ".join(problems)
    return res


# ----------------------------------------------------------------- (b)
def test_b_train_forward_identity(args):
    """Training forward bit-identical with the switch off; grads within the run-to-run floor."""
    core = build_core(args.ffn_type, args.out_dim)
    core2 = build_core(args.ffn_type, args.out_dim).cuda()  # same weights (seeded build), no deepcopy
    plain = HandRolledDP(core.cuda(), broadcast=False)
    swap_student_linears(core2.backbone)
    swapped = HandRolledDP(core2, broadcast=False)
    crops = make_crops(args.n_train)
    res = {'n_train': args.n_train}

    o1, o2 = train_forward(plain, crops), train_forward(swapped, crops)
    eq, mx = forward_equal(o1, o2)
    del o1, o2
    res['forward_equal'], res['forward_max_abs_diff'] = eq, mx
    print("  eager forward: equal=%s, max abs diff %.3e" % (eq, mx))
    assert eq, "training forward differs with the switch off (max abs diff %.3e)" % mx

    ga = train_step_grads(plain, crops)
    floor = max_grad_diff(ga, train_step_grads(plain, crops))
    diff = max_grad_diff(ga, train_step_grads(swapped, crops))
    del ga
    plain.zero_grad(set_to_none=True)
    swapped.zero_grad(set_to_none=True)
    res['grad_floor_run_to_run'], res['grad_diff_swapped_vs_plain'] = floor, diff
    print("  grads: run-to-run floor (unswapped twice) %.3e, swapped vs unswapped %.3e" % (floor, diff))
    assert diff <= floor, "swapped-vs-unswapped grad diff %.3e > run-to-run floor %.3e" % (diff, floor)

    res['compiled_forward_equal'] = 'n/a'
    if args.compile:
        torch._dynamo.reset()
        compile_blocks([plain, swapped])
        o1, o2 = train_forward(plain, crops), train_forward(swapped, crops)
        eq_c, mx_c = forward_equal(o1, o2)
        del o1, o2
        res['compiled_forward_equal'], res['compiled_forward_max_abs_diff'] = eq_c, mx_c
        print("  compiled forward (finding, not a failure): equal=%s, max abs diff %.3e" % (eq_c, mx_c))
        torch._dynamo.reset()
    res['key'] = "forward equal=%s; grad floor %.2e, diff %.2e; compiled equal=%s" % (
        eq, floor, diff, res['compiled_forward_equal'])
    return res


# ----------------------------------------------------------------- (c)
def ckpt_roundtrip(args, res):
    """Load --ckpt into a swapped model (strict), save it, load that into an unswapped model (strict)."""
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in ckpt['student'].items()}
    del ckpt
    out_dim = int(sd['classhead.last_layer.weight_g'].shape[0])
    ffn = 'mlp' if 'backbone.blocks.0.mlp.fc1.weight' in sd else 'swiglu'
    depth = len({k.split('.')[2] for k in sd if k.startswith('backbone.blocks.')})
    print("  ckpt: out_dim %d, ffn %s, depth %d, %d keys" % (out_dim, ffn, depth, len(sd)))
    assert depth == 12, "the test builds depth 12; the checkpoint has depth %d" % depth
    model = build_core(ffn, out_dim)                     # CPU is enough for a load
    swap_student_linears(model.backbone)
    missing = sorted(set(model.state_dict()) - set(sd))
    extra = sorted(set(sd) - set(model.state_dict()))
    assert not missing and not extra, "key sets differ: missing %s, extra %s" % (missing[:5], extra[:5])
    model.load_state_dict(sd, strict=True)
    del sd
    fd, tmp = tempfile.mkstemp(prefix='t2_fp8_swapped_', suffix='.pth', dir=args.out_dir)
    os.close(fd)
    try:
        torch.save(model.state_dict(), tmp)
        fresh = build_core(ffn, out_dim)
        assert not any(isinstance(m, ScoutFP8Linear) for m in fresh.modules())
        fresh.load_state_dict(torch.load(tmp, map_location='cpu'), strict=True)
    finally:
        os.remove(tmp)
    res['ckpt'] = {'path': args.ckpt, 'out_dim': out_dim, 'ffn': ffn, 'roundtrip': 'ok'}
    print("  ckpt -> swapped model (strict) -> file -> unswapped model (strict): ok")


def test_c_state_dict(args):
    """state_dict keys/shapes unchanged by the swap; 48 ScoutFP8Linear; flags; capability; ckpt."""
    core = build_core(args.ffn_type, args.out_dim).cuda()
    keys0, meta0 = snapshot(core)
    swapped = swap_student_linears(core.backbone)
    keys1, meta1 = snapshot(core)
    assert keys0 == keys1, "state_dict keys changed with the swap"
    assert meta0 == meta1, "state_dict shapes or dtypes changed with the swap"
    assert len(swapped) == N_SWAPPED, "swapped %d, expected %d" % (len(swapped), N_SWAPPED)
    assert all(type(m) is ScoutFP8Linear for m in swapped), "a swapped module is not ScoutFP8Linear"
    ids = {id(m) for m in swapped}
    other = [(n, m) for n, m in core.named_modules() if isinstance(m, nn.Linear) and id(m) not in ids]
    bad = [n for n, m in other if type(m) is not nn.Linear]
    assert not bad, "non-swapped linears are not plain nn.Linear: %s" % bad[:5]
    names = {id(m): n for n, m in core.backbone.named_modules()}
    expected = {'qkv', 'proj'} | ({'mlp.fc1', 'mlp.fc2'} if args.ffn_type == 'mlp' else {'mlp.w12', 'mlp.w3'})
    got = {names[id(m)].split('.', 2)[2] for m in swapped}
    assert got == expected, "swapped names %s, expected %s" % (sorted(got), sorted(expected))
    print("  %d state_dict keys unchanged; %d ScoutFP8Linear (%s); %d other plain nn.Linear"
          % (len(keys0), len(swapped), ", ".join(sorted(expected)), len(other)))
    res = {'n_keys': len(keys0), 'n_swapped': len(swapped), 'n_other_linear': len(other), 'capability': {}}
    del core

    assert fp8_active_for('v1', True) is False
    assert fp8_active_for('v2', True) is True
    assert fp8_active_for('v2', False) is False
    assert fp8_active_for('weighted', True) is False
    # The trainer's start-up decision for a v1 run with the default flag (fp8_scout True).
    fresh = build_core(args.ffn_type, args.out_dim)
    if fp8_active_for('v1', True):
        swap_student_linears(fresh.backbone)
    assert not any(isinstance(m, ScoutFP8Linear) for m in fresh.modules()), "a v1 run must not swap"
    del fresh
    print("  fp8_active_for: v1/True False, v2/True True, v2/False False, weighted/True False; "
          "v1 start-up leaves every Linear plain")
    for kernel in KERNELS:
        out = check_fp8_capability(kernel)
        assert isinstance(out, tuple) and len(out) == 2, "check_fp8_capability returned %r" % (out,)
        res['capability'][kernel] = [str(out[0]), list(out[1])]
        print("  check_fp8_capability('%s') -> %s" % (kernel, out))
    if args.ckpt:
        ckpt_roundtrip(args, res)
    else:
        print("  no --ckpt given: checkpoint round trip skipped")
        res['ckpt'] = None
    res['key'] = "%d keys unchanged, %d swapped, ckpt %s" % (len(keys0), len(swapped), 'ok' if res['ckpt'] else 'skipped')
    return res


# ----------------------------------------------------------------- (f)
def test_f_graph_count(args):
    """Dynamo graphs per stage; the fp8 scout must not recompile across real optimizer steps."""
    torch._dynamo.reset()
    torch._dynamo.config.cache_size_limit = 16
    COUNT = [0]
    inductor = torch._dynamo.lookup_backend('inductor')

    def counting(gm, inputs):
        COUNT[0] += 1
        return inductor(gm, inputs)

    student, teacher, swapped = build_pair(args.ffn_type, args.out_dim)
    compile_blocks([student, teacher], backend=counting)
    N = args.chunk
    crops = make_crops(args.n_train)
    x1, x3 = make_tiles(N), make_tiles(3 * N, seed=SEED + 2)
    stages = []

    def mark(label):
        torch.cuda.synchronize()
        stages.append([label, COUNT[0]])
        print("  graphs after %-46s %4d" % (label, COUNT[0]))

    train_step_grads(student, crops)
    mark('(1) train fwd+bwd, packed 10 crops, grad ckpt')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        teacher(crops[:2], token_masks=[None, None], mode='dino')
    mark('(2) teacher eval fwd, 2 global crops')
    scout_sigs(student, x1, N)
    mark('(3) v2 eval scout bf16 at N=%d' % N)
    scout_sigs(student, x1, N, fp8_linears=swapped)
    mark('(4) v2 fp8 scout rowwise at N')
    scout_sigs(student, x1, N, eval_mode=False)
    mark('(5a) v1 train-mode scout at N')
    scout_sigs(student, x3, 3 * N, eval_mode=False)
    mark('(5b) v1 train-mode scout at 3N=%d' % (3 * N))
    after_5 = COUNT[0]
    print("  expected after stage 5: 6 graphs, one per guard set (train+grad packed, teacher eval, eval N, "
          "fp8 N, train no-grad N, train no-grad 3N), plus whatever the checkpoint recompute adds. "
          "The 12 blocks share one forward code object, so one guard set is one graph.")

    # 2.10: three more fp8 scouts with a REAL optimizer step between the calls. The fp8 weights are
    # new tensor objects at every call; dynamo must guard on shape/dtype only, never on identity.
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    recompiles = []
    for i in range(1, 4):
        before = COUNT[0]
        train_step_grads(student, crops)
        opt.step()
        opt.zero_grad()
        mark('(2.10) optimizer step %d' % i)
        if COUNT[0] != before:
            recompiles.append(('train step %d' % i, COUNT[0] - before))
        before = COUNT[0]
        scout_sigs(student, x1, N, fp8_linears=swapped)
        mark('(2.10) fp8 scout call %d after the step' % (i + 1))
        if COUNT[0] != before:
            recompiles.append(('fp8 scout call %d' % (i + 1), COUNT[0] - before))
    res = {'chunk': N, 'stages': stages, 'graphs_after_stage_5': after_5, 'expected_after_stage_5': 6,
           'graphs_final': COUNT[0], 'recompiles_in_2_10': recompiles,
           'key': "%d graphs after stage 5 (expected 6 + ckpt recompute), %d recompiles in 2.10"
                  % (after_5, len(recompiles))}
    if recompiles:
        print("  RECOMPILES in 2.10: " + "; ".join("%s: +%d" % r for r in recompiles)
              + "  (TORCH_LOGS=recompiles on stderr has the guard)")
    assert COUNT[0] == after_5, "the fp8 scout recompiled across optimizer steps: %s" % (recompiles,)
    print("  2.10: graph count stayed at %d over 3 optimizer steps + fp8 scouts" % COUNT[0])
    torch._dynamo.reset()
    return res


# ----------------------------------------------------------------- main
def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ffn_type', default='mlp', choices=['mlp', 'swiglu'])
    p.add_argument('--out_dim', type=int, default=1024, help="DINOHead out_dim (the scout never uses it)")
    p.add_argument('--n', type=int, default=256, help="scout tiles for (a)")
    p.add_argument('--n_hook', type=int, default=32, help="tiles for the per-layer hook pass in (a)")
    p.add_argument('--chunk', type=int, default=256, help="scout chunk N = the run's batch_size_per_gpu")
    p.add_argument('--n_train', type=int, default=32, help="tiles per crop in the packed training forward")
    p.add_argument('--compile', action='store_true', help="also run the compiled forward in (b)")
    p.add_argument('--ckpt', default=None, help="student checkpoint for the round trip in (c)")
    p.add_argument('--out_dir', default='/data1/vanderbc/test_dinov2_swaraj/bench')
    p.add_argument('--out', default=None, help="JSON summary path (default <out_dir>/t2_fp8.json)")
    return p.parse_args()


def main():
    global PROTOS
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.out_dir, 't2_fp8.json')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert torch.cuda.is_available(), "this test needs a CUDA device (H100)"
    dev = torch.cuda.get_device_name(0)
    print("device %s, capability %s, torch %s" % (dev, torch.cuda.get_device_capability(0), torch.__version__))
    reset_rng()
    PROTOS = RepresentativePrototypes(K_prime=256, bottleneck_dim=256).cuda()

    checks = [('a', test_a_fp8_vs_bf16), ('b', test_b_train_forward_identity),
              ('c', test_c_state_dict), ('f', test_f_graph_count)]
    summary = {'args': vars(args), 'device': dev, 'torch': torch.__version__, 'checks': {}}
    lines, failed = [], 0
    for tag, fn in checks:
        print("\n=== (%s) %s" % (tag, fn.__doc__))
        t0 = time.time()
        try:
            res, status = fn(args), 'PASS'
        except Exception as e:  # noqa: BLE001 - report every failure, keep going
            traceback.print_exc()
            failed += 1
            msg = "%s: %s" % (type(e).__name__, e)
            res, status = {'error': msg, 'key': msg}, 'FAIL'
        res['seconds'], res['status'] = round(time.time() - t0, 1), status
        summary['checks'][tag] = res
        lines.append("  %s  (%s) %-28s %s  [%.0fs]" % (status, tag, fn.__name__, res.get('key', ''), res['seconds']))
        gc.collect()
        torch.cuda.empty_cache()

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    print("\n" + "\n".join(lines))
    print("\n%d/%d passed; summary written to %s" % (len(checks) - failed, len(checks), out_path))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
