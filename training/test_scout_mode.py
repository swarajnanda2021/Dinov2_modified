"""
GPU tests T2(d) and T2(e) for the v2 scout mode (training/scout_mode.py and the chunked scout).
No pytest. One GPU (xformers attention runs on CUDA only).

Cluster command (1 H100, under 2 min; add --compile for about 80 s more):
  cd /data1/vanderbc/test_dinov2_swaraj/FMC_ViT-B_curv2_base \
  && export TORCHINDUCTOR_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/inductor_cache \
            TRITON_CACHE_DIR=/data1/vanderbc/test_dinov2_swaraj/bench/triton_cache \
  && python training/test_scout_mode.py --N 64 --out_dir /data1/vanderbc/test_dinov2_swaraj/bench
Outputs: <out_dir>/t2_scout_mode.json and the printed PASS/FAIL lines. Nothing in the home directory.

Checks:
  (d) after the scout context every student submodule has its previous training flag, also when a
      submodule was in eval mode before, and also when the scout raises;
  (e) two v2 scout calls on the same tiles give bit-identical fp32 signatures; two v1 (train-mode)
      calls differ (the stochastic-depth defect); one chunk equals the whole forward bit for bit;
      chunks of N/4 match the whole forward within bf16 tolerance; the signature dtype is float32.
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch._dynamo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import CombinedModelDINO, DINOHead, ModernViT  # noqa: E402
from models.vision_transformer.modern_vit import Mlp  # noqa: E402
from training.hand_rolled_dp import HandRolledDP  # noqa: E402
from training.scout_mode import scout_eval_mode, training_flags  # noqa: E402
from typicality import RepresentativePrototypes  # noqa: E402

_RAW_MEAN = torch.tensor([0.6816, 0.5640, 0.7232]).view(1, 3, 1, 1)   # trainer _raw_mean
_RAW_STD = torch.tensor([0.1617, 0.1714, 0.1389]).view(1, 3, 1, 1)    # trainer _raw_std


def build_student(seed=0, out_dim=1024, compile_blocks=False):
    """The rev13 student (trainer L393-453 effective values), wrapped like the trainer."""
    torch.manual_seed(seed)
    backbone = ModernViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12,
                         mlp_ratio=4.0, qkv_bias=True, qk_norm=True, dual_norm=False,
                         drop_path_rate=0.4, drop_path_uniform=True, pre_norm=False,
                         num_register_tokens=4, layerscale_init=1e-5, mlp_layer=Mlp)
    model = CombinedModelDINO(backbone=backbone,
                              classhead=DINOHead(768, out_dim, use_bn=False, norm_last_layer=False),
                              patchhead=DINOHead(768, out_dim, use_bn=False, norm_last_layer=False),
                              num_masks=3, patch_size=16)
    model.set_grad_checkpointing(True)
    student = HandRolledDP(model, broadcast=False).cuda()
    if compile_blocks:
        torch._dynamo.config.cache_size_limit = 16
        for _blk in student.module.backbone.blocks:
            _blk.forward = torch.compile(_blk.forward, dynamic=False)
    return student


def normalize_raw(raw_u8):
    x = raw_u8.to('cuda', non_blocking=True).float().div_(255.0)
    return (x - _RAW_MEAN.cuda()) / _RAW_STD.cuda()


def scout_sigs(student, repr_protos, raw_u8, chunk=None, eval_mode=False):
    """The trainer's v2 scout forward (eval mode when eval_mode; bf16 autocast around the forward
    only; compute_signatures outside autocast), in chunks of `chunk` tiles (None = whole pool)."""
    n = raw_u8.shape[0]
    chunk = n if chunk is None else chunk
    sigs = []
    with torch.no_grad():
        with (scout_eval_mode(student) if eval_mode else _nullctx()):
            for c0 in range(0, n, chunk):
                x = normalize_raw(raw_u8[c0:c0 + chunk])
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    so = student([x], token_masks=[None], mode='dino', bottleneck_only=True)
                assert so['bottleneck'].dtype == torch.float32, so['bottleneck'].dtype
                sigs.append(repr_protos.compute_signatures(so['bottleneck'].detach()).detach())
                del so, x
    return torch.cat(sigs, dim=0)


class _nullctx(object):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_d_flags_restored(student, raw):
    repr_protos = RepresentativePrototypes(K_prime=256, bottleneck_dim=256).cuda()
    before = training_flags(student)
    assert all(before.values()), "a fresh student must be in train mode everywhere"
    # a submodule deliberately in eval mode before the scout must stay in eval mode after it
    student.module.backbone.blocks[3].drop_path.eval()
    before = training_flags(student)
    with scout_eval_mode(student):
        inside = training_flags(student)
    assert all(v is False for v in inside.values()), "every submodule must be in eval inside"
    after = training_flags(student)
    assert after == before, "training flags not restored exactly"
    assert student.module.backbone.blocks[3].drop_path.training is False
    student.module.backbone.blocks[3].drop_path.train()
    assert training_flags(student) == {k: True for k in before}
    # exception path: the finally block restores the flags
    try:
        with scout_eval_mode(student):
            raise RuntimeError("scout failed on purpose")
    except RuntimeError:
        pass
    assert all(training_flags(student).values()), "flags not restored after an exception"
    # the full scout path also restores the flags
    scout_sigs(student, repr_protos, raw, eval_mode=True)
    assert all(training_flags(student).values())
    return {"modules": len(before)}


def test_e_deterministic_signatures(student, raw):
    repr_protos = RepresentativePrototypes(K_prime=256, bottleneck_dim=256).cuda()
    N = raw.shape[0]
    s1 = scout_sigs(student, repr_protos, raw, chunk=N, eval_mode=True)
    s2 = scout_sigs(student, repr_protos, raw, chunk=N, eval_mode=True)
    assert s1.dtype == torch.float32 and s1.shape == (N, 256), (s1.dtype, s1.shape)
    assert torch.equal(s1, s2), "two v2 scout calls must give bit-identical signatures"
    # v1 (train mode, stochastic depth): two calls differ
    t1 = scout_sigs(student, repr_protos, raw, chunk=N, eval_mode=False)
    t2 = scout_sigs(student, repr_protos, raw, chunk=N, eval_mode=False)
    rows_differ = float((t1 != t2).any(dim=1).float().mean().item())
    max_abs = float((t1 - t2).abs().max().item())
    assert not torch.equal(t1, t2), "train-mode scout calls were identical (expected DropPath noise)"
    # the context equals a plain eval forward
    snap = training_flags(student)
    student.eval()
    s_ref = scout_sigs(student, repr_protos, raw, chunk=N, eval_mode=False)
    for name, m in student.named_modules():
        m.training = snap[name]
    assert torch.equal(s1, s_ref), "context result differs from a plain eval forward"
    # chunking: one chunk == whole forward bit for bit; N/4 chunks within bf16 tolerance
    s_whole = scout_sigs(student, repr_protos, raw, chunk=None, eval_mode=True)
    assert torch.equal(s_whole, s1)
    s_q = scout_sigs(student, repr_protos, raw, chunk=max(1, N // 4), eval_mode=True)
    chunk_max_abs = float((s_whole - s_q).abs().max().item())
    assert s_q.dtype == torch.float32
    assert torch.allclose(s_whole, s_q, atol=2e-2, rtol=0.0), chunk_max_abs
    torch.cuda.reset_peak_memory_stats()
    scout_sigs(student, repr_protos, raw, chunk=None, eval_mode=True)
    mem_whole = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    scout_sigs(student, repr_protos, raw, chunk=max(1, N // 4), eval_mode=True)
    mem_chunk = torch.cuda.max_memory_allocated() / 1e6
    return {"v1_rows_differ": rows_differ, "v1_max_abs": max_abs,
            "chunk_vs_whole_max_abs": chunk_max_abs,
            "peak_mb_whole": mem_whole, "peak_mb_chunk": mem_chunk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N', type=int, default=64)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--out_dir', default='/data1/vanderbc/test_dinov2_swaraj/bench')
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    g = torch.Generator().manual_seed(123)
    raw = torch.randint(0, 256, (a.N, 3, 224, 224), dtype=torch.uint8, generator=g)
    student = build_student(compile_blocks=a.compile)
    results, failed = {}, 0
    for t in (test_d_flags_restored, test_e_deterministic_signatures):
        try:
            results[t.__name__] = t(student, raw)
            print("  PASS  %s %s" % (t.__name__, results[t.__name__]))
        except Exception as e:  # noqa: BLE001
            failed += 1
            results[t.__name__] = "FAIL: %s: %s" % (type(e).__name__, e)
            print("  FAIL  %s: %s: %s" % (t.__name__, type(e).__name__, e))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, 't2_scout_mode.json'), 'w') as f:
        json.dump({"N": a.N, "compile": a.compile, "torch": torch.__version__,
                   "device": torch.cuda.get_device_name(), "results": results}, f, indent=1)
    print("\n%d/2 passed" % (2 - failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
