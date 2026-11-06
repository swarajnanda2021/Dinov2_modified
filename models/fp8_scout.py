"""fp8 (E4M3) scout linears for the thinned scout pass (--fp8_scout, active only with
--keep_rule v2). Written from scratch. No new dependency.

Scope: ONLY the scout forward (training/trainer.py _scout_and_bank, v2 path) and ONLY the
student backbone block linears (qkv, proj, and the two MLP linears: fc1/fc2 for the Mlp FFN,
w12/w3 for the SwiGLU FFN). The training forward and backward, the teacher, the patch
embedding, the norms, the attention, the heads, the R projection and the cache never see fp8.

Mechanism: ScoutFP8Linear is an nn.Linear subclass whose forward is the nn.Linear.forward
statement (F.linear) unless the module-level switch _FP8_ON is True AND the module carries a
cached fp8 weight. The fp8_scout() context manager quantizes the 48 weights once (per scout
call), raises the switch, and clears both on exit. The swap is a __class__ assignment, so the
module object, its Parameter objects, its registration order and its state_dict keys do not
change.

Numerics: per-row (token) activation scales s_x = max|x_row| / 448, per-output-channel weight
scales s_w = max|w_row| / 448, fp32 accumulation (use_fast_accum=False), bf16 output, bias
added in bf16 after the matmul. Two ways to apply the scales, selected by FP8_KERNEL:
    'rowwise'    torch._scaled_mm(x8, w8.t(), s_x [M,1], s_w [1,N], out_dtype=bf16)
    'tensorwise' torch._scaled_mm(x8, w8.t(), 1.0, 1.0, out_dtype=bf16), then * s_x * s_w,
                 then cast to bf16 (the tensorwise cuBLAS kernel can be faster; B1 times both).

torch._scaled_mm contract (torch 2.6 / 2.8):
    torch._scaled_mm(mat_a, mat_b, scale_a, scale_b, bias=None, scale_result=None,
                     out_dtype=None, use_fast_accum=False)
    mat_a: fp8 [M, K] row-major;  mat_b: fp8 [K, N] column-major (pass w8.t());
    K % 16 == 0 and N % 16 == 0 (M is free);  scales fp32 and contiguous;
    rowwise: scale_a [M, 1], scale_b [1, N], out_dtype bf16;  tensorwise: both scalar.
ViT-B: K in {768, 3072} (SwiGLU: 768, 2048), N in {2304, 768, 3072} (SwiGLU: 4096, 768).
"""
import contextlib

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0                 # largest finite float8_e4m3fn value
FP8_DTYPE = torch.float8_e4m3fn
_AMAX_FLOOR = 1e-12              # keeps s > 0 on an all-zero row (no division by zero)
FP8_KERNEL = 'rowwise'           # 'rowwise' or 'tensorwise' (module constant, no CLI flag)
MIN_CAPABILITY = (8, 9)          # fp8 tensor cores: H100 (9,0) yes, A100 (8,0) no

_FP8_ON = False                  # module-level switch; True only inside fp8_scout()
_ONE = {}                        # device -> fp32 scalar 1.0 (tensorwise kernel scales)


def fp8_active_for(keep_rule, fp8_scout):
    """fp8 acts only in the v2 scout: True when --fp8_scout is True AND --keep_rule is v2.
    With --keep_rule v1, or in a weighted / off / baseline run, the flag has no effect."""
    return bool(fp8_scout) and keep_rule == 'v2'


def _one(device):
    t = _ONE.get(device)
    if t is None:
        t = torch.ones((), dtype=torch.float32, device=device)
        _ONE[device] = t
    return t


def quantize_rowwise(x):
    """x: [R, K] fp32 or bf16. Returns (x8, s): x8 fp8 [R, K] row-major, s fp32 [R, 1],
    with x ~= x8 * s. The amax is exact in any float dtype (a max selects an element); the
    scale and the scaled values are computed in fp32."""
    x2 = x if x.is_contiguous() else x.contiguous()
    amax = x2.abs().amax(dim=1, keepdim=True).float().clamp_(min=_AMAX_FLOOR)   # [R,1] fp32
    s = amax / E4M3_MAX                                                          # [R,1] fp32
    x8 = (x2.float() / s).clamp_(-E4M3_MAX, E4M3_MAX).to(FP8_DTYPE)              # [R,K] fp8
    return x8, s


def fp8_matmul(x8, s_x, w8, s_w, kernel=None):
    """y = (x8 * s_x) @ (w8 * s_w).T in bf16. x8 fp8 [M,K]; s_x fp32 [M,1]; w8 fp8 [N,K]
    row-major; s_w fp32 [1,N]. w8.t() is the column-major [K,N] operand _scaled_mm needs."""
    kernel = FP8_KERNEL if kernel is None else kernel
    if kernel == 'rowwise':
        return torch._scaled_mm(x8, w8.t(), s_x, s_w,
                                out_dtype=torch.bfloat16, use_fast_accum=False)
    if kernel == 'tensorwise':
        one = _one(x8.device)
        y = torch._scaled_mm(x8, w8.t(), one, one,
                             out_dtype=torch.bfloat16, use_fast_accum=False)
        return (y * s_x * s_w).to(torch.bfloat16)
    raise ValueError("FP8_KERNEL must be 'rowwise' or 'tensorwise', got %r" % (kernel,))


def fp8_linear(x, w8, s_w, bias, kernel=None):
    """y = x @ w.T + bias with x quantized per row here and w pre-quantized per output channel.
    x: [..., K] fp32 or bf16; w8: fp8 [N, K]; s_w: fp32 [1, N]; bias: [N] or None.
    Returns bf16 [..., N], the dtype F.linear returns under bf16 autocast."""
    K = x.shape[-1]
    x8, s_x = quantize_rowwise(x.reshape(-1, K))
    y = fp8_matmul(x8, s_x, w8, s_w, kernel)
    if bias is not None:
        y = y + bias.to(torch.bfloat16)
    return y.reshape(*x.shape[:-1], w8.shape[0])


class ScoutFP8Linear(nn.Linear):
    """nn.Linear with an fp8 scout path. Installed by __class__ assignment, so the object, its
    Parameter objects (weight, bias), its registration order and its state_dict keys are those
    of the original nn.Linear. _w8 / _s_w are plain attributes (not buffers): they are not in
    state_dict(), not in buffers(), and exist only inside fp8_scout()."""
    _w8 = None       # fp8 [N, K] row-major, set by fp8_scout()
    _s_w = None      # fp32 [1, N] contiguous, set by fp8_scout()

    def forward(self, input):
        if _FP8_ON and self._w8 is not None:
            return fp8_linear(input, self._w8, self._s_w, self.bias)
        return F.linear(input, self.weight, self.bias)     # the nn.Linear.forward statement


def _smoke_matmul(device, kernel):
    """One tiny on-device fp8 matmul with the selected kernel. Raises on failure."""
    x = torch.randn(32, 64, device=device)
    w = torch.randn(48, 64, device=device)
    x8, s_x = quantize_rowwise(x)
    w8, s = quantize_rowwise(w)
    y = fp8_matmul(x8, s_x, w8, s.t().contiguous(), kernel)
    ref = x @ w.t()
    err = float((y.float() - ref).abs().max().item()) / max(float(ref.abs().max().item()), 1e-6)
    if not (err < 0.1):
        raise RuntimeError("fp8 result differs from fp32 by %.3f (relative to max)" % err)
    torch.cuda.synchronize(device)


def check_fp8_capability(kernel=None):
    """Stop at start-up, on every rank together, when the GPU cannot run the fp8 scout.
    (a) compute capability >= (8, 9); (b) one small on-device _scaled_mm with the selected
    kernel; (c) a pass/fail flag is all-reduced across ranks before any rank raises, so a mixed
    allocation stops instead of hanging."""
    kernel = FP8_KERNEL if kernel is None else kernel
    reason = None
    name = "no CUDA device"
    cap = None
    if not torch.cuda.is_available():
        reason = "no CUDA device is available"
    else:
        device = torch.device('cuda', torch.cuda.current_device())
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        if cap < MIN_CAPABILITY:
            reason = ("the compute capability %s is below %s (fp8 tensor cores are needed)"
                      % (cap, MIN_CAPABILITY))
        elif not hasattr(torch, '_scaled_mm'):
            reason = "this torch build has no torch._scaled_mm"
        else:
            try:
                _smoke_matmul(device, kernel)
            except Exception as e:  # noqa: BLE001 - report every failure in plain words
                reason = ("the test fp8 matmul (kernel '%s') failed: %s: %s"
                          % (kernel, type(e).__name__, e))
    ok_local = reason is None
    ok_all = ok_local
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        flag = torch.tensor([1.0 if ok_local else 0.0], device='cuda')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok_all = bool(flag.item() > 0.5)
    if not ok_all:
        own = ("this rank: device %s, capability %s, %s" % (name, cap, reason)
               if not ok_local else "this rank passed; another rank failed the fp8 check")
        raise RuntimeError(
            "--fp8_scout True cannot run here (%s). Launch with --fp8_scout False to run the v2 "
            "scout in bf16 on this GPU." % own)
    return name, cap


def swap_student_linears(backbone):
    """Turn qkv, proj and every nn.Linear direct child of block.mlp (Mlp: fc1, fc2; SwiGLU:
    w12, w3) of every block into ScoutFP8Linear by __class__ assignment. Returns the list of
    swapped modules (4 per block). Call on the STUDENT backbone only, after the teacher deepcopy,
    after .cuda(), before torch.compile and before HandRolledDP."""
    swapped = []
    for i, blk in enumerate(backbone.blocks):
        cands = [("blocks.%d.qkv" % i, blk.qkv), ("blocks.%d.proj" % i, blk.proj)]
        cands += [("blocks.%d.mlp.%s" % (i, n), m) for n, m in blk.mlp.named_children()
                  if isinstance(m, nn.Linear)]
        for name, m in cands:
            if type(m) is not nn.Linear:
                raise TypeError("fp8 swap: %s is %s, expected nn.Linear" % (name, type(m).__name__))
            if m.in_features % 16 or m.out_features % 16:
                raise ValueError("fp8 swap: %s weight %s; _scaled_mm needs K %% 16 == 0 and "
                                 "N %% 16 == 0" % (name, tuple(m.weight.shape)))
            m.__class__ = ScoutFP8Linear
            swapped.append(m)
    expected = 4 * len(backbone.blocks)
    if len(swapped) != expected:
        raise RuntimeError("fp8 swap: swapped %d linears, expected %d" % (len(swapped), expected))
    return swapped


@contextlib.contextmanager
def fp8_scout(linears):
    """Quantize the weights of `linears` ONCE, raise the switch, yield, then clear both.
    The weights do not change inside one scout call (the optimizer step comes after the scout),
    so one quantization serves every chunk of the call. Memory while inside: the fp8 weights
    (85 MB for ViT-B) plus one small fp32 scale vector per linear."""
    global _FP8_ON
    if _FP8_ON:
        raise RuntimeError("fp8_scout() is not re-entrant")
    try:
        with torch.no_grad():
            for m in linears:
                w8, s = quantize_rowwise(m.weight.detach())   # w [N,K] -> s [N,1]
                m._w8 = w8                                     # fp8 [N,K] row-major
                m._s_w = s.t().contiguous()                    # fp32 [1,N] contiguous
        _FP8_ON = True
        yield
    finally:
        _FP8_ON = False
        for m in linears:
            m._w8 = None
            m._s_w = None
