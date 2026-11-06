"""
CPU unit tests for the v2 keep rule (typicality/keep_rule_v2.py). No GPU, no training,
no pytest. Run with:  python typicality/test_keep_rule_v2.py

Covers: the sure share (>= u0; == ceil(u0 n)/n when there are no ties), m >= u0, f_next in
[1, cap] with cap = 1/u0 or min(1/u0, keep_f_max), n_draws = ceil(f_next), the capped flag,
the zero guard, q == 1 at p_hat <= c and strictly decreasing above c, the refill count of a
simulated keep step, and rank agreement (4 simulated ranks on the same P give the same c, m
and n_draws). Runs in seconds.
"""
import os, sys, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typicality.keep_rule_v2 import keep_chance_v2, f_update_v2, f_cap_v2  # noqa: E402

W, N = 4, 256
U0S = (0.01, 0.05, 0.10)
AS = (1.0, 1.5)


def synthetic_P(n, z, seed, sigma=1.5):
    """Log-normal p_hat with a share z of exact zeros. float32 like the bank readout."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(n, generator=g, dtype=torch.float64).mul(sigma).exp().float()
    nz = int(round(z * n))
    if nz > 0:
        P[torch.randperm(n, generator=g)[:nz]] = 0.0
    return P


def test_sure_share_and_m():
    n = W * 6 * N
    for seed in range(5):
        for z in (0.0, 0.02):
            for u0 in U0S:
                for a in AS:
                    P = synthetic_P(n, z, seed)
                    q, c, guard, sure, zero, m = keep_chance_v2(P, a, u0)
                    k = max(1, math.ceil(u0 * n))
                    n_sure = int((P <= c).sum().item())
                    assert n_sure >= k, (n_sure, k)
                    assert sure >= u0 - 1e-6, (sure, u0)
                    assert m >= u0 - 1e-6, (m, u0)
                    assert abs(m - float(q.mean())) < 1e-6
                    assert abs(zero - z) <= 1.0 / n + 1e-6
                    assert bool((q > 0).all()) and bool((q <= 1).all())
                    if z < u0:
                        # zeros do not reach the quantile; continuous values above -> no ties
                        assert not guard
                        ties = int((P == c).sum().item())
                        assert ties == 1 and n_sure == k, (ties, n_sure, k)
                        assert bool((q[P <= c] == 1.0).all()) and bool((q[P > c] < 1.0).all())


def test_zero_guard():
    n = W * 6 * N
    P = synthetic_P(n, 0.05, seed=11)                        # 5% zeros
    q, c, guard, sure, zero, m = keep_chance_v2(P, 1.0, 0.01)   # u0 < z -> quantile is 0
    assert guard is True
    assert c == float(P[P > 0].min())
    assert bool((q[P == 0] == 1.0).all())
    assert sure >= zero and m >= 0.01
    q, c, guard, sure, zero, m = keep_chance_v2(torch.zeros(100), 1.5, 0.05)   # no positive value
    assert guard is True and c == 0.0 and sure == 1.0 and zero == 1.0 and m == 1.0
    assert bool((q == 1.0).all())
    P1 = torch.zeros(100); P1[7] = 3.0                       # one positive value
    q, c, guard, sure, zero, m = keep_chance_v2(P1, 1.0, 0.05)
    assert guard is True and c == 3.0 and bool((q == 1.0).all())


def test_q_shape():
    c0 = 0.75                                                 # exact in binary
    above = c0 * (2.0 ** torch.arange(1, 12, dtype=torch.float32))   # strictly increasing above c
    below = c0 * torch.tensor([0.0, 0.1, 0.5, 0.999, 1.0])
    P = torch.cat([below, above])
    n = P.numel(); nb = below.numel()
    u0 = (nb - 0.5) / n                                       # ceil(u0 n) = nb -> c == c0
    for a in AS:
        q, c, guard, sure, zero, m = keep_chance_v2(P, a, u0)
        assert c == c0 and not guard and abs(sure - nb / n) < 1e-6
        assert bool((q[:nb] == 1.0).all())
        qa = q[nb:]
        assert bool((qa < 1.0).all())
        assert bool((qa[1:] < qa[:-1]).all())                 # strictly decreasing above c
        assert torch.allclose(qa, (c0 / above).pow(a))


def test_f_bounds_and_cap():
    for u0 in U0S:
        cap0 = 1.0 / u0
        assert f_cap_v2(u0) == cap0 and f_cap_v2(u0, 0.0) == cap0 and f_cap_v2(u0, -1.0) == cap0
        assert f_cap_v2(u0, 3.0) == min(cap0, 3.0) and f_cap_v2(u0, 1e9) == cap0
        for m0 in (u0, 0.5 * (u0 + 1.0), 1.0, 1e-6):           # 1e-6 is out of contract; must clamp
            m_ema, f_raw, f, nd, capped = f_update_v2(m0, None, 0.99, 1.1, u0)
            assert m_ema == m0                                # first v2 step
            assert f_raw >= 1.0 and 1.0 - 1e-12 <= f <= cap0 + 1e-9
            assert 1 <= nd <= math.ceil(cap0) and nd == math.ceil(f)
            assert capped == (f_raw > cap0) and f == min(f_raw, cap0)
    m_ema, f_raw, f, nd, capped = f_update_v2(0.2, 0.4, 0.99, 1.1, 0.05)
    assert abs(m_ema - (0.99 * 0.4 + 0.01 * 0.2)) < 1e-15
    assert abs(f - 1.1 / m_ema) < 1e-12 and nd == math.ceil(f) and not capped
    r = f_update_v2(0.001, None, 0.99, 1.1, 0.05)           # clamps at 1/u0
    assert r[2] == 1.0 / 0.05 and r[4] is True and r[1] > r[2]
    r = f_update_v2(1.0, None, 0.99, 0.5, 0.05)             # clamps at 1 (f_raw = max(.., 1))
    assert r[1] == 1.0 and r[2] == 1.0 and r[3] == 1 and r[4] is False
    # --keep_f_max: extra cap below 1/u0; a cap above 1/u0 has no effect
    r = f_update_v2(0.05, None, 0.99, 1.1, 0.01, f_max=16.0)
    assert r[1] == 1.1 / 0.05 and r[2] == 16.0 and r[3] == 16 and r[4] is True
    r = f_update_v2(0.05, None, 0.99, 1.1, 0.01, f_max=500.0)
    assert r[2] == 1.1 / 0.05 and r[3] == 22 and r[4] is False
    r = f_update_v2(0.5, None, 0.99, 1.1, 0.01, f_max=16.0)  # f_raw 2.2 < cap -> not capped
    assert r[2] == 1.1 / 0.5 and r[3] == 3 and r[4] is False
    m_ema = None                                              # EMA converges to a constant m
    for _ in range(2000):
        m_ema, f_raw, f, nd, capped = f_update_v2(0.25, m_ema, 0.99, 1.1, 0.05)
    assert abs(m_ema - 0.25) < 1e-9 and nd == 5


def _keep_step(q_loc, g):
    """The trainer keep step (Bernoulli, first N, refill from the highest-q rejects)."""
    pool_n = q_loc.numel()
    acc = torch.nonzero(torch.rand(pool_n, generator=g) < q_loc, as_tuple=False).flatten()
    if acc.numel() >= N:
        return acc[:N], acc, 0
    rej = torch.ones(pool_n, dtype=torch.bool); rej[acc] = False
    rej_idx = torch.nonzero(rej, as_tuple=False).flatten()
    order = torch.argsort(q_loc[rej_idx], descending=True)
    keep = torch.cat([acc, rej_idx[order[:N - acc.numel()]]])
    return keep, acc, int(N - acc.numel())


def test_refill_count():
    g = torch.Generator().manual_seed(3)
    pool_n = 6 * N
    for u0 in U0S:
        for a in AS:
            P = synthetic_P(W * pool_n, 0.01, seed=5)
            q, c, *_ = keep_chance_v2(P, a, u0)
            for r in range(W):
                q_loc = q[r * pool_n:(r + 1) * pool_n]        # _local_rows
                for scale in (1.0, 0.05):                     # 0.05 forces the refill path
                    keep, acc, refill = _keep_step(q_loc * scale, g)
                    assert keep.numel() == N and keep.unique().numel() == N
                    assert refill == max(0, N - int(acc.numel()))
                    if refill > 0:
                        filled = keep[acc.numel():]
                        rest = torch.ones(pool_n, dtype=torch.bool); rest[keep] = False
                        if bool(rest.any()):
                            assert float(q_loc[filled].min()) >= float(q_loc[rest].max())


def test_rank_agreement():
    for seed in range(3):
        P = synthetic_P(W * 6 * N, 0.01, seed)
        outs = [keep_chance_v2(P.clone(), 1.5, 0.05) for _ in range(W)]   # 4 simulated ranks
        nds, emas = set(), set()
        for q, c, guard, sure, zero, m in outs:
            assert c == outs[0][1] and m == outs[0][5] and torch.equal(q, outs[0][0])
            m_ema, f_raw, f, nd, capped = f_update_v2(m, 0.3, 0.99, 1.1, 0.05)
            nds.add(nd); emas.add(m_ema)
        assert len(nds) == 1 and len(emas) == 1
        q64, c64, *_ = keep_chance_v2(P.double(), 1.5, 0.05)  # dtype-agnostic threshold
        assert c64 == outs[0][1]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
