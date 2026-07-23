"""
CPU unit tests for the counted-coverage typicality bank. No GPU, no training, no pytest --
run with:  python typicality/test_counted_coverage.py

Covers the mechanics (hit assignment, two-hit graduation, LFU eviction, reserve bounds +
exit accounting, cold-start->counted), the self-tuning s sweep, the self-tuning j (variogram
L + under-resolution), the noise-aware soft rank (limits), the Var(lambda_hat) identity,
determinism (incl. the j sweep and soft-rank), checkpoint round-trip, and that the distance
bank is gone.
"""
import os, sys, io, math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typicality import CountedCoverageBank, TypicalityScorer  # noqa: E402

torch.manual_seed(0)


# ----------------------------------------------------------------- helpers
def far_points(n, kp=2, step=10.0):
    x = torch.zeros(n, kp)
    x[:, 0] = torch.arange(n).float() * step
    return x


def counted(M=4, kp=2, s=0.5, j=2, H=4, T_need=3, res=2, readout='pit', pit=8,
            grad_hits=2, pool_selftune=False, soft_rank=False):
    """Counted bank with the hit radius PINNED to s for the mechanics tests (self-tuning is
    exercised separately). pool_selftune off and soft_rank off by default here so mechanics
    are isolated from the readout self-tuning."""
    b = CountedCoverageBank(M=M, K_prime=kp, pool_j=j, halflife_steps=H,
                            reserve_residency=T_need, reserve_size=res, readout=readout,
                            pit_buffer=pit, s_buffer_size=2000, graduation_hits=grad_hits,
                            pool_selftune=pool_selftune, soft_rank=soft_rank)
    b.s.fill_(float(s)); b.s_ready.fill_(1)
    return b


def fill_established(b):
    pts = far_points(b.M, b.K_prime, step=10.0)
    b.score_and_update(pts, 0)
    assert b._ne() == b.M, f"fill failed: n_est={b._ne()} != M={b.M}"
    return pts


def _seed_on_miss_ref(buf, s, M):
    centers = []
    for i in range(buf.shape[0]):
        if len(centers) >= M:
            break
        x = buf[i]
        if not centers:
            centers.append(x); continue
        if float((torch.stack(centers) - x).abs().sum(dim=1).min()) > s:
            centers.append(x)
    return len(centers)


def _ar_field_bank(M, kp, spacing, L0, seed=0, halflife=250):
    """Full, mature bank: positions on a line at `spacing`; lambda an AR(1) field with
    phi=exp(-spacing/L0) so Corr(lam_i,lam_j)=exp(-r/L0) -> a known correlation length L0."""
    torch.manual_seed(seed)
    b = CountedCoverageBank(M=M, K_prime=kp, pool_selftune=True, pool_ema=1.0, halflife_steps=halflife)
    pos = torch.zeros(M, kp); pos[:, 0] = torch.arange(M).float() * spacing
    phi = math.exp(-spacing / L0)
    ar = torch.zeros(M); ar[0] = torch.randn(1).item()
    nz = math.sqrt(max(1e-6, 1 - phi * phi))
    for i in range(1, M):
        ar[i] = phi * ar[i - 1] + nz * torch.randn(1).item()
    lam = (1.0 + 0.3 * ar).clamp(min=0.05)
    b.n_est.fill_(M)
    b.est_b[:M] = pos
    b.est_E[:M] = b.mature_E
    b.est_S[:M] = lam * b.mature_E                   # so S/E = lam
    return b


# ----------------------------------------------------------------- tests
def test_hit_assignment():
    """Within s -> nearest signature's S += 1; beyond s -> seed (established during fill)."""
    b = counted(M=8)
    batch = torch.cat([torch.tensor([[0., 0.]]), torch.tensor([[0.1, 0.]]),
                       torch.tensor([[10., 0.]])], 0)
    b.score_and_update(batch, 0)
    assert b._ne() == 2
    assert abs(b.est_S[0].item() - 1.0) < 1e-6 and abs(b.est_S[1].item()) < 1e-6


def test_graduation_two_hits():
    """A reserve signature is NOT promoted on one hit; it is on two. Established stays <= M."""
    b = counted(M=2, s=0.5, grad_hits=2, T_need=100, res=6)
    fill_established(b)                                       # est at x=0, 10
    b.score_and_update(torch.tensor([[100., 0.]]), 1)        # seed a reserve signature
    assert b._nr() == 1 and int(b.graduations.item()) == 0
    b.score_and_update(torch.tensor([[100.05, 0.]]), 2)      # hit #1 -> not yet
    assert int(b.graduations.item()) == 0, "one hit must not graduate"
    assert b._nr() == 1, "reserve signature still pending after one hit"
    b.score_and_update(torch.tensor([[100.05, 0.]]), 3)      # hit #2 -> graduate
    assert int(b.graduations.item()) == 1, "two hits must graduate"
    assert b._ne() == b.M and b._nr() == 0
    assert any(abs(b.est_b[i, 0].item() - 100.0) < 1e-6 for i in range(b.M))


def test_reserve_bounded_and_exit_counts():
    """Reserve occupancy stays <= cap under continuous novel input; per-step exit counters
    (graduated/expired/flushed) sum to the number of departures that step."""
    b = counted(M=2, s=0.5, T_need=3, res=5, grad_hits=2)
    fill_established(b)
    total_departures = 0
    for k in range(60):
        b.score_and_update(torch.tensor([[100.0 + 5.0 * k, 0.0]]), k)   # each new, > s from all
        assert b._ne() == b.M and b._nr() <= b.reserve_cap
        h = b.health()
        dep = h['step_grad'] + h['step_expired'] + h['step_flushed']
        assert len(b._step_exit_ages) == dep, "exit ages recorded must equal exit-counter sum"
        total_departures += dep
    assert total_departures > 0, "with small T_need, entries must have departed"


def test_lfu_evicts_idle_first():
    """On a graduation into a full established set, the lowest-rate (idle) signature is evicted."""
    b = counted(M=3, s=0.5, T_need=100, res=2, grad_hits=1)   # 1-hit graduation to force eviction
    fill_established(b)                                        # est at x=0,10,20
    for _ in range(3):
        b.score_and_update(torch.tensor([[10.05, 0.], [20.05, 0.]]), 0)   # hit 1 & 2, not 0
    idle = b.est_b[0].clone()
    assert b.est_S[0].item() == 0.0
    b.score_and_update(torch.tensor([[200., 0.], [200.05, 0.]]), 0)       # seed + hit -> graduate
    assert int(b.graduations.item()) >= 1
    assert not any(abs(b.est_b[i, 0].item() - idle[0].item()) < 1e-6 for i in range(b.M))
    assert any(abs(b.est_b[i, 0].item() - 200.0) < 1e-6 for i in range(b.M))


def test_coldstart_then_counted_and_range():
    """Not ready while filling; cold-start distance readout when full-but-immature; counted
    readout after maturation. t in [0,1] throughout."""
    b = counted(M=4, s=0.5, H=4, readout='pit', pit=64)
    out = b.score_and_update(far_points(3, b.K_prime, step=10.0), 0)
    assert out['ready'] is False and out['t'] is None
    b.score_and_update(far_points(4, b.K_prime, step=10.0)[3:4], 1)
    assert b._ne() == b.M and b._matured() is False
    out = b.score_and_update(torch.tensor([[5.0, 0.0], [15.0, 0.0]]), 2)
    assert out['ready'] and torch.all(out['t'] >= 0.0) and torch.all(out['t'] <= 1.0)
    for it in range(3, 12):
        b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.], [20.05, 0.], [30.05, 0.]]), it)
    assert b._matured() is True
    out = b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.]]), 12)
    assert torch.all(out['t'] >= 0.0) and torch.all(out['t'] <= 1.0)
    assert int(b.pit_filled.item()) > 0 and torch.isfinite(out['t']).all()


def test_j_selftune_recovers_L():
    """On synthetic AR(1) rates with a known correlation length, the variogram fit recovers L
    and j lands near the count of signatures within one L (~2 L/spacing on a line)."""
    M, kp, spacing, L0 = 400, 3, 1.0, 12.0
    b = _ar_field_bank(M, kp, spacing, L0, seed=0)
    b._selftune_pool()
    L = float(b.L_est.item())
    assert L0 / 3 < L < L0 * 3, f"fitted L={L:.2f} did not recover L0={L0}"
    j = int(b.j.item())
    expected = 2 * L0 / spacing
    assert 0.3 * expected < j < 3 * expected, f"j={j} not near count-within-L ~{expected:.0f}"
    assert int(b.underresolved.item()) == 0, "well-resolved (spacing < L) must not flag under-res"


def test_j_selftune_underresolved():
    """Coarse budget: uniform spacing (30) far above the field's structure (L0=3), so even the
    closest sampled pairs are decorrelated (a flat variogram). underresolved is set and j falls
    back to the variance target, not to 1. A short half-life makes that floor well above 1."""
    b = _ar_field_bank(M=200, kp=3, spacing=30.0, L0=3.0, seed=1, halflife=2)
    b._selftune_pool()
    L = float(b.L_est.item())
    assert int(b.underresolved.item()) == 1, f"coarse budget must flag under-res (L={L:.2f})"
    assert int(b.j.item()) >= 2, f"fallback must be the variance target, not 1 (j={int(b.j.item())})"


def test_soft_rank_limits():
    """soft rank -> hard PIT as sigma -> 0 ; -> 1/2 as noise dominates ; t in [0,1]."""
    torch.manual_seed(2)
    b = CountedCoverageBank(M=8, K_prime=3, pit_buffer=600, soft_rank=True)
    W = 400
    b.pit_ring[:W] = torch.randn(W) * 5.0                     # well-spread reference
    b.pit_sig2_ring[:W] = torch.zeros(W)                      # ~0 reference noise
    b.pit_filled.fill_(W)
    q = torch.randn(32) * 5.0
    soft0 = b._soft_rank_score(q, torch.zeros(32))
    pit = b._pit_score(q)
    assert torch.allclose(soft0, pit, atol=1e-2), f"soft(sig->0) != PIT: {(soft0 - pit).abs().max():.3f}"
    soft_big = b._soft_rank_score(q, torch.full((32,), 1e12))
    assert torch.allclose(soft_big, torch.full((32,), 0.5), atol=1e-3), "noise-dominated -> 1/2"
    assert torch.all(soft0 >= 0) and torch.all(soft0 <= 1)


def test_variance_identity():
    """A direct simulation of S <- eta S + Poisson(lambda) reproduces
    Var(lambda_hat) = lambda (1-eta)/(1+eta) at steady state (E = 1/(1-eta))."""
    torch.manual_seed(0)
    eta, lam, nsteps, ntrials = 0.9, 3.0, 1500, 6000
    S = torch.zeros(ntrials)
    for _ in range(nsteps):
        S = eta * S + torch.poisson(torch.full((ntrials,), lam))
    E = 1.0 / (1.0 - eta)
    lam_hat = S / E
    var_emp = float(lam_hat.var(unbiased=True).item())
    var_theory = lam * (1.0 - eta) / (1.0 + eta)
    rel = abs(var_emp - var_theory) / var_theory
    assert rel < 0.10, f"Var(lam_hat)={var_emp:.4f} vs theory {var_theory:.4f} (rel {rel:.3f})"


def _selftune_cfg():
    return dict(M=8, K_prime=3, halflife_steps=3, s_buffer_size=64, s_min_buffer=32,
                s_sweep_interval=5, pit_buffer=64, pool_selftune=True, soft_rank=True, reserve_size=6)


def test_determinism_full_path():
    """Two instances fed the identical sequence end byte-identical -- including the s and j
    sweeps and the soft-rank readout."""
    torch.manual_seed(7)
    seqs = [torch.rand(16, 3) * 5.0 for _ in range(120)]
    b1 = CountedCoverageBank(**_selftune_cfg()); b2 = CountedCoverageBank(**_selftune_cfg())
    for it, s in enumerate(seqs):
        b1.score_and_update(s.clone(), it); b2.score_and_update(s.clone(), it)
    assert int(b1.s_ready.item()) == 1 and b1._ne() == b1.M, "path should have activated + filled"
    assert torch.equal(b1.sync_fingerprint(), b2.sync_fingerprint()), "banks diverged"


def test_checkpoint_roundtrip():
    """state_dict save/load restores all state (incl. the new j / soft-rank buffers) exactly."""
    torch.manual_seed(3)
    a = CountedCoverageBank(**_selftune_cfg())
    for it in range(80):
        a.score_and_update(torch.rand(16, 3) * 5.0, it)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = CountedCoverageBank(**_selftune_cfg())
    b.load_state_dict(sd)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint())
    nxt = torch.rand(16, 3) * 5.0
    a.score_and_update(nxt.clone(), 80); b.score_and_update(nxt.clone(), 80)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint()), "restored state must continue identically"


def test_checkpoint_size_change_on_resume():
    """Reproduces the resume crash: an OLDER checkpoint (smaller reserve, and without the buffers
    added later -- res_hits, pit_sig2_ring, j, ...) must load into the current model without a
    shape-mismatch RuntimeError. The established set, counters, and s restore; the reserve fits the
    new cap; the missing new buffers keep their init values."""
    torch.manual_seed(4)
    a = CountedCoverageBank(**{**_selftune_cfg(), 'reserve_size': 4})
    for it in range(80):
        a.score_and_update(torch.rand(16, 3) * 5.0, it)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    for k in ('res_hits', 'pit_sig2_ring', 'j', 'j_smooth', 'L_est', 'h_over_L',
              'underresolved', 's_delta'):
        sd.pop(k, None)                                     # simulate a pre-refactor checkpoint
    b = CountedCoverageBank(**{**_selftune_cfg(), 'reserve_size': 12})   # larger reserve now
    b.load_state_dict(sd)                                   # must NOT raise (was RuntimeError)
    assert b.res_b.shape[0] == 12 and int(b.n_res.item()) <= 12
    assert torch.equal(a.est_b, b.est_b) and torch.equal(a.est_S, b.est_S)
    assert torch.equal(a.n_est, b.n_est) and float(a.s.item()) == float(b.s.item())
    nr = int(a.n_res.item())
    if nr > 0:
        assert torch.equal(a.res_b[:nr], b.res_b[:nr]), "valid reserve prefix must carry over"


def test_distance_bank_gone():
    """The distance bank is removed; only the counted bank is exported."""
    import typicality
    assert 'TypicalityBank' not in typicality.__all__
    try:
        from typicality import TypicalityBank  # noqa: F401
        raise AssertionError("TypicalityBank should not be importable")
    except ImportError:
        pass
    bank = CountedCoverageBank(M=8, K_prime=4)
    for m in ('score_and_update', 'sync_fingerprint', 'health', 'compact_line', 'stats'):
        assert hasattr(bank, m), f"missing {m}"


def test_compact_line_renders():
    """compact_line renders a single 'typ | ... | OK/flags' status string from synthetic state."""
    b = counted(M=4, s=19.6)
    b.s.fill_(19.63); b.s_delta.fill_(-0.08); b.j.fill_(64); b.h_over_L.fill_(1.05)
    line = b.compact_line(grad_per_step=2.1, t_mean=0.501, t_std=0.288, beta=0.5)
    assert line.startswith('typ |') and ('OK' in line or '!' in line)
    assert 's=19.63(-0.08)' in line and 'j=64' in line and 'h/L=1.05' in line


# ---------------------------------------------- self-tuning s (unchanged machinery)
def test_s_recovers_scale():
    torch.manual_seed(11)
    M, kp = 64, 4
    base = torch.rand(1200, kp)

    def edge_for(scale):
        b = CountedCoverageBank(M=M, K_prime=kp, s_buffer_size=1400, s_min_buffer=1)
        b._push_buffer(base * scale)
        return b._sweep_edge()

    e1, e6 = edge_for(1.0), edge_for(6.0)
    assert 5.4 < e6 / e1 < 6.6, f"edge ratio {e6 / e1:.3f} not ~6x"


def test_s_edge_matches_bruteforce():
    torch.manual_seed(5)
    M, kp, n = 40, 3, 500
    b = CountedCoverageBank(M=M, K_prime=kp, s_grid_points=9, s_min_buffer=1, s_buffer_size=n)
    b._push_buffer(torch.rand(n, kp))
    edge = b._sweep_edge()
    buf = b.sig_ring[:n].float()
    m = b._buffer_scale(buf); lo, hi = b.grid_span
    fine = torch.linspace(lo * m, hi * m, 300).tolist()
    bf = max((sc for sc in fine if _seed_on_miss_ref(buf, sc, M) >= M), default=lo * m)
    assert abs(edge - bf) <= (hi * m - lo * m) / 8.0


def test_s_determinism():
    torch.manual_seed(13)
    buf = torch.rand(800, 4)
    b1 = CountedCoverageBank(M=64, K_prime=4, s_buffer_size=800, s_min_buffer=1)
    b2 = CountedCoverageBank(M=64, K_prime=4, s_buffer_size=800, s_min_buffer=1)
    b1._push_buffer(buf.clone()); b2._push_buffer(buf.clone())
    assert b1._sweep_edge() == b2._sweep_edge()


def test_sweep_read_only():
    torch.manual_seed(17)
    b = CountedCoverageBank(M=16, K_prime=3, s_buffer_size=600, s_min_buffer=1)
    b.s.fill_(0.2); b.s_ready.fill_(1)
    for it in range(6):
        b.score_and_update(torch.rand(30, 3), it)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    _ = b._sweep_edge()
    for k, v in before.items():
        assert torch.equal(b.state_dict()[k], v), f"_sweep_edge mutated {k}"


def test_startup_gate():
    b = CountedCoverageBank(M=8, K_prime=3, s_buffer_size=200, s_min_buffer=50, s_sweep_interval=500)
    it = 0
    while int(b.sig_filled.item()) < 40:
        out = b.score_and_update(torch.rand(10, 3), it); it += 1
        assert out['ready'] is False and out['t'] is None
        assert b._ne() == 0 and float(b.s.item()) == 0.0 and int(b.s_ready.item()) == 0
    while int(b.s_ready.item()) == 0:
        b.score_and_update(torch.rand(10, 3), it); it += 1
    assert float(b.s.item()) > 0.0 and b._ne() > 0


def test_s_ema_smoothing():
    torch.manual_seed(19)
    b = CountedCoverageBank(M=32, K_prime=3, s_buffer_size=600, s_min_buffer=300,
                            s_sweep_interval=5, s_ema_alpha=0.2)
    it = 0
    while int(b.s_ready.item()) == 0:
        b.score_and_update(torch.rand(60, 3), it); it += 1
    s_prev = float(b.s.item()); last = int(b.last_sweep_step.item())
    b.sig_ring.zero_(); b.sig_ptr.zero_(); b.sig_filled.zero_()
    b._push_buffer(torch.rand(600, 3) * 3.0)
    raw_edge = b._sweep_edge()
    assert raw_edge > 1.5 * s_prev
    b.score_and_update(torch.rand(60, 3) * 3.0, last + b.s_sweep_interval)
    s_new = float(b.s.item())
    assert s_prev < s_new < raw_edge
    assert s_new < s_prev + 0.5 * (raw_edge - s_prev)


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
