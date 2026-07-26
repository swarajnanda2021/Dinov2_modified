"""
CPU unit tests for the counted-coverage typicality bank. No GPU, no training, no pytest --
run with:  python typicality/test_counted_coverage.py

Covers the mechanics (hit assignment, two-hit graduation, LFU eviction, reserve bounds +
exit accounting, cold-start->counted), the self-tuning s sweep, the self-tuning j (variogram
L + under-resolution; now a diagnostic), the fixed-radius absolute readout (scale sensitivity --
the property the old fixed-count readout lacked -- zero-neighbour finite weight, p_ref init/EMA,
weight scale-invariance through the real DINOLoss weighted path), the Var(lambda_hat) identity,
determinism (incl. the j sweep and p_ref), checkpoint round-trip, and that the distance bank is
gone.
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


def counted(M=4, kp=2, s=0.5, j=2, H=4, T_need=3, res=2,
            grad_hits=2, pool_selftune=False, radius_mult=1.5):
    """Counted bank with the hit radius PINNED to s for the mechanics tests (self-tuning is
    exercised separately). pool_selftune off by default here so mechanics are isolated from the
    readout self-tuning."""
    b = CountedCoverageBank(M=M, K_prime=kp, pool_j=j, halflife_steps=H,
                            reserve_residency=T_need, reserve_size=res,
                            s_buffer_size=2000, graduation_hits=grad_hits,
                            pool_selftune=pool_selftune, radius_mult=radius_mult)
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


def test_coldstart_then_counted_and_pref():
    """Not ready while filling; ready-but-NO-modulation (p_hat is None) when full-but-immature;
    fixed-radius density p_hat (>=0, finite -- NOT a bounded rank) after maturation; and p_ref
    initialised from the first matured batch's median, not before."""
    b = counted(M=4, s=0.5, H=4)
    out = b.score_and_update(far_points(3, b.K_prime, step=10.0), 0)
    assert out['ready'] is False and out['p_hat'] is None
    b.score_and_update(far_points(4, b.K_prime, step=10.0)[3:4], 1)
    assert b._ne() == b.M and b._matured() is False
    out = b.score_and_update(torch.tensor([[5.0, 0.0], [15.0, 0.0]]), 2)
    assert out['ready'] and out['p_hat'] is None, "cold start must apply no modulation"
    assert int(b.p_ref_init.item()) == 0, "p_ref must not be set before maturation"
    for it in range(3, 12):
        b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.], [20.05, 0.], [30.05, 0.]]), it)
    assert b._matured() is True
    out = b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.]]), 12)
    ph = out['p_hat']
    assert ph is not None and torch.all(ph >= 0.0) and torch.isfinite(ph).all()
    assert int(b.p_ref_init.item()) == 1 and float(b.p_ref.item()) > 0.0


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


def _matured_bank(positions, lam, kp, s, radius_mult=1.5, H=250):
    """Full, mature bank with est_b = positions and S/E = lam, hit radius pinned to s. Lets the
    fixed-radius readout be exercised directly (p_ref starts unset; _counted_readout inits it)."""
    M = positions.shape[0]
    b = CountedCoverageBank(M=M, K_prime=kp, halflife_steps=H, radius_mult=radius_mult,
                            pool_selftune=False)
    b.n_est.fill_(M)
    b.est_b[:M] = positions
    b.est_E[:M] = b.mature_E
    b.est_S[:M] = lam * b.mature_E                            # so S/E = lam
    b.s.fill_(float(s)); b.s_ready.fill_(1)
    return b


def test_scale_sensitivity():
    """The property the OLD fixed-count readout lacked (it was exactly invariant to local
    rescaling): p_hat MUST respond to how crowded the neighbourhood is. Scale every query->anchor
    distance by c; within the fixed radius R_rad, p_hat must move monotonically -- up as the
    neighbourhood contracts (c down), down as it spreads (c up), and 0 once it clears the radius."""
    torch.manual_seed(21)
    kp = 3
    q = torch.zeros(1, kp)
    offsets = torch.randn(12, kp) * 0.2                      # anchors clustered near the query
    s, radius_mult = 1.0, 1.5                                # R_rad = 1.5
    p = {}
    for c in (0.5, 1.0, 2.0, 10.0):
        b = _matured_bank(offsets * c, lam=1.0, kp=kp, s=s, radius_mult=radius_mult)
        p[c] = float(b._counted_readout(q)[0].item())
    assert p[0.5] > p[1.0] > p[2.0] > p[10.0], f"p_hat not monotone in crowding: {p}"
    assert p[10.0] == 0.0, f"neighbourhood scaled past the radius must give p_hat=0: {p}"


def test_zero_neighbours_finite_weight():
    """No anchor inside R_rad -> p_hat = 0, and absolute_weights still returns a FINITE weight
    1/c^a (no clipping needed)."""
    kp = 3
    anchors = torch.zeros(6, kp); anchors[:, 0] = torch.arange(6).float() * 5.0
    b = _matured_bank(anchors, lam=1.0, kp=kp, s=1.0, radius_mult=1.5)     # R_rad = 1.5
    q = torch.tensor([[100.0, 0.0, 0.0]])                                  # far from every anchor
    ph = b._counted_readout(q)
    assert float(ph[0].item()) == 0.0, "no neighbour in radius -> p_hat = 0"
    p_ref = torch.tensor(2.0)
    for a in (0.5, 1.0):
        w = TypicalityScorer.absolute_weights(ph, p_ref, a=a, c_frac=0.25)
        c = 0.25 * 2.0
        assert torch.isfinite(w).all() and abs(float(w[0].item()) - 1.0 / c ** a) < 1e-5


def test_pref_init_then_ema():
    """First matured readout initialises p_ref from the batch median (never 0); the next step
    EMAs it toward the new median with weight 0.001."""
    torch.manual_seed(22)
    kp = 3
    anchors = torch.randn(16, kp)
    b = _matured_bank(anchors, lam=1.0, kp=kp, s=1.0, radius_mult=1.5)
    assert int(b.p_ref_init.item()) == 0 and float(b.p_ref.item()) == 0.0
    q1 = torch.randn(64, kp) * 0.2
    med1 = float(torch.median(b._counted_readout(q1)).item())
    assert int(b.p_ref_init.item()) == 1
    assert abs(float(b.p_ref.item()) - med1) < 1e-6, "p_ref must init from the first batch median"
    q2 = torch.randn(64, kp) * 0.2
    med2 = float(torch.median(b._counted_readout(q2)).item())
    expected = 0.999 * med1 + 0.001 * med2
    assert abs(float(b.p_ref.item()) - expected) < 1e-5, "p_ref must EMA at weight 0.001 after init"


def test_pref_never_zero_on_empty_first_batch():
    """1c guardrail: if the FIRST matured batch's median p_hat is 0 (no median-tile neighbour --
    a cold/mistuned bank), p_ref must still initialise strictly positive so c > 0 and the weights
    stay finite, rather than latching p_ref = 0 and diverging to inf."""
    kp = 3
    anchors = torch.zeros(6, kp); anchors[:, 0] = torch.arange(6).float() * 5.0
    b = _matured_bank(anchors, lam=1.0, kp=kp, s=1.0, radius_mult=1.5)      # R_rad = 1.5
    far = torch.tensor([[100.0, 0.0, 0.0]]).repeat(8, 1)                    # every tile empty
    ph = b._counted_readout(far)
    assert float(torch.median(ph).item()) == 0.0 and int(b.p_ref_init.item()) == 1
    assert float(b.p_ref.item()) > 0.0, "p_ref must not latch to 0 on an all-empty first batch"
    w = TypicalityScorer.absolute_weights(ph, b.p_ref, a=1.0, c_frac=0.25)
    assert torch.isfinite(w).all(), "weights must stay finite (c > 0)"


def test_weight_scale_invariance_dino():
    """Scaling every p_hat AND p_ref by the same constant leaves the NORMALISED weighted DINO
    loss unchanged: w = 1/(k*p_hat + k*c)^a = k^-a * w, and (loss*w).sum()/w.sum() cancels the
    constant. Verified through the real DINOLoss.forward weighted path, not a reimplementation."""
    from losses.dino_loss import DINOLoss
    torch.manual_seed(23)
    ncrops, B, K = 2, 16, 64
    dino = DINOLoss(ncrops=ncrops, warmup_teacher_temp=0.04, teacher_temp=0.04,
                    warmup_teacher_temp_iters=10, student_temp=0.1, n_iterations=3)
    student = torch.randn(ncrops * B, K)
    teacher = torch.randn(ncrops * B, K) * 0.02      # small: Sinkhorn exp(teacher/temp) must not overflow
    p_hat = torch.rand(B)
    p_hat[:3] = 0.0                                          # include empty-neighbourhood tiles
    p_ref = torch.tensor(0.7)
    a, c_frac, k = 1.0, 0.25, 37.0
    w1 = TypicalityScorer.absolute_weights(p_hat, p_ref, a, c_frac)
    w2 = TypicalityScorer.absolute_weights(k * p_hat, k * p_ref, a, c_frac)
    l1 = dino(student, teacher, 0, sample_weights=w1)
    l2 = dino(student, teacher, 0, sample_weights=w2)
    assert torch.allclose(l1, l2, atol=1e-5, rtol=1e-4), f"scale-variant: {float(l1)} vs {float(l2)}"
    l0 = dino(student, teacher, 0, sample_weights=None)
    assert not torch.allclose(l1, l0, atol=1e-4), "weighted path must differ from the mean path"


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
                s_sweep_interval=5, pool_selftune=True, reserve_size=6)


def test_determinism_full_path():
    """Two instances fed the identical sequence end byte-identical -- including the s and j
    sweeps and the fixed-radius readout's p_ref update (p_ref is in the fingerprint)."""
    torch.manual_seed(7)
    seqs = [torch.rand(16, 3) * 5.0 for _ in range(120)]
    b1 = CountedCoverageBank(**_selftune_cfg()); b2 = CountedCoverageBank(**_selftune_cfg())
    for it, s in enumerate(seqs):
        b1.score_and_update(s.clone(), it); b2.score_and_update(s.clone(), it)
    assert int(b1.s_ready.item()) == 1 and b1._ne() == b1.M, "path should have activated + filled"
    assert int(b1.p_ref_init.item()) == 1, "path should have matured and exercised the p_ref update"
    assert torch.equal(b1.sync_fingerprint(), b2.sync_fingerprint()), "banks diverged"


def test_checkpoint_roundtrip():
    """state_dict save/load restores all state (incl. the new j / p_ref buffers) exactly."""
    torch.manual_seed(3)
    a = CountedCoverageBank(**_selftune_cfg())
    for it in range(80):
        a.score_and_update(torch.rand(16, 3) * 5.0, it)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = CountedCoverageBank(**_selftune_cfg())
    b.load_state_dict(sd)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint())
    assert float(a.p_ref.item()) == float(b.p_ref.item()) and \
        int(a.p_ref_init.item()) == int(b.p_ref_init.item()), "p_ref must round-trip exactly"
    nxt = torch.rand(16, 3) * 5.0
    a.score_and_update(nxt.clone(), 80); b.score_and_update(nxt.clone(), 80)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint()), "restored state must continue identically"


def test_checkpoint_size_change_on_resume():
    """Reproduces the resume crash: an OLDER checkpoint (smaller reserve, and without the buffers
    added later -- res_hits, p_ref / p_ref_init, j, ...) must load into the current model without
    a shape-mismatch RuntimeError. The established set, counters, and s restore; the reserve fits
    the new cap; the missing new buffers keep their init values."""
    torch.manual_seed(4)
    a = CountedCoverageBank(**{**_selftune_cfg(), 'reserve_size': 4})
    for it in range(80):
        a.score_and_update(torch.rand(16, 3) * 5.0, it)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    for k in ('res_hits', 'p_ref', 'p_ref_init', 'j', 'j_smooth', 'L_est', 'h_over_L',
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
    """compact_line renders a single 'typ | ... | OK/flags' status string from synthetic state,
    now reporting p/ref/w/ess (not t±std)."""
    b = counted(M=4, s=19.6)
    b.s.fill_(19.63); b.s_delta.fill_(-0.08); b.j.fill_(64); b.h_over_L.fill_(1.05)
    line = b.compact_line(grad_per_step=2.1, p_median=0.42, p_ref=0.55, w_mean=1.8, ess=0.69)
    assert line.startswith('typ |') and ('OK' in line or '!' in line)
    assert 's=19.63(-0.08)' in line and 'j=64' in line and 'h/L=1.05' in line
    assert 'ess=69%' in line and 'w=1.8' in line


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
        assert out['ready'] is False and out['p_hat'] is None
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
