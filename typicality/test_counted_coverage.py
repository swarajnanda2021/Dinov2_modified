"""
CPU unit tests for the switchable typicality bank (Algorithm 1 distance vs Algorithm 2
counted-coverage). No GPU, no training, no pytest — run with:  python typicality/test_counted_coverage.py

Each test is a function that asserts; main() runs them and prints PASS/FAIL.
"""
import os, sys, io
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typicality import TypicalityBank, CountedCoverageBank, TypicalityScorer  # noqa: E402

torch.manual_seed(0)


# ----------------------------------------------------------------- helpers
def far_points(n, kp=2, step=10.0):
    """n points each >> spot radius apart along axis 0."""
    x = torch.zeros(n, kp)
    x[:, 0] = torch.arange(n).float() * step
    return x


def counted(M=4, kp=2, s=0.5, j=2, H=4, T_need=3, res=2, readout='pit', pit=8):
    """Build a counted bank with the hit radius PINNED to `s` for the mechanics tests.
    Self-tuning is exercised separately (test_s_*); here we bypass the sweep by setting s
    directly and marking it ready, so seeding/hits use the fixed radius. s_buffer_size is
    kept tiny so construction is cheap."""
    b = CountedCoverageBank(M=M, K_prime=kp, pool_j=j, halflife_steps=H,
                            reserve_residency=T_need, reserve_size=res, readout=readout,
                            pit_buffer=pit, s_buffer_size=2000)
    b.s.fill_(float(s))
    b.s_ready.fill_(1)
    return b


def fill_established(b, s=0.5):
    """Seed b.M distinct far-apart established stored signatures (direct-to-established during fill)."""
    pts = far_points(b.M, b.K_prime, step=10.0)
    b.score_and_update(pts, 0)
    assert b._ne() == b.M, f"fill failed: n_est={b._ne()} != M={b.M}"
    return pts


def _seed_on_miss_ref(buf, s, M):
    """Independent brute-force seed-on-miss reference (pure-python, no chunking): a point
    seeds iff its min L1 distance to all placed centers is > s, up to M. Returns count."""
    centers = []
    for i in range(buf.shape[0]):
        if len(centers) >= M:
            break
        x = buf[i]
        if not centers:
            centers.append(x); continue
        C = torch.stack(centers)
        if float((C - x).abs().sum(dim=1).min()) > s:
            centers.append(x)
    return len(centers)


# ----------------------------------------------------------------- tests
def test_distance_regression():
    """Wrapped score_and_update == inline update_and_score + compute_scores (Algorithm 1).
    Distance-bank numerics must stay byte-identical."""
    torch.manual_seed(1)
    A = TypicalityBank(M=6, K_prime=3, replace_fraction=0.5)
    B = TypicalityBank(M=6, K_prime=3, replace_fraction=0.5)
    # fill both identically
    for _ in range(2):
        s = torch.randn(6, 3)
        A.update_and_score(s); B.update_and_score(s)
    assert torch.equal(A.bank, B.bank)
    q = torch.randn(6, 3)
    out = A.update_and_score(q)                       # inline path
    t_inline = TypicalityScorer.compute_scores(out['d'], out['mu'], out['sigma'])
    res = B.score_and_update(q)                        # wrapped path
    assert res['ready'] and torch.allclose(t_inline, res['t'])
    assert torch.equal(A.bank, B.bank), "wrapper must apply the same churn"
    # the wrapper now also surfaces the calibration stats (numerics unchanged)
    assert torch.equal(res['mu'], out['mu']) and torch.equal(res['sigma'], out['sigma'])
    assert torch.equal(res['d'], out['d'])


def test_hit_assignment():
    """Within s -> nearest stored signature's S += 1; beyond s -> seed (established during fill)."""
    b = counted(M=8)
    a = torch.tensor([[0.0, 0.0]])
    near = torch.tensor([[0.1, 0.0]])                 # < s=0.5 from a
    farr = torch.tensor([[10.0, 0.0]])                # > s from a
    batch = torch.cat([a, near, farr], 0)
    b.score_and_update(batch, 0)
    assert b._ne() == 2, f"expected 2 established (a-seed + far-seed), got {b._ne()}"
    assert abs(b.est_S[0].item() - 1.0) < 1e-6, f"nearest hit not counted: S0={b.est_S[0].item()}"
    assert abs(b.est_S[1].item() - 0.0) < 1e-6, "far seed should have no hit"


def test_graduation_and_cap():
    """Reserve stored signature graduates to established on first hit; |established| never exceeds M."""
    b = counted(M=2, s=0.5)
    far0, far1 = torch.tensor([[0., 0.]]), torch.tensor([[10., 0.]])
    far2, far2b = torch.tensor([[20., 0.]]), torch.tensor([[20.1, 0.]])   # far2b within s of far2
    batch = torch.cat([far0, far1, far2, far2b], 0)
    b.score_and_update(batch, 0)
    assert b._ne() == 2, f"established must stay <= M=2, got {b._ne()}"
    assert b._nr() == 0, f"reserve should be empty after graduation, got {b._nr()}"
    assert int(b.graduations.item()) == 1, f"expected 1 graduation, got {int(b.graduations.item())}"
    # the graduated far2 replaced the LFU established slot (index 0, lowest lambda tie)
    assert abs(b.est_b[0, 0].item() - 20.0) < 1e-6, "graduate should occupy the evicted LFU slot"


def test_reserve_bounded_and_expiry():
    """Ungraduated reserve entries expire at age T_need; |reserve| stays bounded, no growth."""
    b = counted(M=2, s=0.5, T_need=3, res=2)
    fill_established(b)                                 # n_est == M == 2
    # feed many steps of a single novel tile each, always far from everything and never repeated
    max_res = 0
    for k in range(40):
        novel = torch.tensor([[100.0 + 5.0 * k, 0.0]])   # each new, >s from all
        b.score_and_update(novel, k)
        assert b._ne() == b.M, "established must stay at M"
        assert b._nr() <= b.reserve_cap, f"reserve exceeded cap: {b._nr()} > {b.reserve_cap}"
        max_res = max(max_res, b._nr())
    assert max_res <= b.reserve_cap
    # with T_need small, old ungraduated entries must have been expired (no monotonic growth)
    assert b._nr() <= b.reserve_cap


def test_lfu_evicts_idle_first():
    """On a full-established graduation, the lowest-rate (idle) stored signature is evicted first."""
    b = counted(M=3, s=0.5, T_need=100, res=2)
    # fill 3 established stored signatures, then hit 1 and 2, leave signature 0 idle
    fill_established(b)                                 # est at x=0,10,20
    for _ in range(3):
        # hits near stored signatures 1 (x=10) and 2 (x=20); nothing near signature 0 (x=0)
        b.score_and_update(torch.tensor([[10.05, 0.], [20.05, 0.]]), 0)
    idle_pos = b.est_b[0].clone()
    assert b.est_S[0].item() == 0.0, "signature 0 should be idle (no hits)"
    # now force a graduation: seed a reserve stored signature then hit it
    b.score_and_update(torch.tensor([[200., 0.], [200.05, 0.]]), 0)
    assert int(b.graduations.item()) >= 1
    # the idle stored signature (x=0) must be gone; the graduate (x=200) present
    present0 = any(abs(b.est_b[i, 0].item() - idle_pos[0].item()) < 1e-6 for i in range(b.M))
    grad_present = any(abs(b.est_b[i, 0].item() - 200.0) < 1e-6 for i in range(b.M))
    assert not present0, "idle stored signature should have been evicted first"
    assert grad_present, "graduate should be in established"


def test_coldstart_then_counted_and_range():
    """Not ready while filling; cold-start distance readout when full-but-immature; counted
    (PIT) after maturation. t in [0,1] throughout; no div-by-zero for newborns."""
    b = counted(M=4, s=0.5, H=4, readout='pit', pit=64)
    # filling: not ready
    part = far_points(3, b.K_prime, step=10.0)
    out = b.score_and_update(part, 0)
    assert out['ready'] is False and out['t'] is None
    # complete the fill -> ready, and immature -> cold-start distance readout
    b.score_and_update(far_points(4, b.K_prime, step=10.0)[3:4], 1)
    assert b._ne() == b.M
    assert b._matured() is False
    out = b.score_and_update(torch.tensor([[5.0, 0.0], [15.0, 0.0]]), 2)
    assert out['ready'] and out['t'] is not None
    assert torch.all(out['t'] >= 0.0) and torch.all(out['t'] <= 1.0)
    # run enough steps (hits so stored signatures persist) to mature, then confirm counted path runs
    for it in range(3, 12):
        b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.], [20.05, 0.], [30.05, 0.]]), it)
    assert b._matured() is True, "median E should have passed the maturation threshold"
    out = b.score_and_update(torch.tensor([[0.05, 0.], [10.05, 0.]]), 12)
    assert torch.all(out['t'] >= 0.0) and torch.all(out['t'] <= 1.0)
    assert int(b.pit_filled.item()) > 0, "PIT ring should have been populated by the counted readout"
    assert torch.isfinite(out['t']).all()


def test_determinism():
    """Two fresh instances fed the identical gathered-batch sequence end byte-identical."""
    torch.manual_seed(7)
    seqs = [torch.randn(10, 2) * 3.0 for _ in range(30)]
    b1, b2 = counted(M=5, s=1.0, res=3), counted(M=5, s=1.0, res=3)
    for it, s in enumerate(seqs):
        b1.score_and_update(s.clone(), it)
        b2.score_and_update(s.clone(), it)
    assert torch.equal(b1.sync_fingerprint(), b2.sync_fingerprint()), "banks diverged across identical runs"


def test_checkpoint_roundtrip():
    """state_dict save/load restores all counted state exactly, and continues in lock-step."""
    torch.manual_seed(3)
    a = counted(M=5, s=1.0, res=3)
    for it in range(20):
        a.score_and_update(torch.randn(8, 2) * 3.0, it)
    sd = {k: v.clone() for k, v in a.state_dict().items()}
    b = counted(M=5, s=1.0, res=3)
    b.load_state_dict(sd)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint())
    nxt = torch.randn(8, 2) * 3.0
    a.score_and_update(nxt.clone(), 20); b.score_and_update(nxt.clone(), 20)
    assert torch.equal(a.sync_fingerprint(), b.sync_fingerprint()), "restored state must continue identically"


def test_import_and_flag_construction():
    """import typicality; build both banks by a flag, as the trainer will."""
    import typicality  # noqa: F401
    for flag in ('distance', 'counted'):
        if flag == 'distance':
            bank = TypicalityBank(M=8, K_prime=4, replace_fraction=0.1)
        else:
            bank = CountedCoverageBank(M=8, K_prime=4)
        assert hasattr(bank, 'score_and_update') and hasattr(bank, 'sync_fingerprint')


# --------------------------------------------- self-tuning s (Part 1a) tests
def test_s_recovers_scale():
    """The sweep edge tracks the live signature scale: the SAME points scaled 6x give a ~6x
    larger s (the frozen-2.75 bug is exactly this failure). Hits are non-trivial at each scale."""
    torch.manual_seed(11)
    M, kp = 64, 4
    base = torch.rand(1200, kp)

    def edge_for(scale):
        b = CountedCoverageBank(M=M, K_prime=kp, s_buffer_size=1400, s_min_buffer=1)
        b._push_buffer(base * scale)
        return b._sweep_edge()

    e1, e6 = edge_for(1.0), edge_for(6.0)
    ratio = e6 / e1
    assert 5.4 < ratio < 6.6, f"edge ratio {ratio:.3f} not ~6x (e1={e1:.4f}, e6={e6:.4f})"
    # hits non-trivial at each scale's own s
    for scale, e in [(1.0, e1), (6.0, e6)]:
        b = CountedCoverageBank(M=M, K_prime=kp, s_buffer_size=1400, s_min_buffer=1)
        b.s.fill_(e); b.s_ready.fill_(1)
        for it in range(60):
            b.score_and_update(torch.rand(64, kp) * scale, it)
        assert b._ne() == M, f"bank should fill at scale {scale} (n_est={b._ne()})"
        assert b.est_S[:M].sum().item() > 0.0, f"no hits at scale {scale} with s={e:.4f}"


def test_s_edge_matches_bruteforce():
    """The grid+bisect edge matches a fine brute-force 'largest s that fills to M' over the
    sweep's own search range, using an independent (unchunked) seed-on-miss reference."""
    torch.manual_seed(5)
    M, kp, n = 40, 3, 500
    b = CountedCoverageBank(M=M, K_prime=kp, s_grid_points=9, s_min_buffer=1, s_buffer_size=n)
    b._push_buffer(torch.rand(n, kp))
    edge = b._sweep_edge()
    buf = b.sig_ring[:n].float()
    m = b._buffer_scale(buf)
    lo, hi = b.grid_span
    fine = torch.linspace(lo * m, hi * m, 300).tolist()
    bf = max((sc for sc in fine if _seed_on_miss_ref(buf, sc, M) >= M), default=lo * m)
    tol = (hi * m - lo * m) / 8.0
    assert abs(edge - bf) <= tol, f"sweep edge {edge:.4f} vs brute-force {bf:.4f} (m={m:.4f}, tol={tol:.4f})"


def test_s_determinism():
    """Identical buffer contents -> byte-identical s (rank-identical sweep, no randomness)."""
    torch.manual_seed(13)
    buf = torch.rand(800, 4)
    b1 = CountedCoverageBank(M=64, K_prime=4, s_buffer_size=800, s_min_buffer=1)
    b2 = CountedCoverageBank(M=64, K_prime=4, s_buffer_size=800, s_min_buffer=1)
    b1._push_buffer(buf.clone()); b2._push_buffer(buf.clone())
    assert b1._sweep_edge() == b2._sweep_edge(), "sweep edge diverged on identical buffers"


def test_sweep_read_only():
    """A sweep updates only the scalar s (in score_and_update); _sweep_edge itself mutates no
    bank state — established/reserve/counters/pit/buffer all unchanged."""
    torch.manual_seed(17)
    b = CountedCoverageBank(M=16, K_prime=3, s_buffer_size=600, s_min_buffer=1)
    b.s.fill_(0.2); b.s_ready.fill_(1)                 # get the bank into a non-trivial state
    for it in range(6):
        b.score_and_update(torch.rand(30, 3), it)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    _ = b._sweep_edge()
    for k, v in before.items():
        assert torch.equal(b.state_dict()[k], v), f"_sweep_edge mutated bank state: {k}"


def test_startup_gate():
    """Before the first sweep sets s the bank scores t=0 and seeds nothing; once the buffer
    reaches s_min_buffer the first sweep sets s>0 and the fill begins."""
    b = CountedCoverageBank(M=8, K_prime=3, s_buffer_size=200, s_min_buffer=50, s_sweep_interval=500)
    it = 0
    while int(b.sig_filled.item()) < 40:               # below the gate: no seed, no score
        out = b.score_and_update(torch.rand(10, 3), it); it += 1
        assert out['ready'] is False and out['t'] is None
        assert b._ne() == 0, "must not seed before the first sweep"
        assert float(b.s.item()) == 0.0 and int(b.s_ready.item()) == 0
    while int(b.s_ready.item()) == 0:                  # cross the gate -> first sweep sets s
        b.score_and_update(torch.rand(10, 3), it); it += 1
    assert float(b.s.item()) > 0.0, "first sweep must set s>0"
    assert b._ne() > 0, "the counted fill begins once s is set (same step, fall-through)"


def test_s_ema_smoothing():
    """A periodic sweep moves s toward the new edge by an EMA step, not a snap: after a 3x
    scale jump s advances only ~alpha of the gap."""
    torch.manual_seed(19)
    b = CountedCoverageBank(M=32, K_prime=3, s_buffer_size=600, s_min_buffer=300,
                            s_sweep_interval=5, s_ema_alpha=0.2)
    it = 0
    while int(b.s_ready.item()) == 0:                  # startup on scale-1 data
        b.score_and_update(torch.rand(60, 3), it); it += 1
    s_prev = float(b.s.item())
    last = int(b.last_sweep_step.item())
    # replace the buffer with scale-3 data so the next edge jumps ~3x
    b.sig_ring.zero_(); b.sig_ptr.zero_(); b.sig_filled.zero_()
    b._push_buffer(torch.rand(600, 3) * 3.0)
    raw_edge = b._sweep_edge()                          # what an un-smoothed update would snap to
    assert raw_edge > 1.5 * s_prev, f"sanity: new edge {raw_edge:.3f} should dwarf s_prev {s_prev:.3f}"
    b.score_and_update(torch.rand(60, 3) * 3.0, last + b.s_sweep_interval)   # trigger one periodic sweep
    s_new = float(b.s.item())
    assert s_prev < s_new < raw_edge, f"s should move toward the edge, not snap (prev={s_prev}, new={s_new}, edge={raw_edge})"
    halfway = s_prev + 0.5 * (raw_edge - s_prev)
    assert s_new < halfway, f"EMA (alpha=0.2) must move <<half the gap: s_new={s_new:.3f} halfway={halfway:.3f}"


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
