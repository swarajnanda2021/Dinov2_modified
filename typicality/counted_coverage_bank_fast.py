"""
FastCountedCoverageBank -- a drop-in, semantics-preserving optimisation of
typicality/counted_coverage_bank.py::CountedCoverageBank.

Only two methods are overridden:

  * ``_update``       -- the per-step sequential loop over the all-gathered batch
                         (B = world*oversample*N = 6144 rows in production).
  * ``_seed_on_miss`` -- the greedy covering replay driven by ``_sweep_edge``
                         (runs every ``s_sweep_interval`` steps, but is O(1e4)
                         GPU launches + syncs when it does).

Everything else (``_counted_readout``, ``_decay_age``, ``_selftune_pool``,
``_sweep_edge``, ``_push_buffer``, telemetry, checkpoint I/O) is inherited
unchanged.

WHY THE ORIGINAL IS SLOW
------------------------
``_update`` is a Python ``for k in range(B)`` loop.  Per iteration it issues
2 ``torch.cdist`` launches, 2 ``argmin`` launches and 2-6 ``.item()`` calls.
Every ``.item()`` is a device->host synchronisation, so a single step costs
~6144 * (6 launches + ~4 syncs).  The *arithmetic* is trivial (a [1,256] vs
[8192,256] L1 reduction, ~2 MFLOP); the wall time is 100% launch + sync
overhead.  ``_seed_on_miss`` has the same pathology: one cdist + one
``.item()`` per *candidate point* (up to 60000 per grid value, 9-11 grid
values per sweep).

WHY THE REPLACEMENT IS FAST
---------------------------
The loop is genuinely order-dependent (row k can seed a signature that becomes
row k+1's nearest), so it cannot be replaced by a single batched argmin.  It is
instead made *speculative and exact*:

  1. The batch is cut into chunks of ``_fast_chunk`` rows (default 256).
  2. For each chunk, three batched cdists are issued on the GPU:
       De = cdist(chunk, est_b[:ne])   -> per-row (min, argmin) over the
                                          established set as it stands at
                                          chunk entry
       Dr = cdist(chunk, res_b[:nr])   -> full [C, nr] matrix
       G  = cdist(chunk, chunk)        -> distances to signatures that this
                                          chunk itself will seed
  3. Those three (small) results, plus CPU mirrors of the mutable scalar state
     (est_S/est_E/est_age, res_*, counters), are pulled to host **once per
     chunk** (~24 syncs per step instead of ~25k).
  4. The sequential loop then runs in pure host code over numpy views.  Every
     mutation is applied incrementally to the cached distances:
       - seeding established slot j with chunk row k  ->  new column G[:, k]
       - seeding reserve   slot j with chunk row k    ->  Dr[:, j] = G[:, k]
       - graduating reserve r into established slot j ->  new column Dr[:, r]
     For the established side only (min, argmin) is cached, so a column write
     is folded in with the *identical* lowest-index tie-break rule; the one
     case that cannot be folded (the evicted slot *was* this row's argmin and
     the replacement is strictly farther) marks the row "dirty" and it falls
     back to a single exact GPU cdist when the loop reaches it.  Evictions are
     rare and LFU-chosen, so this path fires almost never.
  5. Mutated ``est_b`` rows are flushed to the device with one ``index_copy_``
     per chunk; all other state is written back once at the end of the step.

``_seed_on_miss`` gets the same treatment: instead of one cdist+sync per
candidate, candidates are processed in blocks of ``_sweep_block``; for a block
we fetch (a) the running min distance to the already-placed centers and (b) the
block's intra-block distance matrix, then run the *exact same* greedy
accept/reject scan on the host.  After each block the running min for the
remaining candidates is updated against the centers just placed with one cdist.

EXACTNESS
---------
Every decision the original makes is reproduced bit-for-bit:
  * all distances still come from ``torch.cdist(..., p=1)`` -- never from a
    hand-rolled ``(a-b).abs().sum()``, whose reduction order could differ;
  * argmin/argmax keep the lowest-index tie-break (numpy ``argmin`` and
    ``torch.argmin`` both return the first occurrence);
  * ``est_S += 1.0`` / ``res_S += 1.0`` / ``res_hits += 1.0`` are float32
    increments on host mirrors -- IEEE-identical to the device increments;
  * ``lam = est_S/(est_E+eps)`` for LFU eviction is evaluated with *torch* on
    the host mirror (not numpy) so the python-scalar `eps` stays weak-typed and
    the arithmetic stays float32, exactly as on device;
  * telemetry lists are appended in the original order, so ``health()``'s
    order-dependent float sums are identical.

The single assumption is that ``torch.cdist(A, B, p=1)[i, j]`` depends only on
``A[i]`` and ``B[j]``, not on ``A.shape[0]`` or ``B.shape[0]``.  This is
verified on CPU by ``verify_bank.py`` and must be re-verified on the GPU (see
``assert_cdist_shape_invariant`` below).  Note that even if it did not hold,
cross-rank byte-identity -- the property the bank actually needs for DDP -- is
unaffected, because every rank runs the identical code on the identical
all-gathered batch.

NOT CHANGED (deliberately)
--------------------------
``_selftune_pool`` keeps its 32-iteration python grid search over L.  It runs
only on the sweep cadence, each iteration is a 2x2 solve, and the
``try/except`` around ``torch.linalg.solve`` plus the strict ``res < best_res``
tie-break would have to be reproduced exactly by a batched solve.  Not worth
the equivalence risk for <1 ms/step amortised.
"""

import os
import sys

import numpy as np
import torch

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'repo')
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from .counted_coverage_bank import (  # noqa: E402
    CountedCoverageBank, _argmin_lowest, _argmax_lowest,
)

_INF = float('inf')


def _l1(a, b):
    """The one and only distance primitive -- identical op to the original."""
    return torch.cdist(a, b, p=1)


def assert_cdist_shape_invariant(device='cpu', dtype=torch.float32, seed=0,
                                 n_a=64, n_b=300, k=256):
    """Check the single assumption FastCountedCoverageBank relies on.

    Run this ON THE GPU before trusting bit-identity of the fast path.
    Returns True on success, raises AssertionError otherwise.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    A = torch.randn(n_a, k, generator=g).to(device=device, dtype=dtype)
    B = torch.randn(n_b, k, generator=g).to(device=device, dtype=dtype)
    full = _l1(A, B)
    for i in (0, 1, n_a // 2, n_a - 1):
        assert torch.equal(_l1(A[i:i + 1], B)[0], full[i]), f'A-row {i} not invariant'
    for kk in (1, 3, n_b // 2, n_b - 1, n_b):
        assert torch.equal(_l1(A, B[:kk]), full[:, :kk]), f'B-prefix {kk} not invariant'
    assert torch.equal(_l1(A[3:4], B[7:8])[0, 0], full[3, 7]), 'pairwise not invariant'
    return True


class FastCountedCoverageBank(CountedCoverageBank):
    """Drop-in replacement.  Same constructor, same state, same numbers."""

    # rows of the gathered batch processed per GPU round-trip.
    # Set to None to fall back to the base-class implementation.
    _fast_chunk = 256
    # candidates processed per GPU round-trip inside _seed_on_miss.
    # Set to None to fall back to the base-class implementation.
    # NOTE: on a *CPU* device the base _seed_on_miss can be faster (its
    # per-candidate cdist is cheap when there is no launch/sync tax).  On CUDA
    # the base version is one kernel launch + one D2H sync per candidate point,
    # i.e. O(1e5-1e6) syncs per sweep, and this version is the large win.
    _sweep_block = 512

    # ------------------------------------------------------------ update loop
    @torch.no_grad()
    def _update(self, s_global):
        if self._fast_chunk is None:
            return CountedCoverageBank._update(self, s_global)
        B = int(s_global.shape[0])
        if B == 0:
            return
        M = self.M
        cap = self.reserve_cap
        eps = self.eps
        s = float(self.s.item())
        dev = self.est_b.device
        C = int(self._fast_chunk)

        # ---- host mirrors of everything the loop mutates -------------------
        est_S_t = self.est_S.detach().to('cpu').clone()
        est_E_t = self.est_E.detach().to('cpu').clone()
        est_age_t = self.est_age.detach().to('cpu').clone()
        res_b_t = self.res_b.detach().to('cpu').clone()
        res_S_t = self.res_S.detach().to('cpu').clone()
        res_E_t = self.res_E.detach().to('cpu').clone()
        res_age_t = self.res_age.detach().to('cpu').clone()
        res_hits_t = self.res_hits.detach().to('cpu').clone()

        est_S = est_S_t.numpy()
        est_E = est_E_t.numpy()
        est_age = est_age_t.numpy()
        res_b = res_b_t.numpy()
        res_S = res_S_t.numpy()
        res_E = res_E_t.numpy()
        res_age = res_age_t.numpy()
        res_hits = res_hits_t.numpy()

        ne = self._ne()
        nr = self._nr()

        n_grad = 0
        n_admits = 0
        n_flushed = 0
        evict_lams = []
        exit_ages = []
        grad_hits = self.graduation_hits
        n_dirty = 0           # diagnostic only (see _fast_dirty_recomputes)

        pending = {}          # est_b slot -> host row (float32, K')

        def flush_pending():
            if not pending:
                return
            slots = sorted(pending)
            idx = torch.tensor(slots, dtype=torch.long, device=dev)
            rows = torch.stack([pending[i] for i in slots]).to(dev)
            self.est_b.index_copy_(0, idx, rows)
            pending.clear()

        k0 = 0
        while k0 < B:
            k1 = min(k0 + C, B)
            cs = k1 - k0
            chunk = s_global[k0:k1]
            chunk_cpu = chunk.detach().to('cpu').clone()
            chunk_np = chunk_cpu.numpy()

            # ---- batched distances for this chunk (3 kernels, 1 sync) ------
            if ne > 0:
                De = _l1(chunk, self.est_b[:ne])
                be_dev = torch.argmin(De, dim=1)
                de_dev = De.gather(1, be_dev.unsqueeze(1)).squeeze(1)
                de = de_dev.to('cpu').numpy().astype(np.float64)
                be = be_dev.to('cpu').numpy().astype(np.int64)
                del De, be_dev, de_dev
            else:
                de = np.full(cs, _INF, dtype=np.float64)
                be = np.full(cs, -1, dtype=np.int64)

            Dres = np.empty((cs, max(cap, 1)), dtype=np.float64)
            if nr > 0:
                res_dev = res_b_t[:nr].to(dev)
                Dres[:, :nr] = _l1(chunk, res_dev).to('cpu').numpy().astype(np.float64)
                del res_dev

            G = _l1(chunk, chunk).to('cpu').numpy().astype(np.float64)

            dirty = np.zeros(cs, dtype=bool)

            # ---- fold a new/overwritten established column into (de, be) ---
            def apply_est_col(t, slot, col):
                lo = t + 1
                if lo >= cs:
                    return
                d = col[lo:]
                cur = de[lo:]
                cb = be[lo:]
                upd = (d < cur) | ((d == cur) & (slot <= cb))
                nd = (cb == slot) & (d > cur)
                de[lo:] = np.where(upd, d, cur)
                be[lo:] = np.where(upd, slot, cb)
                if nd.any():
                    dirty[lo:] |= nd

            # ------------------------------------------------ the exact loop
            for t in range(cs):
                if ne == 0 and nr == 0:
                    # _seed_established
                    slot = ne
                    pending[slot] = chunk_cpu[t].clone()
                    est_S[slot] = 0.0
                    est_E[slot] = 0.0
                    est_age[slot] = 0
                    ne += 1
                    n_admits += 1
                    apply_est_col(t, slot, G[:, t])
                    continue

                if dirty[t]:
                    # rare: this row's cached argmin slot was evicted underneath it
                    n_dirty += 1
                    flush_pending()
                    d1 = _l1(chunk[t:t + 1], self.est_b[:ne])[0]
                    bi = _argmin_lowest(d1)
                    be[t] = bi
                    de[t] = float(d1[bi].item())
                    dirty[t] = False

                dist_est = de[t]
                best_est = int(be[t])
                if nr > 0:
                    row = Dres[t, :nr]
                    best_res = int(row.argmin())
                    dist_res = float(row[best_res])
                else:
                    best_res = -1
                    dist_res = _INF

                in_reserve = dist_res < dist_est          # est wins ties
                nearest_dist = dist_res if in_reserve else dist_est

                if nearest_dist <= s:
                    if in_reserve:
                        res_S[best_res] += 1.0
                        res_hits[best_res] += 1.0
                        if float(res_hits[best_res]) >= grad_hits:
                            # ---------------------------------- _graduate(r)
                            r = best_res
                            if ne >= M:
                                lam = est_S_t / (est_E_t + eps)
                                slot = _argmin_lowest(lam[:M])
                                evict_lams.append(
                                    float((est_S_t[slot] / (est_E_t[slot] + eps)).item()))
                            else:
                                slot = ne
                                ne += 1
                            pending[slot] = res_b_t[r].clone()
                            est_S[slot] = res_S[r]
                            est_E[slot] = res_E[r]
                            est_age[slot] = res_age[r]
                            n_grad += 1
                            exit_ages.append(int(res_age[r]))
                            # distances to the promoted signature are exactly
                            # the reserve column r (same vector).
                            apply_est_col(t, slot, Dres[:, r].copy())
                            # compact the reserve (shift left by one)
                            if r < nr - 1:
                                res_b[r:nr - 1] = res_b[r + 1:nr].copy()
                                res_S[r:nr - 1] = res_S[r + 1:nr].copy()
                                res_E[r:nr - 1] = res_E[r + 1:nr].copy()
                                res_age[r:nr - 1] = res_age[r + 1:nr].copy()
                                res_hits[r:nr - 1] = res_hits[r + 1:nr].copy()
                                Dres[:, r:nr - 1] = Dres[:, r + 1:nr].copy()
                            res_b[nr - 1] = 0.0
                            res_S[nr - 1] = 0.0
                            res_E[nr - 1] = 0.0
                            res_age[nr - 1] = 0
                            res_hits[nr - 1] = 0.0
                            nr -= 1
                    else:
                        est_S[best_est] += 1.0
                else:
                    if ne < M:
                        # ---------------------------------- _seed_established
                        slot = ne
                        pending[slot] = chunk_cpu[t].clone()
                        est_S[slot] = 0.0
                        est_E[slot] = 0.0
                        est_age[slot] = 0
                        ne += 1
                        n_admits += 1
                        apply_est_col(t, slot, G[:, t])
                    else:
                        # -------------------------------------- _seed_reserve
                        if nr < cap:
                            j = nr
                            nr += 1
                        else:
                            j = _argmax_lowest(res_age_t[:nr])
                            n_flushed += 1
                            exit_ages.append(int(res_age[j]))
                        res_b[j] = chunk_np[t]
                        res_S[j] = 0.0
                        res_E[j] = 0.0
                        res_age[j] = 0
                        res_hits[j] = 0.0
                        Dres[:, j] = G[:, t]
                        n_admits += 1

            flush_pending()
            k0 = k1

        # ---- write host mirrors back --------------------------------------
        flush_pending()
        self.est_S.copy_(est_S_t)
        self.est_E.copy_(est_E_t)
        self.est_age.copy_(est_age_t)
        self.res_b.copy_(res_b_t)
        self.res_S.copy_(res_S_t)
        self.res_E.copy_(res_E_t)
        self.res_age.copy_(res_age_t)
        self.res_hits.copy_(res_hits_t)
        self.n_est.fill_(ne)
        self.n_res.fill_(nr)
        if n_grad:
            self.graduations.add_(n_grad)
        self._step_admits += n_admits
        self._step_grad += n_grad
        self._step_flushed += n_flushed
        if evict_lams:
            self._step_evict_lams.extend(evict_lams)
        if exit_ages:
            self._step_exit_ages.extend(exit_ages)
        # diagnostic (NOT part of the bank state / fingerprint / telemetry)
        self._fast_dirty_recomputes = getattr(self, '_fast_dirty_recomputes', 0) + n_dirty

    # ------------------------------------------------------ covering replay
    @torch.no_grad()
    def _seed_on_miss(self, buf, s):
        """Exact seed-on-miss replay; returns #centers placed (capped at M).

        Identical greedy semantics to the base class, but the accept/reject
        scan runs on the host over a block of candidates whose distances to the
        already-placed centers and to each other were fetched in one round-trip.
        """
        if self._sweep_block is None:
            return CountedCoverageBank._seed_on_miss(self, buf, s)
        n = buf.shape[0]
        M = self.M
        centers = buf.new_empty((M, buf.shape[1]))
        nc = 0
        CH = 4096
        Q = int(self._sweep_block)
        i = 0
        while i < n and nc < M:
            j = min(i + CH, n)
            chunk = buf[i:j]
            if nc > 0:
                dmin = _l1(chunk, centers[:nc]).min(dim=1).values
                cand = torch.nonzero(dmin > s, as_tuple=False).flatten()
                dcur = dmin[cand].to('cpu').numpy().astype(np.float64)
            else:
                cand = torch.arange(chunk.shape[0], device=chunk.device)
                dcur = np.full(int(cand.numel()), _INF, dtype=np.float64)

            dev = chunk.device
            while nc < M and int(cand.numel()) > 0:
                q = min(Q, int(cand.numel()))
                dc = dcur[:q]
                # Candidates already killed by a previously placed center can
                # never be accepted (centers only grow -> dcur only shrinks),
                # so the original's per-candidate cdist on them is dead work.
                sv = np.flatnonzero(dc > s)
                new_centers = None
                if sv.size:
                    svi = cand[:q].index_select(0, torch.as_tensor(sv, dtype=torch.long,
                                                                   device=dev))
                    pts = chunk.index_select(0, svi)
                    Gb = _l1(pts, pts).to('cpu').numpy().astype(np.float64)
                    dacc = np.full(sv.size, _INF, dtype=np.float64)
                    acc = []
                    for a in range(sv.size):
                        if nc + len(acc) >= M:
                            break
                        if dacc[a] > s:
                            acc.append(a)
                            np.minimum(dacc, Gb[:, a], out=dacc)
                    if acc:
                        sel = torch.as_tensor(acc, dtype=torch.long, device=dev)
                        new_centers = pts.index_select(0, sel)
                        centers[nc:nc + len(acc)] = new_centers
                        nc += len(acc)
                cand = cand[q:]
                dcur = dcur[q:]
                if new_centers is not None and nc < M and int(cand.numel()) > 0:
                    rpts = chunk.index_select(0, cand)
                    dn = _l1(rpts, new_centers).min(dim=1).values
                    dcur = np.minimum(dcur, dn.to('cpu').numpy().astype(np.float64))
                    keep = np.flatnonzero(dcur > s)
                    if keep.size != dcur.size:
                        cand = cand.index_select(
                            0, torch.as_tensor(keep, dtype=torch.long, device=dev))
                        dcur = dcur[keep]
            i = j
        return nc
