"""
Counted-coverage bank (typicality/technique.md, section 3.5) -- the typicality method.

It stores a covering set of stored signatures and reads crowding from explicit,
exponentially-decayed hit counts (not from nearest-neighbour distance), so the estimate is
faithful without requiring the stored set to be a representative sample.

Per-signature state: position b, decayed hit count S, decayed exposure E (steps alive), age.
Split into an ESTABLISHED set (capacity M) and a separate RESERVE buffer. Ratio
lambda_hat = S / (E + eps) is the decayed per-signature hit-RATE; the readout sums
lambda_hat * K over the established signatures within a FIXED radius R_rad = radius_mult * s
(unnormalized kernel sum, triweight clamped to zero beyond R_rad) and returns the absolute
local density p_hat directly. A fixed radius -- not a fixed count -- is what makes p_hat carry
the crowding: dividing by the j-th nearest distance (the old fixed-count bandwidth) cancels the
local scale exactly, so a fixed-count sum is invariant to how crowded the neighbourhood is.

Two things are measured online rather than fixed:
  * Hit radius s (section 3.5, Placement): a rolling ~60k-signature buffer is swept every
    ~500 steps -- seed-on-miss over a grid re-centered on the live scale -- to find the fill
    knee; s tracks that edge, EMA-smoothed. It also sets the readout radius (R_rad =
    radius_mult * s): the measured signature spacing / s ~ 1, so s stands in for anchor spacing.
  * Pooling count j (Algorithm 3): on the same cadence, L (the density correlation length) is
    fitted by a variogram-with-nugget and j is set to the count of stored signatures within L,
    clamped by a variance target; under-resolution (spacing > L) is flagged. j no longer feeds
    the readout (the radius is fixed); it is retained as a resolution diagnostic only.

Graduation requires `graduation_hits` corroborating hits (default 2), not one, so a candidate
is not promoted over an established signature on a single observation.

Determinism: on the global (all-gathered) batch every rank runs the identical update, sweep,
and pool/soft-rank computation over identical state, so the bank stays byte-identical across
ranks. Every tie-break is pinned to LOWEST index; the self-tune uses no RNG.

NOTE (performance): the update loop is sequential over the gathered batch (each tile can seed
a signature that changes the nearest answer for the next); it is a per-step Python loop.
"""

import math

import torch
import torch.nn as nn

from .typicality_scorer import TypicalityScorer


def _argmin_lowest(x):
    """argmin with lowest-index tie-break (torch.argmin returns the first/lowest index)."""
    return int(torch.argmin(x).item())


def _argmax_lowest(x):
    """argmax with lowest-index tie-break (first/lowest index on ties)."""
    return int(torch.argmax(x).item())


_INV_SQRT2 = 1.0 / math.sqrt(2.0)


class CountedCoverageBank(nn.Module):
    """The counted-coverage typicality bank.
        score_and_update(s_global, current_iteration) -> {'ready': bool, 'p_hat': Tensor|None}
        sync_fingerprint() -> Tensor ; health()/stats()/compact_line() for telemetry.
    """

    def __init__(self, M, K_prime, pool_j=64, halflife_steps=250,
                 reserve_residency=300, reserve_size=550, eps=1e-8,
                 s_buffer_size=60000, s_sweep_interval=500, s_grid_points=9,
                 grid_span=(0.3, 2.0), s_ema_alpha=0.2, s_min_buffer=60000,
                 s_headroom=0.0, strong_lambda=0.1, scale_sample=2048,
                 graduation_hits=2, pool_selftune=True, pool_rse_target=0.05,
                 pool_max=256, pool_ema=0.2, radius_mult=1.5):
        super().__init__()
        self.M = int(M)
        self.K_prime = int(K_prime)
        self.eta = 0.5 ** (1.0 / float(halflife_steps))      # per-step decay
        self.T_need = int(reserve_residency)
        self.reserve_cap = int(reserve_size)
        self.eps = float(eps)
        self.mature_E = 0.5 / (1.0 - self.eta)
        self.n_eff = (1.0 + self.eta) / (1.0 - self.eta)     # estimator variance floor (half-life only)
        # ---- self-tuning s hyperparameters ----
        self.s_buffer_size = int(s_buffer_size)
        self.s_sweep_interval = int(s_sweep_interval)
        self.s_grid_points = int(s_grid_points)
        self.grid_span = (float(grid_span[0]), float(grid_span[1]))
        self.s_ema_alpha = float(s_ema_alpha)
        self.s_min_buffer = int(s_min_buffer)
        self.s_headroom = float(s_headroom)
        self.strong_lambda = float(strong_lambda)
        self.scale_sample = int(scale_sample)
        # ---- Part 1: graduation ----
        self.graduation_hits = int(graduation_hits)
        # ---- Part 2: self-tuning j ----
        self.pool_selftune = bool(pool_selftune)
        self.pool_rse_target = float(pool_rse_target)
        self.pool_max = int(pool_max)
        self.pool_ema = float(pool_ema)
        # ---- fixed-radius readout ----
        self.radius_mult = float(radius_mult)                # R_rad = radius_mult * s

        # ---- established set (capacity M) ----
        self.register_buffer('est_b', torch.zeros(self.M, self.K_prime))
        self.register_buffer('est_S', torch.zeros(self.M))
        self.register_buffer('est_E', torch.zeros(self.M))
        self.register_buffer('est_age', torch.zeros(self.M, dtype=torch.long))
        self.register_buffer('n_est', torch.tensor(0, dtype=torch.long))
        # ---- reserve buffer (separate, on top of M) ----
        cap = max(1, self.reserve_cap)
        self.register_buffer('res_b', torch.zeros(cap, self.K_prime))
        self.register_buffer('res_S', torch.zeros(cap))
        self.register_buffer('res_E', torch.zeros(cap))
        self.register_buffer('res_age', torch.zeros(cap, dtype=torch.long))
        self.register_buffer('res_hits', torch.zeros(cap))       # raw (undecayed) hit count for graduation
        self.register_buffer('n_res', torch.tensor(0, dtype=torch.long))
        # ---- reference scale for the absolute-density weight (checkpointed) ----
        # p_hat has no natural units, so the weight w = 1/(p_hat + c)^a needs one: c = c_frac *
        # p_ref, with p_ref a slow EMA of the batch-median density. Initialised from the first
        # matured batch's median (NEVER 0 -> c=0 -> weight diverges on empty-neighbourhood tiles).
        self.register_buffer('p_ref', torch.tensor(0.0))
        self.register_buffer('p_ref_init', torch.tensor(0, dtype=torch.long))
        # ---- self-tuning s state ----
        self.register_buffer('s', torch.tensor(0.0))                 # hit radius (unset until 1st sweep)
        self.register_buffer('s_delta', torch.tensor(0.0))           # change at last sweep
        self.register_buffer('s_ready', torch.tensor(0, dtype=torch.long))
        self.register_buffer('last_sweep_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('sig_ring', torch.zeros(self.s_buffer_size, self.K_prime, dtype=torch.float16))
        self.register_buffer('sig_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('sig_filled', torch.tensor(0, dtype=torch.long))
        # ---- self-tuning j state ----
        self.register_buffer('j', torch.tensor(int(pool_j), dtype=torch.long))
        self.register_buffer('j_smooth', torch.tensor(float(pool_j)))
        self.register_buffer('L_est', torch.tensor(0.0))
        self.register_buffer('h_over_L', torch.tensor(0.0))
        self.register_buffer('underresolved', torch.tensor(0, dtype=torch.long))
        # ---- diagnostics ----
        self.register_buffer('graduations', torch.tensor(0, dtype=torch.long))
        self._reset_step_telemetry()

    def _reset_step_telemetry(self):
        # per-step, plain attrs (rank-identical, not checkpointed)
        self._step_admits = 0
        self._step_kappa = 0
        self._step_evict_lams = []
        self._step_grad = 0
        self._step_expired = 0
        self._step_flushed = 0
        self._step_exit_ages = []

    # ---------------------------------------------------------------- helpers
    def _ne(self):
        return int(self.n_est.item())

    def _nr(self):
        return int(self.n_res.item())

    def _triweight(self, u):
        w = (1.0 - u * u).clamp(min=0.0)
        return w * w * w

    # ------------------------------------------------------- rolling s buffer
    @torch.no_grad()
    def _push_buffer(self, s_global):
        x = s_global.detach().to(self.sig_ring.dtype)
        n = x.shape[0]
        cap = self.s_buffer_size
        if n >= cap:
            self.sig_ring.copy_(x[-cap:]); self.sig_ptr.fill_(0); self.sig_filled.fill_(cap)
            return
        ptr = int(self.sig_ptr.item()); end = ptr + n
        if end <= cap:
            self.sig_ring[ptr:end] = x
        else:
            first = cap - ptr
            self.sig_ring[ptr:] = x[:first]
            self.sig_ring[:end - cap] = x[first:]
        self.sig_ptr.fill_(end % cap)
        self.sig_filled.fill_(min(cap, int(self.sig_filled.item()) + n))

    @torch.no_grad()
    def _buffer_scale(self, buf):
        n = buf.shape[0]
        k = min(n, self.scale_sample)
        idx = torch.linspace(0, n - 1, k).round().long()
        sub = buf[idx]
        D = torch.cdist(sub, sub, p=1)
        D.diagonal().fill_(float('inf'))
        return max(float(D.min(dim=1).values.median().item()), self.eps)

    @torch.no_grad()
    def _seed_on_miss(self, buf, s):
        """Exact seed-on-miss replay; returns #centers placed (capped at M). Read-only scratch."""
        n = buf.shape[0]; M = self.M
        centers = buf.new_empty((M, buf.shape[1])); nc = 0; CH = 4096; i = 0
        while i < n and nc < M:
            j = min(i + CH, n)
            chunk = buf[i:j]
            if nc > 0:
                dmin = torch.cdist(chunk, centers[:nc], p=1).min(dim=1).values
                cand = torch.nonzero(dmin > s, as_tuple=False).flatten().tolist()
            else:
                cand = list(range(chunk.shape[0]))
            for local in cand:
                if nc >= M:
                    break
                if nc == 0:
                    centers[0] = chunk[local]; nc = 1; continue
                if torch.cdist(chunk[local:local + 1], centers[:nc], p=1).min().item() > s:
                    centers[nc] = chunk[local]; nc += 1
            i = j
        return nc

    @torch.no_grad()
    def _sweep_edge(self):
        n = int(self.sig_filled.item())
        buf = self.sig_ring[:n].float()
        m = self._buffer_scale(buf)
        lo, hi = self.grid_span
        grid = torch.exp(torch.linspace(math.log(lo * m), math.log(hi * m), self.s_grid_points))
        fills = [self._seed_on_miss(buf, float(sc)) >= self.M for sc in grid.tolist()]
        true_idx = [i for i, f in enumerate(fills) if f]
        if not true_idx:
            return float(grid[0].item())
        last_true = max(true_idx)
        if last_true == self.s_grid_points - 1:
            return float(grid[-1].item())
        lo_s = float(grid[last_true].item()); hi_s = float(grid[last_true + 1].item())
        mid = math.sqrt(lo_s * hi_s)
        return mid if (self._seed_on_miss(buf, mid) >= self.M) else lo_s

    # ---------------------------------------------- Part 2: self-tuning j
    @torch.no_grad()
    def _selftune_pool(self):
        """Fit L (variogram-with-nugget on lambda_hat over mature signature pairs), set
        j = median count within L, clamp by a variance target, flag under-resolution. Updates
        only j / j_smooth / L_est / h_over_L / underresolved. Deterministic (no RNG)."""
        ne = self._ne()
        if ne < 8:
            return
        b = self.est_b[:ne]
        E = self.est_E[:ne]
        lam = self.est_S[:ne] / (E + self.eps)
        # query sample over all established: spacing, count-within-L, bandwidth h
        qn = min(256, ne)
        qidx = torch.linspace(0, ne - 1, qn, device=b.device).round().long()
        Dq = torch.cdist(b[qidx], b, p=1)                              # [qn, ne]
        Dnn = Dq.clone(); Dnn[Dnn == 0] = float('inf')
        spacing = float(torch.median(Dnn.min(dim=1).values).item())
        # mature subset for the variogram (young signatures are the noisiest); fall back to all
        mi = torch.nonzero(E >= self.mature_E, as_tuple=False).flatten()
        if mi.numel() < 8:
            mi = torch.arange(ne, device=b.device)
        bm = b[mi]; lm = lam[mi]; m = bm.shape[0]
        # deterministic pair subsample: pair index i with (i+off)%m over a set of offsets (no RNG)
        n_off = min(32, m - 1)
        base = torch.arange(m, device=b.device)
        offs = torch.arange(1, n_off + 1, device=b.device)
        ii = base.repeat(n_off)
        jj = ((base.unsqueeze(0) + offs.unsqueeze(1)) % m).reshape(-1)
        r = (bm[ii] - bm[jj]).abs().sum(dim=1)                          # L1 separations
        g = 0.5 * (lm[ii] - lm[jj]) ** 2                               # semivariance contributions
        P = 30000
        if r.numel() > P:
            sel = torch.linspace(0, r.numel() - 1, P, device=b.device).round().long()
            r = r[sel]; g = g[sel]
        r_hi = float(torch.quantile(r, 0.9).item())
        if r_hi <= self.eps:
            return
        nb = 24
        edges = torch.linspace(0.0, r_hi, nb + 1, device=b.device)
        binidx = torch.bucketize(r, edges[1:-1]).clamp(max=nb - 1)
        rb = torch.zeros(nb, device=b.device); gb = torch.zeros(nb, device=b.device)
        cnt = torch.zeros(nb, device=b.device)
        rb.scatter_add_(0, binidx, r); gb.scatter_add_(0, binidx, g)
        cnt.scatter_add_(0, binidx, torch.ones_like(r))
        v = cnt > 0
        rb = rb[v] / cnt[v]; gb = gb[v] / cnt[v]; wts = cnt[v]
        if rb.numel() < 4:
            return
        # fit gamma(r)=c0+c1(1-exp(-r/L)) by a grid over L + weighted linear LS for (c0,c1).
        # Anchor the grid low-end on the spacing so a true L BELOW the spacing is reachable --
        # that is exactly the under-resolution regime the flag below must be able to detect.
        L_lo = min(max(0.1 * spacing, self.eps), 0.5 * r_hi)
        L_grid = torch.exp(torch.linspace(math.log(L_lo), math.log(r_hi), 32, device=b.device))
        eye2 = torch.eye(2, device=b.device)
        best_L = float(L_grid[0].item()); best_res = float('inf')
        for L in L_grid.tolist():
            feat = 1.0 - torch.exp(-rb / L)
            A = torch.stack([torch.ones_like(feat), feat], dim=1)      # [nb', 2]
            AtW = A.t() * wts                                          # [2, nb']
            try:
                c = torch.linalg.solve(AtW @ A + self.eps * eye2, AtW @ gb)
            except Exception:
                continue
            res = float((wts * (A @ c - gb) ** 2).sum().item())
            if res < best_res:
                best_res = res; best_L = float(L)
        L = max(best_L, self.eps)
        self.L_est.fill_(L)
        # under-resolution: the design is coarser than the structure. Two robust signals --
        # (a) the median nearest-signature spacing exceeds the fitted L, or (b) the variogram is
        # already at its sill at the smallest lag (even the closest pairs are decorrelated, so
        # the structure is below the point spacing and L is not fit-recoverable at this budget).
        sill = float(gb.max().item())
        flat = sill > self.eps and float(gb[0].item()) >= 0.7 * sill
        underres = (spacing > L) or flat
        self.underresolved.fill_(1 if underres else 0)
        # j = median count within L (excl. self)
        j_count = float(torch.median((Dq <= L).sum(dim=1).float() - 1.0).item())
        # variance-target floor: rel SE(log p_hat) ~ sqrt( (Var(lam)/lam^2)_bar / j )
        lam_med = max(float(torch.median(lam).item()), self.eps)
        rel = (1.0 - self.eta) / (1.0 + self.eta) / lam_med
        j_min = max(1, min(int(math.ceil(rel / max(self.pool_rse_target ** 2, self.eps))), self.pool_max))
        # under-resolution falls back to the variance target (never 1)
        j_raw = float(j_min) if underres else max(j_count, 1.0)
        j_raw = min(max(j_raw, float(j_min)), float(self.pool_max))
        self.j_smooth.mul_(1.0 - self.pool_ema).add_(self.pool_ema * j_raw)
        j_new = int(round(min(max(float(self.j_smooth.item()), float(j_min)), float(self.pool_max))))
        self.j.fill_(j_new)
        # achieved bandwidth h (j-th nearest distance) in units of L
        kth = min(j_new, ne - 1)
        hq = torch.sort(Dq, dim=1).values[:, kth]
        self.h_over_L.fill_(float(torch.median(hq).item()) / L)

    # ------------------------------------------------------------ decay / age
    @torch.no_grad()
    def _decay_age(self):
        ne, nr = self._ne(), self._nr()
        if ne > 0:
            self.est_S[:ne].mul_(self.eta)
            self.est_E[:ne].mul_(self.eta).add_(1.0)
            self.est_age[:ne].add_(1)
        if nr > 0:
            self.res_S[:nr].mul_(self.eta)
            self.res_E[:nr].mul_(self.eta).add_(1.0)
            self.res_age[:nr].add_(1)                                  # res_hits: raw, NOT decayed
            keep = self.res_age[:nr] <= self.T_need
            n_keep = int(keep.sum().item())
            if n_keep < nr:
                gone = torch.nonzero(~keep, as_tuple=False).flatten()
                self._step_expired += int(gone.numel())
                self._step_exit_ages.extend(self.res_age[gone].tolist())
                idx = torch.nonzero(keep, as_tuple=False).flatten()    # ascending -> order preserved
                self.res_b[:n_keep] = self.res_b[:nr][idx]
                self.res_S[:n_keep] = self.res_S[:nr][idx]
                self.res_E[:n_keep] = self.res_E[:nr][idx]
                self.res_age[:n_keep] = self.res_age[:nr][idx]
                self.res_hits[:n_keep] = self.res_hits[:nr][idx]
                self.res_b[n_keep:nr].zero_(); self.res_S[n_keep:nr].zero_()
                self.res_E[n_keep:nr].zero_(); self.res_age[n_keep:nr].zero_()
                self.res_hits[n_keep:nr].zero_()
                self.n_res.fill_(n_keep)

    # ------------------------------------------------------------ read-out
    @torch.no_grad()
    def _matured(self):
        if self._ne() < self.M:
            return False
        return bool((torch.median(self.est_E[:self.M]) >= self.mature_E).item())

    @torch.no_grad()
    def _coldstart_readout(self, s_global):
        """Distance readout (section 3.3) over the established set: t = 1 - Phi((d-mu)/sigma).
        Used until the counters have matured; the same readout the covering set motivates."""
        bank = self.est_b[:self.M]
        d = torch.cdist(s_global, bank, p=1).min(dim=1).values
        Dbank = torch.cdist(bank, bank, p=1)
        Dbank.diagonal().fill_(float('inf'))
        nn = Dbank.min(dim=1).values
        return TypicalityScorer.compute_scores(d, nn.mean(), nn.std())

    @torch.no_grad()
    def _counted_readout(self, s_global):
        """Fixed-radius kernel sum. p_hat(x) = sum_i lambda_hat_i * K(d_i / R_rad) over the
        established signatures within R_rad = radius_mult * s. The triweight clamps to zero
        beyond R_rad, so the radius test is implicit and no argsort is needed. Returns the
        ABSOLUTE local density p_hat (not a rank): the fixed-volume domain makes the kernel sum
        report crowding, which a fixed-count sum cannot (dividing by the j-th nearest distance
        cancels the local scale). Also updates p_ref, the reference scale for the weight."""
        ne = self._ne()
        bank = self.est_b[:ne]
        D = torch.cdist(s_global, bank, p=1)                              # [B, ne]
        R_rad = (self.radius_mult * self.s).clamp(min=self.eps)          # fixed radius (guard s=0)
        lam = self.est_S[:ne] / (self.est_E[:ne] + self.eps)              # [ne] hit-rate
        Kw = self._triweight(D / R_rad)                                   # [B, ne], zero beyond R_rad
        p_hat = (Kw * lam[None, :]).sum(dim=1)                            # [B]
        # p_ref: slow EMA of the batch-median density (section 3.5). The bank already runs on the
        # all-gathered batch, so this update is identical on every rank by construction -- no
        # collective. Init from the first matured batch's median (never 0 -> c=0 -> weight blows
        # up on empty-neighbourhood tiles).
        med = torch.median(p_hat)
        if int(self.p_ref_init.item()) == 0:
            # never 0 -> c=0 -> weight diverges. If the batch median is itself 0 (the median tile
            # has no neighbour -- only in a cold/mistuned bank; the deployed radius measures ~0
            # empty-neighbourhood fraction, so the median is positive), fall back to the batch
            # mean, then eps, so p_ref stays strictly positive.
            init = med if float(med.item()) > 0.0 else p_hat.mean()
            if float(init.item()) <= 0.0:
                init = p_hat.new_tensor(self.eps)
            self.p_ref.copy_(init)
            self.p_ref_init.fill_(1)
        else:
            self.p_ref.mul_(0.999).add_(0.001 * med)
        return p_hat

    # ------------------------------------------------------------ mutation ops
    @torch.no_grad()
    def _seed_established(self, x):
        i = self._ne()
        self.est_b[i] = x; self.est_S[i] = 0.0; self.est_E[i] = 0.0; self.est_age[i] = 0
        self.n_est.add_(1); self._step_admits += 1

    @torch.no_grad()
    def _evict_lfu_established(self):
        lam = self.est_S[:self.M] / (self.est_E[:self.M] + self.eps)
        return _argmin_lowest(lam)

    @torch.no_grad()
    def _seed_reserve(self, x):
        nr = self._nr()
        if nr < self.reserve_cap:
            self.res_b[nr] = x; self.res_S[nr] = 0.0; self.res_E[nr] = 0.0
            self.res_age[nr] = 0; self.res_hits[nr] = 0.0
            self.n_res.add_(1)
        else:
            oldest = _argmax_lowest(self.res_age[:nr])                   # capacity flush: oldest out
            self._step_flushed += 1
            self._step_exit_ages.append(int(self.res_age[oldest].item()))
            self.res_b[oldest] = x; self.res_S[oldest] = 0.0; self.res_E[oldest] = 0.0
            self.res_age[oldest] = 0; self.res_hits[oldest] = 0.0
        self._step_admits += 1

    @torch.no_grad()
    def _graduate(self, r):
        """Promote reserve signature r to established (evict LFU if full), then compact reserve."""
        if self._ne() >= self.M:
            slot = self._evict_lfu_established()
            self._step_evict_lams.append(
                float((self.est_S[slot] / (self.est_E[slot] + self.eps)).item()))
        else:
            slot = self._ne(); self.n_est.add_(1)
        self.est_b[slot] = self.res_b[r]
        self.est_S[slot] = self.res_S[r]
        self.est_E[slot] = self.res_E[r]
        self.est_age[slot] = self.res_age[r]
        self.graduations.add_(1)
        self._step_grad += 1
        self._step_exit_ages.append(int(self.res_age[r].item()))
        nr = self._nr()
        if r < nr - 1:
            self.res_b[r:nr - 1] = self.res_b[r + 1:nr].clone()
            self.res_S[r:nr - 1] = self.res_S[r + 1:nr].clone()
            self.res_E[r:nr - 1] = self.res_E[r + 1:nr].clone()
            self.res_age[r:nr - 1] = self.res_age[r + 1:nr].clone()
            self.res_hits[r:nr - 1] = self.res_hits[r + 1:nr].clone()
        self.res_b[nr - 1].zero_(); self.res_S[nr - 1] = 0.0
        self.res_E[nr - 1] = 0.0; self.res_age[nr - 1] = 0; self.res_hits[nr - 1] = 0.0
        self.n_res.sub_(1)

    # ------------------------------------------------------------ update loop
    @torch.no_grad()
    def _update(self, s_global):
        B = s_global.shape[0]
        s = float(self.s.item())
        for k in range(B):
            x = s_global[k:k + 1]
            ne, nr = self._ne(), self._nr()
            if ne == 0 and nr == 0:
                self._seed_established(x[0]); continue
            d_est = torch.cdist(x, self.est_b[:ne], p=1)[0] if ne > 0 else None
            d_res = torch.cdist(x, self.res_b[:nr], p=1)[0] if nr > 0 else None
            best_est = _argmin_lowest(d_est) if ne > 0 else -1
            best_res = _argmin_lowest(d_res) if nr > 0 else -1
            dist_est = float(d_est[best_est].item()) if ne > 0 else float('inf')
            dist_res = float(d_res[best_res].item()) if nr > 0 else float('inf')
            in_reserve = dist_res < dist_est                             # est wins ties
            nearest_dist = dist_res if in_reserve else dist_est
            if nearest_dist <= s:
                if in_reserve:
                    self.res_S[best_res] += 1.0                          # decayed rate (carried on graduation)
                    self.res_hits[best_res] += 1.0                       # raw hits (graduation threshold)
                    if float(self.res_hits[best_res].item()) >= self.graduation_hits:
                        self._graduate(best_res)                        # Part 1: promote only after k hits
                else:
                    self.est_S[best_est] += 1.0
            else:
                if ne < self.M:
                    self._seed_established(x[0])
                else:
                    self._seed_reserve(x[0])

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def score_and_update(self, s_global, current_iteration=0):
        s_global = s_global.float()
        self._push_buffer(s_global)
        self._reset_step_telemetry()
        self._step_kappa = int(s_global.shape[0])

        if int(self.s_ready.item()) == 0:
            if int(self.sig_filled.item()) < self.s_min_buffer:
                return {'ready': False, 'p_hat': None}
            edge = self._sweep_edge()
            self.s.fill_(edge * (1.0 - self.s_headroom))
            self.s_delta.fill_(0.0)
            self.s_ready.fill_(1)
            self.last_sweep_step.fill_(int(current_iteration))
        elif int(current_iteration) - int(self.last_sweep_step.item()) >= self.s_sweep_interval:
            edge = self._sweep_edge()
            target = edge * (1.0 - self.s_headroom)
            old_s = float(self.s.item())
            self.s.mul_(1.0 - self.s_ema_alpha).add_(self.s_ema_alpha * target)
            self.s_delta.fill_(float(self.s.item()) - old_s)
            self.last_sweep_step.fill_(int(current_iteration))
            if self.pool_selftune and self._matured():
                self._selftune_pool()                                   # Part 2: same cadence as s sweep

        self._decay_age()
        ready = (self._ne() == self.M)
        if not ready:
            self._update(s_global)
            return {'ready': False, 'p_hat': None}
        if not self._matured():
            # cold start: counters immature, no p_hat -> apply NO modulation (explicit; the
            # trainer passes sample_weights=None). _coldstart_readout (distance, section 3.3) is
            # retained but no longer feeds a weight, since it produces no density.
            self._update(s_global)
            return {'ready': True, 'p_hat': None}
        p_hat = self._counted_readout(s_global)
        self._update(s_global)
        return {'ready': True, 'p_hat': p_hat}

    def load_state_dict(self, state_dict, strict=False):
        """Tolerate buffer-size changes across config edits on resume (e.g. reserve_size 300->550,
        or a changed s_buffer_size): copy the overlapping prefix of each mismatched buffer and
        leave the remainder at its init value, so a checkpoint saved under a different size still
        restores the established set, counters, s, and j. New buffers absent from an older
        checkpoint (res_hits, p_ref / p_ref_init, j / j_smooth / L_est / ...) keep their init
        values. Without this, torch's load_state_dict raises on the reserve shape mismatch even
        with strict=False (strict only tolerates missing/unexpected keys, not size changes)."""
        own = self.state_dict()
        reconciled = {}
        for k, v in state_dict.items():
            if k in own and hasattr(v, 'shape') and own[k].shape != v.shape:
                cur = own[k].clone()
                sl = tuple(slice(0, min(a, b)) for a, b in zip(cur.shape, v.shape))
                cur[sl] = v[sl]
                reconciled[k] = cur
            else:
                reconciled[k] = v
        out = super().load_state_dict(reconciled, strict=False)
        self.n_res.clamp_(max=self.reserve_cap)          # keep occupancy within the (maybe smaller) cap
        return out

    @torch.no_grad()
    def sync_fingerprint(self):
        """Flat tensor of all cross-rank state (byte-identical on every rank)."""
        parts = [
            self.est_b.reshape(-1), self.est_S, self.est_E, self.est_age.float(),
            self.n_est.float().reshape(1),
            self.res_b.reshape(-1), self.res_S, self.res_E, self.res_age.float(),
            self.res_hits, self.n_res.float().reshape(1),
            self.p_ref.reshape(1), self.p_ref_init.float().reshape(1),
            self.graduations.float().reshape(1),
            self.s.reshape(1), self.s_delta.reshape(1),
            self.s_ready.float().reshape(1), self.last_sweep_step.float().reshape(1),
            self.sig_ptr.float().reshape(1), self.sig_filled.float().reshape(1),
            self.j.float().reshape(1), self.j_smooth.reshape(1),
            self.L_est.reshape(1), self.h_over_L.reshape(1), self.underresolved.float().reshape(1),
        ]
        return torch.cat(parts)

    @torch.no_grad()
    def stats(self):
        return {
            'n_est': self._ne(), 'n_reserve': self._nr(),
            'graduations': int(self.graduations.item()),
            's': float(self.s.item()), 'j': int(self.j.item()),
            'L': float(self.L_est.item()), 'underresolved': int(self.underresolved.item()),
            'p_ref': float(self.p_ref.item()),
        }

    @torch.no_grad()
    def health(self):
        """Raw per-step + state fields for the compact log line and the scalar meters."""
        ne = self._ne()
        if ne > 0:
            lam = self.est_S[:ne] / (self.est_E[:ne] + self.eps)
            q = torch.quantile(lam, torch.tensor([0.1, 0.5, 0.9], device=lam.device))
            q10, med, q90 = float(q[0]), float(q[1]), float(q[2])
        else:
            q10 = med = q90 = 0.0
        spread = q90 / q10 if q10 > self.eps else 0.0
        ev = self._step_evict_lams
        evict_mean = float(sum(ev) / len(ev)) if ev else 0.0
        evict_over_q10 = evict_mean / q10 if q10 > self.eps else 0.0
        ages = self._step_exit_ages
        kappa = max(1, self._step_kappa)
        return {
            's': float(self.s.item()), 's_delta': float(self.s_delta.item()),
            'j': int(self.j.item()), 'h_over_L': float(self.h_over_L.item()),
            'L': float(self.L_est.item()), 'underresolved': int(self.underresolved.item()),
            'p_ref': float(self.p_ref.item()),
            'n_est': ne, 'n_reserve': self._nr(), 'reserve_cap': self.reserve_cap,
            'admits': int(self._step_admits), 'kappa': int(self._step_kappa),
            'hit_frac': 1.0 - self._step_admits / kappa,
            'graduations': int(self.graduations.item()),
            'lam_q10': q10, 'lam_median': med, 'lam_q90': q90, 'lam_spread': spread,
            'evict_lam_mean': evict_mean, 'evict_over_q10': evict_over_q10,
            'step_grad': int(self._step_grad), 'step_expired': int(self._step_expired),
            'step_flushed': int(self._step_flushed),
            'exit_age_mean': float(sum(ages) / len(ages)) if ages else 0.0,
            'n_eff': float(self.n_eff),
        }

    @torch.no_grad()
    def compact_line(self, grad_per_step, p_median, p_ref, w_mean, ess):
        """Render the one-line 'typ | ...' status from health() + trainer-derived rates
        (grad_per_step; p_median/p_ref/w_mean/ess from the fixed-radius readout and the realized
        weights). ESS (effective fraction of the batch, sum(w)^2 / [sum(w^2)*B]) is the health
        signal -- t_std was uninformative because a rank statistic is uniform on [0,1] whatever
        the estimator is doing, so it sat at sqrt(1/12) and could not move. ESS can move."""
        h = self.health()
        turnover = self.M / grad_per_step if grad_per_step > 1e-9 else float('inf')
        turn_ratio = turnover / h['n_eff'] if h['n_eff'] > 0 else 0.0
        flags = []
        if turnover < 2.0 * h['n_eff']:
            flags.append('!churn')
        if h['evict_over_q10'] >= 1.0 and h['lam_q10'] > 0:
            flags.append('!evict')
        if h['underresolved']:
            flags.append('!underres')
        if h['n_est'] < self.M:
            flags.append('!fill')
        if 0.0 < ess < 0.5:
            flags.append('!ess')                             # >half the effective batch collapsed
        status = ' '.join(flags) if flags else 'OK'
        turn_str = ("inf" if turnover == float('inf')
                    else (f"{turnover / 1000:.1f}k" if turnover >= 1000 else f"{turnover:.0f}"))
        return (
            f"typ | s={h['s']:.2f}({h['s_delta']:+.2f}) j={h['j']} h/L={h['h_over_L']:.2f} | "
            f"n={h['n_est']} res={h['n_reserve']}/{h['reserve_cap']} | "
            f"hit={h['hit_frac'] * 100:.1f}% grad={grad_per_step:.1f}/st turn={turn_str}({turn_ratio:.1f}x) | "
            f"lam={h['lam_q10']:.3f}/{h['lam_median']:.3f}/{h['lam_q90']:.3f} ({h['lam_spread']:.1f}x) "
            f"evict={h['evict_over_q10']:.2f}q10 | "
            f"p={p_median:.3g}/ref{p_ref:.3g} w={w_mean:.2f} ess={ess * 100:.0f}% | "
            f"res_exit=grad{h['step_grad']:.0f}/exp{h['step_expired']:.0f}/flush{h['step_flushed']:.0f} "
            f"age={h['exit_age_mean']:.1f} | {status}"
        )
