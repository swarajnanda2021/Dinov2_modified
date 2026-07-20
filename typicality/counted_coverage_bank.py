"""
Counted-coverage bank (Algorithm 2 of typicality/technique.md, section 3.5).

A second, switchable variant of the typicality bank. It stores a covering set of
stored signatures but reads crowding from explicit, exponentially-decayed hit counts
rather than from nearest-neighbour distance (Algorithm 1), so the estimate is faithful
without requiring the bank to be a representative sample.

Per-signature state: position b, decayed hit count S, decayed exposure E (steps alive),
age. Split into an ESTABLISHED set (capacity M) and a separate RESERVE buffer. Ratio
lambda_hat = S / (E + eps) is the decayed per-signature hit-RATE; the readout sums
lambda_hat * K_h over the j nearest established signatures (unnormalized centered kernel
sum) and maps log p_hat through a decayed empirical rank (PIT) or a probit.

Self-tuning hit radius s (section 3.5, Placement): s is NOT a fixed constant. The offline
synthetic-R knee was s ~ 2.75, but the live online-trained R produces signatures ~6x
larger, so any frozen radius is wrong and drifts. Instead the bank keeps a rolling ring
buffer of the most recent ~60,000 global signatures and, every ~500 steps, sweeps it: for
a grid of radii re-centered on the live signature scale it replays seed-on-miss from empty
(a tile seeds a new entry iff it is farther than s, in L1, from every entry placed so far,
up to capacity M) and finds the largest radius that still fills the bank to M -- the fill
knee. s tracks that edge, lightly EMA-smoothed. The sweep is read-only on the live bank
(scratch tensors only); only the scalar s is updated.

Determinism: on the global (all-gathered) batch every rank runs the identical update and
the identical sweep over the identical buffer, so the bank -- and s -- stay byte-identical
across ranks. Every tie-break is pinned to LOWEST index.

Fill policy (per the build decision for this fork): while the established set is filling
(n_est < M) a novel tile seeds DIRECTLY into the established set; the reserve and the
graduate-on-first-hit path are enabled only once n_est == M.

Startup (matters for resume-from-warmup): at activation the buffer is empty, so from the
first activated step signatures are pushed to the buffer while the module scores t = 0 and
the bank does not seed. Once the buffer holds >= s_min_buffer signatures the first sweep
runs, sets s directly, and only then does the counted bank begin its normal fill.

NOTE (performance): the update loop is sequential over the gathered batch, as Algorithm 2
specifies (each tile can seed a signature that changes the nearest answer for the next
tile). This is a per-step Python loop; profile it cluster-side before long runs.
"""

import math

import torch
import torch.nn as nn

from .typicality_scorer import TypicalityScorer


def _argmin_lowest(x):
    """argmin with lowest-index tie-break (torch.argmin already returns the first/lowest
    index on ties; kept explicit so it can't silently change)."""
    return int(torch.argmin(x).item())


def _argmax_lowest(x):
    """argmax with lowest-index tie-break (first/lowest index on ties)."""
    return int(torch.argmax(x).item())


class CountedCoverageBank(nn.Module):
    """Algorithm 2. Uniform interface with TypicalityBank:
        score_and_update(s_global, current_iteration) -> {'ready': bool, 't': Tensor|None}
        sync_fingerprint() -> Tensor
    """

    def __init__(self, M, K_prime, pool_j=64, halflife_steps=250,
                 reserve_residency=300, reserve_size=300, readout='pit',
                 pit_buffer=20000, eps=1e-8,
                 s_buffer_size=60000, s_sweep_interval=500, s_grid_points=9,
                 grid_span=(0.3, 2.0), s_ema_alpha=0.2, s_min_buffer=60000,
                 s_headroom=0.0, strong_lambda=0.1, scale_sample=2048):
        super().__init__()
        assert readout in ('pit', 'probit')
        self.M = int(M)
        self.K_prime = int(K_prime)
        self.j = int(pool_j)
        self.eta = 0.5 ** (1.0 / float(halflife_steps))      # per-step decay
        self.T_need = int(reserve_residency)
        self.reserve_cap = int(reserve_size)
        self.readout = readout
        self.pit_buffer = int(pit_buffer)
        self.eps = float(eps)
        # Maturation gate: counted readout activates once full AND median exposure has
        # passed one half-life's accumulation (the E ceiling 1/(1-eta) is never reached).
        self.mature_E = 0.5 / (1.0 - self.eta)
        # probit decayed moments: half-life matched to the PIT ring (pit_buffer tiles),
        # NOT the counter half-life -> per-tile decay rho.
        self.pb_rho = 0.5 ** (1.0 / max(1, self.pit_buffer))
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
        self.register_buffer('n_res', torch.tensor(0, dtype=torch.long))
        # ---- PIT reference: FIFO ring of recent log p_hat ----
        self.register_buffer('pit_ring', torch.zeros(self.pit_buffer))
        self.register_buffer('pit_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('pit_filled', torch.tensor(0, dtype=torch.long))
        # ---- probit decayed moments of log p_hat ----
        self.register_buffer('pb_w', torch.tensor(0.0))     # total weight
        self.register_buffer('pb_wm', torch.tensor(0.0))    # weighted sum of logp
        self.register_buffer('pb_wm2', torch.tensor(0.0))   # weighted sum of logp^2
        # ---- self-tuning s state (runtime, checkpointed, rank-identical) ----
        self.register_buffer('s', torch.tensor(0.0))                 # hit radius (unset until 1st sweep)
        self.register_buffer('s_ready', torch.tensor(0, dtype=torch.long))
        self.register_buffer('last_sweep_step', torch.tensor(0, dtype=torch.long))
        # rolling signature ring buffer (global signatures, fp16 to save memory)
        self.register_buffer('sig_ring', torch.zeros(self.s_buffer_size, self.K_prime, dtype=torch.float16))
        self.register_buffer('sig_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('sig_filled', torch.tensor(0, dtype=torch.long))
        # ---- diagnostics (logged per checkpoint / per interval) ----
        self.register_buffer('graduations', torch.tensor(0, dtype=torch.long))
        # per-step telemetry (plain attrs, rank-identical, not checkpointed)
        self._step_admits = 0
        self._step_evict_lams = []

    # ---------------------------------------------------------------- helpers
    def _ne(self):
        return int(self.n_est.item())

    def _nr(self):
        return int(self.n_res.item())

    def _triweight(self, u):
        """k(u) = (1 - u^2)^3 on u <= 1, else 0 (compact support)."""
        w = (1.0 - u * u).clamp(min=0.0)
        return w * w * w

    # ------------------------------------------------------- rolling s buffer
    @torch.no_grad()
    def _push_buffer(self, s_global):
        """FIFO-push the gathered signatures into the rolling ring (stored fp16)."""
        x = s_global.detach().to(self.sig_ring.dtype)
        n = x.shape[0]
        cap = self.s_buffer_size
        if n >= cap:
            self.sig_ring.copy_(x[-cap:])
            self.sig_ptr.fill_(0)
            self.sig_filled.fill_(cap)
            return
        ptr = int(self.sig_ptr.item())
        end = ptr + n
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
        """Live signature scale m = median nearest-neighbour L1 distance among a
        deterministic evenly-spaced subsample of the buffer (subsample for speed)."""
        n = buf.shape[0]
        k = min(n, self.scale_sample)
        idx = torch.linspace(0, n - 1, k).round().long()        # deterministic, spans buffer
        sub = buf[idx]
        D = torch.cdist(sub, sub, p=1)
        D.diagonal().fill_(float('inf'))
        nn = D.min(dim=1).values
        return max(float(nn.median().item()), self.eps)

    @torch.no_grad()
    def _seed_on_miss(self, buf, s):
        """Exact seed-on-miss replay over buf in fixed order: a tile seeds a new center
        iff its L1 distance to every center placed so far is > s, up to capacity M. Returns
        the number of centers placed (capped at M). Read-only scratch; touches no bank
        state. Chunked for speed but exact -- points uncovered by the pre-chunk centers are
        resolved sequentially so intra-chunk seeds still cover their successors."""
        n = buf.shape[0]
        M = self.M
        centers = buf.new_empty((M, buf.shape[1]))
        nc = 0
        CH = 4096
        i = 0
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
                    centers[0] = chunk[local]
                    nc = 1
                    continue
                dm = torch.cdist(chunk[local:local + 1], centers[:nc], p=1).min().item()
                if dm > s:
                    centers[nc] = chunk[local]
                    nc += 1
            i = j
        return nc

    @torch.no_grad()
    def _sweep_edge(self):
        """Read-only sweep of the current buffer: return the fill knee -- the largest s
        that still fills a bank to M under seed-on-miss. Grid is re-centered on the live
        scale every call (never a fixed grid), then one bisection refinement across the
        fill->underfill transition. Deterministic -> rank-identical."""
        n = int(self.sig_filled.item())
        buf = self.sig_ring[:n].float()                          # upcast fp16 -> fp32 scratch
        m = self._buffer_scale(buf)
        lo, hi = self.grid_span
        grid = torch.exp(torch.linspace(math.log(lo * m), math.log(hi * m), self.s_grid_points))
        fills = [self._seed_on_miss(buf, float(sc)) >= self.M for sc in grid.tolist()]
        true_idx = [i for i, f in enumerate(fills) if f]
        if not true_idx:
            return float(grid[0].item())                         # degenerate: even smallest s underfills
        last_true = max(true_idx)
        if last_true == self.s_grid_points - 1:
            return float(grid[-1].item())                        # even largest s fills: edge >= grid max
        lo_s = float(grid[last_true].item())                     # fills
        hi_s = float(grid[last_true + 1].item())                 # underfills
        mid = math.sqrt(lo_s * hi_s)                             # one bisection (geometric midpoint)
        return mid if (self._seed_on_miss(buf, mid) >= self.M) else lo_s

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
            self.res_age[:nr].add_(1)
            # expire ungraduated one-offs whose age exceeds T_need (compact, order-preserving)
            keep = self.res_age[:nr] <= self.T_need
            n_keep = int(keep.sum().item())
            if n_keep < nr:
                idx = torch.nonzero(keep, as_tuple=False).flatten()   # ascending index -> order preserved
                self.res_b[:n_keep] = self.res_b[:nr][idx]
                self.res_S[:n_keep] = self.res_S[:nr][idx]
                self.res_E[:n_keep] = self.res_E[:nr][idx]
                self.res_age[:n_keep] = self.res_age[:nr][idx]
                self.res_b[n_keep:nr].zero_(); self.res_S[n_keep:nr].zero_()
                self.res_E[n_keep:nr].zero_(); self.res_age[n_keep:nr].zero_()
                self.n_res.fill_(n_keep)

    # ------------------------------------------------------------ read-out
    @torch.no_grad()
    def _matured(self):
        if self._ne() < self.M:
            return False
        return bool((torch.median(self.est_E[:self.M]) >= self.mature_E).item())

    @torch.no_grad()
    def _coldstart_readout(self, s_global):
        """Distance readout of section 3.3 over the (full) established set."""
        bank = self.est_b[:self.M]
        d = torch.cdist(s_global, bank, p=1).min(dim=1).values            # [B]
        Dbank = torch.cdist(bank, bank, p=1)
        Dbank.diagonal().fill_(float('inf'))
        nn = Dbank.min(dim=1).values                                      # [M]
        mu, sigma = nn.mean(), nn.std()
        return TypicalityScorer.compute_scores(d, mu, sigma)              # reuse, no duplication

    @torch.no_grad()
    def _counted_readout(self, s_global):
        """p_hat = sum over j-nearest established signatures of lambda_hat * K_h; t = F(log p_hat)."""
        ne = self._ne()
        bank = self.est_b[:ne]
        D = torch.cdist(s_global, bank, p=1)                              # [B, ne]
        k = min(self.j, ne)
        order = torch.argsort(D, dim=1, stable=True)[:, :k]               # lowest-index tie-break
        vals = torch.gather(D, 1, order)                                  # [B, k] distances (ascending)
        h = vals[:, -1:].clamp(min=self.eps)                             # bandwidth = dist to j-th nearest
        Kw = self._triweight(vals / h)                                    # [B, k]
        lam = (self.est_S[:ne] / (self.est_E[:ne] + self.eps))[order]     # [B, k] hit-rate at those signatures
        p_hat = (lam * Kw).sum(dim=1)                                     # [B] unnormalized kernel sum
        logp = torch.log(p_hat + self.eps)
        t = self._pit_score(logp) if self.readout == 'pit' else self._probit_score(logp)
        # fold this step's log p_hat into the reference AFTER scoring (rank vs. history)
        self._pit_push(logp)
        self._probit_push(logp)
        return t.clamp(0.0, 1.0)

    # -------------------------------------------------------- PIT reference
    @torch.no_grad()
    def _pit_score(self, logp):
        filled = int(self.pit_filled.item())
        ref = self.pit_ring[:filled] if filled > 0 else logp             # first step: self-rank
        ref_sorted, _ = torch.sort(ref)                                  # ascending
        rank = torch.searchsorted(ref_sorted, logp, right=True).float()  # # ref <= logp
        return rank / max(1, ref_sorted.numel())

    @torch.no_grad()
    def _pit_push(self, logp):
        cap = self.pit_buffer
        flat = logp[-cap:] if logp.numel() > cap else logp              # keep only most recent cap
        n = int(flat.numel())
        ptr = int(self.pit_ptr.item())
        end = ptr + n
        if end <= cap:
            self.pit_ring[ptr:end] = flat
        else:
            first = cap - ptr
            self.pit_ring[ptr:] = flat[:first]
            self.pit_ring[:end - cap] = flat[first:]
        self.pit_ptr.fill_(end % cap)
        self.pit_filled.fill_(min(cap, int(self.pit_filled.item()) + n))

    # -------------------------------------------------------- probit fallback
    @torch.no_grad()
    def _probit_score(self, logp):
        w = float(self.pb_w.item())
        if w <= 0.0:
            # first step: standardize within the batch
            m = logp.mean(); v = logp.var(unbiased=False)
        else:
            m = self.pb_wm / self.pb_w
            v = (self.pb_wm2 / self.pb_w - m * m).clamp(min=0.0)
        sigma = v.clamp(min=self.eps).sqrt()
        z = (logp - m) / sigma
        return 0.5 * (1.0 + torch.erf(z * (1.0 / (2.0 ** 0.5))))

    @torch.no_grad()
    def _probit_push(self, logp):
        # decayed moments; half-life matched to the PIT ring (pb_rho per tile).
        decay = self.pb_rho ** logp.numel()
        self.pb_w.mul_(decay).add_(float(logp.numel()))
        self.pb_wm.mul_(decay).add_(logp.sum())
        self.pb_wm2.mul_(decay).add_((logp * logp).sum())

    # ------------------------------------------------------------ mutation ops
    @torch.no_grad()
    def _seed_established(self, x):
        i = self._ne()
        self.est_b[i] = x; self.est_S[i] = 0.0; self.est_E[i] = 0.0; self.est_age[i] = 0
        self.n_est.add_(1)
        self._step_admits += 1

    @torch.no_grad()
    def _evict_lfu_established(self):
        """Return the established slot of lowest hit-rate S/(E+eps) (lowest index on ties)."""
        lam = self.est_S[:self.M] / (self.est_E[:self.M] + self.eps)
        return _argmin_lowest(lam)

    @torch.no_grad()
    def _seed_reserve(self, x):
        nr = self._nr()
        if nr < self.reserve_cap:
            self.res_b[nr] = x; self.res_S[nr] = 0.0; self.res_E[nr] = 0.0; self.res_age[nr] = 0
            self.n_res.add_(1)
        else:
            # reserve full: evict its oldest entry (max age; lowest index on ties), reuse the slot
            oldest = _argmax_lowest(self.res_age[:nr])
            self.res_b[oldest] = x; self.res_S[oldest] = 0.0
            self.res_E[oldest] = 0.0; self.res_age[oldest] = 0
        self._step_admits += 1

    @torch.no_grad()
    def _graduate(self, r):
        """Move reserve signature r into the established set (evict LFU if full), then compact reserve."""
        if self._ne() >= self.M:
            slot = self._evict_lfu_established()
            # record the evicted signature's hit-rate (health telemetry: should stay near zero)
            self._step_evict_lams.append(
                float((self.est_S[slot] / (self.est_E[slot] + self.eps)).item()))
        else:
            slot = self._ne()
            self.n_est.add_(1)
        self.est_b[slot] = self.res_b[r]
        self.est_S[slot] = self.res_S[r]
        self.est_E[slot] = self.res_E[r]
        self.est_age[slot] = self.res_age[r]
        self.graduations.add_(1)
        # remove r from reserve (compact, order-preserving)
        nr = self._nr()
        if r < nr - 1:
            self.res_b[r:nr - 1] = self.res_b[r + 1:nr].clone()
            self.res_S[r:nr - 1] = self.res_S[r + 1:nr].clone()
            self.res_E[r:nr - 1] = self.res_E[r + 1:nr].clone()
            self.res_age[r:nr - 1] = self.res_age[r + 1:nr].clone()
        self.res_b[nr - 1].zero_(); self.res_S[nr - 1] = 0.0
        self.res_E[nr - 1] = 0.0; self.res_age[nr - 1] = 0
        self.n_res.sub_(1)

    # ------------------------------------------------------------ update loop
    @torch.no_grad()
    def _update(self, s_global):
        B = s_global.shape[0]
        s = float(self.s.item())                                         # current self-tuned radius
        for k in range(B):                                               # sequential, fixed row order
            x = s_global[k:k + 1]                                        # [1, K']
            ne, nr = self._ne(), self._nr()
            if ne == 0 and nr == 0:
                self._seed_established(x[0]); continue
            # nearest over ALL live stored signatures (established + reserve), lowest-index tie-break
            d_est = torch.cdist(x, self.est_b[:ne], p=1)[0] if ne > 0 else None
            d_res = torch.cdist(x, self.res_b[:nr], p=1)[0] if nr > 0 else None
            best_est = _argmin_lowest(d_est) if ne > 0 else -1
            best_res = _argmin_lowest(d_res) if nr > 0 else -1
            dist_est = float(d_est[best_est].item()) if ne > 0 else float('inf')
            dist_res = float(d_res[best_res].item()) if nr > 0 else float('inf')
            in_reserve = dist_res < dist_est                             # est wins ties (lower "index space")
            nearest_dist = dist_res if in_reserve else dist_est
            if nearest_dist <= s:
                if in_reserve:
                    self.res_S[best_res] += 1.0                          # hit (exposure aged already)
                    self._graduate(best_res)                            # graduate on first hit
                else:
                    self.est_S[best_est] += 1.0
            else:
                # novel tile
                if ne < self.M:
                    self._seed_established(x[0])                        # direct-to-established during fill
                else:
                    self._seed_reserve(x[0])                            # reserve + graduation active at capacity

    # ------------------------------------------------------------ public API
    @torch.no_grad()
    def score_and_update(self, s_global, current_iteration=0):
        s_global = s_global.float()
        self._push_buffer(s_global)                                     # rolling buffer (always)
        self._step_admits = 0                                           # per-step telemetry reset
        self._step_evict_lams = []

        if int(self.s_ready.item()) == 0:
            # startup: fill the buffer, score t=0, do NOT seed until the first sweep sets s
            if int(self.sig_filled.item()) < self.s_min_buffer:
                return {'ready': False, 't': None}
            edge = self._sweep_edge()
            self.s.fill_(edge * (1.0 - self.s_headroom))               # first sweep sets s directly
            self.s_ready.fill_(1)
            self.last_sweep_step.fill_(int(current_iteration))
            # fall through: the counted bank may now begin its normal fill this step
        elif int(current_iteration) - int(self.last_sweep_step.item()) >= self.s_sweep_interval:
            edge = self._sweep_edge()                                   # periodic sweep -> EMA-smoothed s
            target = edge * (1.0 - self.s_headroom)
            self.s.mul_(1.0 - self.s_ema_alpha).add_(self.s_ema_alpha * target)
            self.last_sweep_step.fill_(int(current_iteration))

        self._decay_age()
        ready = (self._ne() == self.M)
        if not ready:
            self._update(s_global)                                      # keep filling; no scoring yet
            return {'ready': False, 't': None}
        t = self._counted_readout(s_global) if self._matured() else self._coldstart_readout(s_global)
        self._update(s_global)
        return {'ready': True, 't': t}

    @torch.no_grad()
    def sync_fingerprint(self):
        """Flat tensor of all cross-rank state (must be byte-identical on every rank).
        The rolling sig_ring is rank-identical by construction (global signatures) and its
        derived scalar s is included here, so the ring itself is omitted to keep this cheap."""
        parts = [
            self.est_b.reshape(-1), self.est_S, self.est_E, self.est_age.float(),
            self.n_est.float().reshape(1),
            self.res_b.reshape(-1), self.res_S, self.res_E, self.res_age.float(),
            self.n_res.float().reshape(1),
            self.pit_ring, self.pit_ptr.float().reshape(1), self.pit_filled.float().reshape(1),
            self.pb_w.reshape(1), self.pb_wm.reshape(1), self.pb_wm2.reshape(1),
            self.graduations.float().reshape(1),
            self.s.reshape(1), self.s_ready.float().reshape(1), self.last_sweep_step.float().reshape(1),
            self.sig_ptr.float().reshape(1), self.sig_filled.float().reshape(1),
        ]
        return torch.cat(parts)

    @torch.no_grad()
    def stats(self):
        return {
            'n_est': self._ne(),
            'n_reserve': self._nr(),
            'graduations': int(self.graduations.item()),
            's': float(self.s.item()),
        }

    @torch.no_grad()
    def health(self):
        """Per-interval health telemetry. lambda_hat = S/E over the established set (core
        strength), the fraction above a small threshold (the 'strong core'), plus admits and
        the hit-rate of entries evicted this step (should stay near zero -- junk turning over)."""
        ne = self._ne()
        if ne > 0:
            lam = self.est_S[:ne] / (self.est_E[:ne] + self.eps)
            q = torch.quantile(lam, torch.tensor([0.1, 0.5, 0.9], device=lam.device))
            lam_q10, lam_median, lam_q90 = float(q[0]), float(q[1]), float(q[2])
            strong = float((lam > self.strong_lambda).float().mean().item())
        else:
            lam_q10 = lam_median = lam_q90 = strong = 0.0
        ev = self._step_evict_lams
        evict_lam_mean = float(sum(ev) / len(ev)) if ev else 0.0
        return {
            's': float(self.s.item()),
            'n_est': ne,
            'n_reserve': self._nr(),
            'graduations': int(self.graduations.item()),
            'admits': int(self._step_admits),
            'lam_q10': lam_q10,
            'lam_median': lam_median,
            'lam_q90': lam_q90,
            'strong_core_frac': strong,
            'evict_lam_mean': evict_lam_mean,
        }
