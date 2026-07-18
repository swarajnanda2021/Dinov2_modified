"""
Counted-coverage bank (Algorithm 2 of typicality/technique.md, section 3.5).

A second, switchable variant of the typicality bank. It stores a covering set of
anchors but reads crowding from explicit, exponentially-decayed hit counts rather than
from nearest-neighbour distance (Algorithm 1), so the estimate is faithful without
requiring the bank to be a representative sample.

Per-anchor state: position b, decayed hit count S, decayed exposure E (blocks alive),
age. Split into an ESTABLISHED set (capacity M) and a separate RESERVE buffer.
Ratio lambda_hat = S / (E + eps) is the decayed per-anchor hit-RATE; the readout sums
lambda_hat * K_h over the j nearest established anchors (unnormalized centered kernel
sum) and maps log p_hat through a decayed empirical rank (PIT) or a probit.

Determinism: on the global (all-gathered) batch every rank runs the identical update, so
the bank stays byte-identical across ranks. Every tie-break is pinned to LOWEST index.

Fill policy (per the build decision for this fork): while the established set is filling
(n_est < M) a novel tile seeds DIRECTLY into the established set; the reserve and the
graduate-on-first-hit path are enabled only once n_est == M.

NOTE (performance): the update loop is sequential over the gathered batch, as Algorithm 2
specifies (each tile can seed an anchor that changes the nearest-anchor answer for the
next tile). This is a per-block Python loop; profile it cluster-side before long runs.
"""

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

    def __init__(self, M, K_prime, spot_radius=2.75, pool_j=64, halflife_blocks=250,
                 reserve_residency=300, reserve_size=300, readout='pit',
                 pit_buffer=20000, eps=1e-8):
        super().__init__()
        assert readout in ('pit', 'probit')
        self.M = int(M)
        self.K_prime = int(K_prime)
        self.s = float(spot_radius)
        self.j = int(pool_j)
        self.eta = 0.5 ** (1.0 / float(halflife_blocks))     # per-block decay
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
        # ---- diagnostics (logged per checkpoint) ----
        self.register_buffer('graduations', torch.tensor(0, dtype=torch.long))

    # ---------------------------------------------------------------- helpers
    def _ne(self):
        return int(self.n_est.item())

    def _nr(self):
        return int(self.n_res.item())

    def _triweight(self, u):
        """k(u) = (1 - u^2)^3 on u <= 1, else 0 (compact support)."""
        w = (1.0 - u * u).clamp(min=0.0)
        return w * w * w

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
        """p_hat = sum over j-nearest established anchors of lambda_hat * K_h; t = F(log p_hat)."""
        ne = self._ne()
        bank = self.est_b[:ne]
        D = torch.cdist(s_global, bank, p=1)                              # [B, ne]
        k = min(self.j, ne)
        order = torch.argsort(D, dim=1, stable=True)[:, :k]               # lowest-index tie-break
        vals = torch.gather(D, 1, order)                                  # [B, k] distances (ascending)
        h = vals[:, -1:].clamp(min=self.eps)                             # bandwidth = dist to j-th nearest
        Kw = self._triweight(vals / h)                                    # [B, k]
        lam = (self.est_S[:ne] / (self.est_E[:ne] + self.eps))[order]     # [B, k] hit-rate at those anchors
        p_hat = (lam * Kw).sum(dim=1)                                     # [B] unnormalized kernel sum
        logp = torch.log(p_hat + self.eps)
        t = self._pit_score(logp) if self.readout == 'pit' else self._probit_score(logp)
        # fold this block's log p_hat into the reference AFTER scoring (rank vs. history)
        self._pit_push(logp)
        self._probit_push(logp)
        return t.clamp(0.0, 1.0)

    # -------------------------------------------------------- PIT reference
    @torch.no_grad()
    def _pit_score(self, logp):
        filled = int(self.pit_filled.item())
        ref = self.pit_ring[:filled] if filled > 0 else logp             # first block: self-rank
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
            # first block: standardize within the batch
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

    @torch.no_grad()
    def _graduate(self, r):
        """Move reserve anchor r into the established set (evict LFU if full), then compact reserve."""
        slot = self._evict_lfu_established() if self._ne() >= self.M else self._ne()
        if self._ne() < self.M:
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
        for k in range(B):                                               # sequential, fixed row order
            x = s_global[k:k + 1]                                        # [1, K']
            ne, nr = self._ne(), self._nr()
            if ne == 0 and nr == 0:
                self._seed_established(x[0]); continue
            # nearest anchor over ALL live anchors (established + reserve), lowest-index tie-break
            d_est = torch.cdist(x, self.est_b[:ne], p=1)[0] if ne > 0 else None
            d_res = torch.cdist(x, self.res_b[:nr], p=1)[0] if nr > 0 else None
            best_est = _argmin_lowest(d_est) if ne > 0 else -1
            best_res = _argmin_lowest(d_res) if nr > 0 else -1
            dist_est = float(d_est[best_est].item()) if ne > 0 else float('inf')
            dist_res = float(d_res[best_res].item()) if nr > 0 else float('inf')
            in_reserve = dist_res < dist_est                             # est wins ties (lower "index space")
            nearest_dist = dist_res if in_reserve else dist_est
            if nearest_dist <= self.s:
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
        """Flat tensor of all cross-rank state (must be byte-identical on every rank)."""
        parts = [
            self.est_b.reshape(-1), self.est_S, self.est_E, self.est_age.float(),
            self.n_est.float().reshape(1),
            self.res_b.reshape(-1), self.res_S, self.res_E, self.res_age.float(),
            self.n_res.float().reshape(1),
            self.pit_ring, self.pit_ptr.float().reshape(1), self.pit_filled.float().reshape(1),
            self.pb_w.reshape(1), self.pb_wm.reshape(1), self.pb_wm2.reshape(1),
            self.graduations.float().reshape(1),
        ]
        return torch.cat(parts)

    @torch.no_grad()
    def stats(self):
        return {
            'n_est': self._ne(),
            'n_reserve': self._nr(),
            'graduations': int(self.graduations.item()),
        }
