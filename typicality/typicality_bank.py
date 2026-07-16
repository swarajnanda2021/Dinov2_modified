"""
FIFO keep-last-M reservoir for redundancy estimation.
Stores the M most recent morphology signatures as a first-in-first-out ring buffer.
Computes within-bank nearest-neighbor distances for Gaussian calibration of typicality scores.
"""

import torch
import torch.nn as nn


class TypicalityBank(nn.Module):
    """
    FIFO keep-last-M signature reservoir for redundancy estimation.

    A fixed-capacity ring buffer of the M most recent signatures. Each step the
    incoming batch is appended and the oldest entries are evicted (FIFO), so the
    bank is a running, content-independent sample of the signature distribution.

    Args:
        M: Bank capacity (number of stored signatures)
        K_prime: Signature dimensionality (number of representative prototypes)
    """
    def __init__(self, M, K_prime):
        super().__init__()
        self.M = M
        self.K_prime = K_prime

        # Ring buffer — persists across steps, saved in checkpoints.
        self.register_buffer('bank', torch.zeros(M, K_prime))
        self.register_buffer('bank_filled', torch.tensor(0, dtype=torch.long))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))

    def _fifo_insert(self, s_batch):
        """Append s_batch to the ring buffer, evicting the oldest entries."""
        B = s_batch.shape[0]
        if B >= self.M:
            # Only the most recent M signatures survive.
            self.bank.copy_(s_batch[-self.M:])
            self.write_ptr.fill_(0)
            self.bank_filled.fill_(self.M)
            return
        ptr = int(self.write_ptr.item())
        end = ptr + B
        if end <= self.M:
            self.bank[ptr:end] = s_batch
        else:
            first = self.M - ptr
            self.bank[ptr:] = s_batch[:first]
            self.bank[:end - self.M] = s_batch[first:]
        self.write_ptr.fill_(end % self.M)
        self.bank_filled.fill_(min(self.M, int(self.bank_filled.item()) + B))

    @torch.no_grad()
    def update_and_score(self, s_query, s_insert=None):
        """
        Score s_query against the current bank, then FIFO-insert s_insert.

        Scoring happens BEFORE insertion, so the batch never matches itself.
        s_insert defaults to s_query (single-rank / local bank); with a global
        bank the caller scores the local s_query but inserts the all-gathered
        s_insert, so every rank inserts the identical batch and banks stay in sync.
        During filling (bank not yet full): insert only, no scoring.

        Args:
            s_query: [B, K'] detached local signatures to score
            s_insert: [B_ins, K'] detached signatures to insert (defaults to s_query)

        Returns:
            dict with:
                ready: bool — True once the ring has filled (bank_filled >= M)
                d: [B] min L1 distance from each batch sample to its nearest bank entry
                bank_nn_dists: [M] within-bank nearest-neighbor L1 distances
                mu: scalar mean of bank_nn_dists
                sigma: scalar std of bank_nn_dists
        """
        if s_insert is None:
            s_insert = s_query
        s_query = s_query.float()
        s_insert = s_insert.float()  # resolve autocast dtype

        # ---- Filling phase: bank not yet full -> insert only, not ready ----
        if int(self.bank_filled.item()) < self.M:
            self._fifo_insert(s_insert)
            return {
                'ready': False,
                'd': None,
                'bank_nn_dists': None,
                'mu': None,
                'sigma': None,
            }

        # ---- Bank full: score s_query against the current bank, then insert ----

        # Batch-to-bank L1 distances: [B, M] -> [B]
        d = torch.cdist(s_query, self.bank, p=1).min(dim=1).values  # [B]

        # Within-bank NN L1 distances: [M, M] -> [M]
        D_bank = torch.cdist(self.bank, self.bank, p=1)
        D_bank.diagonal().fill_(float('inf'))
        bank_nn_dists = D_bank.min(dim=1).values  # [M]

        mu = bank_nn_dists.mean()
        sigma = bank_nn_dists.std()

        # ---- FIFO churn: keep the most recent M signatures ----
        self._fifo_insert(s_insert)

        return {
            'ready': True,
            'd': d,
            'bank_nn_dists': bank_nn_dists,
            'mu': mu,
            'sigma': sigma,
        }
