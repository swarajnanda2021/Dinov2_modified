"""
Diversity-managed bank for redundancy estimation.
Stores morphology signatures and maintains diversity via batch-to-bank L1 churn.
Computes within-bank nearest-neighbor distances for Gaussian calibration of typicality scores.
"""

import torch
import torch.nn as nn


class TypicalityBank(nn.Module):
    """
    Diversity-managed signature bank for redundancy estimation.
    
    Not a FIFO queue. Entries are replaced based on L1 redundancy:
    the most novel batch samples displace the bank entries closest to them.
    
    Args:
        M: Bank capacity (number of stored signatures)
        K_prime: Signature dimensionality (number of representative prototypes)
        replace_fraction: Fraction of bank to refresh per step (default 0.1 = 10%)
    """
    def __init__(self, M, K_prime, replace_fraction=0.1):
        super().__init__()
        self.M = M
        self.K_prime = K_prime
        self.N_replace = max(1, int(M * replace_fraction))
        
        # Bank buffer — persists across steps, saved in checkpoints
        self.register_buffer('bank', torch.zeros(M, K_prime))
        self.register_buffer('bank_filled', torch.tensor(0, dtype=torch.long))
    
    @property
    def is_full(self):
        return self.bank_filled.item() >= self.M
    
    @torch.no_grad()
    def update_and_score(self, s_batch):
        """
        Insert signatures into the bank and return scoring data.
        
        During filling: signatures fill empty slots, no scoring.
        Once full: compute L1 distances, churn, return data for Gaussian scoring.
        
        Args:
            s_batch: [B, K'] detached morphology signatures
            
        Returns:
            dict with:
                ready: bool — True if bank is full and scoring data is valid
                d: [B] min L1 distance from each batch sample to its nearest bank entry
                bank_nn_dists: [M] within-bank nearest-neighbor L1 distances
                mu: scalar mean of bank_nn_dists
                sigma: scalar std of bank_nn_dists
        """
        B = s_batch.shape[0]
        s_batch = s_batch.float() # Convert to float for resolving autocast issues
        filled = self.bank_filled.item()
        
        # ---- Filling phase: bank not yet full ----
        if filled < self.M:
            n_insert = min(B, self.M - filled)
            self.bank[filled:filled + n_insert] = s_batch[:n_insert]
            self.bank_filled += n_insert
            
            return {
                'ready': False,
                'd': None,
                'bank_nn_dists': None,
                'mu': None,
                'sigma': None,
            }
        
        # ---- Bank is full: score and churn ----
        
        # Batch-to-bank L1 distances: [B, M]
        D_batch_bank = torch.cdist(s_batch, self.bank, p=1)
        d = D_batch_bank.min(dim=1).values       # [B]
        d_argmin = D_batch_bank.argmin(dim=1)     # [B] — nearest bank entry per sample
        
        # Within-bank NN L1 distances: [M, M] -> [M]
        D_bank = torch.cdist(self.bank, self.bank, p=1)
        D_bank.diagonal().fill_(float('inf'))
        bank_nn_dists = D_bank.min(dim=1).values  # [M]
        
        mu = bank_nn_dists.mean()
        sigma = bank_nn_dists.std()
        
        # ---- Churn: insert N most novel batch samples ----
        N = min(self.N_replace, B)
        _, novel_idx = d.topk(N, largest=True)     # [N] indices into batch
        evict_idx = d_argmin[novel_idx]             # [N] bank entries to evict
        
        # Deduplicate: if multiple novel samples target the same bank entry,
        # keep the most novel (largest d). Small loop on CPU.
        novel_idx_cpu = novel_idx.cpu()
        evict_idx_cpu = evict_idx.cpu()
        d_novel_cpu = d[novel_idx].cpu()
        
        claimed = {}  # bank_idx -> (batch_idx_in_novel, d_value)
        for i in range(N):
            b_idx = evict_idx_cpu[i].item()
            d_val = d_novel_cpu[i].item()
            if b_idx not in claimed or d_val > claimed[b_idx][1]:
                claimed[b_idx] = (novel_idx_cpu[i].item(), d_val)
        
        # Batched GPU write
        if claimed:
            bank_indices = list(claimed.keys())
            batch_indices = [claimed[k][0] for k in bank_indices]
            self.bank[bank_indices] = s_batch[batch_indices]
        
        return {
            'ready': True,
            'd': d,
            'bank_nn_dists': bank_nn_dists,
            'mu': mu,
            'sigma': sigma,
        }