"""
Learned representative prototypes for morphology signature computation.
Compresses the K=65536 DINO prototype space into K' representative directions.
Trained with its own optimizer, fully detached from DINO gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentativePrototypes(nn.Module):
    """
    Learnable representative prototype matrix R ∈ R^{K' x D}.
    Rows are constrained to the unit sphere.
    
    Args:
        K_prime: Number of representative prototypes
        bottleneck_dim: Dimension of DINO head bottleneck (default 256)
    """
    def __init__(self, K_prime, bottleneck_dim=256):
        super().__init__()
        self.K_prime = K_prime
        self.bottleneck_dim = bottleneck_dim
        
        self.R = nn.Parameter(torch.empty(K_prime, bottleneck_dim))
        nn.init.xavier_normal_(self.R)
        with torch.no_grad():
            self.R.data = F.normalize(self.R.data, p=2, dim=1)
    
    def project_to_sphere(self):
        """Re-project rows onto the unit sphere. Call after each optimizer step."""
        with torch.no_grad():
            self.R.data = F.normalize(self.R.data, p=2, dim=1)
    
    def compute_signatures(self, z_x):
        """
        Compute morphology signatures.
        
        Args:
            z_x: [B, D] L2-normalized bottleneck embeddings (detached by caller)
            
        Returns:
            [B, K'] morphology signatures (cosine similarities to representatives)
        """
        R_hat = F.normalize(self.R, p=2, dim=1)
        return z_x @ R_hat.t()
    
    def compute_loss(self, P):
        """
        Compute L_repr = L_nn + L_cov.
        
        L_nn: each representative should be close to at least one actual prototype.
        L_cov: penalize off-diagonal entries of the Gram matrix (VICReg/Barlow Twins style).
        
        Args:
            P: [K, D] prototype weight matrix from student classhead (detached by caller)
            
        Returns:
            loss: scalar
        """
        R_hat = F.normalize(self.R, p=2, dim=1)          # [K', D]
        P_hat = F.normalize(P.detach(), p=2, dim=1)       # [K, D]
        
        # Nearest-neighbour loss: Eq. 4 in technical note
        sim = R_hat @ P_hat.t()                            # [K', K]
        L_nn = -(1.0 / self.K_prime) * sim.max(dim=1).values.sum()
        
        # Covariance loss: Eq. 5 in technical note
        gram = R_hat @ R_hat.t()                           # [K', K']
        mask = ~torch.eye(self.K_prime, dtype=torch.bool, device=gram.device)
        L_cov = gram[mask].pow(2).sum() / (self.K_prime * (self.K_prime - 1))
        
        return L_nn + L_cov, L_nn, L_cov