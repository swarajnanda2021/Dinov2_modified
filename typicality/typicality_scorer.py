"""
Typicality scorer with Gaussian calibration.
Converts raw L1 distances into redundancy scores t(x) ∈ [0, 1],
then produces per-sample modulation (temperature or loss weight).
"""

import math
import torch


class TypicalityScorer:
    """
    Stateless scorer. Converts bank output into redundancy scores and modulation values.
    
    Scoring formula:
        z = (d(x) - mu_bank) / (sigma_bank + eps)
        t(x) = 1 - Phi(z)
    
    where Phi is the standard normal CDF.
    
    t ≈ 1: sample is deep inside well-represented territory (dampen)
    t ≈ 0: sample is far from the bank (preserve full gradient)
    """
    
    INV_SQRT2 = 1.0 / math.sqrt(2.0)
    
    @staticmethod
    def compute_scores(d, mu, sigma, eps=1e-8):
        """
        Compute redundancy scores from raw L1 distances.
        
        Args:
            d: [B] min L1 distance from each sample to nearest bank entry
            mu: scalar mean of within-bank NN distances
            sigma: scalar std of within-bank NN distances
            eps: numerical stability
            
        Returns:
            t: [B] redundancy scores in [0, 1]
        """
        z = (d - mu) / (sigma + eps)
        # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        phi = 0.5 * (1.0 + torch.erf(z * TypicalityScorer.INV_SQRT2))
        t = 1.0 - phi
        return t
    
    @staticmethod
    def adaptive_temperature(t, tau_base, alpha):
        """
        Variant I: Per-sample student temperature.
        
        tau_s(x) = tau_base * (1 + alpha * t(x))
        
        Args:
            t: [B] redundancy scores
            tau_base: scalar baseline student temperature (e.g. 0.1)
            alpha: dampening strength (alpha > 0)
            
        Returns:
            tau_s: [B] per-sample student temperatures
        """
        return tau_base * (1.0 + alpha * t)
    
    @staticmethod
    def sample_weights(t, beta):
        """
        Variant II: Per-sample loss weights.
        
        w(x) = 1 - beta * t(x)
        
        Args:
            t: [B] redundancy scores
            beta: dampening strength (0 <= beta < 1)
            
        Returns:
            w: [B] per-sample loss weights
        """
        return 1.0 - beta * t