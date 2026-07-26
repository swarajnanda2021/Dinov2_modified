"""
Typicality scorer.
Turns the counted-coverage bank's fixed-radius density p_hat into the per-tile
weighted-loss weight  w(x) = 1 / (p_hat(x) + c)^a,  c = c_frac * p_ref.
"""

import torch


class TypicalityScorer:
    """
    Stateless scorer for the fixed-radius readout: maps the bank's absolute local density
    p_hat to the per-tile DINO loss weight (the weighted-loss modulation of section 3.6).
    """

    @staticmethod
    def absolute_weights(p_hat, p_ref, a, c_frac):
        """
        Weight from the absolute local density.

        w(x) = 1 / (p_hat(x) + c)^a ,   c = c_frac * p_ref

        p_hat is the fixed-radius kernel-sum density from CountedCoverageBank (an absolute
        rate density, not a rank). c anchors the denominator to the running scale of p_hat
        (p_ref, an EMA of the batch median), which is what makes the weight scale-invariant:
        multiply every p_hat and p_ref by the same constant and the normalised weighted loss is
        unchanged. p_hat = 0 (no established signature within the radius) gives a finite weight
        1 / c^a, so no clipping is required.

        Args:
            p_hat: [B] absolute local density (>= 0)
            p_ref: scalar reference density (bank.p_ref)
            a:     tilt exponent (larger -> more emphasis on rare tiles)
            c_frac: reference fraction setting c = c_frac * p_ref

        Returns:
            w: [B] per-sample loss weights (> 0), averaging order 1-3 (fed to the
               weight-normalised DINO loss, which depends only on weight ratios)
        """
        c = c_frac * p_ref
        return 1.0 / (p_hat + c).pow(a)
