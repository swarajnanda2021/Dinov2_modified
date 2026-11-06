"""
v2 keep rule for stream thinning (thinned mode, gate open). Pure functions. No random numbers.
No global state. The same input gives the same output on CPU and on GPU.

Symbols:
    P      1-D tensor [n] of cache densities p_hat, all >= 0. P is the gathered pool. It is
           identical on every rank.
    a      tilt exponent (--typicality_a), a > 0.
    u0     share of P kept for sure (--keep_u0), 0 < u0 < 1.
    c      keep threshold of the step. c is the k-th smallest value of P with
           k = max(1, ceil(u0 * n)). So at least a share u0 of P satisfies p_hat <= c.
    q      keep chance of a tile. q = 1 where p_hat <= c. q = (c / p_hat)^a where p_hat > c.
    m      mean of q over P. m >= u0 always, because at least k rows have q = 1.
    f      pool size in units of N. f_next = min(max(margin / m_ema, 1), cap), with
           cap = 1/u0, or min(1/u0, f_max) when f_max > 0 (--keep_f_max).
"""
import math

import torch


def keep_chance_v2(P, a, u0):
    """Compute the v2 keep chance q for one step.

    Args:
        P: tensor [n], n >= 1, values >= 0, any device. A non-float input is cast to float32.
        a: tilt exponent, float > 0.
        u0: sure share, float in (0, 1).

    Returns:
        q: float tensor [n] on the device of P, 0 < q <= 1.
        c: float. The keep threshold used for q.
        c_guard: bool. True when the zero guard replaced the quantile.
        sure_share: float. Share of P with q == 1, that is p_hat <= c.
        zero_share: float. Share of P with p_hat == 0.
        m: float. Mean of q over P.

    Quantile: c is the k-th smallest value of P, k = max(1, ceil(u0 * n)) (torch.kthvalue).
    Zero guard: if that value is 0, c becomes the smallest positive value in P. If P has no
    positive value, q = 1 everywhere, c = 0 and c_guard = True.
    """
    P = P.detach().reshape(-1)
    if not torch.is_floating_point(P):
        P = P.float()
    n = int(P.numel())
    assert n >= 1, "P must hold at least one value"
    assert 0.0 < float(u0) < 1.0, "u0 must be in (0, 1), got %r" % (u0,)
    assert float(a) > 0.0, "a must be > 0, got %r" % (a,)
    zero_share = float((P == 0).float().mean().item())
    k = max(1, int(math.ceil(float(u0) * n)))
    c = float(torch.kthvalue(P, k).values.item())
    c_guard = False
    if c <= 0.0:
        c_guard = True
        pos = P[P > 0]
        if pos.numel() == 0:
            return torch.ones_like(P), 0.0, True, 1.0, zero_share, 1.0
        c = float(pos.min().item())
    # ratio = max(p_hat, c) / c is exactly 1.0 where p_hat <= c (tensor / scalar divides exactly;
    # scalar / tensor would go through reciprocal() * scalar and miss 1.0 by one ulp) and
    # strictly above 1.0 where p_hat > c. q = ratio^(-a).
    q = (P.clamp(min=c) / c).pow(-float(a))
    sure_share = float((P <= c).float().mean().item())
    m = float(q.mean().item())
    return q, c, c_guard, sure_share, zero_share, m


def f_cap_v2(u0, f_max=0.0):
    """The upper bound on f. 1/u0, or min(1/u0, f_max) when f_max > 0 (--keep_f_max)."""
    cap = 1.0 / float(u0)
    if float(f_max) > 0.0:
        cap = min(cap, float(f_max))
    return cap


def f_update_v2(m, m_ema, decay, margin, u0, f_max=0.0):
    """Update the EMA of m and set the pool size of the next step.

    Args:
        m: float. Mean keep chance of this step.
        m_ema: float or None. None at the first v2 step.
        decay: float in [0, 1). EMA decay (--keep_m_ema).
        margin: float > 0 (--keep_f_margin).
        u0: float in (0, 1) (--keep_u0).
        f_max: float. Extra cap on f (--keep_f_max). A value <= 0 means no extra cap.

    Returns:
        m_ema_new: float. m when m_ema is None. decay * m_ema + (1 - decay) * m otherwise.
        f_raw: float. max(margin / m_ema_new, 1).
        f_next: float. min(f_raw, cap), cap = f_cap_v2(u0, f_max).
        n_draws_next: int. ceil(f_next), at least 1.
        capped: bool. True when f_raw > cap.

    Pure Python float arithmetic. Ranks that get the same inputs get the same outputs.
    """
    m = float(m)
    if m_ema is None:
        m_ema_new = m
    else:
        m_ema_new = float(decay) * float(m_ema) + (1.0 - float(decay)) * m
    cap = f_cap_v2(u0, f_max)
    f_raw = max(float(margin) / max(m_ema_new, 1e-12), 1.0)
    f_next = min(f_raw, cap)
    capped = bool(f_raw > cap)
    n_draws_next = max(1, int(math.ceil(f_next)))
    return m_ema_new, f_raw, f_next, n_draws_next, capped
