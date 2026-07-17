# Typicality Dampening

*Method section (manuscript draft). This document describes the typicality-dampening module
in full: the morphology-signature representation it operates on, the online estimator of
tile redundancy at its core, the empirical study of the training stream that fixes the
estimator's design, the resulting curation policy, and the two ways its output modulates the
learning objective. Quantitative statements are inference-only measurements on our
ViT-B/16 pathology model; the experimental protocol for each is given in Appendix A.*

---

## 3. Typicality Dampening

Whole-slide images are annotated at the slide level but consumed at the tile level, and the
resulting training stream is heavily redundant. A single slide contributes thousands of
near-duplicate views of stroma, adipose, and background, while diagnostically informative
morphology is comparatively rare. Under the shared softmax temperature of the DINO objective,
these common tiles contribute per-example gradients of the same magnitude as rare ones, so
the informative signal is simultaneously diluted by, and redundantly reinforced against, a
long tail of near-duplicates. This is the tile-level analogue of a problem that modern
self-supervised recipes otherwise solve *offline*: the DINOv2 corpus, for instance, was built
by an explicit de-duplication and retrieval-rebalancing pipeline before training began
(Oquab et al., 2024). Tile streams admit no such pre-curation at scale.

Typicality dampening supplies this curation *online and in-domain*. At each step it estimates,
for every tile in the batch, how typical that tile is of the morphology the model has recently
encountered, and attenuates the contribution of typical tiles to the image-level DINO
objective. The module is a stop-gradient side-channel on the standard student/teacher pipeline
(Figure 1): it reads an intermediate student representation, produces a scalar typicality
score, and multiplies the DINO cross-entropy by a weight that decreases with typicality. The
patch-level iBOT objective is left unmodulated, so dense per-patch supervision is preserved
while the image-level objective is rebalanced toward rare morphology.

### 3.1 Overview

The module comprises three components, applied in sequence to each tile `x` in the batch.

1. **Signature.** The student's L2-normalized 256-dimensional DINO-head bottleneck
   `z(x)` is taken under stop-gradient and projected onto a small set of learned
   *representative prototypes* `R` to yield a compact **morphology signature** `s(x)`
   (§3.2).
2. **Redundancy estimate.** `s(x)` is compared against a bounded **memory bank** of recently
   observed signatures to produce a scalar **typicality score** `t(x) ∈ [0,1]`, high when the
   tile lies in a densely populated region of signature space and low when it is rare (§3.3).
3. **Modulation.** `t(x)` is mapped to a per-tile modulation of the DINO objective — either a
   loss weight or a softmax temperature (§3.6).

Of these, the signature and the modulation are lightweight and largely determined by the
surrounding recipe; the memory bank is the technical crux, because whether `t(x)` faithfully
measures redundancy depends entirely on what the bank stores and how it is maintained. The
bulk of this section (§3.3–§3.5) is therefore devoted to the estimator: the requirement it
must satisfy, an empirical characterization of the tile stream that determines its parameters,
and the curation policy we derive.

### 3.2 Morphology signatures

The redundancy estimate is computed in a low-dimensional morphology space rather than on raw
backbone features, for two reasons: the raw `[CLS]` representation is high-dimensional and
continues to reorganize late into training (§3.4), whereas the DINO-head bottleneck is both
compact and stable; and comparing tiles against a learned set of anchors gives an
interpretable, bounded representation well suited to a nearest-neighbor memory.

Let `z(x) ∈ S^{255}` be the student's unit-norm DINO-head bottleneck for the first global crop,
taken under stop-gradient so the typicality path never contributes to the backbone gradient.
A matrix of `K' = 256` **representative prototypes** `R ∈ ℝ^{K'×256}`, with unit-norm rows,
defines the signature

```
s(x) = R · z(x) ∈ ℝ^{K'} ,        s_k(x) = ⟨R_k, z(x)⟩ ,
```

the vector of similarities between the tile and each anchor. Signatures are *peaky* for
well-formed morphology (one anchor dominates) and *diffuse* for representations that resemble
no anchor (all similarities near the small-angle floor of a random projection); this
distinction is used diagnostically in §3.4.

The anchors are trained online, alongside the backbone but by a dedicated optimizer, to tile
the occupied region of bottleneck space. Two terms are used:

```
L_R = L_nn + λ_cov · L_cov ,
```

where `L_nn` pulls each anchor toward its nearest DINO output prototype (so anchors track the
morphology directions the model itself has learned), and `L_cov` penalizes the off-diagonal
entries of `R Rᵀ` (so anchors spread out rather than collapse together). The rows of `R` are
re-projected to the unit sphere after each step, keeping `R` close to an orthonormal frame.
Because the anchors and the loss both derive from the stop-gradient bottleneck, the signature
space is a passive readout of the representation, not a target the backbone optimizes toward.

### 3.3 Tile redundancy as online density estimation

Given signatures, estimating a tile's redundancy is a density-estimation problem. Let `μ` be
the distribution of signatures induced by the current model over the tile stream, with local
density `p`. A tile is redundant precisely when it lies where `p` is large — where many other
tiles produce nearly the same signature — so the typicality score should be a monotone
increasing function of `p(s(x))`. We estimate `p` with a bounded memory: a bank `B` of at most
`M` signatures, from which each incoming tile receives a score.

The shipped estimator is a nearest-neighbor density readout. For a query signature `s(x)`, let
`d(x) = min_{b∈B} ‖s(x) − b‖₁` be the distance to the nearest bank entry, and let `μ_B, σ_B`
be the mean and standard deviation of the within-bank nearest-neighbor distances. The score is

```
t(x) = 1 − Φ( (d(x) − μ_B) / σ_B ) ,
```

with `Φ` the standard normal CDF, so a tile that lands close to the bank (small `d`) is scored
typical (`t → 1`) and a tile far from every bank entry is scored rare (`t → 0`). The
calibration by the bank's own spacing makes the score scale-free.

The estimator is only as good as the bank's contents, and this is where the design becomes
non-trivial. For `d(x)` to track *frequency*, the bank's stationary distribution must reflect
the data distribution `μ`: nearest-neighbor distance grows as local density falls only if the
bank samples dense and sparse regions in proportion to how often the stream visits them. Two
natural maintenance policies both fail this requirement in opposite ways. A reservoir that
admits every tile and evicts uniformly reproduces `μ` in distribution, but with `M ≪ N` it
draws almost all of its entries from the dense majority and represents sparse morphology only
noisily. A novelty-driven policy that admits the most dissimilar tiles and evicts the nearest
incumbent — the policy of our initial implementation — instead relaxes the bank toward a
space-filling cover of the occupied region: nearest-neighbor distance becomes roughly constant
everywhere and carries little information about density, and isolated entries, never the
nearest neighbor of any future tile, are never evicted. On our own model this second failure
mode is severe: the bank saturates at a bimodal state in which the majority of entries are
diffuse seed signatures left over from early training and the score is nearly constant across
the stream (§3.4, §3.7). The remainder of this section derives a maintenance policy that
estimates density faithfully within a fixed memory budget, and the parameters that policy
requires are fixed by a direct measurement of the stream.

### 3.4 Characterizing the tile stream

The estimator's design — when to activate it, how to lay out the bank, how fast to forget —
is determined by six properties of the tile stream, all measured directly on our model and
summarized in Table 1. Two findings are structural (they change the algorithm) and the rest
are quantitative (they fix its constants).

**When to activate: representation stability.** The signature space is a projection of the
DINO-head bottleneck, and the bottleneck stabilizes far earlier in training than the raw
backbone. Measuring linear centered kernel alignment (CKA; Kornblith et al., 2019) between a
fixed probe set encoded at successive checkpoints and at the final model, the raw `[CLS]`
representation is only 55% converged at 50k iterations, whereas the bottleneck is 94%
converged by the same point. A control that applies the *final* projection head to every
checkpoint's `[CLS]` reaches 98% by 50k — higher than with the co-evolving head — which
establishes that the early stability is a genuine property of a low-dimensional, early-forming
subspace of the backbone rather than an artifact of the head adapting to a drifting
representation. Because the bank is only meaningful once the signatures it stores are stable,
the module is inactive for the first `T_warm ≈ 50k` iterations and the bank is filled only
thereafter. (Local neighborhood structure continues to reorganize until roughly 90k, a
residual we return to in §3.8.)

**Geometry: low dimension, mild skew, no isolated modes.** The intrinsic dimension of the
signature support, estimated both by the two-nearest-neighbor ratio method (Facco et al., 2017)
and by the participation ratio of the feature covariance, is approximately 9.5 — far below the
ambient 256. The dynamic range of local density across the support, estimated from k-nearest-
neighbor distances, spans roughly three orders of magnitude (a factor of `10^3` between the
1st and 99th percentiles), a figure independently corroborated by the variance of the log-
density field. The support is a connected continuum rather than a set of isolated clusters:
the effective number of resolvable modes grows smoothly with resolution and no dominant mode
emerges. Together these mean the estimation problem is benign — density is readable, the skew
is moderate, and the absorbing-outlier pathology of a novelty-driven bank has no atomic modes
to latch onto (the isolation rate decays with sample size with no floor, extrapolating to a
one-off rate near `10^{-5}`).

**Temporal structure: near-i.i.d., slow drift.** At the batch level the stream is close to
i.i.d.: the variance ratio between batches is near unity and the lag-one autocorrelation of
batch statistics is small (a positive control that groups tiles by slide raises both sharply,
confirming the measurement is sensitive). At the tile level the loader's interleave leaves only
a weak, short-range correlation — a burst factor near 1.2 that decays within a single batch —
so redundant tiles do not arrive in long runs. Finally, the signature distribution itself
drifts slowly after activation: the representation moves by a correlation length only over tens
of thousands of iterations, two to three orders of magnitude slower than the bank's forgetting
timescale, so stored statistics never become stale relative to the encoder that produced them.

**Table 1.** Measured properties of the signature stream (ViT-B/16, post-activation).
| Property | Symbol | Value | Design role |
|---|---|---|---|
| Bottleneck stabilization | — | 94% CKA at 50k (vs. 55% for `[CLS]`) | sets activation `T_warm ≈ 50k` |
| Intrinsic dimension | `d*` | ≈ 9.5 | sets readout viability and cell scale |
| Local-density dynamic range | `R` | ≈ `10^3` | places the operating regime; moderate |
| Mode structure | — | connected continuum, no dominant mode | removes the coverage/outlier hazard |
| Tile-level burst factor | `b` | ≈ 1.2, decays within one batch | permits simple exponential forgetting |
| Log-density correlation length | `L` | ≈ 0.16 | sets the memory-bound cell scale |
| One-off (artifact) rate | `ρ_out` | ≈ `10^{-5}`, no floor | sizes the transient reserve |

### 3.5 Curation policy

The characterization in §3.4 dictates a specific bank design. Because a single set of stored
points cannot encode both *where* the occupied region is and *how densely* each part is
populated — the failure at the root of §3.3 — we separate the two roles. The bank stores a set
of anchor signatures that cover the support, and augments each anchor with two decayed scalar
counters that record how much traffic it receives. Coverage is carried by the anchor
positions; frequency is carried by the counters.

**Anchor state.** Each anchor `i` maintains a position `b_i` and two exponentially decayed
counters: a *hit count* `S_i`, incremented when a tile falls within the cell of `b_i`, and an
*exposure* `E_i`, incremented whenever `b_i` is the nearest anchor to an incoming tile. Their
ratio `λ̂_i = S_i / E_i` estimates the local traffic rate at `b_i`. This pair is the sufficient
statistic for the redundancy estimate under a Poisson-arrival model, and it dominates the two
degenerate alternatives: a recency timestamp alone (which supports only a coarse "seen
recently" test and discards magnitude) and an undecayed cumulative count (which retains anchors
in regions the stream has long since left).

**Placement.** New anchors are admitted where the stream visits regions not yet covered — a
tile farther than the cell scale `s` from every anchor seeds a new anchor — so the anchor set
tiles the occupied support. On the measured continuum the optimal density-estimation bandwidth
places anchors in near-proportion to the data density (formally, anchor density `∝ p^{γ}` with
`γ = d*/(d*+2) ≈ 0.83`, the classical quantization exponent; Graf and Luschgy, 2000), rather
than as the uniform cover of a novelty-driven policy. The cell scale is set by the memory
budget: with `M = 8192` anchors over a `d* ≈ 9.5`-dimensional support the achievable scale
`s ≈ 0.15` coincides with the density correlation length `L`, which — as noted in §3.8 — is
the resolution floor of any bounded summary in this regime.

**Readout.** The typicality score is read from the counters rather than from distance:
`t(x)` is a monotone map of a kernel-weighted, query-centered average of `λ̂_i` over the
`j ≈ 32–64` nearest anchors. Centering the average at the query cancels the leading-order bias
that a single-anchor readout would incur. Early in training, before the counters have
accumulated, the distance readout of §3.3 serves as a cold-start estimator; once the counters
fill — within a few dozen batches — the counted readout takes over, and the distance readout
is retained only as a consistency probe whose disagreement with the counted estimate localizes
drift or miscalibration.

**Eviction and forgetting.** When the anchor set is full, the anchor with the lowest traffic
rate `λ̂_i` is evicted — a least-frequently-used policy with aging, so anchors that stop
receiving traffic decay out while active anchors persist. A small transient reserve (of order
1% of `M`, sized by the measured one-off rate `ρ_out`) holds newly admitted anchors so that a
one-off tile cannot displace an established anchor before it has had the chance to accumulate
traffic. The counters decay with a half-life of a few hundred batches — long relative to the
batch autocorrelation, so recurring morphology accumulates standing evidence, yet short
relative to the representation drift horizon, so the estimate tracks the current distribution.

Algorithm 1 states the policy. It is applied once per batch, after the activation iteration,
to the batch of signatures gathered across data-parallel workers (§3.7).

```
Algorithm 1  Typicality bank update and scoring (one batch of signatures X)

  if iteration < T_warm:                       # signatures not yet stable (§3.4)
      return t(x) = 0 for all x                 #   module inactive; bank untouched

  decay all counters:  S_i ← η · S_i ,  E_i ← η · E_i      # η set by half-life

  for each tile x in X:                         # score before updating
      N ← the j anchors nearest to s(x)
      if counters are still filling:
          t(x) ← monotone( distance readout of §3.3 )       # cold-start
      else:
          λ̂(x) ← centered kernel average of {S_i/E_i : i ∈ N}
          t(x) ← monotone( λ̂(x) )

  for each tile x in X:                         # update after scoring
      i* ← nearest anchor to s(x) ;  E_{i*} ← E_{i*} + 1
      if ‖s(x) − b_{i*}‖ ≤ s:
          S_{i*} ← S_{i*} + 1                   # a hit
      else:
          admit a new anchor at s(x) into the transient reserve
          if the anchor set is full:
              evict the anchor of lowest rate S_i/E_i outside the reserve

  return { t(x) : x ∈ X }
```

### 3.6 Modulating the objective

The typicality score modulates the image-level DINO cross-entropy in one of two ways, which
differ in what they change about the learning signal. Both leave the iBOT objective untouched;
the total loss is `L = m(x) · CE_DINO + CE_iBOT + λ_sem · CE_iBOT^{sem}`, where the modulation
`m(x)` is one of the following.

**Weighted loss (magnitude).** The DINO term is scaled per tile by
`w(x) = 1 − β · t(x)`, `β ∈ [0,1]`. A typical tile contributes a smaller-magnitude gradient,
but the target it is trained toward is unchanged. This is a direct importance weighting: it
reshapes the *effective sampling distribution* toward a flatter distribution over morphology
while leaving each tile's learning signal intact. It is bounded and simple to reason about,
and is the setting we adopt by default.

**Adaptive temperature (distribution).** The per-tile student softmax temperature is scaled,
`τ(x) = τ_base · (1 + α · t(x))`, so a typical tile receives a flatter target distribution. This
changes not only the gradient magnitude but the *shape* of the target, redistributing
probability mass across output prototypes rather than merely down-scaling the tile's
contribution. It is a stronger intervention — it can actively flatten over-represented modes in
the assignment itself — but it couples the redundancy estimate into the representation geometry
and is correspondingly harder to bound. We report it as an alternative and use the weighted-loss
form for our main results.

### 3.7 Implementation and design progression

**Synchronization.** The bank is a single global structure, maintained on the union of
signatures gathered across all data-parallel workers, so that all workers share one estimator
rather than each maintaining a private, `1/W`-resolution bank. All update decisions are made
deterministically (ties broken by index) so the global bank remains bit-identical across
workers and reproducible across nodes; a periodic cross-worker checksum guards this invariant.

**Activation and monitoring.** Consistent with §3.4, the entire module — signature
computation, bank update, and scoring — is gated off until `T_warm ≈ 50k` iterations, so the
bank is never seeded from an unstable representation. Two health statistics are logged
throughout training: the fraction of the bank that is diffuse, and the fraction of tiles scored
extremely rare; a healthy run keeps both away from their degenerate limits.

**Design progression.** The policy of §3.5 is the endpoint of two revisions, which together
form an ablation of the estimator. The initial implementation maintained a private per-worker
bank, filled it from the first iteration, admitted the most novel tiles, and evicted the nearest
incumbent. Filling before the representation stabilized seeded the bank with diffuse signatures,
and the evict-nearest rule — unable to remove isolated entries — never cleared them: measured at
the end of training, the bank was 78% diffuse seed material, and the resulting score was nearly
constant across the stream, so the modulation was effectively inert. An intermediate revision
corrected the two implementation faults — it made the bank global and deterministic and deferred
filling to `T_warm` — but retained the novelty-admission and distance readout, and therefore the
structural flatness of a coverage-based estimator. The policy of §3.5 replaces the churn rule
with least-frequently-used eviction and the readout with the counted rate, closing the gap
between the estimated and the true redundancy. §3.4 shows that on the measured stream the
distance readout is in fact usable once the bank is faithfully maintained, so the two readouts
agree after the counters fill; their disagreement before then is what the consistency probe
monitors.

### 3.8 Limitations

Three limitations bound the method. First, the estimate is resolution-limited: with a memory
budget of `M` anchors over a `d* ≈ 9.5`-dimensional support, the smallest resolvable cell scale
coincides with the density correlation length `L`, so structure finer than `L` is invisible to
*any* bounded summary of this size — refining it would require exponentially more memory
(`∝ 3^{d*}` per halving) or a lower-dimensional signature. Second, the empirical constants of
§3.4 were measured on a single, morphologically homogeneous slice of the stream; a
substantially more skewed corpus could shift the operating regime, although the counted readout
is by construction insensitive to the exact anchor-placement exponent. Third, the redundancy
estimate modulates the objective that trains the encoder that produces the signatures, so the
distribution the module measures is not exogenous. Deferring activation until the representation
has stabilized (§3.4) is the safeguard we rely on; a formal analysis of the coupled dynamics is
left to future work, and the adaptive-temperature modulation (§3.6), which feeds back through
the target distribution, tightens this coupling relative to the weighted-loss form.

---

### References

Arthur and Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA 2007.
Bifet and Gavaldà. *Learning from Time-Changing Data with Adaptive Windowing.* SDM 2007.
Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.
Che, Tung, and Wang. *Hierarchical Web Caching Systems.* IEEE JSAC 2002.
Facco et al. *Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information.* Scientific Reports 2017.
Graf and Luschgy. *Foundations of Quantization for Probability Distributions.* Springer LNM 1730, 2000.
Kornblith et al. *Similarity of Neural Network Representations Revisited.* ICML 2019.
Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer.* ICLR 2022.
