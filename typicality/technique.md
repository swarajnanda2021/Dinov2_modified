# Typicality Dampening

*Method section (manuscript draft). This document describes the typicality-dampening module:
the morphology-signature representation it operates on, the memory bank at its core as
currently implemented, a structural limitation of that bank, a proposed alternative that
addresses the limitation, and the two ways the module's output modulates the learning
objective. The two bank designs and the two modulations define the four configurations
evaluated in Section 4. Quantitative statements are inference-only measurements on a baseline
ViT-B/16 DINOv2 model with all extensions of this work disabled; the empirical basis is
described in §3.3 and full protocols are given in Appendix A.*

---

## 3. Typicality Dampening

Whole-slide images are annotated at the slide level but consumed at the tile level, and the
resulting training stream is heavily redundant. A single slide contributes thousands of
near-duplicate views of stroma, adipose, and background, while diagnostically informative
morphology is comparatively rare. Under the shared softmax temperature of the DINO objective,
these common tiles contribute per-example gradients of the same magnitude as rare ones, so the
informative signal is simultaneously diluted by, and redundantly reinforced against, a long
tail of near-duplicates. This is the tile-level analogue of a problem that modern
self-supervised recipes otherwise address *offline*: the DINOv2 corpus, for example, was built
by an explicit de-duplication and retrieval-rebalancing pipeline before training began
(Oquab et al., 2024). Tile streams admit no such pre-curation at scale.

Typicality dampening supplies this curation *online and in-domain*. At each step it estimates,
for every tile in the batch, how typical that tile is of the morphology the model has recently
encountered, and attenuates the contribution of typical tiles to the image-level DINO
objective. The module is a stop-gradient side-channel on the standard student/teacher pipeline
(Figure 1): it reads an intermediate student representation, produces a scalar typicality
score, and multiplies the DINO cross-entropy by a factor that decreases with typicality. The
patch-level iBOT objective is left unmodulated, so dense per-patch supervision is preserved
while the image-level objective is rebalanced toward rare morphology.

### 3.1 Overview

The module has three stages, applied in sequence to each tile `x` in the batch.

1. **Signature.** The student's L2-normalized 256-dimensional DINO-head bottleneck `z(x)` is
   taken under stop-gradient and projected onto a small set of learned *representative
   prototypes* `R` to yield a compact morphology signature `s(x)` (§3.2).
2. **Typicality score.** `s(x)` is compared against a bounded memory bank of recently observed
   signatures to produce a scalar typicality score `t(x) ∈ [0,1]`, high when the tile lies in a
   densely populated region of signature space and low when it is rare (§3.3).
3. **Modulation.** `t(x)` modulates the DINO objective for that tile, either as a loss weight
   or as a softmax temperature (§3.6).

The signature (§3.2) and the modulation (§3.6) are shared across all configurations of the
method. The memory bank is the component that varies: §3.3 describes the bank as currently
implemented, §3.4 identifies a structural limitation of it, and §3.5 describes a proposed
alternative. The two bank designs, crossed with the two modulations, give the four
configurations studied in Section 4.

### 3.2 Morphology signatures

The typicality score is computed in a compact morphology space rather than on raw backbone
features. This choice is deliberate: the raw `[CLS]` representation is high-dimensional and
continues to reorganize late into training, whereas the DINO-head bottleneck is both compact
and stabilizes early (§3.3), which is precisely the property a nearest-neighbor memory requires.

Let `z(x) ∈ S^{255}` be the student's unit-norm DINO-head bottleneck for the first global crop,
taken under stop-gradient so the typicality path never contributes to the backbone gradient. A
matrix of `K' = 256` representative prototypes `R ∈ ℝ^{K'×256}`, with unit-norm rows, defines
the signature

```
s(x) = R · z(x) ∈ ℝ^{K'} ,        s_k(x) = ⟨R_k, z(x)⟩ ,
```

the vector of similarities between the tile and each anchor. Signatures are *peaky* for
well-formed morphology (one anchor dominates) and *diffuse* for representations that resemble
no anchor (all similarities near the small-angle floor of a random projection), a distinction
used diagnostically below.

The anchors are trained online, alongside the backbone but by a dedicated optimizer, to tile
the occupied region of bottleneck space, using

```
L_R = L_nn + λ_cov · L_cov ,
```

where `L_nn` pulls each anchor toward its nearest DINO output prototype, so anchors track the
morphology directions the model itself has learned, and `L_cov` penalizes the off-diagonal
entries of `R Rᵀ`, so anchors spread out rather than collapse. The rows of `R` are re-projected
to the unit sphere after each step, keeping `R` close to an orthonormal frame. Because both the
anchors and their loss derive from the stop-gradient bottleneck, the signature space is a
passive readout of the representation, not a target the backbone optimizes toward.

### 3.3 The typicality bank

The score measures how densely populated a tile's neighborhood is in signature space. Let the
tile stream have length `N` — the total number of tiles seen over training, on the order of
`10^8` — and let the bank `B` hold at most `M` signatures, `M ≪ N` (in our configuration
`M = 8192`). For each incoming tile the bank yields a nearest-neighbor distance, which the
scorer converts to a typicality value.

**Scoring.** For a query signature `s(x)`, let `d(x) = min_{b ∈ B} ‖s(x) − b‖₁` be the
distance to the nearest bank entry, and let `μ_B, σ_B` be the mean and standard deviation of the
within-bank nearest-neighbor distances. The score is

```
t(x) = 1 − Φ( (d(x) − μ_B) / σ_B ) ,
```

with `Φ` the standard normal CDF: a tile that lands close to the bank is scored typical
(`t → 1`), a tile far from every entry is scored rare (`t → 0`), and calibrating by the bank's
own spacing makes the score scale-free.

**Maintenance.** Empty slots are filled first; once the bank is full, each step admits the
tiles of the current batch whose distance `d(x)` is largest — the most novel — and, for each
admission, evicts the bank entry nearest to it. This novelty-admission, evict-nearest rule keeps
the bank spread across the occupied region of signature space.

**Empirical basis.** The properties of the signature stream that inform the module's design —
the stabilization point here, and the geometry and temporal structure of Table 1 below — were
measured on a *baseline* model: a standard DINOv2 ViT-B/16 trained on pathology tiles with all
four extensions of this work disabled, under a recipe following UNI (Chen et al., 2024) at
ViT-B rather than its ViT-L scale. Measuring on this un-modulated baseline is deliberate: it
characterizes the signature distribution the module takes as *input*, before the module itself
perturbs it, which both isolates the design target and avoids the closed-loop confound of §3.8.
All measurements are inference-only — a fixed probe of 76,800 tiles, drawn from the training
stream in dataloader order (so arrival order is preserved for the temporal measurements) and
encoded through a ladder of checkpoints spanning training (10k–124k iterations) — with no
retraining.

**Activation.** The bank is only meaningful once the signatures it stores are stable. The
signature space is a projection of the DINO-head bottleneck, and the bottleneck stabilizes far
earlier than the raw backbone: measuring linear centered kernel alignment (CKA; Kornblith et
al., 2019) between the probe encoded at successive checkpoints and at the final model,
the raw `[CLS]` representation is only 55% converged at 50k iterations whereas the bottleneck is
94% converged, and a control applying the final projection head to every checkpoint's `[CLS]`
reaches 98% by 50k — establishing that the early stability is a property of a low-dimensional,
early-forming subspace of the backbone rather than of the head adapting to a drifting
representation. Accordingly the entire module is inactive for the first `T_warm ≈ 50k`
iterations, and the bank is filled only thereafter.

**Implementation.** The bank is a single global structure maintained on the union of signatures
gathered across all data-parallel workers, so that all workers share one estimator rather than
each maintaining a private, reduced-resolution bank; all update decisions are deterministic
(ties broken by index) so the bank remains identical across workers and reproducible across
nodes. Two health statistics — the fraction of the bank that is diffuse and the fraction of
tiles scored extremely rare — are logged throughout training. Algorithm 1 states the procedure.

```
Algorithm 1  Typicality bank (implemented): update and scoring for one batch X

  if iteration < T_warm:                                # signatures not yet stable
      return t(x) = 0 for all x                          #   module inactive

  gather signatures of X across workers  →  X_global
  for each x in X_global:
      d(x)  ← min over b in B of ‖s(x) − b‖₁             # distance to nearest entry
      t(x)  ← 1 − Φ( (d(x) − μ_B) / σ_B )                # μ_B, σ_B: within-bank NN spacing

  A ← the tiles of X_global with the largest d(x)        # most novel
  for each x in A:
      j ← argmin over b in B of ‖s(x) − b‖₁              # nearest incumbent
      B[j] ← s(x)                                        # evict-nearest, admit x

  return { t(x) : x in this worker's rows }
```

### 3.4 A limitation: non-eviction of isolated signatures

The evict-nearest rule has a structural consequence. An entry that is far from all others is,
by definition, never the nearest neighbor of an incoming tile, and so is never selected for
eviction: isolated entries are absorbing. Over training the bank therefore drifts toward a
space-filling cover of the occupied region rather than a sample of it. This degrades the score
in the way that matters most: once the bank approximates a cover, the nearest-neighbor distance
`d(x)` is close to the cover's spacing almost everywhere on the support and no longer varies
with local density, so the typicality score flattens and ceases to distinguish common tiles
from rare ones — exactly the discrimination the module exists to provide.

Stated in estimator terms, the score of §3.3 is a nearest-neighbor density estimate, and such
an estimate is faithful only if the bank it queries is itself a representative sample of the
signature distribution. A representative sample is produced by any *content-independent*
maintenance rule — retaining the most recent `M` signatures, which is valid here because the
stream is near-independent at the tile scale (Table 1), or a uniform reservoir over those seen
— because such a rule evicts by age or at random rather than by position, leaving the surviving
set distributed as the data. Evict-nearest is content-dependent by construction: it evicts by
position, which is precisely what drives the bank away from a representative sample and toward
a cover. The limitation therefore admits two resolutions: restore faithful sampling with a
content-independent rule, under which the existing distance readout becomes a valid density
estimate; or estimate density from explicit counts rather than distance. The first is simpler
and, on the mild stream we measured (§3.5), likely adequate on its own; we nonetheless develop
the second (§3.5), because explicit counts retain coverage of rare morphology and remain valid
if the full corpus departs from the mild regime that faithful sampling relies on.

The effect is observable. In an earlier implementation that additionally filled the bank before
the representation had stabilized — seeding it with diffuse signatures from an untrained encoder
— the two effects compounded: the seed entries were mutually isolated, were never evicted, and
at the end of training occupied 78% of the bank, whose signatures formed a bimodal distribution
(a diffuse cluster at the random-projection floor and a smaller peaky remainder). Every real
tile then scored far from this bank, the typicality score was nearly constant across the stream,
and the modulation was effectively inert. Deferring the fill to `T_warm` (§3.3) removes the
seeding component of this failure, but not its cause: the non-eviction of isolated entries is a
property of the evict-nearest rule itself and persists for any admission schedule. The design in
§3.5 is motivated by removing it.

### 3.5 A proposed alternative: the counted-coverage bank

We describe a bank design, not yet implemented, that retains a covering set of anchors but
estimates density from explicit traffic counters rather than from nearest-neighbor distance,
thereby removing the non-eviction pathology of §3.4. Its parameters are fixed by a direct
characterization of the tile stream, summarized in Table 1. Each quantity was estimated on the
same baseline model and probe as above (§3.3), inference-only, by a standard estimator:
intrinsic dimension by the two-nearest-neighbor ratio method (Facco et al., 2017), cross-checked
against the covariance participation ratio; the local-density dynamic range from k-nearest-
neighbor density estimates; the mode structure from a clustering sweep; the burst factor from
the signature autocorrelation and the run-length of same-slide tiles in arrival order; the
correlation length from the semivariogram of the log-density field; and the one-off rate from
the isolation rate as a function of sample size.

**Table 1.** Properties of the signature stream, measured on the baseline model and probe of
§3.3 (standard DINOv2 ViT-B/16, all extensions disabled), and the design quantity each
determines.
| Property | Symbol | Value | Design role |
|---|---|---|---|
| Intrinsic dimension | `d*` | ≈ 9.5 | readout viability; cell scale |
| Local-density dynamic range | `R` | scale-dependent (§3.7); ≈ 20 at radius `L` | places the operating regime (moderate) |
| Mode structure | — | connected continuum, no dominant mode | removes the isolated-outlier hazard |
| Tile-level burst factor | `b` | ≈ 1.2, decays within one batch | permits simple exponential forgetting |
| Log-density correlation length | `L` | ≈ 0.26 | sets the readout bandwidth |
| One-off (artifact) rate | `ρ_out` | ≈ `10^{-5}`, no floor | sizes the transient reserve |

Two structural findings from Table 1 guide the design. First, the support is a connected
continuum of intrinsic dimension near ten with a moderate density range (scale-dependent, and
≈ 20× at the operating bandwidth `L`; §3.7) and no isolated modes; the absorbing-outlier hazard of §3.4 thus has no atomic
modes to attach to, and the design need not defend against extreme skew. Second, the stream is
near-independent at the tile level (a burst factor near 1.2 that decays within one batch) and
the signature distribution drifts slowly after activation — by one correlation length only over
tens of thousands of iterations — so stored statistics can be forgotten by simple exponential
decay without becoming stale relative to the encoder that produced them.

**Design.** Because a single set of stored points cannot encode both *where* the occupied region
lies and *how densely* each part is populated — the tension underlying §3.4 — the two roles are
separated. The bank stores anchor signatures that cover the support, and augments each anchor
with two exponentially decayed counters: a hit count `S_i`, incremented when a tile falls in the
cell of anchor `i`, and an *exposure* `E_i` that accumulates the anchor's decayed lifetime —
incremented once per block for as long as the anchor is live. Their ratio `λ̂_i = S_i / E_i` is
then a decayed hit-*rate* (hits per unit lifetime), which is the correct per-anchor quantity: it
scales as `p / g` (data density divided by anchor density), so summing it over a window recovers
the data density independently of how the anchors are placed (§Readout). Defining the exposure as
lifetime rather than as traffic is essential — were `E_i` a count of tiles nearest to `i`, it
would scale as `p / g` just as `S_i` does, their ratio would collapse to a near-constant geometric
factor carrying no density, and both the readout and the staleness eviction below would fail.
Coverage is carried by the anchor positions; frequency is carried by the rate `λ̂_i`, which —
unlike a nearest-neighbor distance under a cover — retains its dependence on density.

- *Placement.* A tile farther than the cell scale `s` from every anchor seeds a new anchor, so
  the anchor set tiles the occupied support. On the measured continuum the density-estimation-
  optimal bandwidth places anchors in near-proportion to the data density (formally, anchor
  density `∝ p^{d*/(d*+2)}`, the classical quantization exponent; Graf and Luschgy, 2000). The
  cell scale — the single radius `s` used for *both* seeding and hits (Algorithm 2) — is set at the
  *fill knee*, the value at which seed-on-miss populates exactly `M` anchors (empirically `s ≈ 0.22`
  here; §3.7). This exceeds the covering-radius estimate `s_M ≈ (V/M)^{1/d*} ≈ 0.155` by about 1.4×,
  because seed-on-miss packs anchors at spacing `s` rather than covering at radius `s`; the estimate
  is a lower bound and `s` should be set empirically. The resulting anchor spacing (≈ 0.6 `L`) is
  finer than the correlation length, but the readout pools anchors to an effective bandwidth of
  ≈ `L` (§3.7).
- *Readout.* The local density at a query is read from the counters as an *unnormalized*
  kernel sum of the anchor rates,
  `p̂(x) = Σ_i λ̂_i · K_h(x − b_i)`,
  where `K_h` is a smooth, radially symmetric, compactly supported kernel whose bandwidth
  `h(x)` encloses `j ≈ 32–64` anchors. Summing rather than averaging is essential: each `λ̂_i`
  is a per-anchor mass, so the sum has expectation `∫ p(y) K_h(x − y) dy`, *independent of the
  anchor placement density* — the estimate is the same whether anchors are placed proportionally
  or as a cover, because the number of nearby anchors and the mass each carries are reciprocal
  and cancel. Normalizing the sum (dividing by `Σ_i K_h`) would cancel the anchor density and
  instead estimate the mean per-cell mass `∝ p^{1−γ}`, collapsing the density range and
  vanishing entirely at proportional placement; this is why the readout must be an unnormalized
  sum. Centering the kernel on the query and using a symmetric profile cancels the leading
  (gradient) term of the bias exactly — anchors on either side of the query balance — leaving a
  curvature-order residual, whereas reading only the nearest anchor incurs a first-order,
  spatially frozen bias of up to half a cell. The effective readout bandwidth is the pooling window
  — the radius enclosing the `j` anchors, ≈ `L` (§3.7) — and going finer resolves nothing, since the
  density field has no structure below `L`; in `d* ≈ 9.5` those `j` anchors already lie within about
  `1.5×` the local spacing, so pooling costs almost nothing in resolution. The score is the
  monotone map `t(x) = F̂(log p̂(x))` — the probability integral transform, i.e. the
  decayed empirical rank of `log p̂(x)` among its values at recent tiles — so that `t` is the
  fraction of the distribution at lower density than `x` (a density percentile in `[0,1]`) and
  is invariant to the estimate's unknown multiplicative constant, to error in `d*`, and to
  exposure normalization. A two-moment probit on `log p̂` is the cheap parametric fallback (the
  same functional form as the §3.3 scorer, but applied to the log-density rather than to raw
  distance). Before the counters have filled, the distance readout of §3.3 serves as a
  cold-start estimator; thereafter it is retained only as a consistency probe.
- *Eviction and forgetting.* When the anchor set is full, the anchor of lowest traffic rate
  `λ̂_i` is evicted — a least-frequently-used rule with aging — so an anchor that stops receiving
  traffic decays out while active anchors persist. This directly removes the non-eviction of
  §3.4: an isolated anchor accrues no traffic and is the first evicted, rather than the last. A
  transient reserve of order 1% of `M`, sized by the measured one-off rate `ρ_out`, holds newly
  admitted anchors so a one-off tile cannot displace an established anchor before accumulating
  traffic. The counters decay with a half-life of a few hundred batches — long relative to the
  batch autocorrelation, so recurring morphology accumulates standing evidence, yet short
  relative to the drift horizon, so the estimate tracks the current distribution.

```
Algorithm 2  Counted-coverage bank (proposed): update and scoring for one batch X

  if iteration < T_warm:  return t(x) = 0 for all x

  for every live anchor i:                                    # decay, and age the exposure
      S_i ← η · S_i                                           #   S: decayed hits
      E_i ← η · E_i + 1                                       #   E: decayed lifetime (blocks alive)

  gather signatures of X across workers  →  X_global
  for each x in X_global:                                     # score before updating
      if counters are still filling:
          t(x) ← distance readout of §3.3                      # cold-start
      else:
          p̂(x) ← Σ_i (S_i / E_i) · K_h( s(x) − b_i )           # unnormalized centered kernel sum
          t(x) ← F̂( log p̂(x) )                                # decayed empirical rank (PIT)

  for each x in X_global:                                      # update after scoring
      i* ← nearest anchor to s(x)
      if ‖s(x) − b_{i*}‖ ≤ s:
          S_{i*} ← S_{i*} + 1                                  # a hit (exposure was aged above)
      else:
          admit a new anchor at s(x) into the transient reserve
          if the anchor set is full:  evict argmin_i S_i/E_i outside the reserve

  return { t(x) : x in this worker's rows }
```

A remark on resolution. With `M = 8192` anchors the anchor spacing is finer than the correlation
length (`s_M ≈ 0.6 L`), but the readout pools `j ≈ 64` anchors to an effective bandwidth of ≈ `L`
(§3.7), so every readout is `L`-smoothed. Structure finer than `L` is therefore invisible to any
bounded summary of this size, whatever its readout; refining it would require exponentially more
memory or a lower-dimensional signature. This bounds both bank designs equally and is a property
of the regime, not of either policy.

### 3.6 Modulating the objective

Both bank designs produce a typicality score `t(x)`, which modulates the image-level DINO
cross-entropy in one of two ways. The iBOT objective is left untouched; the total loss is
`L = m(x) · CE_DINO + CE_iBOT + λ_sem · CE_iBOT^{sem}`, where the modulation `m(x)` is one of
the following.

**Weighted loss.** The DINO term is scaled per tile by `w(x) = 1 − β · t(x)`, `β ∈ [0,1]`. A
typical tile contributes a smaller-magnitude gradient while the target it is trained toward is
unchanged. This is a direct importance weighting — it reshapes the effective sampling
distribution toward a flatter distribution over morphology while leaving each tile's learning
signal intact — and is bounded and simple to reason about.

**Adaptive temperature.** The per-tile student softmax temperature is scaled,
`τ(x) = τ_base · (1 + α · t(x))`, so a typical tile receives a flatter target distribution. This
changes not only the gradient magnitude but the shape of the target, redistributing probability
mass across output prototypes rather than only down-scaling the tile's contribution. It is a
stronger intervention that can actively flatten over-represented modes in the assignment itself,
at the cost of coupling the typicality estimate more tightly into the representation geometry.

The two bank designs (§3.3, §3.5) and the two modulations thus define four configurations. Their
comparison is the subject of Section 4; the sensitivity of the leading configuration to its
principal hyperparameters (`β` or `α`, the warmup `T_warm`, and, for the counted-coverage bank,
the decay half-life and cell scale) is studied thereafter.

### 3.7 Empirical determination of the constants, and offline validation

The counted-coverage design rests on two empirical claims: that the constants of Table 1 are
properties of the model rather than of one sample, and that the bank, run end to end, actually
recovers tile redundancy. We establish both by inference-only study on the baseline model — a
convergence analysis of the constants and an offline run of the bank on cached signatures — with
no retraining.

**Convergence of the constants.** Each constant of Table 1 was first estimated on the single
76,800-tile probe of §3.3. To rule out sampling artifacts, we re-estimated each on a fresh,
independent 384,000-tile sample (a disjoint dataloader seed, same tap and checkpoint), reporting
a bootstrap 95% confidence interval and the estimate as a function of subsample size. A constant
is taken as converged when the two independent samples agree and the estimate is flat in `N`.

**Table 2.** Convergence of the design constants: original probe vs. a 5× larger independent sample.
| constant | probe (76.8k) | independent sample (384k), 95% CI | vs. `N` | verdict |
|---|---|---|---|---|
| `d*` (participation ratio) | 9.52 | 9.58 [9.54, 9.61] | flat | converged, ≈ 9.5 |
| `d*` (two-NN) | 9.23 | 9.29 [9.14, 9.45] | mild estimator drift | converged, ≈ 9.3 |
| burst factor `b` | 1.20 | 1.15 | — | converged |
| autocorr length `τ_ac` | < 1 batch | < 1 batch | — | converged |
| one-off rate `ρ_out` | ~10⁻⁵ (slope −0.80) | 1.6×10⁻⁵ (slope −0.87) | power law to 128k, no floor | converged |
| correlation length `L` | 0.26 | 0.257 / 0.259 | flat | converged, ≈ 0.26 |
| density skew `R` | — | p99/p1 and log-variance grow with `N` | not flat | **scale-dependent** |

Five constants (`d*`, `b`, `τ_ac`, `ρ_out`, `L`) agree across the two samples and are flat in `N`.
Two points require care. The correlation length is `L ≈ 0.26` from the neighbor-gradient estimator;
an earlier value of 0.16 came from a random-pair estimator that is unstable in this dimension and
is superseded. The density dynamic range `R` is genuinely *not* a fixed constant: both p99/p1 and
the variance of log-density grow monotonically with sample size (the variance rises from 1.7 at
5k tiles to 3.1 at 160k), because a `k`-nearest-neighbor estimate's bandwidth shrinks as `N` grows
and resolves finer structure. The converged, operationally meaningful quantity is the skew at a
*fixed* bandwidth equal to the bank's resolution: at radius ≈ `L`, the log-density variance is 1.41
(stable across sample size) and the 90/10 density ratio is ≈ 20. The design does not depend on
pinning `R`, because the rank/PIT readout (§3.5) is invariant to any monotone rescaling of density.

The same study fixes the resolution scales. The bank's anchor spacing is `s_M ≈ 0.155` (≈ 0.6 `L`),
but the readout pools `j ≈ 64` anchors, whose enclosing radius — the effective bandwidth — is
`≈ 0.28 ≈ L`. So although anchors are placed finer than the correlation length, every readout is
`L`-smoothed, and structure below `L` is unresolvable at this memory budget; this is the ceiling of
§3.8, and it is set by the pooling bandwidth, not by the finer anchor spacing.

**Offline validation of the bank.** We ran the counted-coverage bank (Algorithm 2, with the
corrected lifetime exposure) over the 384,000 signatures in stream order and compared its score,
per tile, against an offline `k`-nearest-neighbor density on the full sample — the best available
proxy for ground-truth redundancy. With the hit radius set at the fill knee (`s ≈ 0.22`), the
online score recovers the offline density with **Spearman ρ = 0.75**, on a bounded memory holding
8,192 of 384,000 tiles; this is close to the ceiling the resolution allows, since the score is an
`L`-smoothed estimate correlated against a finer reference. The per-anchor rate `λ̂` tracks the
density at its own location with ρ = 0.72, confirming that the counting itself — not merely the
kernel smoothing — carries the signal. The bank reaches steady state (8,192 anchors, modest
turnover).

**Table 3.** Offline bank on 384k signatures: recovery of the offline density vs. the hit radius `s`.
| hit radius `s` | Spearman(`t`, density) | Spearman(`λ̂`ₐₙ𝒸ₕₒᵣ, density) | anchors | admits/block |
|---|---|---|---|---|
| 0.14 | +0.07 | — | 8192 | very high |
| 0.18 | +0.70 | +0.18 | 8192 | 222 |
| **0.22** | **+0.75** | **+0.72** | 8192 | 25 |
| 0.26 | +0.68 | +0.92 | 4967 (underfilled) | ~0 |
| 0.30 | +0.61 | +0.90 | 2117 (underfilled) | ~0 |

Three controls confirm the design decisions. The normalized (Nadaraya–Watson) readout — which the
theory of §3.5 predicts collapses toward `p^{1−γ}` — yields ρ ≈ 0, so the unnormalized sum is
necessary as claimed. Replacing the lifetime exposure with the discarded traffic-count exposure
(§3.5), everything else held at `s = 0.22`, collapses the per-anchor rate to a constant (coefficient
of variation 0.00, versus 1.27 for the corrected form): the counters then carry no density, and
recovery falls to ρ = 0.57 — the residual coverage-only signal of §3.4 — forfeiting the counting
contribution that lifts the corrected readout to 0.75. This directly measures the failure the
lifetime-exposure definition was introduced to avoid, rather than resting on the analogous
normalization control. Finally, setting the hit radius below the fill knee (`s = 0.14`) drives
constant admission and eviction that prevents the counters from stabilizing, collapsing recovery to
ρ = 0.07; setting `s` at the fill knee restores it. The single parameter that must be set with care
is therefore the hit radius, at the fill knee (≈ 0.22 here, where admissions per block collapse),
which exceeds the covering-radius estimate `s_M ≈ 0.155` by ~1.4× and should be set empirically. This
offline run is the
prerequisite we place before any training integration (§3.8): it exercises the full mechanism on
real signatures at low cost, and it is where a readout-inverting error surfaces as a flat,
uncorrelated score — as the normalized control and the mis-set-radius run both illustrate.

**Portability: which changes invalidate the constants.** The constants above are properties of the
*signature distribution* — of the composition (encoder × prototypes × data stream) — so it matters
which configuration changes leave that distribution intact and which do not. Two curator-internal
knobs leave every constant unchanged. The **bank size `M`** sets only the anchor spacing
`s_M ∝ M^{−1/d*}` and, through the pooling count, the readout bandwidth; because `d* ≈ 9.5` this
dependence is very weak (halving `M` coarsens the spacing by ~7.5%, and reaching the `L` ceiling or
the pooling-locality floor takes order-of-magnitude changes), so `M` is a soft knob over a wide band,
with the hit radius `s` the only coupled parameter — it must be re-tuned to the fill knee, which
scales with `M` as `s_M` does. The **prototype count `K'`**
also leaves the constants intact provided `K' ≥` the effective prototype rank (measured ≈ 44): the
signature is then a rotation (`K' = 256`) or a projection that retains the occupied subspace, and the
intrinsic dimension on which every `d*`-dependent formula rests is preserved. Reducing `K'` below the
effective rank projects out real structure and does change `d*` and everything downstream; the
collapse of `R` to effective rank ≈ 44 is evidence that any `K'` in roughly `[64, 256]` behaves
identically, so an undercomplete choice near 64 is safe and cheaper.

Everything else that alters the learned representation or the stream ordering shifts the constants:

| change | constants affected | note |
|---|---|---|
| backbone scale / architecture / SSL recipe | `d*`, `R`, `L`, effective rank | the largest effect; measured at ViT-B, so a ViT-L model — the scale much of the pathology-FM literature uses — will have a different manifold and must be re-measured before porting |
| training data mix / tissue diversity / QC | `R`, `ρ_out`, mode structure | the single-slice caveat of §3.8; a broader or less-filtered corpus raises the skew and the artifact rate |
| dataloader: interleave, shard size, batch size, source mixing | `b`, `τ_ac` | burstiness and autocorrelation are properties of the stream *order*; reducing the interleave raises `b` and lengthens `τ_ac`, which the decay half-life must then accommodate |
| magnification / tile size / augmentation | `d*`, `R` (secondary) | the characterization uses the clean 448→224 tap; heavy train-time augmentation shifts the signature distribution |
| `R`-training weights (`L_cov`, prototype LR) | effective rank → the `K'` floor | these set how orthogonal and spread the prototypes are |

The practical rule: **`M` and `K' (≥ effective rank)` may be swept freely, but a change of backbone,
data, or dataloader requires re-running the characterization of §3.7 before the design constants — and
the parameters `s`, `L`, and the half-life derived from them — can be trusted.** Re-measurement is
inexpensive (inference-only on cached signatures), and the offline prototype is itself the guard: if
the constants have drifted under a configuration change, the density-recovery correlation falls, which
flags the need to re-measure before committing a training run.

### 3.8 Limitations

Three limitations bound the method. First, as noted in §3.5, the estimate is resolution-limited:
with a fixed memory budget over a support of intrinsic dimension near ten, structure finer than
the density correlation length is invisible to any bounded summary. Second, the empirical
constants of Table 1 were measured on a single, morphologically homogeneous slice of the stream;
a substantially more skewed corpus could shift the operating regime, although the counted readout
is by construction insensitive to the exact anchor-placement exponent. Third, the typicality
estimate modulates the objective that trains the encoder that produces the signatures, so the
distribution the module measures is not exogenous. Deferring activation until the representation
has stabilized (§3.3) is the safeguard we rely on; a formal analysis of the coupled dynamics is
left to future work, and the adaptive-temperature modulation, which feeds back through the
target distribution, tightens this coupling relative to the weighted-loss form.

---

### References

Arthur and Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA 2007.

Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.

Chen et al. *Towards a General-Purpose Foundation Model for Computational Pathology (UNI).* Nature Medicine 2024.

Facco et al. *Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information.* Scientific Reports 2017.

Graf and Luschgy. *Foundations of Quantization for Probability Distributions.* Springer LNM 1730, 2000.

Kornblith et al. *Similarity of Neural Network Representations Revisited.* ICML 2019.

Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.

Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer.* ICLR 2022.
