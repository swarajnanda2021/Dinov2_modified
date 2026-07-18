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
   signatures to produce a scalar typicality score `t(x) ∈ [0,1]`, high when the tile's signature
   lands in a crowded part of the bank — many stored signatures nearby — and low when it is
   isolated (§3.3).
3. **Modulation.** `t(x)` modulates the DINO objective for that tile, either as a loss weight
   or as a softmax temperature (§3.6).

Throughout, a signature is a 256-dimensional vector, the bank is a set of such vectors, and every
"nearby" or "distance" below is an L1 distance between signatures (a `cdist`); "crowded" means many
stored signatures fall within a short L1 distance. The signature (§3.2) and the modulation (§3.6)
are shared across all configurations of the method. The memory bank is the component that varies:
§3.3 describes the bank as currently implemented, §3.4 identifies a structural limitation of it,
and §3.5 describes a proposed alternative. The two bank designs, crossed with the two modulations,
give the four configurations studied in Section 4.

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

the vector of similarities (cosines) between the tile and each representative prototype. Signatures
are *peaky* for well-formed morphology (one prototype dominates) and *diffuse* for representations
that resemble no prototype (all similarities near the small-angle floor of a random projection),
a distinction used diagnostically below. (These `K'` representative prototypes — the rows of `R` —
are distinct from, and should not be confused with, the *anchors* of the memory bank in §3.5,
which are stored signatures; the word "anchor" below always denotes a bank entry.)

The representative prototypes are trained online, alongside the backbone but by a dedicated
optimizer, to spread across the region of bottleneck space the data occupies, using

```
L_R = L_nn + λ_cov · L_cov ,
```

where `L_nn` pulls each representative prototype toward its nearest DINO output prototype, so they
track the morphology directions the model itself has learned, and `L_cov` penalizes the off-diagonal
entries of `R Rᵀ`, so they spread out rather than collapse. The rows of `R` are re-projected
to the unit sphere after each step, keeping `R` close to an orthonormal frame. Because both the
prototypes and their loss derive from the stop-gradient bottleneck, the signature space is a
passive readout of the representation, not a target the backbone optimizes toward.

The bottleneck is 256-dimensional, so at most 256 directions can be mutually orthogonal: `K' = 256`
is the largest orthonormal frame the space admits, at which `R` is a rotation (a complete orthonormal
basis). This upper bound complements the lower bound of §3.7 — `R` collapses under training to an
effective rank ≈ 44, below which morphology structure would be projected out. In a full run the rows
of `R` are grown online by `L_R` above (the DINO output prototypes it references number
`out_dim = 65,536`); the offline study of §3.7 instead uses a frozen baseline with no online `L_R`,
as detailed there.

### 3.3 The typicality bank

The score measures how crowded a tile's neighborhood is: how many stored signatures sit close to it
in L1 distance. Let the tile stream have length `N` — the total number of tiles seen over training,
on the order of `10^8` — and let the bank `B` hold at most `M` signatures, `M ≪ N` (in our
configuration `M = 8192`). For each incoming tile the bank yields the L1 distance to its single
nearest stored signature, which the scorer converts to a typicality value — a short distance means
a crowded neighborhood, hence a typical tile.

**Scoring.** For a query signature `s(x)`, let `d(x) = min_{b ∈ B} ‖s(x) − b‖₁` be the
distance to the nearest bank entry, and let `μ_B, σ_B` be the mean and standard deviation of the
within-bank nearest-neighbor distances (how far apart the stored signatures typically sit). The
score is

```
t(x) = 1 − Φ( (d(x) − μ_B) / σ_B ) ,
```

with `Φ` the standard normal CDF: a tile that lands close to the bank is scored typical
(`t → 1`), a tile far from every entry is scored rare (`t → 0`), and calibrating by the bank's
own spacing makes the score scale-free.

**Maintenance.** Empty slots are filled first; once the bank is full, each step admits the
tiles of the current batch whose distance `d(x)` is largest — the most novel — and, for each
admission, evicts the bank entry nearest to it. This novelty-admission, evict-nearest rule keeps
the bank spread across the region of signature space the data occupies.

**Empirical basis.** The properties of the signature stream that inform the module's design were
measured on a *baseline* model: a standard DINOv2 ViT-B/16 trained on pathology tiles with all four
extensions of this work disabled, under a recipe following UNI (Chen et al., 2024) at ViT-B rather
than its ViT-L scale. Measuring on this un-modulated baseline is deliberate: it characterizes the
signature distribution the module takes as *input*, before the module itself perturbs it, which both
isolates the design target and avoids the closed-loop confound of §3.8. All measurements are
inference-only, drawn from the training stream in dataloader order (so arrival order is preserved
for the temporal constants), with no retraining: the stabilization point below is measured on a
fixed probe encoded through a checkpoint ladder (10k–124k iterations), and the geometry and temporal
constants of Table 1 are determined on a larger independent sample at the final checkpoint (§3.7).

**Activation.** The bank is only meaningful once the signatures it stores are stable. The
signature space is a projection of the DINO-head bottleneck, and the bottleneck stabilizes far
earlier than the raw backbone: measuring linear centered kernel alignment (CKA; Kornblith et
al., 2019) between the probe encoded at successive checkpoints and at the final model,
the raw `[CLS]` representation is only 55% converged at 50k iterations whereas the bottleneck is
94% converged, and a control applying the final projection head to every checkpoint's `[CLS]`
reaches 98% by 50k — establishing that the early stability is a property of a low-dimensional,
early-forming subspace of the backbone rather than of the head adapting to a drifting
representation. Accordingly the entire module is inactive for the first `T_warm ≈ 50k`
iterations (the warmup), and the bank is filled only thereafter.

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
eviction: isolated entries are absorbing. Over training the bank therefore drifts toward an even
grid that fills the occupied space (a *cover*) rather than a set distributed like the data itself
(a *sample*). This degrades the score in the way that matters most: once the bank approximates a
cover, the nearest-neighbor distance `d(x)` is close to the cover's spacing almost everywhere and
no longer varies with crowding, so the typicality score flattens and ceases to distinguish common
tiles from rare ones — exactly the discrimination the module exists to provide.

Stated in estimator terms, reading crowding from the distance to the single nearest stored
signature is faithful only if the bank is itself a representative *sample* of the signature
distribution — a set distributed like the data. Such a sample is produced by any *content-independent*
maintenance rule — one that evicts by age or at random rather than by position — for instance
retaining the most recent `M` signatures (valid here because the stream is near-independent at the
tile scale, Table 1) or a uniform reservoir over those seen; the surviving set is then distributed
as the data. Evict-nearest is content-dependent by construction: it evicts by position, which is
precisely what drives the bank away from a sample and toward a cover. The limitation therefore
admits two resolutions: restore faithful sampling with a content-independent rule, under which the
existing distance readout becomes valid; or read crowding from explicit counts rather than distance.
The first is simpler and, on the mild stream we measured (§3.5), likely adequate on its own; we
nonetheless develop the second (§3.5), because explicit counts retain coverage of rare morphology
and remain valid if the full corpus departs from the mild regime that faithful sampling relies on.

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

We describe a bank design, not yet implemented, that retains a covering set of anchors but reads
crowding from explicit hit counts rather than from nearest-neighbor distance, thereby removing the
non-eviction pathology of §3.4. Its parameters are fixed by a direct characterization of the tile
stream, summarized in Table 1 and determined and validated in §3.7.

Two counting terms recur below. A *spot* is a small L1 ball of radius `s` around a point (`s` is
the spot radius, fixed by the bank; §3.7). Within a spot we distinguish *tile-crowding* — how many
streaming tiles land in it, i.e. how common that morphology is — from *anchor-crowding* — how many
bank entries sit in it, i.e. how the memory happens to be placed. The design turns on keeping these
two apart. An anchor's *cell* is the spot (ball of radius `s`) around it.

**Table 1.** Design constants of the signature stream, determined on the baseline model (standard
DINOv2 ViT-B/16, all extensions disabled) over an independent 384,000-tile sample with bootstrap
95% confidence intervals (§3.7), and the design quantity each fixes. Distance-valued constants
(`L`, and the spacings below) are in L1 units on the signatures `s = R·z`, the metric the bank
queries in (§3.7); the dimensionless constants (`d*`, `R`, `b`) are metric-invariant.
| Property | Symbol | Value (95% CI) | Design role |
|---|---|---|---|
| Intrinsic dimension | `d*` | 9.3 [9.1–9.4] | readout viability; cell scale |
| Local-density dynamic range | `R` | scale-dependent; ≈ 18 at radius `L` | operating regime (moderate) |
| Mode structure | — | connected continuum, no dominant mode | removes the isolated-outlier hazard |
| Tile-level burst factor | `b` | ≈ 1.2 (decays within one batch) | permits exponential forgetting |
| Log-density correlation length | `L` | 3.34 | sets the readout bandwidth |
| One-off (artifact) rate | `ρ_out` | ≈ 10⁻³ (no floor) | sizes the transient reserve |

In Table 1, `d*` is the intrinsic dimension — the number of effective directions the signatures
actually occupy (far below the ambient 256); `R` is how much the tile-crowding varies from the
emptiest to the most crowded spots; and `L` is the correlation length — the L1 distance over which
crowding changes appreciably. Two structural findings guide the design. First, the support is a
connected continuum of intrinsic dimension near ten, the crowding varies only moderately across
spots (`R` ≈ 18× at the operating scale `L`; §3.7), and there are no isolated clumps; the
absorbing-outlier hazard of §3.4 thus has no isolated clumps to attach to, and the design need not
defend against extreme skew. Second, the stream is near-independent at the tile level (a burst
factor `b` near 1.2 that decays within one batch — similar tiles do not arrive in long runs) and
the signature distribution drifts slowly after activation — by one correlation length only over
tens of thousands of iterations — so stored statistics can be forgotten by simple exponential decay
without becoming stale relative to the encoder that produced them.

**Design.** Because a single set of stored points cannot encode both *where* the occupied region
lies and *how crowded* each part is — the tension underlying §3.4 — the two roles are separated.
The bank stores anchor signatures that cover the support, and augments each anchor with two
exponentially decayed counters: a hit count `S_i` — decayed number of tiles that have landed in
anchor `i`'s cell — and an *exposure* `E_i` that accumulates the anchor's decayed lifetime,
incremented once per *block* (one update step, i.e. one processing of a gathered batch; the unit
the half-life is quoted in) for as long as the anchor is live. Their ratio `λ̂_i = S_i / E_i` is
the anchor's decayed hit-*rate* — tiles landing in its cell per block — which equals the
tile-crowding divided by the anchor-crowding at its spot. This ratio is the correct per-anchor
quantity: summing it over the anchors near a query cancels the anchor-crowding and leaves the
tile-crowding (§Readout), so the estimate does not depend on how the memory happens to be placed.
Defining the exposure as lifetime rather than as traffic is essential — were `E_i` instead a count
of tiles nearest to `i`, it would grow with tile-crowding just as `S_i` does, their ratio would
collapse to a near-constant carrying no crowding information, and both the readout and the staleness
eviction below would fail. Coverage is carried by the anchor positions; crowding is carried by the
rate `λ̂_i`, which — unlike a nearest-neighbor distance under a cover — keeps its dependence on how
common a tile is.

- *Placement.* A tile farther than the spot radius `s` from every anchor seeds a new anchor, so
  the anchor set tiles the occupied support. On the measured continuum the estimation-optimal
  placement puts anchors in near-proportion to the tile-crowding (formally, anchor density
  `∝ p^{d*/(d*+2)}`, the classical quantization exponent of Graf and Luschgy, 2000 — this motivates
  the placement but never enters the implementation). The spot radius — the single `s` used for
  *both* seeding and hit-counting (Algorithm 2) — is set at the *fill knee*, the value at which
  seed-on-miss populates exactly `M` anchors (empirically `s ≈ 2.75` in L1 units here; §3.7). This
  exceeds the ideal even-tiling estimate `s_M = (V/M)^{1/d*} ≈ 1.97` by about 1.4×, because
  seed-on-miss packs anchors at spacing `s` rather than tiling at radius `s`; the estimate is a lower
  bound and `s` should be set empirically. Because seed-on-miss forces every anchor at least `s` from
  the others, the *achieved* spacing is ≈ `s` ≈ 2.75 (≈ 0.82 `L`), comparable to the correlation
  length; the readout then pools anchors over a window of order `L` (§3.7).
- *Readout.* The tile-crowding at a query is estimated from the counters as an *unnormalized*
  kernel sum — a distance-weighted sum over the nearby anchors,
  `p̂(x) = Σ_i λ̂_i · K_h(x − b_i)` (`p̂` = estimated tile-crowding at `x`),
  where `K_h` is a smooth, radially symmetric, compactly supported weight that falls off with
  distance and whose bandwidth `h(x)` encloses the `j ≈ 32–64` nearest anchors. Summing rather than
  averaging is essential, and the reason is the tile-crowding ÷ anchor-crowding structure of `λ̂`: a
  crowded spot holds more anchors, each carrying a smaller rate, so **summing** the rates cancels the
  anchor-crowding and leaves the tile-crowding, whereas **averaging** (dividing by `Σ_i K_h`) divides
  the anchor-crowding straight back out — it would instead estimate the mean per-anchor rate
  `∝ p^{1−γ}`, which vanishes at *proportional* placement (`γ → 1`). The gap is therefore
  placement-dependent, and seed-on-miss produces a near-uniform cover (`γ ≈ 0`), where `p^{1−γ} = p`
  and the two nearly coincide: the offline run (§3.7) measures ρ = 0.74 for the normalized readout
  versus 0.75 for the sum. We nonetheless use the unnormalized sum because it is placement-independent
  — it recovers crowding at *any* `γ`, hence stays valid if the placement drifts from this mild
  regime. Centering the kernel
  on the query and using a symmetric profile cancels the leading (gradient) term of the bias exactly
  — anchors on either side of the query balance — leaving a curvature-order residual, whereas reading
  only the nearest anchor incurs a first-order, spatially frozen bias of up to half a cell. The
  effective readout bandwidth is this pooling window — the radius enclosing the `j` anchors, ≈ `L`
  (§3.7) — and going finer resolves nothing, since the crowding field has no structure below `L`; in
  `d* ≈ 9.3` those `j` anchors already lie within about `1.5×` the local spacing, so pooling costs
  almost nothing in resolution. The score is the monotone map `t(x) = F̂(log p̂(x))` — the probability
  integral transform: the percentile of the query's crowding among recent tiles (more crowded than
  75% of them means `t = 0.75`). Being a percentile, `t` is invariant to the estimate's unknown
  multiplicative constant, to error in `d*`, and to exposure normalization. A two-moment probit on
  `log p̂` is the cheap parametric fallback (the same functional form as the §3.3 scorer, but applied
  to the log-crowding rather than to raw distance). Before the counters have filled, the distance
  readout of §3.3 serves as a cold-start estimator; thereafter it is retained only as a consistency
  probe.
- *Eviction and forgetting.* When the anchor set is full, the anchor of lowest hit-rate `λ̂_i` is
  evicted — a least-frequently-used rule with aging — so an anchor that stops receiving tiles decays
  out while active anchors persist. This directly removes the non-eviction of §3.4: an isolated
  anchor accrues no hits and is the first evicted, rather than the last. A transient reserve of order
  1% of `M`, sized by the measured one-off rate `ρ_out`, holds newly admitted anchors so a one-off
  tile cannot displace an established anchor before accumulating hits. The counters decay with a
  half-life of a few hundred blocks — long relative to the batch autocorrelation, so recurring
  morphology accumulates standing evidence, yet short relative to the drift horizon, so the estimate
  tracks the current distribution.

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

Here `η` is the per-block decay factor (set by the half-life), `S_i`/`E_i` the decayed hit count and
lifetime of anchor `i`, and `p̂` the estimated tile-crowding at the query; the kernel sum runs over
the `j` nearest anchors and `F̂` is the recent-crowding percentile of the previous paragraph.

A remark on resolution. With `M = 8192` anchors the achieved spacing is ≈ 0.82 `L` (set by the spot
radius `s`; §3.7), and the readout pools `j ≈ 64` anchors over a window of order `L`, so every
readout is smoothed at scale `L`. Structure finer than `L` is therefore invisible to any bounded
summary of this size, whatever its readout; refining it would require exponentially more memory or a
lower-dimensional signature. This bounds both bank designs equally and is a property of the regime,
not of either policy.

### 3.6 Modulating the objective

Both bank designs produce a typicality score `t(x)`, which modulates the image-level DINO
cross-entropy in one of two ways. The iBOT objective is left untouched; the total loss is
`L = m(x) · CE_DINO + CE_iBOT + λ_sem · CE_iBOT^{sem}`, where the modulation `m(x)` is one of
the following.

**Weighted loss.** The DINO term is scaled per tile by `w(x) = 1 − β · t(x)`, `β ∈ [0,1]`. A
typical tile contributes a smaller-magnitude gradient while the target it is trained toward is
unchanged. This is a direct importance weighting — it reshapes the effective sampling
distribution to be flatter over morphology while leaving each tile's learning signal intact —
and is bounded and simple to reason about.

**Adaptive temperature.** The per-tile student softmax temperature is scaled,
`τ(x) = τ_base · (1 + α · t(x))`, so a typical tile receives a flatter target distribution. This
changes not only the gradient magnitude but the shape of the target, redistributing probability
mass across output prototypes rather than only down-scaling the tile's contribution. It is a
stronger intervention that can actively flatten over-represented modes in the assignment itself,
at the cost of coupling the typicality estimate more tightly into the representation geometry.

The two bank designs (§3.3, §3.5) and the two modulations thus define four configurations. Their
comparison is the subject of Section 4; the sensitivity of the leading configuration to its
principal hyperparameters (`β` or `α`, the warmup `T_warm`, and, for the counted-coverage bank,
the decay half-life and spot radius) is studied thereafter.

### 3.7 Empirical determination of the constants, and offline validation

The counted-coverage design rests on two empirical claims: that the constants of Table 1 are
properties of the model rather than of one sample, and that the bank, run end to end, actually
recovers tile redundancy. We establish both by inference-only study on the baseline model — a
convergence analysis of the constants and an offline run of the bank on cached signatures — with
no retraining. As the reference notion of crowding in this section we use an offline
`k`-nearest-neighbor *density* on the full sample — how many tiles sit near each tile — computed
once and treated as ground truth; "density" below always means this offline crowding reference.

**Determination of the constants.** The constants of Table 1 were determined on the signatures
`s = R·z` under the L1 metric (the bank's metric; see the provenance note below), on an independent
384,000-tile sample (a disjoint dataloader seed, same tap and checkpoint), each with a bootstrap
95% confidence interval and verified stable across subsample size; the intervals are those in
Table 1. The estimators are standard: intrinsic dimension by the two-nearest-neighbor ratio method
(Facco et al., 2017), cross-checked against the covariance participation ratio; the density range
from k-nearest-neighbor density; the burst factor from the signature autocorrelation and same-slide
run-lengths in arrival order; the correlation length from the log-density field; and the one-off
rate from the isolation rate as a function of sample size. Two points on the estimators. The
correlation length is `L ≈ 3.34` (L1 units) from a neighbor-gradient estimator; a random-pair
estimator is unstable in this dimension. The density range `R` is not a fixed constant: both p99/p1
and the variance of log-density grow with sample size, because a `k`-nearest-neighbor estimate's
bandwidth shrinks as `N` grows and resolves finer structure. The converged, operationally meaningful
quantity is the skew at a *fixed* bandwidth equal to the bank's resolution: at radius ≈ `L` the
log-density variance is 1.33 and the 90/10 density ratio is ≈ 18 (Table 1), stable across sample
size. The design does not depend on pinning `R`, because the rank/PIT readout (§3.5) is invariant to
any monotone rescaling of crowding.

The same study fixes the resolution scales (L1 units on `s`). The ideal-tiling estimate
`s_M = (V/M)^{1/d*} ≈ 1.97` (≈ 0.59 `L`) is a lower bound; because seed-on-miss packs anchors at the
spot radius, the *achieved* spacing is ≈ 2.75 (≈ 0.82 `L`, the ~1.4× gap of §3.5). The readout pools
`j ≈ 64` anchors over a window of order `L` (measured 64th-NN ≈ 3.5), so every readout is smoothed at
scale `L` and structure below `L` is unresolvable at this memory budget; this is the ceiling of §3.8,
set by the pooling window, not by the anchor spacing.

**Offline validation of the bank.** We ran the counted-coverage bank (Algorithm 2, with the
corrected lifetime exposure) over the 384,000 signatures `s = R·z` in stream order, under L1, and
compared its score, per tile, against an offline `k`-nearest-neighbor density on the full sample —
the best available proxy for ground-truth redundancy. With the hit radius set at the fill knee
(`s ≈ 2.75` in L1 units), the online score recovers the offline density with **Spearman ρ = 0.75**,
on a bounded memory holding 8,192 of 384,000 tiles; this is close to the ceiling the resolution
allows, since the score is smoothed at scale `L` and correlated against a finer reference. The
per-anchor rate `λ̂` tracks the density at its own location with ρ = 0.67, confirming that the
counting itself — not merely the kernel smoothing — carries the signal. The bank reaches steady
state (8,192 anchors, modest turnover).

**Provenance of `R`.** The baseline run had typicality disabled, so `L_R` never ran and the
checkpoint contains no `R`. We therefore constructed a synthetic `R` from the frozen baseline's
`out_dim = 65,536` DINO output prototypes: a column-pivoted QR on the unit-normed prototype matrix
selects the 256 most linearly independent directions, giving `R ∈ ℝ^{256×256}` (condition number 25,
effective rank 202) — a near-orthonormal frame that stands in for the `L_nn`-aligned `R` a run would
grow (`L_nn` pulls the representative prototypes toward exactly these output prototypes). The cached
signatures are `s = R·z`, and the entire study above — constants and validation — is computed on `s`
under L1, the bank's metric. (An earlier iteration of this study ran on the bare bottleneck `z` under
L2; because this `R` is a near-L2-isometry — `‖s₁−s₂‖₂ / ‖z₁−z₂‖₂ = 1.00 ± 0.03` — the two agree on
the dimensionless constants, and the distance-valued constants simply rescale into L1 units by the
common factor ≈ 12.7, with all scale *ratios* preserved; the ρ = 0.75 recovery reproduces under both.)
One caveat remains, stated as unmeasured: this synthetic `R` has effective rank 202, whereas an
`L_R`-trained `R` collapses to effective rank ≈ 44 (§3.2), and the study is on the frozen baseline,
not on the online-`L_R` signatures a training run would grow. Their agreement was not measured; we
assert no equivalence.

**Table 2.** Offline bank on 384k signatures `s = R·z` (L1): recovery of the offline density vs. the
hit radius `s` (L1 units; `s_M ≈ 1.97`).
| hit radius `s` | Spearman(`t`, density) | Spearman(`λ̂`ₐₙ𝒸ₕₒᵣ, density) | anchors | admits/block |
|---|---|---|---|---|
| 1.97 | +0.24 | −0.10 | 8192 | 476 |
| 2.36 | +0.73 | +0.28 | 8192 | 174 |
| **2.75** | **+0.75** | **+0.67** | 8192 | 31 |
| 3.15 | +0.70 | +0.91 | 6684 (underfilled) | ~0 |
| 3.54 | +0.65 | +0.93 | 3178 (underfilled) | ~0 |

Three controls probe the two load-bearing choices and the hit radius. The one that isolates a design
decision is the **exposure definition**: replacing the lifetime exposure with the discarded
traffic-count exposure (§3.5), everything else held at the knee `s = 2.75`, collapses the per-anchor
rate to a constant (coefficient of variation 0.00, versus 1.21 for the corrected form) — the counters
then carry no crowding, and recovery falls to ρ = 0.53, the residual coverage-only signal of §3.4,
forfeiting the counting contribution that lifts the corrected readout to 0.75. So the exposure must
be lifetime, not traffic. The **normalization** control is weaker than we first reported: the
normalized (Nadaraya–Watson) readout recovers ρ = 0.74, nearly matching the unnormalized sum (0.75).
As §3.5 explains, the two coincide at the near-uniform placement seed-on-miss produces (`γ ≈ 0`); the
sum's advantage is placement-*independence*, not a measurable gap here, so this run establishes that
the sum is *safe*, not that it is *necessary* — the necessity is theoretical, showing only under
proportional placement. Finally, the **hit radius**: setting it below the fill knee (`s = 1.57`)
drives constant admission and eviction that prevent the counters from stabilizing, collapsing
recovery to ρ ≈ 0; setting `s` at the knee restores it. The single parameter that must be set with
care is therefore the hit radius, at the fill knee (≈ 2.75 in L1 units, where admissions per block
collapse), which exceeds the ideal-tiling estimate `s_M ≈ 1.97` by ~1.4× and should be set
empirically. This offline run is the prerequisite we place before any training integration (§3.8):
it exercises the full mechanism on real signatures at low cost, and it is where a readout-breaking
error surfaces as a flat, uncorrelated score — as the traffic-exposure control and the
mis-set-radius run both illustrate.

**Portability: which changes invalidate the constants.** The constants above are properties of the
*signature distribution* — of the composition (encoder × prototypes × data stream) — so it matters
which configuration changes leave that distribution intact and which do not. Two curator-internal
knobs leave every constant unchanged. The **bank size `M`** sets only the anchor spacing
`s_M ∝ M^{−1/d*}` and, through the pooling count, the readout window; because `d* ≈ 9.3` this
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
the correlation length is invisible to any bounded summary. Second, the empirical constants of
Table 1 were measured on a single, morphologically homogeneous slice of the stream; a substantially
more skewed corpus could shift the operating regime, although the counted readout is by construction
insensitive to the exact anchor-placement exponent. Third, the typicality estimate modulates the
objective that trains the encoder that produces the signatures, so the distribution the module
measures is not exogenous. Deferring activation until the representation has stabilized (§3.3) is the
safeguard we rely on; a formal analysis of the coupled dynamics is left to future work, and the
adaptive-temperature modulation, which feeds back through the target distribution, tightens this
coupling relative to the weighted-loss form.

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
