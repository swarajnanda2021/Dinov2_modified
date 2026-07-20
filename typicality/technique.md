# Typicality Dampening

*Method section (manuscript draft, readability revision). This document describes the
typicality-dampening module: the morphology-signature representation it operates on, two
interchangeable memory-bank variants that turn a signature into a typicality score, and the two ways
that score modulates the learning objective. The two bank variants crossed with the two modulations
define the four configurations evaluated in Section 4. Except where explicitly attributed to the online-trained run (§3.5), quantitative statements are inference-only
measurements on a baseline ViT-B/16 DINOv2 model with all extensions of this work disabled; the
empirical basis is described in §3.3 and full protocols are given in Appendix A.*

---

## 3. Typicality Dampening

Whole-slide images are labelled at the slide level but consumed at the tile level, and the resulting
training stream is heavily redundant. A single slide contributes thousands of near-duplicate views of
stroma, adipose, and background, while diagnostically informative morphology is comparatively rare.
Under the single shared softmax temperature of the DINO objective, these common tiles contribute
per-example gradients of the same magnitude as rare ones, so the informative signal is at once
diluted by, and redundantly reinforced against, a long tail of near-duplicates. This is the
tile-level version of a problem that modern self-supervised recipes otherwise handle *before*
training: the DINOv2 corpus, for instance, was de-duplicated and retrieval-rebalanced offline (Oquab
et al., 2024). A tile stream cannot be curated that way at scale.

Typicality dampening supplies the missing curation *online, during training*. At each step it
estimates, for every tile in the batch, how typical that tile is of the morphology the model has
recently seen, and turns down the contribution of typical tiles to the image-level DINO objective.
The module is a stop-gradient side-channel on the standard student/teacher pipeline (Figure 1): it
reads an intermediate student representation, produces one scalar typicality score per tile, and
multiplies that tile's DINO cross-entropy by a factor that shrinks as typicality rises. The
patch-level iBOT objective is left untouched, so dense per-patch supervision is preserved while the
image-level objective is rebalanced toward rare morphology.

### 3.1 Overview

**The one quantity both bank variants estimate is local density** — how crowded a tile's
neighbourhood is in morphology space. Common morphology sits in dense neighbourhoods; rare morphology
sits in sparse ones. The typicality score is simply where a tile's density falls in the recent
distribution: high density → typical → dampen; low density → rare → preserve. The two bank variants
are two standard ways of estimating that density, and naming them this way is the cleanest way to see
how they relate:

- The **distance-calibrated bank** (§3.3) is a *nearest-neighbour* density estimator. In a dense
  neighbourhood the stored points are packed close together, so a short distance to the nearest one
  signals high density. It reads density from the **gaps between** stored points.
- The **counted-coverage bank** (§3.5) is a *kernel density estimator on a fixed memory budget*.
  Rather than store all `N` tiles, it keeps `M` reference points spread across the space and lets each
  one *tally* how many tiles land on it; a heavily-tallied neighbourhood signals high density. It
  reads density from the **counts on** stored points.

The module runs in three stages on each tile `x`:

1. **Signature.** The student's L2-normalized 256-dimensional DINO-head bottleneck `z(x)` is taken
   under stop-gradient and projected onto a small set of learned *representative prototypes* `R`,
   giving a compact morphology signature `s(x)` (§3.2).
2. **Typicality score.** `s(x)` is turned into a scalar `t(x) ∈ [0,1]` by one of the two banks — high
   when the tile lands in a dense neighbourhood, low when it lands in a sparse one (§3.3, §3.5).
3. **Modulation.** `t(x)` modulates that tile's DINO objective, either as a loss weight or as a
   softmax temperature (§3.6).

The signature (§3.2) and the modulation (§3.6) are common to the whole method; only the bank varies.
§3.4 is the analysis that relates the two banks — it identifies the condition under which the
nearest-neighbour readout is a valid density estimate, and the maintenance rule that breaks it, which
is what the counted-coverage variant is built for. The two banks crossed with the two modulations
give the four configurations of Section 4; the distance-calibrated bank is the baseline (it is what
is implemented and instrumented), and the counted-coverage bank is the new variant.

### Notation and terminology

*Symbols used in §3, grouped by role; constant values live in Table 1 and the Implementation
defaults, not repeated here. Throughout, one unit of training time is a **step** — one processing of
a gathered batch, one optimizer update. (This replaces the word "block"; it is not a transformer
block, and it is the same unit "iteration" denotes elsewhere.) Three similar-sounding objects are
kept strictly distinct: an **output prototype** is one of the DINO head's `out_dim = 65,536`
output-layer directions (the model's own), a **representative prototype** is one of the 256 rows of
`R`, and a **stored signature** is neither — it is a tile's signature the bank has kept, together with its state `(S, E, age)`.*

**Representation and signature**
| Symbol | Meaning |
|---|---|
| `z(x)` | student DINO-head bottleneck for tile `x` (256-d, unit-norm, stop-gradient) |
| `R` | representative-prototype matrix (`K'×256`, unit-norm rows) |
| representative prototype | a row of `R` |
| output prototype | one of the DINO head's `out_dim = 65,536` output directions |
| `s(x) = R·z(x)` | the signature: the `K'`-vector of cosines to the representative prototypes |
| `K'` | number of representative prototypes |

**The bank (both variants)**
| Symbol | Meaning |
|---|---|
| `B` | the bank — stored signatures (distance-calibrated) or the stored-signature set (counted-coverage) |
| `M` | bank capacity (established signatures, counted-coverage) |
| stored signature | a tile's signature kept in the counted-coverage bank as a reference point, with a state `(S, E, age)` |
| established set / reserve | the counted-coverage bank's main store and its transient staging buffer |
| `N` | total tiles seen over training (order `10⁸`) |
| step | one training step — one gathered batch, one update (the unit half-lives are quoted in) |

**Distance-calibrated bank (§3.3)**
| Symbol | Meaning |
|---|---|
| `d(x)` | L1 distance from `s(x)` to its nearest stored signature |
| `μ_B, σ_B` | mean and standard deviation of the within-bank nearest-neighbour distances |
| `Φ` | standard normal CDF |
| `t(x)` | typicality score, `∈ [0,1]` (high = typical, low = rare) |

**Counted-coverage bank (§3.5)**
| Symbol | Meaning |
|---|---|
| `p` | tile density — how common a morphology is locally (the §3.7 "density" reference) |
| `g` | stored-signature density — how many stored signatures happen to sit locally |
| `s` | hit radius (a small ball of radius `s`; set at the fill knee) |
| `S_i` | stored signature `i`'s decayed **hit count** (tiles that landed on it) |
| `E_i` | stored signature `i`'s decayed **exposure** — steps it has been alive (a clock, not a hit count) |
| `age_i` | stored signature `i`'s **age** — undecayed count of steps since it was created |
| `λ̂_i = S_i/E_i` | stored signature `i`'s decayed **hit-rate** (hits per step of life) |
| `p̂(x)` | estimated tile density at `x` (the unnormalised kernel sum) |
| `K_h, h` | pooling kernel (triweight) and its bandwidth |
| `j` | number of stored signatures pooled in the readout |
| `η` | per-step decay factor (`= 0.5^{1/H}`, half-life `H` steps) |
| `ε` | numerical guard in `λ̂ = S/(E+ε)` |
| `κ` | tiles per step (the gathered batch size) |

**Stream constants (Table 1; §3.7)**
| Symbol | Meaning |
|---|---|
| `d*` | intrinsic dimension of the signature cloud |
| `L` | correlation length — L1 distance over which density changes appreciably |
| `b` | tile burst factor |
| `ρ_out` | one-off (artifact) rate |

The local-density dynamic range (≈ 18× at scale `L`) is written descriptively, not with a symbol, to
keep it distinct from the matrix `R`.

**Terminology.**

- **Density (tile density, `p`).** The local density of tiles in signature space — how many tiles
  land near a point. Common morphology is high `p`, rare morphology low `p`. The offline "density"
  reference of §3.7 and the design's "tile density" are the same quantity.
- **Stored-signature density (`g`).** How many stored signatures sit near a point — a property of where the
  memory was placed, not of the data. The design's central move is to measure `p` without letting `g`
  contaminate it.
- **Sample vs cover.** A *sample* is a point set drawn like the data, so its own local density tracks
  the data density `p`. A *cover* is a point set laid down to fill the occupied region evenly
  (controlled spacing), so its nearest-neighbour distance reflects the spacing, not `p`. Reading
  density from a nearest-neighbour distance is valid for a sample, not for a cover — the distinction
  §3.4 turns on.
- **Voronoi cell.** The region of signature space closer to a given stored signature than to any other.
  The counted-coverage bank assigns each tile to its nearest stored signature, so that signature's catchment — the
  tiles it counts as hits — is its Voronoi cell (capped at radius `s`). A Voronoi cell's volume
  scales as `1/g`, which is what makes the hit count `S ∝ p/g`.
- **Probability integral transform (PIT).** Replacing a number by its percentile in a reference
  distribution, yielding a score uniform on `[0,1]`. The counted-coverage bank scores a tile by the
  percentile of its density among recent tiles ("denser than 75% of recent tiles" → `t = 0.75`), so
  the score is bounded, centred, and immune to the density estimate's arbitrary scale.
- **Kernel and bandwidth (`K_h`, `h`).** A smooth, radially symmetric, compactly supported weight
  that falls off with distance; its bandwidth `h` is the radius enclosing the `j` nearest stored signatures.
  The readout is a kernel-weighted sum over those stored signatures.
- **Effective rank.** The number of significant directions of a matrix (participation ratio of its
  singular values). Used for `R`: an `L_R`-trained `R` collapses to effective rank ≈ 44, whereas the
  synthetic frame of §3.7 has effective rank 202.
- **Stop-gradient.** Detaching a tensor from the backward pass so no gradient flows through it. The
  entire typicality path is stop-gradient, so it never perturbs the backbone.

### 3.2 Morphology signatures

Both banks need a compact, *stable* coordinate for each tile's morphology — stable because a
nearest-neighbour memory that measures distances is only meaningful if the space those distances live
in stops moving. The raw `[CLS]` representation is neither: it is high-dimensional and keeps
reorganising late into training. The DINO-head bottleneck is both compact and settles early (§3.3),
which is exactly what the memory needs, so the signature is built from it.

Let `z(x) ∈ S^{255}` be the student's unit-norm DINO-head bottleneck for the first global crop, taken
under stop-gradient so the typicality path never contributes to the backbone gradient. A matrix of
`K' = 256` representative prototypes `R ∈ ℝ^{K'×256}`, with unit-norm rows, defines the signature

```
s(x) = R · z(x) ∈ ℝ^{K'} ,        s_k(x) = ⟨R_k, z(x)⟩ ,
```

the vector of cosine similarities between the tile and each representative prototype. A signature is
**peaky** for well-formed morphology (one prototype dominates) and **diffuse** for a tile that
resembles no prototype (all similarities near the small-angle floor of a random projection) — a
distinction used diagnostically below.

The representative prototypes are trained online, alongside the backbone but by their own optimizer,
so they spread across the region of bottleneck space the data occupies:

```
L_R = L_nn + λ_cov · L_cov ,
```

where `L_nn` pulls each representative prototype toward its nearest *output* prototype, so the rows
track the morphology directions the model has actually learned, and `L_cov` penalizes the
off-diagonal of `R Rᵀ`, so the rows stay spread rather than collapse; the rows are renormalized to
the unit sphere after each step. Because prototypes and their loss both derive from the stop-gradient
bottleneck, the signature space is a passive readout of the representation, never a target the
backbone is pushed toward.

The bottleneck is 256-dimensional, so at most 256 directions can be mutually orthogonal: `K' = 256`
is the largest orthonormal frame the space admits, at which `R` is a rotation (a complete orthonormal
basis). This upper bound complements the lower bound of §3.7 — a trained `R` collapses to effective
rank ≈ 44, below which morphology structure would be projected away. In a full run the rows of `R`
are grown online by `L_R` above (the output prototypes it references number `out_dim = 65,536`); the
offline study of §3.7 instead uses a frozen baseline with no online `L_R`, as detailed there.

### 3.3 The distance-calibrated bank

This bank is a nearest-neighbour density estimator: it keeps a set of stored signatures and reads a
tile's local density from how close the nearest stored signature is. Let the tile stream have length
`N` (total tiles over training, order `10⁸`), and let the bank `B` hold at most `M` signatures,
`M ≪ N` (here `M = 8192`). For each incoming tile the bank returns the L1 distance to its single
nearest stored signature — short distance ⇒ dense neighbourhood ⇒ typical tile.

**Scoring.** For a query signature `s(x)`, let `d(x) = min_{b ∈ B} ‖s(x) − b‖₁` be the distance to
the nearest bank entry, and let `μ_B, σ_B` be the mean and standard deviation of the *within-bank*
nearest-neighbour distances — how far apart the stored signatures typically sit. The score is

```
t(x) = 1 − Φ( (d(x) − μ_B) / σ_B ) ,
```

with `Φ` the standard normal CDF: a tile that lands close to the bank scores typical (`t → 1`), one
far from every entry scores rare (`t → 0`), and calibrating by the bank's own spacing makes the score
scale-free.

**Maintenance.** Empty slots are filled first; once the bank is full, each step admits the tiles of
the current batch whose distance `d(x)` is largest — the most novel — and, for each admission, evicts
the bank entry nearest to it. This novelty-admission, evict-nearest rule keeps the bank spread across
the region of signature space the data occupies.

**Empirical basis.** The stream properties that inform the module's design were measured on a
*baseline* model: a standard DINOv2 ViT-B/16 trained on pathology tiles with all four extensions of
this work disabled, under a recipe following UNI (Chen et al., 2024) at ViT-B rather than its ViT-L
scale. Measuring on this un-modulated baseline is deliberate — it characterizes the signature
distribution the module takes as *input*, before the module perturbs it, which isolates the design
target and avoids the closed-loop confound of §3.8. All measurements are inference-only, drawn from
the training stream in dataloader order (so arrival order is preserved for the temporal constants),
with no retraining: the stabilization point below is measured on a fixed probe encoded through a
checkpoint ladder (10k–124k steps), and the geometry and temporal constants of Table 1 are determined
on a larger independent sample at the final checkpoint (§3.7).

**Activation.** The bank is only meaningful once the signatures it stores are stable. The signature
space is a projection of the DINO-head bottleneck, and the bottleneck stabilizes far earlier than the
raw backbone: measuring linear centered kernel alignment (CKA; Kornblith et al., 2019) between the
probe encoded at successive checkpoints and at the final model, the raw `[CLS]` representation is only
55% converged at 50k steps whereas the bottleneck is 94% converged, and a control applying the final
projection head to every checkpoint's `[CLS]` reaches 98% by 50k — establishing that the early
stability belongs to a low-dimensional, early-forming subspace of the backbone rather than to the
head adapting to a drifting representation. Accordingly the entire module is inactive for the first
`T_warm ≈ 50k` steps (the warmup), and the bank is filled only thereafter.

**Implementation.** The bank is a single global structure maintained on the union of signatures
gathered across all data-parallel workers, so every worker shares one estimator rather than each
keeping a private, reduced-resolution bank; all update decisions are deterministic (ties broken by
index), so the bank stays identical across workers and reproducible across nodes. Two health
statistics — the fraction of the bank that is diffuse, and the fraction of tiles scored extremely
rare — are logged throughout training. Algorithm 1 states the procedure.

```
Algorithm 1  Distance-calibrated bank: update and scoring for one batch X

  if step < T_warm:                                     # signatures not yet stable
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

### 3.4 Bank maintenance and the density estimate

The evict-nearest rule has a structural consequence. An entry far from all others is, by definition,
never the nearest neighbour of an incoming tile, so it is never chosen for eviction: isolated entries
are absorbing. Over training the bank therefore drifts from a set distributed like the data (a
**sample**) toward an even grid that merely fills the occupied space (a **cover**). This wrecks the
score in the way that matters most: once the bank is a cover, the nearest-neighbour distance `d(x)`
sits near the cover's spacing almost everywhere and no longer varies with density, so the score
flattens and stops distinguishing common tiles from rare ones — the exact discrimination the module
exists to provide.

Put in estimator terms, the nearest-neighbour readout is a valid density estimate *only when the bank
is a sample* — a set distributed like the data — which requires a **content-independent** maintenance
rule (eviction by age or at random, e.g. retaining the most recent `M` signatures; valid here because
the stream is near-independent at the tile scale, Table 1), under which the survivors are distributed
as the data. Evict-nearest is content-*dependent*: it evicts by position, which is precisely what
drives the bank away from a sample and toward a cover. The distance-calibrated variant is therefore
the baseline whose failure mode motivates the counted-coverage variant.

There are two remedies for this, and they are *not* the two bank variants. (i) A content-independent
maintenance rule restores faithfulness with the existing nearest-neighbour readout, and on the mild
stream measured here is adequate on its own — but it is an alternative maintenance rule, not one of
the two evaluated variants. (ii) The counted-coverage bank (§3.5) reads density from explicit counts,
which stays faithful regardless of maintenance and retains coverage of rare morphology, and remains
valid if the full corpus departs from the mild regime that faithful sampling relies on; this is the
second variant, and the one developed here.

The failure mode is directly observable. Seeding the bank before the representation stabilizes — with
diffuse signatures from an untrained encoder — compounds the two effects: the seed entries are
mutually isolated, are never evicted, and by the end of training occupy 78% of the bank, whose
signatures form a bimodal distribution (a diffuse cluster at the random-projection floor and a
smaller peaky remainder). Every real tile then scores far from this bank, the score is nearly
constant across the stream, and the modulation is inert. Deferring the fill to `T_warm` (§3.3) removes
the seeding component but not the cause: non-eviction of isolated entries is a property of the
evict-nearest rule itself and persists for any admission schedule — which is what the counted-coverage
bank removes.

### 3.5 The counted-coverage bank

**The idea in one paragraph.** A nearest-neighbour estimator has to place its stored points *like the
data* for distances to mean anything, and maintenance keeps ruining that placement. The
counted-coverage bank breaks the dependence by separating the two jobs a single point set was being
asked to do at once — mark *where* the data is, and record *how dense* each part is. It keeps `M`
stored signatures laid down as an even cover of the occupied space (the "where"), and attaches to each stored signature a
running tally of how many tiles land on it (the "how dense"). Density is then read from the tallies,
not from the spacing — so the cover that broke the distance bank is exactly what this bank wants, and
maintenance can no longer corrupt the estimate. This is a kernel density estimator compressed onto a
fixed memory budget: instead of a bump on each of `N` tiles, a bump on each of `M` stored signatures, weighted
by its tally.

The rest of this section makes that precise: the tally, why a stored signature's catchment must be its Voronoi
cell, how the tallies become a density, how the stored signatures are added and dropped, and the defaults.

**Two densities to keep apart.** At any point in signature space there is a *tile density* `p` — how
common that morphology is, the thing we want — and a *stored-signature density* `g` — how many stored signatures happen
to sit there, an artifact of placement. A raw count of tiles near a stored signature mixes the two; the design
below cancels `g` and leaves `p`.

**The tally.** Each stored signature `i` carries two decayed counters and one plain counter:

- `S_i` — its **hit count**: the number of tiles that have landed on it (decayed).
- `E_i` — its **exposure**: the number of *steps it has been alive* (decayed). This is a clock — it
  ticks `+1` every step regardless of hits — not a second hit count.
- `age_i` — its **age**: the same steps-alive, but *undecayed*, used only for the reserve deadline
  below.

The ratio `λ̂_i = S_i / E_i` is the stored signature's **hit-rate**: hits per step of life. Because `E` is a
clock, `λ̂` measures how busy the stored signature's cell is — and, as the next paragraph shows, it equals the
tile density divided by the stored-signature density at that location. Defining exposure as lifetime rather than
as traffic is essential: were `E` a count of tiles routed to the stored signature, it would grow with tile
density exactly as `S` does, their ratio would collapse to a near-constant carrying no density
information, and both the readout and the staleness eviction below would fail.

**Why the catchment is a Voronoi cell.** A tile is a hit for its *nearest* stored signature only, within radius
`s` — never for every stored signature inside `s`. This is the load-bearing choice. A stored signature's catchment is
then its Voronoi cell (capped at `s`), whose volume scales as `1/g` (denser stored signatures ⇒ smaller cells).
So the hit count is `S ∝ p · (1/g) = p/g`: tile density divided by stored-signature density. Counting every
tile within a fixed-radius ball instead would give `S ∝ p` and break the cancellation in the readout
below. At the fill knee stored signatures sit ≈ `s` apart, so the Voronoi cell lies inside the ball of radius
`s` and the radius cap rarely binds.

**Table 1.** Design constants of the signature stream, determined on the baseline model (standard
DINOv2 ViT-B/16, all extensions disabled) over an independent 384,000-tile sample with bootstrap 95%
confidence intervals (§3.7), and the design quantity each fixes. Distance-valued constants (`L`, and
the spacings below) are in L1 units on the signatures `s = R·z`, the metric the bank queries in
(§3.7); the dimensionless constants (`d*`, dynamic range, `b`) are metric-invariant.
| Property | Symbol | Value (95% CI) | Design role |
|---|---|---|---|
| Intrinsic dimension | `d*` | 9.3 [9.1–9.4] | readout viability; cell scale |
| Local-density dynamic range | — | scale-dependent; ≈ 18× at radius `L` | operating regime (moderate) |
| Mode structure | — | connected continuum, no dominant mode | removes the isolated-outlier hazard |
| Tile-level burst factor | `b` | ≈ 1.2 (decays within one batch) | permits exponential forgetting |
| Density correlation length | `L` | 3.34 | sets the readout bandwidth |
| One-off (artifact) rate | `ρ_out` | ≈ 10⁻³ (no floor) | sizes the transient reserve |

Here `d*` is the intrinsic dimension — the number of effective directions the signatures occupy, far
below the ambient 256; the dynamic range is how much tile density varies from the emptiest to the
densest regions; and `L` is the correlation length — the L1 distance over which density changes
appreciably. Two structural findings guide the design. First, the support is a connected continuum of
intrinsic dimension near ten, density varies only moderately across the space (≈ 18× at the operating
scale `L`; §3.7), and there are no isolated clumps — so the absorbing-outlier hazard of §3.4 has
nothing to attach to and the design need not defend against extreme skew. Second, the stream is
near-independent at the tile level (a burst factor `b` near 1.2, and an autocorrelation length `τ_ac`
— the lag over which the stream decorrelates — under one batch; similar tiles do not arrive in long
runs), and the signature distribution drifts slowly after activation — by one correlation length only
over tens of thousands of steps — so stored statistics can be forgotten by simple exponential decay
without going stale relative to the encoder that produced them.

**Placement.** A tile farther than the hit radius `s` from every stored signature seeds a new one, so the
stored signatures tile the occupied support. On the measured continuum the estimation-optimal placement puts
stored signatures in near-proportion to tile density (formally, stored-signature density `∝ p^{d*/(d*+2)}`, the classical
quantization exponent of Graf and Luschgy, 2000 — this motivates the placement but never enters the
code). The hit radius `s` — the single radius used for *both* seeding and hit-counting (Algorithm 2)
— is set at the *fill knee*: the largest radius at which seed-on-miss populates exactly `M` stored
signatures. It is **self-tuned online, not a fixed constant.** The bank keeps a rolling buffer of the
most recent ≈ 60,000 signatures and, every ≈ 500 steps, sweeps it — replaying seed-on-miss from empty
over the buffer across a grid of radii *re-centered on the live signature scale* — to find the largest
radius that still fills the bank to `M`, the knee, which `s` then tracks, lightly smoothed (an EMA over
a few sweeps). Measuring the knee live is not optional; it is forced by the same portability logic §3.7
makes. The knee is a property of the signature *scale*, which the online-trained `R` sets and which
*drifts*: the offline synthetic `R` of §3.7 has a knee of `s ≈ 2.75` (L1 units), but the live
trained-`R` signatures are several times larger, and the measured live knee is correspondingly larger: on the online-trained run it is `s ≈ 13`
at activation (iteration 50k), drifting *downward* to ≈ 11 by iteration 70k as `R`'s effective rank and
scale contract over training. A radius frozen at the offline 2.75 would sit far inside every stored
signature's neighbourhood: almost nothing would count as a hit, the counts would stay ≈ 0, and the bank
would go inert — which is exactly why `s` must be measured on the live signatures rather than fixed. Offline, this
knee exceeds the ideal even-tiling estimate `s_M = (V/M)^{1/d*} ≈ 1.97` (a lower bound) by about 1.4× —
seed-on-miss packs stored signatures at spacing `s` rather than tiling at radius `s` — for an achieved
spacing ≈ `s` ≈ 2.75 (≈ 0.82 `L`), comparable to the correlation length and to the scale over which the
readout pools (a window of order `L`; §3.7). This ratio is a property of the offline geometry and is not
assumed to transfer: on the online-trained run `s` is measured directly, so no downstream quantity
depends on the relationship between `s_M` and the knee.

**Readout: sum the rates, don't average them.** The tile density at a query is estimated as an
*unnormalised* kernel sum over the `j` nearest stored signatures,

```
p̂(x) = Σ_i λ̂_i · K_h(x − b_i)   ( sum over the j nearest stored signatures ) ,
```

where `K_h` is a smooth, radially symmetric, compactly supported weight whose bandwidth `h(x)` is the
distance to the `j`-th nearest stored signature. **Summing rather than averaging is essential**, and the reason
is the `p/g` structure of `λ̂`: a dense region holds more stored signatures, each carrying a *smaller* rate
(because `λ̂ ∝ 1/g`), so **summing** the rates cancels the stored-signature density `g` and leaves the tile
density `p`. **Averaging** (dividing by `Σ_i K_h`) would divide `g` straight back out — it would
instead estimate the mean per-signature rate `∝ p^{1−γ}`, which vanishes at *proportional* placement
(`γ → 1`). The gap is thus placement-dependent, and seed-on-miss produces a near-uniform cover
(`γ ≈ 0`), where `p^{1−γ} = p` and the two nearly coincide: the offline run (§3.7) measures ρ = 0.74
for the normalized readout versus 0.75 for the sum. We use the unnormalised sum anyway because it is
placement-*independent* — it recovers density at *any* `γ`, hence stays valid if placement drifts from
this mild regime. Centering the kernel on the query and using a symmetric profile cancels the leading
(gradient) term of the bias exactly — stored signatures on either side of the query balance — leaving a
curvature-order residual, whereas reading only the nearest stored signature incurs a first-order, spatially
frozen bias of up to half a cell. The effective bandwidth is this pooling window — the radius
enclosing the `j` stored signatures, ≈ `L` (§3.7) — and going finer resolves nothing, since the density field
has no structure below `L`; in `d* ≈ 9.3` those `j` stored signatures already lie within about `1.5×` the local
spacing, so pooling costs almost nothing in resolution.

**Score: the density's percentile, not its value.** The kernel sum `p̂` has no meaningful units — it
could come out 4 or 800 depending on the kernel — so a tile is scored not by `p̂` but by its **rank**:

```
t(x) = F̂( log p̂(x) ) ,
```

the fraction of *recent tiles* whose density was below this one's. "Denser than 75% of recent tiles"
means `t = 0.75`. This is the probability integral transform (PIT), and it buys two things: every
score lands in `[0,1]` with the typical tile near the middle (interpretable), and the score is
invariant to `p̂`'s unknown scale, to error in `d*`, and to exposure normalization (robust). "Recent
tiles" is a rolling window of the last ≈ 20,000 `log p̂` values — about twenty steps' worth — long
enough for the rank to be steady, short enough to move with the stream. A two-moment probit on
`log p̂` (a running mean and variance) is the cheap parametric fallback that needs no window. Before
the counters have filled, the nearest-neighbour readout of §3.3 serves as a cold-start estimator;
after that it is kept only as a consistency probe.

**Adding and dropping stored signatures.** The bank is split into an **established set** of capacity `M` and a
small **reserve** buffer of `≈ reserve` slots on top (total stored `≈ M + reserve`); the established
set holds the same `M` signatures the distance-calibrated bank holds, for a like-for-like comparison.

The established set is filled *directly*: while it holds fewer than `M` stored signatures, a novel tile (farther
than `s` from every stored signature) seeds a new established signature, laying the covering set down first — this
is exactly the configuration validated in §3.7. The reserve and its graduation rule switch on only
once the established set is full, which is also the first moment eviction pressure exists (a one-off
can displace an established stored signature only when the set is at capacity).

In steady state, established stored signatures are managed **least-frequently-used with aging**: the stored signature of
lowest hit-rate `λ̂` is evicted, so a stored signature that stops receiving tiles decays out while active ones
persist. This directly reverses the §3.4 pathology — an isolated stored signature accrues no hits and is the
*first* out, not the last. A novel tile is then admitted to the reserve, initialized empty
(`S = 0, E = 0, age = 0`; the decay-and-age step precedes scoring, so `E ≥ 1` before any stored signature is
read, and `λ̂ = S/(E+ε)`), so a one-off cannot displace an established stored signature before earning hits. Two
rules keep the reserve a bounded buffer rather than a growing one: a newborn **graduates** into the
established set on its first hit (evicting the lowest-`λ̂` established stored signature if the set is full), and
an ungraduated stored signature is **evicted when its age exceeds `T_need`**. Its steady-state occupants are
therefore the non-graduating one-offs, and its size is their arrival rate times their residency,
`reserve ≈ ρ_out · κ · T_need`, where `κ` is the tiles per step (one-offs arrive at `ρ_out · κ` per
step, not `ρ_out`). At `ρ_out ≈ 10⁻³` (the L1 value — isolation is metric-dependent, §3.7),
`κ ≈ 10³`, and `T_need` a few hundred steps, that is ≈ 300 slots — a few percent of `M`. The two
decayed counters `S`, `E` fade with a half-life of a few hundred steps — long relative to the batch
autocorrelation, so recurring morphology builds standing evidence, yet short relative to the drift
horizon, so the estimate tracks the current distribution. All tie-breaks — the nearest-signature
argmin, the lowest-`λ̂` eviction, the oldest-reserve eviction, and the `j`-nearest set — resolve by
lowest index, so the bank stays byte-identical across workers (as in §3.3).

```
Algorithm 2  Counted-coverage bank: update and scoring for one batch X
             (established capacity M; the reserve — a separate buffer of ≈ reserve slots —
              is active only once the established set is full)

  if step < T_warm:  return t(x) = 0 for all x

  for every live stored signature i:                                     # decay, age, expire stale reserve entries
      S_i   ← η · S_i                                          #   S: decayed hits
      E_i   ← η · E_i + 1                                      #   E: decayed lifetime (steps alive)
      age_i ← age_i + 1
      if i in reserve and age_i > T_need:  evict i             #   ungraduated one-off expires

  gather signatures of X across workers  →  X_global

  # ---- score against the pre-update state ----
  if |established| < M:                                        # bank still building its cover
      t(x) ← 0 for all x in X_global                           #   module inactive (as in §3.3)
  else:
      for each x in X_global:
          if median over established of E_i  <  0.5/(1−η):     # counters not yet mature
              t(x) ← nearest-neighbour readout of §3.3 over the established set     # cold-start
          else:
              p̂(x) ← Σ_{i ∈ established, j nearest} (S_i / (E_i + ε)) · K_h(s(x) − b_i)   # unnormalized sum
              t(x) ← F̂( log p̂(x) )                            # percentile among recent tiles (PIT)

  # ---- update (runs every step, including fill) ----
  for each x in X_global:
      i* ← nearest stored signature to s(x)                              # over established ∪ reserve
      if ‖s(x) − b_{i*}‖ ≤ s:                                  # a hit
          S_{i*} ← S_{i*} + 1
          if i* in reserve:                                    # graduate on first hit
              if |established| = M:  evict argmin_i S_i/(E_i+ε) over the established set
              move i* from reserve to established set
      else:                                                    # novel tile
          if |established| < M:                                # fill: seed the cover directly
              add a signature to established at s(x):  b ← s(x), S ← 0, E ← 0, age ← 0
          else:                                                # steady state: stage in the reserve
              if reserve is at capacity:  evict its oldest entry
              add a signature to the reserve at s(x):  b ← s(x), S ← 0, E ← 0, age ← 0

  return { t(x) : x in this worker's rows }
```

Here `η` is the per-step decay factor (set by the half-life), `S_i`/`E_i` the decayed hit count and
lifetime of stored signature `i`, and `p̂` the estimated tile density at the query; the kernel sum runs over the
`j` nearest stored signatures, and `F̂` is the recent-tile percentile of the previous paragraph.

A note on resolution. With `M = 8192` stored signatures the achieved spacing is ≈ 0.82 `L` (set by the hit
radius `s`; §3.7), and the readout pools `j ≈ 64` stored signatures over a window of order `L`, so every readout
is smoothed at scale `L`. Structure finer than `L` is invisible to any bounded summary of this size,
whatever its readout; refining it would need exponentially more memory or a lower-dimensional
signature. This bounds both bank variants equally and is a property of the regime, not of either
policy.

**Implementation defaults.** For a build-ready specification we fix the five choices left abstract
above. *Hit radius:* self-tuned online rather than fixed (§3.5, "Placement"). The bank maintains a rolling
buffer of the most recent ≈ 60,000 signatures and, every ≈ 500 steps, replays seed-on-miss over the
buffer on a radius grid re-centered on the current median inter-signature distance — a geometric grid
spanning ≈ 0.3–2× that scale, refined once across the fill-to-underfill transition — and sets `s` to the
largest radius that still fills the bank to `M`, smoothed by an exponential moving average with a
≈ 5-sweep time constant. Because the radius is re-measured on the live signatures, it tracks the scale
drift of the online-trained `R` that a fixed value cannot (§3.7, "Bank size `M`"). *Kernel:* a triweight profile `k(u) = (1 − u²)³` on `u ≤ 1` (smooth and compactly supported)
with bandwidth `h(x)` = the L1 distance to the `j`-th nearest stored signature (`j = 64`). `j = 64` is the value
in the 32–64 range at which the pooling radius reaches `L` — the 64th-nearest stored signature sits at L1 ≈ 3.5
≈ `L` (§3.7), so the readout pools over exactly one correlation length, the scale below which the
density field has no structure; the readout is otherwise insensitive to the profile (the offline study
of §3.7 used a truncated Gaussian and gives the same recovery). *Half-life:* 250 steps
(`η = 0.5^{1/250} ≈ 0.997`) — long relative to the batch autocorrelation, short relative to the drift
horizon (§3.5); the offline study used 100 steps with no material change. *Cold-start switchover:* the
counted readout activates once the bank is full *and* the median exposure has passed one half-life's
accumulation, `median E ≥ 0.5/(1−η)`. (The exposure ceiling `1/(1−η)` is approached from below but
never reached — a literal `E ≥ 1/(1−η)` test would never fire — so the threshold is half the ceiling,
which is the accumulated `E` at exactly one half-life; a *median* avoids waiting on the perpetually
re-admitted `E = 1` newborns.) Until then the §3.3 readout is used. *PIT reference:* a rolling window
of the most recent ≈ 20,000 `log p̂` values, ranked by binary search (this is what §3.7 validated);
the two-moment probit on `log p̂` (running mean and variance, `t = Φ((log p̂ − m̂)/σ̂)`) is the cheaper
fallback that needs no window.

### 3.6 Modulating the objective

Both banks produce a typicality score `t(x)`, which modulates the image-level DINO cross-entropy one
of two ways. The iBOT objective is untouched; the total loss is
`L = m(x) · CE_DINO + CE_iBOT + λ_sem · CE_iBOT^{sem}`, where the modulation `m(x)` is one of:

**Weighted loss.** The DINO term is scaled per tile by `w(x) = 1 − β · t(x)`, `β ∈ [0,1]`. A typical
tile contributes a smaller-magnitude gradient while the target it is trained toward is unchanged.
This is a direct importance weighting — it flattens the effective sampling distribution over
morphology while leaving each tile's learning signal intact — and is bounded and simple to reason
about.

**Adaptive temperature.** The per-tile student softmax temperature is scaled,
`τ(x) = τ_base · (1 + α · t(x))`, so a typical tile receives a flatter target. This changes not only
the gradient magnitude but the shape of the target, redistributing probability mass across output
prototypes rather than only down-scaling the tile's contribution. It is a stronger intervention that
can actively flatten over-represented modes in the assignment itself, at the cost of coupling the
typicality estimate more tightly into the representation geometry.

The two banks (§3.3, §3.5) and the two modulations thus define four configurations. Their comparison
is the subject of Section 4; the sensitivity of the leading configuration to its principal
hyperparameters (`β` or `α`, the warmup `T_warm`, and, for the counted-coverage bank, the decay
half-life and hit radius) is studied thereafter.

### 3.7 Empirical determination of the constants, and offline validation

The counted-coverage design rests on two empirical claims: that the constants of Table 1 are
properties of the model rather than of one sample, and that the bank, run end to end, actually
recovers tile redundancy. We establish both by inference-only study on the baseline model — a
convergence analysis of the constants, and an offline run of the bank on cached signatures — with no
retraining. As the reference notion of density here we use an offline `k`-nearest-neighbour *density*
on the full sample (how many tiles sit near each tile), computed once and treated as ground truth;
"density" below always means this reference.

**Determination of the constants.** The constants of Table 1 were determined on the signatures
`s = R·z` under the L1 metric (the bank's metric; see the provenance note), on an independent
384,000-tile sample (a disjoint dataloader seed, same tap and checkpoint), each with a bootstrap 95%
confidence interval and verified stable across subsample size. The estimators are standard: intrinsic
dimension by the two-nearest-neighbour ratio method (Facco et al., 2017), cross-checked against the
covariance participation ratio; the density range from `k`-nearest-neighbour density; the burst factor
from the signature autocorrelation and same-slide run-lengths in arrival order; the correlation length
from the log-density field; and the one-off rate from the isolation rate versus sample size. Two notes
on the estimators. The correlation length is `L ≈ 3.34` (L1 units) from a neighbour-gradient
estimator; a random-pair estimator is unstable in this dimension. The dynamic range is not a fixed
constant — both p99/p1 and the variance of log-density grow with sample size, because a
`k`-nearest-neighbour estimate's bandwidth shrinks as `N` grows and resolves finer structure. The
converged, operationally meaningful quantity is the skew at a *fixed* bandwidth equal to the bank's
resolution: at radius ≈ `L` the log-density variance is 1.33 and the 90/10 density ratio is ≈ 18
(Table 1), stable across sample size. The design does not depend on pinning the dynamic range, because
the rank/PIT readout (§3.5) is invariant to any monotone rescaling of density.

The same study fixes the resolution scales (L1 units on `s`). The ideal-tiling estimate
`s_M = (V/M)^{1/d*} ≈ 1.97` (≈ 0.59 `L`) is a lower bound; because seed-on-miss packs stored signatures at the
hit radius, the achieved spacing is ≈ 2.75 (≈ 0.82 `L`, the ~1.4× gap of §3.5). The readout pools
`j ≈ 64` stored signatures over a window of order `L` (measured 64th-NN ≈ 3.5), so every readout is smoothed at
scale `L` and structure below `L` is unresolvable at this memory budget; this is the ceiling of §3.8,
set by the pooling window, not by the stored-signature spacing.

**Offline validation of the bank.** We ran the counted-coverage bank (Algorithm 2, with the corrected
lifetime exposure) over the 384,000 signatures `s = R·z` in stream order, under L1, and compared its
score, per tile, against the offline `k`-nearest-neighbour density on the full sample — the best
available proxy for ground-truth redundancy. With the hit radius set at the fill knee (`s ≈ 2.75` in
L1 units), the online score recovers the offline density with **Spearman ρ = 0.75**, on a bounded
memory holding 8,192 of 384,000 tiles; this is close to the ceiling the resolution allows, since the
score is smoothed at scale `L` and correlated against a finer reference. The per-signature rate `λ̂`
tracks the density at its own location with ρ = 0.67, confirming that the counting itself — not merely
the kernel smoothing — carries the signal. The bank reaches steady state (8,192 stored signatures, modest
turnover). This run exercises the core mechanism — direct seeding of the established set, Voronoi
hit-counting, lifetime exposure, unnormalized kernel sum, PIT scoring, and least-frequently-used
eviction — at a full established set of 8,192 stored signatures. Its direct seeding is exactly the fill phase of
Algorithm 2; what it does not exercise is the steady-state reserve and its graduation rule (§3.5),
which activate only once the established set is full and serve only to shield established stored signatures from
one-off tiles. Newborns were initialized `S = E = 1` rather than the `S = E = 0` of §3.5; the offset
decays away before the counters mature, so neither difference bears on the recovery reported here.

**Provenance of `R`.** The baseline run had typicality disabled, so `L_R` never ran and the checkpoint
contains no `R`. We therefore constructed a synthetic `R` from the frozen baseline's `out_dim =
65,536` output prototypes: a column-pivoted QR on the unit-normed prototype matrix selects the 256
most linearly independent directions, giving `R ∈ ℝ^{256×256}` (condition number 25, effective rank
202) — a near-orthonormal frame that stands in for the `L_nn`-aligned `R` a run would grow (`L_nn`
pulls the representative prototypes toward exactly these output prototypes). The cached signatures are
`s = R·z`, and the entire study above — constants and validation — is computed on `s` under L1, the
bank's metric. (An earlier iteration of this study ran on the bare bottleneck `z` under L2; the tight
isometry ratio `‖s₁−s₂‖₂ / ‖z₁−z₂‖₂ = 1.00 ± 0.03` holds *despite* `R`'s condition number 25 because
the data `z` occupies `R`'s well-conditioned top directions — its ~9.5-dimensional manifold sits
within them, leaving the ≈ 54 near-null directions of the rank-202 frame essentially unpopulated, so
`R` acts near-isometrically on the differences that matter. The two metrics therefore agree on the
dimensionless constants, the distance-valued constants rescale into L1 units by the common factor
≈ 12.7 with all scale *ratios* preserved, and the ρ = 0.75 recovery reproduces under both.) One caveat
remains, stated as unmeasured: this synthetic `R` has effective rank 202, whereas an `L_R`-trained `R`
collapses to effective rank ≈ 44 (§3.2), and the study is on the frozen baseline, not on the
online-`L_R` signatures a training run produces. Their agreement was not measured; we assert no
equivalence.

**Table 2.** Offline bank on 384k signatures `s = R·z` (L1): recovery of the offline density vs. the
hit radius `s` (L1 units; `s_M ≈ 1.97`).
| hit radius `s` | Spearman(`t`, density) | Spearman(`λ̂`, density) | stored signatures | admits/step |
|---|---|---|---|---|
| 1.97 | +0.24 | −0.10 | 8192 | 476 |
| 2.36 | +0.73 | +0.28 | 8192 | 174 |
| **2.75** | **+0.75** | **+0.67** | 8192 | 31 |
| 3.15 | +0.70 | +0.91 | 6684 (underfilled) | ~0 |
| 3.54 | +0.65 | +0.93 | 3178 (underfilled) | ~0 |

The two Spearman columns move in *opposite* directions as `s` grows: the per-signature rate correlation
`Spearman(λ̂, density)` rises (0.28 → 0.93) while the readout correlation `Spearman(t, density)` falls
past the knee (0.75 → 0.65). This is the underfill trade-off — a larger hit radius gives each stored signature a
bigger catchment and a cleaner per-signature estimate, but seeds fewer stored signatures (the bank drops to 6,684
then 3,178), leaving it too sparse to cover the space, so the pooled readout coarsens. The knee
`s = 2.75` is where both are jointly good: a full bank (8,192) and the best readout recovery.

Three controls probe the two load-bearing choices and the hit radius. The one that isolates a design
decision is the **exposure definition**: replacing the lifetime exposure with the discarded
traffic-count exposure (§3.5), everything else at the knee `s = 2.75`, collapses the per-signature rate
to a constant (coefficient of variation 0.00, versus 1.21 for the corrected form) — the counters then
carry no density, and recovery falls to ρ = 0.53, the residual coverage-only signal of §3.4,
forfeiting the counting contribution that lifts the corrected readout to 0.75. So exposure must be
lifetime, not traffic. The **normalization** control is weaker than we first reported: the normalized
(Nadaraya–Watson) readout recovers ρ = 0.74, nearly matching the unnormalized sum (0.75). As §3.5
explains, the two coincide at the near-uniform placement seed-on-miss produces (`γ ≈ 0`); the sum's
advantage is placement-*independence*, not a measurable gap here, so this run establishes that the sum
is *safe*, not that it is *necessary* — the necessity is theoretical, showing only under proportional
placement. Finally the **hit radius**: setting it below the fill knee (`s = 1.57`) drives constant
admission and eviction that prevent the counters from stabilizing, collapsing recovery to ρ ≈ 0;
setting `s` at the knee restores it. The single parameter that must be set with care is therefore the
hit radius, at the fill knee (≈ 2.75 in L1 units, where admits per step collapse), which exceeds the
ideal-tiling estimate `s_M ≈ 1.97` by ~1.4× and should be set empirically. This offline run is the
prerequisite we place before any training integration (§3.8): it exercises the full mechanism on real
signatures at low cost, and it is where a readout-breaking error surfaces as a flat, uncorrelated
score — as the traffic-exposure control and the mis-set-radius run both illustrate.

**Portability: which changes invalidate the constants.** The constants above are properties of the
*signature distribution* — of the composition (encoder × prototypes × data stream) — so it matters
which configuration changes leave that distribution intact and which do not. Two curator-internal
knobs leave the *dimensionless* constants unchanged; both couple only to the hit radius `s`.

*Bank size `M`.* It sets the stored-signature spacing `s_M ∝ M^{−1/d*}` and, through the pooling count, the
readout window. Because `d* ≈ 9.3` this is very weak: over `M ∈ {4096, 8192, 16384}` the fill knee
moves only `s ≈ {2.97, 2.75, 2.56}` (≈ ±8% per 2×), and reaching the `L` ceiling or the
pooling-locality floor (`M ≫ j = 64`) takes order-of-magnitude changes. Recovery is essentially flat
over this band (ρ ≈ 0.72–0.75; Table 2 read as an effective-`M` sweep). So changing `M` is a config
change with `s` the only coupled parameter. `s` **is self-tuned online** (§3.5, Placement): every
≈ 500 steps the bank replays seed-on-miss over a rolling ≈ 60,000-signature buffer, on a grid
re-centered on the live signature scale, and sets `s` to the largest radius that still fills the bank
to `M` — the *underfill edge* — lightly smoothed. Targeting that edge, and *not* simply holding
`|B| ≈ M`, is what makes it correct: Table 2 shows the bank is full across `s ∈ [1.57, 2.75]`, so
`|B| = M` alone is a flat signal that would admit the churning `s = 1.57` (ρ ≈ 0) as readily as the
knee `s = 2.75` (ρ = 0.75); the sweep instead pushes `s` up until `|B|` just begins to drop. Because
the edge is re-measured continuously it also absorbs the scale *drift* of the online-trained `R` — the
reason a fixed radius fails — so a change of `M` needs no manual re-tune. The scratch reserve, half-life (in steps), and pooling `j` (a count) all carry over
untouched, and no re-characterization of §3.7 is needed.

*Prototype count `K'`.* Changing it is also a config change — `L_R` trains any `K'`, no structural code
change — but it couples more strongly and has a hard floor. Because the L1 distance sums over `K'`
coordinates, the distance scale runs roughly *linearly* with `K'`: halving `K'` roughly halves `L`,
`s_M`, and the fill knee `s`, so `s` needs a ~proportional re-tune (again absorbed by a self-tuning
`s`). The dimensionless constants stay intact provided `K' ≥` the effective prototype rank (measured
≈ 44): the signature is then a rotation (`K' = 256`) or a projection that retains the occupied
subspace, and the intrinsic dimension every `d*`-dependent formula rests on is preserved; `K'` above
256 is impossible (256 is the orthogonal-frame ceiling, §3.2), and smaller `K'` is cheaper (smaller
`R`, smaller `cdist`). Reducing `K'` below the effective rank projects out real structure and does
change `d*` and everything downstream — no longer a config change, but a re-characterization. The
relevant rank is the ≈ 44 to which an *online-`L_R`* `R` collapses (measured on an `L_R`-active run;
§3.2) — not the rank-202 synthetic `R` the offline study used as a stand-in. Because the intrinsic
dimension `d* ≈ 9.3` lies below *both*, the **dimensionless** constants (`d*`, dynamic range, `b`) are
the same at either rank; that invariance is why any `K'` in roughly `[64, 256]` gives the same
dimensionless regime, and an undercomplete choice near 64 is safe and cheaper. The **distance-valued**
constants (`L`, `s`, `s_M`), being L1 measurements and L1 not being rotation-invariant, can differ
between the two `R`'s and must be re-measured on the trained `R` — which the rule below already forces,
since a change of `R` is a change of the signature distribution. This is the same distinction the
provenance note draws: agreement on the dimensionless constants, no assumed equivalence on the
distance scales.

Everything else that alters the learned representation or the stream ordering shifts the constants:

| change | constants affected | note |
|---|---|---|
| backbone scale / architecture / SSL recipe | `d*`, dynamic range, `L`, effective rank | the largest effect; measured at ViT-B, so a ViT-L model — the scale much of the pathology-FM literature uses — will have a different manifold and must be re-measured before porting |
| training data mix / tissue diversity / QC | dynamic range, `ρ_out`, mode structure | the single-slice caveat of §3.8; a broader or less-filtered corpus raises the skew and the artifact rate |
| dataloader: interleave, shard size, batch size, source mixing | `b`, `τ_ac` | burstiness and autocorrelation are properties of the stream *order*; reducing the interleave raises `b` and lengthens `τ_ac`, which the decay half-life must then accommodate |
| magnification / tile size / augmentation | `d*`, dynamic range (secondary) | the characterization uses the clean 448→224 tap; heavy train-time augmentation shifts the signature distribution |
| `R`-training weights (`L_cov`, prototype LR) | effective rank → the `K'` floor | these set how orthogonal and spread the prototypes are |

The practical rule: **`M` and `K' (≥ effective rank)` may be swept freely, but a change of backbone,
data, or dataloader requires re-running the characterization of §3.7 before the design constants — and
the parameters `s`, `L`, and the half-life derived from them — can be trusted.** Re-measurement is
inexpensive (inference-only on cached signatures), and the offline prototype is itself the guard: if
the constants have drifted under a configuration change, the density-recovery correlation falls, which
flags the need to re-measure before committing a training run.

### 3.8 Limitations

Three limitations bound the method. First, as noted in §3.5, the estimate is resolution-limited: with
a fixed memory budget over a support of intrinsic dimension near ten, structure finer than the
correlation length is invisible to any bounded summary. Second, the empirical constants of Table 1
were measured on a single, morphologically homogeneous slice of the stream; a substantially more
skewed corpus could shift the operating regime, though the counted readout is by construction
insensitive to the exact signature-placement exponent. Third, the typicality estimate modulates the
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
