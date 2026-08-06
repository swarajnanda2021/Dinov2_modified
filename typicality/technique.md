# Typicality Dampening

*Method section (manuscript draft, readability revision). This document describes the
typicality-dampening module: the morphology-signature representation it operates on, the
counted-coverage memory bank that turns a signature into a per-tile density, and the weighted-loss
modulation that density drives. Section 4 evaluates that configuration at two tilt settings.
Except where explicitly attributed to the online-trained run (§3.5), quantitative statements are inference-only
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
sits in sparse ones. The typicality signal is simply a tile's local density: high density → typical →
dampen; low density → rare → preserve. The two bank variants are two standard ways of estimating that
density, and naming them this way is the cleanest way to see how they relate:

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
2. **Local density.** `s(x)` is turned into a local density `p̂(x)` by the counted-coverage bank —
   high when the tile lands in a dense neighbourhood, low when it lands in a sparse one (§3.5).
3. **Modulation.** `p̂(x)` modulates that tile's DINO objective through a reciprocal-density loss
   weight `w = 1/(p̂ + c)^a` (§3.6).

The signature (§3.2) and the modulation (§3.6) are common to the whole method; the bank is the
counted-coverage bank (§3.5). §3.3–§3.4 are the estimator analysis that motivates it: a
nearest-neighbour distance readout is a faithful density estimate only if the stored set is a
representative *sample*, novelty-driven maintenance instead drives that set toward an even *cover*,
and so the counted-coverage bank reads density from explicit hit counts rather than from distance.
The distance readout is no longer used at runtime — §3.3–§3.4 keep it only as the estimator analysis
that motivates the counted-coverage design — and cold start applies no modulation rather than falling
back to it (§3.5). The counted-coverage bank with the weighted-loss modulation is the configuration
evaluated in Section 4, at two tilt settings.

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
| `p̂(x)` | estimated tile density at `x` (the unnormalised fixed-radius kernel sum) |
| `K, R_rad` | pooling kernel (triweight) and the fixed readout radius `R_rad = radius_mult · s` |
| `p_ref` | reference density scale (EMA of the batch-median `p̂`); fixes the weight floor `c = c_frac · p_ref` |
| `w(x), a` | per-tile loss weight `w = 1/(p̂ + c)^a` and its tilt exponent `a` |
| `j` | resolution diagnostic: median count of stored signatures within `L` (no longer pools the readout) |
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
  distribution, yielding a score uniform on `[0,1]`. An earlier version of the readout scored a tile
  this way; it was dropped (§3.5) because a percentile keeps only the *ordering* and discards the
  *magnitude* of the density. The current weight instead references the density's live scale directly
  through `p_ref`.
- **Kernel and readout radius (`K`, `R_rad`).** A smooth, radially symmetric, compactly supported
  weight that falls off with distance and vanishes beyond the fixed readout radius `R_rad = radius_mult
  · s`. The readout is a kernel-weighted sum over the stored signatures within `R_rad`.
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
| Density correlation length | `L` | 3.34 | resolution diagnostic (§3.5, under-resolution) |
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

**Readout: sum the rates over a fixed radius.** The tile density at a query is estimated as an
*unnormalised* kernel sum over the stored signatures inside a **fixed radius** `R_rad = radius_mult · s`,

```
p̂(x) = Σ_i λ̂_i · K( ‖x − b_i‖ / R_rad )   ( sum over stored signatures with ‖x − b_i‖ ≤ R_rad ) ,
```

where `K` is the smooth, radially symmetric, compactly supported triweight, so a signature past `R_rad`
contributes exactly zero — the radius test is implicit and no sorting is needed. **Summing rather than
averaging is essential**, and the reason is the `p/g` structure of `λ̂`: a dense region holds more stored
signatures, each carrying a *smaller* rate (because `λ̂ ∝ 1/g`), so **summing** the rates cancels the
stored-signature density `g` and leaves the tile density `p`; **averaging** (dividing by `Σ_i K`) would
divide `g` straight back out — it would instead estimate the mean per-signature rate `∝ p^{1−γ}`, which
vanishes at *proportional* placement (`γ → 1`). That cancellation is valid only over a domain of **fixed
volume** — and the radius is exactly what supplies one. The argument was always right; the domain was
wrong. An earlier version of this readout summed over the `j` *nearest* stored signatures and set the
kernel bandwidth to the `j`-th nearest distance, `h(x) = d_j(x)`. That makes the domain volume
query-adaptive — it shrinks in dense regions and grows in sparse ones — and dividing every distance by
`d_j` cancels the local scale *exactly*: the sum comes out **bit-for-bit invariant to how crowded the
neighbourhood is** (verified directly on cached signatures; §3.7, "Measured basis"), which is precisely
what a density readout must *not* be. On a near-uniform cover the `j` nearest signatures sit at almost
one distance, so `d_i/d_j ≈ 1` for all of them and the triweight `(1 − u²)³` annihilates every term —
its total kernel weight was measured at 0.158 out of a possible `j`. The fixed radius removes the
query-adaptive normalisation and lets the crowding show. Its resolution is set by `R_rad` — a fixed
multiple `radius_mult = 1.5` of the cover scale `s` (measured `spacing / s ≈ 1.02`, so `s` stands in for
the stored-signature spacing), chosen at the value with the best measured density agreement (§3.7).
Centering the kernel on the query and using a symmetric profile cancels the leading (gradient) term of
the bias exactly — stored signatures on either side balance — leaving a curvature-order residual,
whereas reading only the nearest stored signature incurs a first-order, spatially frozen bias of up to
half a cell.

**Resolution diagnostic (self-tuned `j`, `L`).** `j` no longer feeds the readout — the radius is fixed —
but the machinery that measured it is retained as a **resolution diagnostic**: it reports whether the
fixed radius is wide enough relative to the structure it must resolve. On the radius-sweep cadence,
Algorithm 3 estimates the density correlation length `L` by a **variogram with a nugget** — fitting
`γ(r) = c₀ + c₁(1 − exp(−r/L))` to the squared `λ̂` differences of established-signature pairs binned by
separation `r`, over signatures mature enough to be low-noise — the nugget `c₀` absorbing the counting
noise that otherwise biases `L` down. It then reports `j` (the median number of stored signatures within
`L` of a query, clamped by a variance target — smallest `j` for a target relative SE on `log p̂`,
default `0.05`, `j ≤ j_max = 256`) and, crucially, **under-resolution**: if the median stored-signature
spacing exceeds `L` (equivalently the variogram is already at its sill at the smallest lag), the design
is coarser than the structure — a property of the memory budget, not a tuning failure — and the bank
flags `underresolved` and surfaces `spacing/L`. These are diagnostics on the fixed-radius readout, not
inputs to it.

```
Algorithm 3  Resolution diagnostic: L, j, under-resolution   (on the radius-sweep cadence; NOT the readout bandwidth)
  fit  γ(r) = c₀ + c₁(1 − exp(−r/L))  to binned pair semivariances of λ̂ over mature signatures
       → L         (nugget c₀ absorbs counting noise; subsample pairs, do not form all M²)
  j_count ← median over queries of  #{ i : ‖x − b_i‖₁ ≤ L }
  j_min   ← smallest j with relative SE(log p̂) ≤ rse_target,  from Var(λ̂) = λ̂ (1−η)/(1+η)
  if median-spacing > L  or  variogram flat at the smallest lag:        # under-resolved
        underresolved ← 1 ;  j ← EMA(j, j_min)                          #   report the variance-target j
  else  underresolved ← 0 ;  j ← EMA(j, clamp(j_count, j_min, j_max))
```

**Score: the density itself, weighted by its reciprocal.** The kernel sum `p̂` is now used *as the
density*, not converted to a rank. An earlier version scored a tile by the percentile of `log p̂` among
recent tiles (the probability integral transform); it is dropped because it **discards the magnitude of
the density** — a percentile is uniform on `[0,1]` by construction, so it keeps only the ordering and
throws away *how much* denser one tile is than another, which is exactly the signal the fixed-radius sum
was built to recover. Instead the per-tile weight is a smooth reciprocal of the absolute density,

```
w(x) = 1 / ( p̂(x) + c )^a ,   c = c_frac · p_ref ,
```

with `a` the tilt exponent and `p_ref` a slow EMA of the batch-median `p̂` (the *reference scale*). The
floor `c` does two things. It keeps the weight **finite where `p̂ = 0`** (a tile with no stored signature
inside `R_rad` gets `w = 1/c^a`, not a division by zero), so no clipping is needed. And it makes the
weight **scale-invariant precisely when `c` tracks the scale of `p̂`**: multiply every `p̂` and `p_ref`
by the same constant `k` and `w = 1/(k p̂ + k c)^a = k^{−a} w`, a common factor that the
weight-normalised loss (§3.6) divides straight out. `p̂` has no natural units — it comes out 0.001 or 800
depending on the kernel and the signature scale, and that scale *drifts* as `R` trains — so the
reference cannot be a fixed constant; it must be measured live, which is why `p_ref` is an EMA of the
running median rather than a hyperparameter. `p_ref` is initialised from the first matured batch's
median (never zero) and checkpointed, so it survives preemption. The counter half-life still fixes the
estimator's floor: the effective sample size `n_eff = (1 + η)/(1 − η)` (≈ 721 at half-life 250) is set
by the decay half-life alone, independent of stream length. Before the counters have filled the module
applies **no modulation** at all (the §3.3 nearest-neighbour readout has no `p̂` to weight by); it
activates only once the counters mature.

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
established set once it has accrued **two corroborating hits** (evicting the lowest-`λ̂` established
stored signature if the set is full), and an ungraduated stored signature is **evicted when its age
exceeds `T_need`**. Two hits rather than one is a stability requirement, not a tuning choice. A
candidate promoted on a single hit displaces an established stored signature whose rate is estimated
over hundreds of steps of exposure — one observation set against many. Selecting the genuinely lowest-
rate signature by `argmin` over `M` noisy estimates needs of order `2 ln M / δ²` accumulated counts to
resolve rate from noise; at the observed bottom-decile rate that is several thousand steps of exposure,
whereas one-hit graduation gave a mean slot occupancy several times short of it — so the evicted
signatures were under-sampled, not genuinely quiet (their measured rate sat near the bottom decile).
Its steady-state occupants are the non-graduating one-offs, and its size is their arrival rate times
their residency, `reserve ≈ ρ_out · κ · T_need`, where `κ` is the tiles per step. At `ρ_out ≈ 10⁻³`
(the L1 value — isolation is metric-dependent, §3.7), `κ ≈ 10³`, and `T_need` a few hundred steps,
that is a few hundred slots — a few percent of `M`; because the two-hit rule cuts the graduation rate,
the reserve default is raised to ≈ 550 so occupancy lands just past the stability bound. The two
decayed counters `S`, `E` fade with a half-life of a few hundred steps — long relative to the batch
autocorrelation, so recurring morphology builds standing evidence, yet short relative to the drift
horizon, so the estimate tracks the current distribution. All tie-breaks — the nearest-signature
argmin, the lowest-`λ̂` eviction, the oldest-reserve eviction, and the `j`-nearest set — resolve by
lowest index, so the bank stays byte-identical across workers (as in §3.3).

```
Algorithm 2  Counted-coverage bank: update and scoring for one batch X
             (established capacity M; the reserve — a separate buffer of ≈ reserve slots —
              is active only once the established set is full)

  if step < T_warm:  return p̂(x) = ⊥ for all x        # inactive: no modulation

  for every live stored signature i:                                     # decay, age, expire stale reserve entries
      S_i   ← η · S_i                                          #   S: decayed hits
      E_i   ← η · E_i + 1                                      #   E: decayed lifetime (steps alive)
      age_i ← age_i + 1
      if i in reserve and age_i > T_need:  evict i             #   ungraduated one-off expires

  gather signatures of X across workers  →  X_global

  # ---- score against the pre-update state ----
  if |established| < M:                                        # bank still building its cover
      p̂(x) ← ⊥ for all x in X_global                          #   module inactive: no modulation
  else:
      for each x in X_global:
          if median over established of E_i  <  0.5/(1−η):     # counters not yet mature
              p̂(x) ← ⊥                                        #   cold-start: no modulation (§3.3 has no p̂)
          else:
              R_rad ← radius_mult · s
              p̂(x) ← Σ_{i ∈ established, ‖s(x) − b_i‖ ≤ R_rad} (S_i / (E_i + ε)) · K(‖s(x) − b_i‖ / R_rad)
              p_ref ← 0.999 · p_ref + 0.001 · median_x p̂(x)   # reference scale for the weight (§3.6)

  # ---- update (runs every step, including fill) ----
  for each x in X_global:
      i* ← nearest stored signature to s(x)                              # over established ∪ reserve
      if ‖s(x) − b_{i*}‖ ≤ s:                                  # a hit
          S_{i*} ← S_{i*} + 1
          if i* in reserve:                                    # count corroborating hits
              hits_{i*} ← hits_{i*} + 1
              if hits_{i*} ≥ graduation_hits:                  #   graduate on the k-th hit (default 2)
                  if |established| = M:  evict argmin_i S_i/(E_i+ε) over the established set
                  move i* from reserve to established set
      else:                                                    # novel tile
          if |established| < M:                                # fill: seed the cover directly
              add a signature to established at s(x):  b ← s(x), S ← 0, E ← 0, age ← 0
          else:                                                # steady state: stage in the reserve
              if reserve is at capacity:  evict its oldest entry
              add a signature to the reserve at s(x):  b ← s(x), S ← 0, E ← 0, age ← 0

  return { p̂(x) : x in this worker's rows }        # ⊥ during fill / cold-start → the trainer applies no weight
```

Here `η` is the per-step decay factor (set by the half-life), `S_i`/`E_i` the decayed hit count and
lifetime of stored signature `i`, and `p̂` the estimated tile density at the query; the kernel sum runs over
the stored signatures within the fixed radius `R_rad = radius_mult · s`. The per-tile weight `w(x)` is
formed downstream from `p̂` and `p_ref` (§3.6).

A note on resolution. With `M = 8192` stored signatures the achieved spacing is ≈ 0.82 `L` (set by the hit
radius `s`; §3.7), and the readout sums over the stored signatures inside `R_rad = 1.5 s ≈ 1.5 ×` the
spacing, so every readout is smoothed at scale `R_rad`. Structure finer than that is invisible to any
bounded summary of this size, whatever its readout; refining it would need exponentially more memory or a
lower-dimensional signature. This is a property of the regime, not of the policy.

**Implementation defaults.** For a build-ready specification we fix the choices left abstract
above. *Hit radius:* self-tuned online rather than fixed (§3.5, "Placement"). The bank maintains a rolling
buffer of the most recent ≈ 60,000 signatures and, every ≈ 500 steps, replays seed-on-miss over the
buffer on a radius grid re-centered on the current median inter-signature distance — a geometric grid
spanning ≈ 0.3–2× that scale, refined once across the fill-to-underfill transition — and sets `s` to the
largest radius that still fills the bank to `M`, smoothed by an exponential moving average with a
≈ 5-sweep time constant. Because the radius is re-measured on the live signatures, it tracks the scale
drift of the online-trained `R` that a fixed value cannot (§3.7, "Bank size `M`"). *Kernel and readout
radius:* a triweight profile `k(u) = (1 − u²)³` on `u ≤ 1` (smooth and compactly supported), summed over
the stored signatures within `R_rad = radius_mult · s`, `radius_mult = 1.5`. The radius is parameterised
as a multiple of the **self-tuned hit radius `s`**, not of the anchor spacing directly, because `s` is
what the bank measures online and therefore what tracks the drifting signature scale (§3.5,
"Placement"). The offline sweep, by contrast, was run in units of median anchor spacing, and the
fixed-radius density agreed best with the `k`-NN reference at 1.5× spacing (correlation 0.85, and the
first zero empty-neighbourhood fraction; §3.7, "Measured basis"). The two units differ only by the
measured `spacing / s ≈ 1.02`, so `radius_mult = 1.5` places the radius at ≈ 1.47× spacing rather than
1.50× — an immaterial gap, since 1.25× spacing already measured 0.82. `j` is no longer a readout
parameter (it is a resolution diagnostic; §3.5). *Weight:* `w = 1/(p̂ + c)^a`,
`c = c_frac · p_ref`, with `c_frac = 0.25` and the tilt `a` the one deliberately swept knob — `a = 0.5`
and `a = 1.0` give measured gradient tilts of 2.77× and 7.68× (§3.7, "Measured basis"). *Half-life:* 250 steps
(`η = 0.5^{1/250} ≈ 0.997`) — long relative to the batch autocorrelation, short relative to the drift
horizon (§3.5); the offline study used 100 steps with no material change. This is nonetheless the
method's least-justified constant — hand-picked where every other parameter is self-tuned, budgeted, or
measured, and alone in fixing the estimator's variance floor (§3.8). *Cold-start switchover:* the
counted readout activates once the bank is full *and* the median exposure has passed one half-life's
accumulation, `median E ≥ 0.5/(1−η)`. (The exposure ceiling `1/(1−η)` is approached from below but
never reached — a literal `E ≥ 1/(1−η)` test would never fire — so the threshold is half the ceiling,
which is the accumulated `E` at exactly one half-life; a *median* avoids waiting on the perpetually
re-admitted `E = 1` newborns.) Until then no modulation is applied. *Reference scale:* `p_ref`, an EMA
`p_ref ← 0.999 p_ref + 0.001 · median_x p̂(x)` of the batch-median density, computed on the gathered
batch (so it is identical across workers by construction) and initialised from the first matured batch's
median — never from zero, or `c = 0` and the weight would diverge on a tile with no signature in range.
It is a registered buffer, so it checkpoints and survives mid-run preemption.

### 3.6 Modulating the objective

The counted-coverage bank produces a per-tile density `p̂(x)`, which modulates the image-level DINO
cross-entropy through a per-tile loss weight. The iBOT objective is untouched; the total loss is
`L = w(x) · CE_DINO + CE_iBOT + λ_sem · CE_iBOT^{sem}`, with the weight `w(x)` set from the density as
follows.

**Weighted loss.** The DINO term is scaled per tile by the reciprocal-density weight
`w(x) = 1/(p̂(x) + c)^a` (§3.5), `c = c_frac · p_ref`, and applied as a **weight-normalised mean**,
`Σ w(x)·CE(x) / Σ w(x)`. Because the weight sum divides out, the batch-level gradient magnitude is
unchanged and the modulation acts entirely through the *relative* weights — a uniform weight of any
value is identical to no weighting. That normalisation is also what makes the reciprocal-density weight
well-posed despite `p̂` having no fixed units: scaling every `p̂` and `p_ref` by a common factor scales
every weight by `k^{−a}`, which cancels in the ratio, so only the *spread* of `w` across the batch acts
(§3.5, "Score"). The tilt `a` sets that spread — larger `a` puts more relative weight on the rarest
tiles (measured rarest:commonest gradient-mass ratio 2.77× at `a = 0.5`, 7.68× at `a = 1.0`; §3.7,
"Measured basis") — while the mean weight is absorbed by the normalisation, so the modulation
redistributes emphasis *within* each batch rather than rescaling the objective, avoiding a silent change
to the effective learning rate. This is a direct importance weighting — it flattens the effective
sampling distribution over morphology while leaving each tile's learning signal intact — and is simple
to reason about.

The counted-coverage bank (§3.5) with the weighted-loss modulation, at two tilt settings `a`, is the
configuration carried forward; its comparison is the subject of Section 4, and the sensitivity of the
leading configuration to its principal hyperparameters (`a`, the warmup `T_warm`, and the decay
half-life and hit radius) is studied thereafter.

### 3.7 Empirical determination of the constants, and offline validation

> **Historical note (readout).** The *readout-recovery* figures in this section — the Spearman
> `ρ = 0.75`, Table 2, and the exposure/normalization controls — were produced with the earlier
> **fixed-count percentile** readout (summing the `j` nearest signatures, scoring by the PIT) on an
> **offline synthetic `R`**. That readout has been replaced by the fixed-radius absolute readout of
> §3.5, and the synthetic-`R` artifacts are not reproducible from anything on disk, so `ρ = 0.75`
> cannot be re-derived and is **not a current claim** about the shipped method. It is retained only as
> the record of how the constants were established. The measured basis for the current readout is the
> "Measured basis for the fixed-radius readout" subsection at the end of this section. The *structural*
> constants of Table 1 (intrinsic dimension `d*`, correlation length `L`, dynamic range, burst factor,
> one-off rate) are properties of the signature stream and are unaffected by the readout change.

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
the fixed-radius weight (§3.5) references the density's live scale (`c = c_frac · p_ref`) and so is
invariant to a global rescaling of density.

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
L1 units), the online score *recovered* the offline density with **Spearman ρ = 0.75**, on a bounded
memory holding 8,192 of 384,000 tiles (historical, under the fixed-count percentile readout — see the
note at the head of this section; superseded by the fixed-radius measurements below); this was close to
the ceiling the resolution allows, since the score is smoothed at scale `L` and correlated against a
finer reference. The per-signature rate `λ̂`
tracks the density at its own location with ρ = 0.67, confirming that the counting itself — not merely
the kernel smoothing — carries the signal. The bank reaches steady state (8,192 stored signatures, modest
turnover). This run exercises the core mechanism — direct seeding of the established set, Voronoi
hit-counting, lifetime exposure, unnormalized kernel sum, PIT scoring, and least-frequently-used
eviction — at a full established set of 8,192 stored signatures. Its direct seeding is exactly the fill phase of
Algorithm 2; what it does not exercise is the steady-state reserve and its graduation rule (§3.5),
which activate only once the established set is full and serve only to shield established stored signatures from
one-off tiles. Newborns were initialized `S = E = 1` rather than the `S = E = 0` of §3.5; the offset
decays away before the counters mature, so neither difference bears on the recovery reported here.

**Scope of these numbers.** The recovery figures in this section — `ρ = 0.75`, the exposure and
normalization controls, and Table 2 — were produced with the configuration validated at the time: the
hard PIT score over a fixed count of `j = 64` nearest signatures, one-hit graduation, and the reserve
disabled. That readout has since been replaced entirely: the current method sums the counters over a
**fixed radius** and weights by the absolute density, with no percentile step and with `j` demoted to a
resolution diagnostic (§3.5). So these figures are **not** the current method's — the direct evidence
for the fixed-radius readout is the "Measured basis" subsection above, and re-running the §3.7 protocol
end to end under the current configuration is required before any recovery figure can be quoted for it.
Retiring the distance-calibrated bank likewise does not transfer this evidence to it; nothing here
should be read as the shipped configuration inheriting these numbers.

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

**Measured basis for the fixed-radius readout.** The replacement of the fixed-count percentile readout
by the fixed-radius absolute readout (§3.5) was decided by direct measurement on **59,249 cached tile
signatures**, drawn across four checkpoints of the two rev6 arms — offline, inference-only on the
cached signatures, with no training. Four findings drove it. *(i) The fixed-count readout is exactly
invariant to local rescaling.* Multiplying every distance in a query's neighbourhood by any constant
leaves `p̂` bit-identical, because dividing by the `j`-th nearest distance cancels the scale on the next
line — so the output carries no information about how crowded the neighbourhood is, the opposite of what
a density readout must do. *(ii) Its agreement with the offline `k`-NN reference density fell from 0.70
to 0.29* between iteration 52k (`j = 36`) and 124k (`j = 7`), as the cover became near-uniform (measured
nearest-neighbour-distance p90/p10 = 1.05): with the `j` nearest signatures at almost one distance,
`d_i/d_j ≈ 1` and the triweight `(1 − u²)³` annihilates every term — the measured total kernel weight at
`j = 7` was **0.158**, out of a possible 7. *(iii) The fixed-radius form, same kernel and same counters
summed over `R_rad` instead of a fixed count, agreed with the reference at 0.85* and was ~4× cheaper (no
argsort). The sweep was run in units of median anchor spacing (best agreement at 1.5× spacing), which
the code re-expresses in units of the self-tuned hit radius `s` (the conversion is in §3.5,
Implementation defaults). *(iv) The fixed-radius readout tilts the gradient several-fold harder than the
readout it replaces.* The measure is each decile's share
of the normalised DINO loss, with tiles binned into deciles by the offline reference density — the same
quantity, computed the same way, for both readouts, so the two are directly comparable. Under the
**retired** fixed-count percentile readout (`w = 1 − β · t`) this tilt was slight: the rarest decile
received **11.15%** of the loss at `β = 0.5` against 8.90% for the commonest — a rarest:commonest ratio
of **1.25×**, at 124k — and **12.59%** against 8.12% at `β = 0.9` (a ratio of **1.55×**, at 104k), both
barely above the 10% of no modulation. The **fixed-radius absolute readout** (`w = 1/(p̂ + c)^a`,
`c_frac = 0.25`) reaches a rarest:commonest tilt of **2.77×** at `a = 0.5` and **7.68×** at `a = 1.0`,
with effective sample sizes of **90.5%** and **68.7%** of the batch respectively — a two- to sevenfold
concentration of gradient mass onto the rare tail, where the retired readout reached at most 1.55×.
(The per-decile percentages were not recorded for the `c_frac = 0.25` rows, so for the new arms only the
tilt and the effective sample size are quoted, not a decile breakdown.)

Two things are explicitly **not** claimed. No AUROC or downstream effect has been demonstrated: every
number here is gradient-mass arithmetic on cached signatures, and nothing has yet run inside training.
And an earlier claim that the percentile readout "manufactures rarity in homogeneous batches" was
**retracted after measurement** — the deployed readout ranks a tile against a global rolling reference,
not within its batch, and on within-batch weight spread neither readout showed a consistent advantage;
that argument is not relied on anywhere.

### 3.8 Limitations

Four limitations bound the method. First, as noted in §3.5, the estimate is resolution-limited: with
a fixed memory budget over a support of intrinsic dimension near ten, structure finer than the
correlation length is invisible to any bounded summary. Second, the empirical constants of Table 1
were measured on a single, morphologically homogeneous slice of the stream; a substantially more
skewed corpus could shift the operating regime, though the counted readout is by construction
insensitive to the exact signature-placement exponent. Third, the typicality estimate modulates the
objective that trains the encoder that produces the signatures, so the distribution the module
measures is not exogenous. Deferring activation until the representation has stabilized (§3.3) is the
safeguard we rely on; a formal analysis of the coupled dynamics is left to future work.

Fourth, the counter half-life `H` is the least-justified constant in the method. Every other parameter
is either self-tuned online (the hit radius `s` and the pooling count `j`; §3.5), a memory budget
(`M`), derived from a measured rate (the reserve size; §3.5), or inert in practice (`T_need`); `H`
alone is hand-picked, and it alone sets the variance floor of the whole estimator. The decay fixes an
effective sample size `n_eff = (1 + η)/(1 − η) ≈ 721` steps at `H = 250`, a property of the half-life
alone and independent of stream length — a run of `10⁸` tiles gives the same per-signature precision as
one of `10⁵`, and at typical hit-rates that is ≈ 12% relative noise on each stored signature's rate.
Longer training does not sharpen the density estimate; only a longer half-life does. The value was
*bracketed, not optimised*: it must sit above the tile burst length (which decays within one step;
§3.5) and below the representation's drift horizon (tens of thousands of steps), leaving roughly two
orders of magnitude of slack, with 250 chosen inside that range. The only sensitivity evidence is the
offline `100`-versus-`250` comparison of §3.7, which found no material difference — but that is two
points, measured offline and under the earlier configuration (hard PIT, fixed `j`, one-hit
graduation), so it does not transfer to the present method. A post hoc bound does constrain the value
and is worth recording: a signature's rate must settle before the signature turns over, which requires
the observed turnover to exceed roughly `2·n_eff`. At the measured turnover of ≈ 2800 steps, a
half-life of 500 would give `n_eff ≈ 1442` and a ratio of 1.9, below that bound, while halving to 125
raises per-signature noise to ≈ 17%; the usable window is therefore roughly 150–400 steps, and 250
sits inside it — but this is a constraint identified after the fact, not the reason the value was
chosen, and no measurement establishes that it is the best point within it. Finally, `H` is counted in
*steps*, so it is implicitly tied to the batch size: changing the tiles per step changes the effective
memory measured in tiles, and the natural clock for this quantity is tiles seen rather than steps, so a
change of batch size should rescale `H` accordingly.

### 3.9 Rebalancing by thinning instead of weighting

**The same rebalancing, realized on the stream instead of the loss.** The weighted-loss modulation of
§3.6 scales each tile's DINO gradient by `w(x) = 1/(p̂(x) + c)^a`, flattening the effective sampling
distribution over morphology while every tile still enters the batch. *Thinning* reaches the same target
from the other side: it leaves the loss unweighted and instead admits each candidate tile to the batch
with a probability proportional to that weight. Common tiles are dropped more often, rare tiles almost
always kept, and the morphology mix the encoder trains on is rebalanced by *which tiles are present*
rather than by *how hard each is pushed*.

**Matched mass.** Thinning and weighting are constructed to deliver the same expected per-region
gradient mass. With acceptance probability

```
a(p̂) = w(p̂) / w_max ,   w(p̂) = (p̂ + c)^{-a} ,   c = c_frac · p_ref
```

(`a = 0.5`, `c_frac = 0.25`, matching the `lo` weighted arm; `w_max` the running max of `w` over a short
warmup, then frozen, so the rarest tile is admitted with probability ≈ 1) the thinned measure `a · μ`
over morphology equals the weighted measure `w · μ` up to the constant `1/w_max`. The two arms therefore
share their expected gradient and differ only in *effective sample size / coverage*: weighting keeps
every tile but down-weights the common ones (full coverage, unequal leverage); thinning spends the batch
on fewer, rarer tiles (reduced coverage, equal leverage). Expected consumption per committed batch is
`χ = 1/E[a] = w_max / E[w]`; the candidate pool is over-drawn by `--thin_oversample_factor` (set above
`χ`) so `N` survivors can be admitted, and more batches are pulled if a step falls short.

**The load-bearing rule: the bank sees the true stream.** The bank measures how rare each morphology is
in the *incoming* stream. Thinning flattens that stream, so the bank must be updated on the un-thinned
candidate pool, never on the thinned committed batch. Were it updated on the committed batch, the rare
tiles that thinning concentrates would stop looking rare, `p̂` would rise, `w → 1`, and admission would
decay to uniform sampling — the intervention erasing its own signal. In thinned mode the bank is
therefore updated exactly once per step, on the scout signatures of the whole pool (below), and the
committed forward is bank-read-only.

**The scout: a third, unaugmented global crop.** In the ordinary loop `p̂` is a post-forward quantity —
it exists only after the fused `student(all crops)` pass — so there is no density by which to admit tiles
before committing them. Thinning adds a *scout* crop: a third global view whose transform is `Resize +
Normalize` and nothing else (no random-resized-crop, flip, rotation, colour jitter, grayscale, or blur).
Because the DINO objective learns representations invariant to those augmentations, a tile's density is a
property of its content and is read correctly from the clean view; the augmented-crop-1 reading the bank
used previously was incidental to how it was wired. The scout is forwarded on its own no-grad pass — one
extra embed, not an extra training forward — and its bottleneck yields the signatures that update the
bank and set `p̂`. It feeds the bank only: it never enters the student/teacher crop sets or the
DINO/iBOT losses, is excluded from the student-view count, and, being deterministic, leaves the augmented
crops' random state untouched, so the `weighted` and `off` arms remain byte-identical to the code before
this section.

**Two-scale bias probe (measure-only).** The counted readout is smoothed at the fixed radius `R_rad`, so
`log p̂` carries a resolution-dependent bias. A second, coarse covering bank at ≈ `2s` spacing — a
`CountedCoverageBank` with `M' = 2^{-d*} M`; since `d* ≈ 9.3` this is ≈ `M/630`, floored to `64` because
the class's self-tuning (reserve, sweep, variogram) assumes `M` in the thousands — gives a coarser
estimate `u_2s = log p̂_2s` beside the fine `u_s = log p̂_s`. Their difference `β̂ = u_2s − u_s` is a
pointwise Richardson estimate of the bias, and `u* = 2 u_s − u_2s` the debiased density. Because `M'`
sits far below the bank's working regime the coarse readout is crude, so `β̂` is reported as an
*indicator*, not a calibrated correction: `u*` is implemented but inactive by default
(`--thin_richardson_correct`), and admission uses the plain `p̂`.

**Diagnostics.** Four scalars appear on the canonical log line in thinned mode: `thin_seen`, the
candidates scouted to fill the step (≈ `χ·N`); `thin_accept = N / thin_seen` (≈ `1/χ`, whose drift is
the miscalibration alarm); `thin_bias`, the mean `|β̂|`; and `thin_prof_err`, a binned-histogram L1
between the committed batch's `p̂` distribution and the target profile `a · μ`.

**Flag.** `--balance_mode` selects `weighted` (the §3.6 arm, byte-unchanged and the comparison
baseline), `thinned` (this section), or `off` (no rebalancing). `p_ref`, `w_max`, and the realized `χ`
are re-measured on the unaugmented scout basis during warmup rather than carried from the augmented
weighted arm, whose `p̂` distribution differs.

### 3.10 Free and derived hyperparameters

The bank carries many named constants, but they are not independent knobs. Only two are free: the
bank capacity `M` and the tilt exponent `a`. Every other constant is *derived*, from `M`, from `a`,
or from the data, and is fixed by the requirement that the bank sit at a stable operating point
rather than by a separate search.

**Derived from `M` (the evidence budget).** A cell's hit-rate is estimated from the hits it collects
before they decay, of order `(κ / M) · H` per cell (with `κ` the tiles fed per step and `H` the
half-life), and the probation traffic scales with the number of cells to be discovered. So the
half-life `H`, the reserve size, and the reserve residency scale linearly with `M`. Quadrupling `M`
gives each cell a quarter of the hits, which a four-fold longer memory restores: the effective hit
count each estimate integrates, `n_eff = (1 + η)/(1 − η)`, scales with the half-life, from ≈ 721 at
`H = 250` to ≈ 2884 at `H = 1000`. A four-fold larger reserve absorbs the four-fold larger inflow of
newborns. Holding `(κ / M) · H` and `reserve / M` fixed keeps the per-cell estimator variance and the
graduation dynamics invariant as resolution grows. Raising `M` alone, with memory and reserve
unchanged, starves the rarest cells first and collapses the dynamic range toward `lam_spread → 1`.
The graduation threshold rises with `M`: resolving the genuinely lowest-rate cell by `argmin` over
`M` noisy estimates needs of order `2 ln M` accumulated counts, so the corroboration to promote a
newborn grows logarithmically, from two hits at `M = 8192` to three at `M = 32768`. The four-fold
longer memory makes the extra hit reachable rather than exclusionary: the rate a newborn must sustain
to graduate, `graduation_hits` over its exposure window, actually falls, since three hits over a
four-fold window is a lower bar than two over the base window. Raising the count therefore buys
argmin stability at finer resolution without starving the rare cells it is meant to protect.

**Derived from the data.** The hit radius `s` (the fill knee of §3.5), the readout neighbourhood size
`j`, and the candidate-pool self-tuning are read off the stream at run time. As signatures pack
denser they shrink to track the smaller spacing without intervention.

**Derived from `a` (fill).** Thinning over-draws `χ = 1 / E[a(p̂)]` candidates per committed tile
(§3.9), and `χ` grows with the tilt, so the oversample factor tracks `a`: a stronger tilt lowers
acceptance and needs a proportionally larger pool to admit `N` survivors. Its one side effect is on
the rate at which the bank is fed, which the experiments hold under test rather than assume away.

This is a parameterisation, not a confound. Co-scaling a derived constant with the free knob it
depends on is not a second, uncontrolled change; it is precisely what holds the bank's operating
point fixed while the free knob moves. A confound would be perturbing an *independent* quantity by
accident. These are dependent by construction, so the design space is genuinely two-dimensional in
`(M, a)`, with the remainder determined.

Table 2 records the full run matrix in the same layout, and the forward sensitivity sweep is its
`thinned` rows: `M` and `a` vary across rows, the derived constants take the values the scaling law
assigns, and the fixed constants are noted below.

<table>
<thead>
<tr>
  <th rowspan="2">run (rev)</th>
  <th rowspan="2">mode</th>
  <th colspan="2">Free (swept)</th>
  <th colspan="6">Derived (set by the scaling law)</th>
</tr>
<tr>
  <th><code>M</code></th><th><code>a</code></th>
  <th><code>n_eff</code> (∝ M)</th><th>half-life <code>H</code> (∝ M)</th><th>reserve (∝ M)</th><th>residency (∝ M)</th><th>graduation hits (∝ ln M)</th><th>oversample (∝ a)</th>
</tr>
</thead>
<tbody>
<tr><td>weightedloss_lo (rev7)</td><td>weighted</td><td>8,192</td><td>0.5</td><td>721</td><td>250</td><td>550</td><td>300</td><td>2</td><td>n/a</td></tr>
<tr><td>weightedloss_hi (rev7)</td><td>weighted</td><td>8,192</td><td>1.0</td><td>721</td><td>250</td><td>550</td><td>300</td><td>2</td><td>n/a</td></tr>
<tr><td>thinned_lo (rev10)</td><td>thinned</td><td>8,192</td><td>0.5</td><td>721</td><td>250</td><td>550</td><td>300</td><td>2</td><td>3×</td></tr>
<tr><td>thinned_lo (rev10, 6×)</td><td>thinned</td><td>8,192</td><td>0.5</td><td>721</td><td>250</td><td>550</td><td>300</td><td>2</td><td>6×</td></tr>
<tr><td>thinned_hi (rev11)</td><td>thinned</td><td>8,192</td><td>1.0</td><td>721</td><td>250</td><td>550</td><td>300</td><td>2</td><td>6×</td></tr>
<tr><td>thinned_lo (16k, rev13)</td><td>thinned</td><td>16,384</td><td>0.5</td><td>1,442</td><td>500</td><td>1,100</td><td>600</td><td>2</td><td>6×</td></tr>
<tr><td>thinned_hi (16k, rev13)</td><td>thinned</td><td>16,384</td><td>1.0</td><td>1,442</td><td>500</td><td>1,100</td><td>600</td><td>2</td><td>~6×</td></tr>
<tr><td>thinned_lo (32k, rev12)</td><td>thinned</td><td>32,768</td><td>0.5</td><td>2,884</td><td>1,000</td><td>2,200</td><td>1,200</td><td>3</td><td>8×</td></tr>
<tr><td>thinned_hi (32k, rev12)</td><td>thinned</td><td>32,768</td><td>1.0</td><td>2,884</td><td>1,000</td><td>2,200</td><td>1,200</td><td>3</td><td>~8×</td></tr>
</tbody>
</table>

**Table 2.** The complete run matrix as a free/derived grid, and the forward sensitivity sweep (its
`thinned` rows). The two free hyperparameters vary across rows; the derived constants show the values
fixed by the scaling law (`n_eff`, half-life, reserve, and residency scale linearly with `M`;
oversample tracks `a`; `s` and the readout pool are auto-tuned). Graduation hits track the `2 ln M`
argmin-stability bound: held at two through 16k and raised to three at 32k, with the longer memory
keeping the extra hit reachable. The `M` sweep is the three thinned levels 8k / 16k / 32k at each
tilt (8k done, 32k running, 16k queued as rev13). Fixed across every run: `c_frac = 0.25`,
`radius_mult = 1.5`, `K' = 256`.
Weighted arms commit every tile and do
not over-draw (`n/a`). The `thinned_lo (rev10, 6×)` row was configured as the `hi` arm but admitted
at `a ≈ 0.5` because of a pre-`rev11` tilt hardcode, so it is really the `a = 0.5` arm at a larger
pool; read against `thinned_lo (rev10)` at 3× it is an `a = 0.5` feed comparison, not an `a = 1.0`
point. Oversample at `M = 32768` is a starting estimate, raised if the realised acceptance
under-fills `N`.

---

## 4. Evaluation: grounded figures (to be measured)

The figures below are the planned evaluation. Each pairs a theoretical law with the measured log
quantity from the Table 2 runs that anchors it, and is only reported once the run points sit on it.
All points are read at a matched post-gate age with settled bank statistics (§3.10); a point whose
`s` has not flattened is provisional. Every block marked `⟨THIS NEEDS TO BE MEASURED⟩` is a
reserved placeholder pending the completed 8k / 16k / 32k sweep.

### 4.1 Resolution laws (the `M` sweep)

**Figure 1. Fill-knee radius vs bank size.** `s` versus `M`, log-log, one series per tilt. Theory:
`s ∝ M^{-1/d*}` (§3.5), so the slope is `-1/d*`. Anchor: settled `s` at each of the six thinned
`(M, a)` points. Inference: the fitted slope measures an *effective* `d*` at the bank's operating
scale, to compare against the TwoNN intrinsic dimension `d* ≈ 9.3` (Table 1). The 8k→32k pair
already implies a steeper slope (effective `d* ≈ 4`); the 16k point tests whether the law is a clean
power or curves.
`⟨THIS NEEDS TO BE MEASURED⟩`

**Figure 2. Density dynamic range vs bank size.** `lam_spread` (and the `q10 / median / q90`
decomposition) versus `M`. Theory: finer catchments resolve a wider range of local density, so the
stored hit-rate range widens with `M`. Anchor: settled `lam_spread` at each `(M, a)`. The
decomposition guards against a widening that is bottom-body decay rather than true resolution.
`⟨THIS NEEDS TO BE MEASURED⟩`

**Figure 3. Hit fraction vs bank size.** `hit_frac` versus `M`. Theory: the knee hit-fraction falls
as catchments shrink. Anchor: settled `hit_frac` at each `(M, a)`. This is the resolution cost that
accompanies Figures 1 and 2.
`⟨THIS NEEDS TO BE MEASURED⟩`

### 4.2 Tilt laws (the `a` sweep)

**Figure 4. Acceptance and over-draw vs tilt.** `E[a(p̂)] = ∫ (1 + p̂/c)^{-a} μ(p̂)` and
`χ = 1/E[a]` versus `a`. Anchor: measured `thin_accept` at each `(a, M)`.
`⟨THIS NEEDS TO BE MEASURED⟩`

**Figure 5. Weight contrast vs tilt.** Rarest-to-commonest gradient-mass ratio versus `a`, log-y.
Theory: contrast `= R^a`, so doubling `a` squares the contrast. Anchor: `R` measured as the density
dynamic range `lam_spread`; the contrast points measured as the gradient-mass ratio in the weighted
arms (2.77 at `a = 0.5`, 7.68 at `a = 1.0`).
`⟨THIS NEEDS TO BE MEASURED⟩`

**Figure 6. The cost duality.** Effective sample size and over-draw versus `a` on shared axes.
Theory: matched mass reaches one committed diet by two mechanisms whose costs are orthogonal,
weighting pays in `ESS`, thinning pays in `χ`. Anchor: measured `ess` in the weighted arms
(0.89, 0.67) and `χ = 1/thin_accept` in the thinned arms.
`⟨THIS NEEDS TO BE MEASURED⟩`

### 4.3 What each figure needs from the logs

Gathered once the sweep is complete, per run, at matched post-gate age (settled): `s`, `lam_spread`
with `lam_q10 / lam_median / lam_q90`, `hit_frac`, `thin_accept`, `ess`, `w_mean`, `grad_per_step`,
`p_ref`, `thin_prof_err`, and the config `M / a / H / oversample`. The matched-mass reshaping
(committed versus pool `p̂` distribution) is deferred: it needs the two histograms behind
`thin_prof_err` logged, a small trainer change, not part of this pull.

### 4.4 Oversample sizing (to be measured)

Oversample must clear two floors: the **fill floor** `χ = 1/thin_accept` (the minimum pool that
admits `N`), and a **feed floor** (enough hits per cell to hold a stable `s`, provisionally
`per-cell feed ≥ 0.25`, i.e. `oversample ≥ M/4096`). The `per-cell feed = oversample × batch_gpu ×
n_GPU / M` column is arithmetic and filled below; `χ` and the stability outcome are measured per run.
The feed floor is not yet a validated law, it rests on a single stable point, so the claim to test is
narrow: runs whose per-cell feed clears the floor settle, and the one below it (32k lo at 4×,
per-cell 0.125) did not, which is why it was re-run at 8×.

<table>
<thead>
<tr>
  <th>run (rev)</th><th><code>M</code></th><th><code>a</code></th><th>oversample</th>
  <th>per-cell feed</th><th><code>χ</code> (fill, measured)</th><th>s / lam_spread settled?</th>
</tr>
</thead>
<tbody>
<tr><td>thinned_lo (rev10)</td><td>8,192</td><td>0.5</td><td>3×</td><td>0.375</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
<tr><td>thinned_lo-6× (rev10) †</td><td>8,192</td><td>0.5</td><td>6×</td><td>0.75</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
<tr><td>thinned_hi (rev11)</td><td>8,192</td><td>1.0</td><td>6×</td><td>0.75</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
<tr><td>thinned_hi (rev12)</td><td>32,768</td><td>1.0</td><td>8×</td><td>0.25</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
<tr><td>thinned_lo (rev12)</td><td>32,768</td><td>0.5</td><td>8× (was 4×)</td><td>0.25 (was 0.125)</td><td>⟨measure⟩</td><td>4× did NOT settle; 8× ⟨measure⟩</td></tr>
<tr><td>thinned_lo (rev13)</td><td>16,384</td><td>0.5</td><td>6×</td><td>0.375</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
<tr><td>thinned_hi (rev13)</td><td>16,384</td><td>1.0</td><td>6×</td><td>0.375</td><td>⟨measure⟩</td><td>⟨measure⟩</td></tr>
</tbody>
</table>

**Table 3.** Oversample sizing per thinned run. `per-cell feed` is arithmetic (`oversample × 256 × 4
/ M`); `χ` (the measured fill floor, `1/thin_accept`) and the stability outcome are measured as each
run settles. `⟨measure⟩` cells fill in as the runs finish (rev10 / rev11 / rev12-hi first, then
rev12-lo at 8×, then rev13). `†` admitted at `a ≈ 0.5` via the pre-`rev11` hardcode.

### 4.5 A closed form for the presentation multiplier

The multiplier `f` (the oversample) obeys `f = max(f_fill, f_feed)`, the larger of a fill floor and a
feed floor.

**Fill (closed, exact).** Present `fN` points and retain each independently with probability `r(ξ)`.
Marginalising over `ξ ∼ μ`, the retained count is *exactly* `Z ∼ Binomial(fN, 1/χ)`, `χ = 1/E_μ[r]`;
the density-spread contribution `Var_μ(r)` cancels against the Poisson-binomial term, so there is no
correction from the heaviness of the tail. Requiring `P(Z < N) ≤ ε` per block, over `K` independent
blocks,

```
f_fill = χ · ( 1 + z_{ε/K} · sqrt( (χ − 1) / (χ N) ) ),
```

or, for small `ε`, the exact Chernoff form `f·D(1/f ‖ 1/χ) = ln(K/ε)/N` with `D` the binary relative
entropy. `χ = 1/thin_accept` is measured per run (size against its worst value in the settled
window); `K` enters only through `ln(K/ε)`. This floor is complete.

**Feed (scaling law, one exponent pending).** Bank health requires the rarest resolved cells to keep
re-inducting, `φ_c · r_⋆ · T ≳ g`, where `r_⋆ = ς · M · μ(B(a_⋆, s))` is the relative arrival mass at
a near-minimal-intensity (low-density) cell. With `μ(B(a_⋆,s)) ∝ p_⋆ · s^{d_⋆}` and `s ∝ M^{−1/δ}`,

```
r_⋆ ∝ p_⋆ · M^{1 − d_⋆/δ},        f_feed = φ_c · M / (KN) ∝ ln M · M^{d_⋆/δ − 1}.
```

`d_⋆` is the local mass exponent at the occupied lower-decile sites, fixed by regressing
`[ln q_{0.1}(λ) − ln φ − ln M]` on `ln s_M` across the sweep (per-`M` data: `s_M`, `φ`, the `λ`
deciles, `|Q|/M`). The measured stability bracket forces `d_⋆ > 4.4`. A point value is not yet
recoverable: the only settled runs are both at `M = 8192`, and the single `M = 32768` arm has not
plateaued (its `q_{0.1}(λ)` is still rising), which throws the fitted slope to a non-physical value
above `d`. So for now `d_⋆ ∈ (4.4, ~9)` and the feed exponent `d_⋆/δ − 1 ∈ (0.1, 1.25)`.

**Open item (closes §4.5).** Once one 32k arm settles (`rev12 hi`, or the `rev12 lo` restart at 8×,
roughly 10+ half-lives to a plateau in `lam_q10` / `lam_spread`), re-pull the six per-`M` settled
values (`s_M`, `φ`, `lam_q10 / median / q90`, `|Q|/M`) and regress `[ln q_{0.1}(λ) − ln φ − ln M]` on
`ln s_M` across `M` to fix `d_⋆`; substitute into `f_feed` and into the symbolic `φ_c` from part (b).
This is the only quantity left open. Oversample is meanwhile set operationally by the stability
bracket alone: `3×` at 8k, `4–6×` at 16k, `8×` at 32k.

**Resolution of the `δ ≈ 4` vs `d ≈ 9` gap** (Table 1, §6). The self-tuned radius exponent
`s ∝ M^{−1/δ}` with `δ ≈ 4` is the *typical local mass dimension at occupied sites*, not the
box-counting dimension `d ≈ 9` of the support: the radius balance holds the captured mass fraction
roughly constant, which forces `s ∝ M^{−1/d_typ}` with `d_typ ≈ δ`. Low-density sites carry thinner
mass, so their local dimension `d_⋆ > d_typ`, and it is precisely `d_⋆ > δ` that makes the feed floor
grow with `M` at all; were the support tiled at its box dimension the floor would be `M`-independent.
Local dimension in the sense of Rossi (2013).

---

### References

Rossi. *Local dimensions of measures on infinitely generated self-affine sets.* arXiv:1302.1435, 2013.


Arthur and Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA 2007.

Caron et al. *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021.

Chen et al. *Towards a General-Purpose Foundation Model for Computational Pathology (UNI).* Nature Medicine 2024.

Facco et al. *Estimating the Intrinsic Dimension of Datasets by a Minimal Neighborhood Information.* Scientific Reports 2017.

Graf and Luschgy. *Foundations of Quantization for Probability Distributions.* Springer LNM 1730, 2000.

Kornblith et al. *Similarity of Neural Network Representations Revisited.* ICML 2019.

Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.

Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer.* ICLR 2022.
