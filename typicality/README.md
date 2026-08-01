# Curating the Heavy Tail

*A readable companion to [`technique.md`](technique.md). Where `technique.md` is the formal
method section, this is the walkthrough: how the typicality module estimates local density
online and rebalances the training objective toward rare morphology, built from first
principles, through both roads, weighted and thinned.*

Modern self-supervised recipes curate their data *before* training: de-duplicate, rebalance,
retrieve. A tile stream from whole-slide images is far too large to curate that way. This module
does it **online, during training**. It is a stop-gradient density estimator that measures how
typical each tile is and turns down the typical, so the model spends its capacity on the tail.

```
z(x)  →  s = R·z  →  λ̂ = S/E = p/g  →  p̂ = Σ λ̂·K(d/R)  →  w = 1/(p̂+c)^a  →  weighted, or thinned
```

---

## 01 · The problem: a stream you cannot curate in advance

A whole-slide image is labelled once, at the slide level, but consumed as thousands of tiles.
The stream is **heavily redundant**: one slide contributes thousands of near-duplicate views of
stroma, adipose, and background, while diagnostically informative morphology is comparatively
rare. Under the single shared softmax temperature of the DINO objective, a common tile
contributes a per-example gradient of the same magnitude as a rare one, so the informative
signal is at once *diluted by*, and redundantly reinforced against, a long tail of near-duplicates.

Large pipelines normally fix this offline (the DINOv2 corpus was de-duplicated and
retrieval-rebalanced before training). A pathology tile stream, of order `10⁸` tiles, cannot be
curated that way at scale. This module supplies the missing curation online: at each step it
estimates, for every tile, how typical it is of the morphology the model has recently seen, and
reshapes the effective sampling distribution toward the tail as it trains.

## 02 · The module: a stop-gradient side-channel

The module never touches the backbone's gradient. It is a **stop-gradient side-channel** on the
standard student/teacher pipeline: it reads an intermediate student representation under
stop-gradient, produces one scalar per tile, and uses it to reweight that tile's loss. Because
the read is detached, the representation is a passive input it *measures*, never a target the
model is pushed toward.

That scalar is **local density** in morphology space: how crowded a tile's neighbourhood is.
Common morphology sits in dense neighbourhoods; rare morphology in sparse ones. So typicality
*is* density: high density → typical → dampen; low density → rare → preserve. The dense per-patch
iBOT objective is left untouched; only the image-level DINO cross-entropy is rebalanced.

Three stages on each tile `x`: form a compact **signature** of its morphology (§03); turn the
signature into a **density** `p̂(x)` with a memory bank (§05 to §08); let that density drive a
**weight** that either rescales the loss (§10) or rescales admission to the batch (§11).

## 03 · The signature: a stable coordinate for morphology

A memory that measures distances is only meaningful if the space those distances live in *stops
moving*. The raw `[CLS]` representation is unsuitable: it is high-dimensional and still
reorganising late in training (about 55% converged by linear CKA at 50k steps). The DINO-head
**bottleneck** is compact and settles early (about 94% at 50k), living in a low-dimensional,
early-forming subspace. So the signature is built from it.

Let `z(x) ∈ S²⁵⁵` be the student's unit-norm bottleneck for the tile, taken under stop-gradient.
A matrix `R` of `K′ = 256` unit-norm **representative prototypes** defines the signature as the
vector of cosine similarities:

```
s(x) = R · z(x) ∈ ℝ^{K'} ,     s_k(x) = ⟨ R_k , z(x) ⟩
```

A signature is **peaky** for well-formed morphology (one prototype dominates) and **diffuse** for
a tile resembling nothing (all similarities near the small-angle floor of a random projection).
The prototypes are trained online by their own optimizer, `L_R = L_nn + λ_cov · L_cov`: `L_nn`
pulls each row toward its nearest *output* prototype so the frame tracks directions the model has
actually learned; `L_cov` penalises the off-diagonal of `R Rᵀ` so the rows stay spread; rows are
renormalised to the sphere each step. Both prototypes and their loss derive from the detached
bottleneck, so the signature space is a passive readout. A trained `R` spans an effective rank of
about 44, comfortably above the intrinsic dimension of the data it must resolve (§08).

## 04 · The estimand: two ways to estimate a density

Everything hinges on one estimation problem: given a tile's signature, how *dense* is the
neighbourhood it landed in? There are two standard estimators, and the design is the story of why
the obvious one fails and the second is built.

| | **Estimator A, distance-calibrated** | **Estimator B, counted-coverage** |
|---|---|---|
| what it is | nearest-neighbour density estimate | kernel density estimate on a fixed memory budget |
| reads density from | the **gaps between** stored points | the **counts on** stored points |
| signal carrier | spacing of the points | tallies attached to the points |
| faithful when | the stored set is a **sample** of the data | **regardless** of how the points are maintained |

Both are legitimate. They differ in what carries the signal, the spacing or the counts.

## 05 · Why the obvious estimator fails: a sample becomes a cover

A nearest-neighbour readout is a valid density estimate only when its stored set is distributed
*like the data*, a **sample**. Bounded memory means evicting, and the natural rule is "admit a
novel tile, evict the stored point nearest to it." That rule has a fatal structural consequence.

> **Isolated points are absorbing.** A stored point far from all others is, by definition, never
> the nearest neighbour of an incoming tile, so it is never chosen for eviction. Isolated entries
> can only accumulate. Over training the memory drifts from a **sample** (distributed like the
> data) to a **cover** (an even grid that merely fills the occupied space).

Once the memory is a cover, the nearest-neighbour distance sits near the cover's spacing almost
*everywhere* and stops varying with density. The score flattens; it can no longer tell a common
tile from a rare one, the exact discrimination the module exists to provide. Deferring the fill
past warmup removes a seeding artifact but not the cause: non-eviction of isolated points is a
property of the eviction rule itself.

This is the pivot. The distance readout is kept only as the analysis that *motivates* the real
design. The fix: stop asking one point set to do two jobs at once, marking *where* the data is and
recording *how dense* each part is, and separate them.

## 06 · The counted-coverage bank: separate *where* from *how dense*

The bank keeps `M ≈ 8,192` stored signatures laid down as an even **cover** of the occupied space
(the "where") and attaches to each a running **tally** of how many tiles land on it (the "how
dense"). Density is read from the tallies, not the spacing, so the very cover that broke the
distance bank is exactly what this bank wants. It is a KDE compressed onto a fixed budget: instead
of a bump on each of `N` tiles, a bump on each of `M` stored points, weighted by its tally.

Each stored signature `i` carries two decayed counters:

```
λ̂_i = S_i / E_i
```

- `S`, the decayed **hit count** (tiles that landed on it).
- `E`, the decayed **exposure**, a *clock*: `+1` every step the point is alive, decayed by
  `η = ½^(1/halflife)`.

Their ratio `λ̂` is the point's **hit-rate**: hits per step of life. Defining `E` as lifetime, not
traffic, is essential: were it a tile count it would grow with density exactly as `S` does, their
ratio would collapse to a constant, and it would carry no density signal at all.

### The one idea that makes it work: two densities, and a cancellation

At any point in signature space there are *two* densities. The one we want is **tile density** `p`,
how common that morphology is. The nuisance is **stored-signature density** `g`, how many
reference points happen to sit there, an artifact of placement. A raw count of nearby tiles mixes
them. The design cancels `g` and leaves `p`, through one deliberate rule:

> **A tile hits only its nearest stored point, within radius `s`.** Not every point inside `s`,
> only the single nearest. A point's catchment is then its **Voronoi cell**, whose volume scales
> as `1/g` (denser points ⇒ smaller cells). So its hit count picks up density times cell volume:
> `S ∝ p · (1/g) = p/g`. The hit-rate `λ̂` measures tile density *divided by* the local crowding
> of the reference points, exactly the structure the readout needs.

Counting every tile in a fixed-radius ball instead would give `S ∝ p` and break the cancellation
about to happen. The nearest-only rule is doing real work here, not decoration.

## 07 · The readout: sum the rates, don't average them

The density at a query is an *unnormalised* kernel sum of the hit-rates over the stored points
inside a **fixed radius** `R = radius_mult · s`:

```
p̂(x) = Σ_i  λ̂_i · K( ‖ s(x) − b_i ‖ / R )        ( K = smooth, compactly supported triweight )
```

> **Summing cancels `g`; averaging would divide it back out.** A dense region holds *more* stored
> points, each carrying a *smaller* rate (because `λ̂ ∝ 1/g`). **Summing** multiplies the two,
> more points times smaller rates, and `g` cancels, leaving `p̂ ∝ p`. **Averaging** (dividing by
> `ΣK`) divides `g` straight back in and estimates the mean per-point rate, which vanishes exactly
> where placement is proportional to density. Sum, never average, and the cancellation is valid
> only over a domain of *fixed volume*, which the fixed radius supplies.

**Why not the `j` nearest neighbours?** An earlier readout summed over the `j` nearest points and
set the kernel bandwidth to the `j`-th distance. That makes the domain *query-adaptive*, small in
dense regions and large in sparse ones, and dividing every distance by that bandwidth cancels the
local scale *exactly*: the sum comes out bit-for-bit invariant to how crowded the neighbourhood
is, which is precisely what a density estimate must *not* be. On a near-uniform cover the `j`
nearest points sit at almost one distance, and the triweight `(1−u²)³` annihilates nearly every
term (measured total weight: 0.158 out of a possible `j`). The **fixed radius** removes that
self-cancelling normalisation and lets the crowding show. Centering the kernel on the query with a
symmetric profile also cancels the leading (gradient) term of the smoothing bias.

The output is an **absolute** density, a real rate, not a percentile. That matters: an earlier
version scored by the percentile of `log p̂`, which is uniform by construction and throws away
*how much* denser one tile is than another, the very magnitude the fixed-radius sum recovers.

## 08 · The self-tuned scale: the bank measures its own geometry

Two radii, both set by the bank rather than by hand. The **hit radius** `s` governs seeding and
hit-counting; the **readout radius** `R = 1.5·s` governs smoothing. The natural scale of the
signatures *drifts* as `R` trains, so no fixed radius can be right for the whole run.

**The fill knee.** `s` is set at the largest radius for which seed-on-miss populates exactly `M`
points, the knee between under- and over-filling. It is **measured live**: every ~500 steps the
bank replays seed-on-miss over a rolling buffer of the ~60,000 most recent signatures, across a
grid of radii re-centred on the current scale, and tracks the knee with a light EMA. On the
running arms the knee sits near `s ≈ 17.7` when the gate opens, and the `a = 0.5` arm drifts *down*
about 9% to `s ≈ 16.1` by iteration 93k as the signature scale contracts (the `a = 1.0` arm holds
near 17.7 over its shorter run so far); a radius frozen at an early value would soon fall inside
every point's neighbourhood, nothing would count as a hit, and the bank would go inert.

**The resolution check.** A variogram fit estimates the density **correlation length** `L`, the
distance over which density changes appreciably. If the stored points are spaced *wider* than `L`,
the bank is coarser than the structure it must resolve, and it raises an `underresolved` flag: a
memory-budget limit, curable only by more points, not a different radius.

| Measured constant | What it is | Value |
|---|---|---|
| `d*` | intrinsic dimension: effective directions the signatures occupy (≪ 256) | ≈ 9.3 |
| `L` | density correlation length (L1 units on the signatures) | ≈ 3.34 |
| dyn. range | density spread, emptiest to densest, at scale `L` (a moderate regime) | ≈ 18× |
| `n_eff` | settled evidence in each `λ̂`, fixed by the half-life alone | ≈ 721 |
| `ρ_out` | one-off (artifact) arrival rate, which sizes the reserve | ≈ 10⁻³ |

*Measured on a baseline ViT-B/16 over an independent 384k-tile sample (inference-only, before the
module perturbs the stream). The support is a connected continuum of intrinsic dimension near ten
with only moderate density variation, and no isolated clumps for the §05 outlier hazard to attach
to.*

## 09 · The lifecycle: how a stored point earns, keeps, and loses its place

```
miss (> s from all)  →  reserve (probation, ~550)  →  [2 hits]  →  established (~8,192)  →  [lowest λ̂]  →  evicted
```

**Eviction reverses the pathology.** Established points are managed **least-frequently-used with
aging**: the point of lowest hit-rate `λ̂` leaves. This is the exact inverse of §05: an isolated
point accrues no hits and is the *first* out, not the last. Content-dependent placement, which
corrupted the distance bank, is harmless here because density no longer rides on placement.

**Two hits to graduate, a stability requirement rather than a knob.** A newborn enters the reserve
empty and must earn **two corroborating hits** before it displaces an established point. Promotion
on a *single* hit sets one observation against a rate estimated over hundreds of steps; resolving
the genuinely lowest-rate point by `argmin` over `M` noisy estimates needs on the order of
`2 ln M / δ²` counts, and one-hit graduates arrive far short of that, so they evict points that
were merely *under-sampled*, not truly quiet. An ungraduated one-off is dropped when its age
exceeds a deadline, keeping the reserve bounded at `reserve ≈ ρ_out · κ · T_need`, a few hundred
slots, a few percent of `M`.

**Decay sets the memory horizon.** The half-life (a few hundred steps) is long relative to the
batch autocorrelation, so recurring morphology builds standing evidence, yet short relative to the
representation's drift, so the estimate tracks the current encoder. It fixes the estimator's floor:
`n_eff = (1+η)/(1−η) ≈ 721` steps of effective evidence, independent of run length.

## The algorithm: the counted-coverage bank in one step

One training step on the all-gathered batch `X`. `p̂` is read against the pre-update state; the
update then runs on every tile. Each block is tagged with the section that explains it.

```
[ warmup · §03 ]
if step < T_warm: return p̂ = ⊥ for all x          # inactive: signatures not yet stable

[ decay & age · §06 §09 ]
for each live signature i:
    S_i   ← η · S_i                                # decayed hit count
    E_i   ← η · E_i + 1                            # decayed lifetime (the clock)
    age_i ← age_i + 1
    if i in reserve and age_i > T_need: evict i    # ungraduated one-off expires

[ all-gather · §13 ]
gather signatures of X across workers → X_global

[ readout · §07 §08 ]
if |established| = M and counters mature:
    R_rad ← radius_mult · s
    p̂(x) ← Σ_{ ‖s(x)−b_i‖ ≤ R_rad }  (S_i / E_i) · K( ‖s(x)−b_i‖ / R_rad )
else: p̂(x) ← ⊥                                    # fill / cold-start: no weight

[ reference scale · §10 ]
p_ref ← 0.999 · p_ref + 0.001 · median_x p̂(x)

[ hit · seed · graduate · evict · §09 ]
for each x in X_global:
    i* ← nearest signature to s(x)
    if ‖s(x) − b_{i*}‖ ≤ s:                        # a hit
        S_{i*} ← S_{i*} + 1
        if i* in reserve and its hits ≥ 2:         # graduate (two corroborating hits)
            if |established| = M: evict argmin_i λ̂_i
            move i* → established
    else:                                          # novel tile
        if |established| < M: seed into established     # fill the cover
        else: seed into reserve (flush oldest if full)
```

The per-tile weight `w = 1/(p̂+c)^a` is formed downstream from `p̂` and `p_ref` (§10); the weighted
road (§10) scales the loss by it, the thinned road (§11) admits by it. In thinned mode the update
runs on the *un-thinned* scout pool (§12), and the committed forward is read-only.

## 10 · The weight, and road one: weight the loss

Rarity is density read upside-down:

```
w(x) = 1 / ( p̂(x) + c )^a ,     c = c_frac · p_ref
```

**The floor `c` earns its keep twice.** It keeps `w` finite where `p̂ = 0` (an empty-neighbourhood
tile gets `1/c^a`, no divide-by-zero, no clipping). And, because it rides `p_ref` (a slow EMA of
the batch-median `p̂`), it makes the weight **scale-invariant**: multiply every `p̂` and `p_ref` by
any `k` and `w → k^(−a) w`, a common factor. Since `p̂` has no natural units (it reads 0.001 or
800 depending on the kernel and the drifting signature scale) the reference *cannot* be a
constant; it must be measured live.

**Road one, the weighted loss.** Scale each tile's DINO cross-entropy by `w` and apply it as a
**weight-normalised mean**, `Σ w·CE / Σ w`. The weight sum divides out, so the batch's gradient
magnitude is unchanged and modulation acts *only* through the relative weights: a uniform weight
of any value is identical to no weighting, and the common factor `k^(−a)` cancels. Only the
*spread* of `w` acts: the tilt `a` tunes it (measured rarest-to-commonest gradient-mass ratio
2.77× at `a=0.5`, 7.68× at `a=1.0`). This is importance weighting in its standard form: it
flattens the effective sampling distribution while leaving each tile's learning signal intact.

The cost is **effective sample size**, `ESS = (Σw)² / (B·Σw²)`. When a few rare tiles hoard the
weight, the gradient is effectively computed from those few. You keep full coverage of the data at
a reduced statistical batch size. *Measured* on the two completed weighted runs: `ESS = 0.89` at
`a = 0.5` and `0.67` at `a = 1.0` (mean weight 1.65 and 3.12); at the harder tilt a batch of 256
pulls like about 171. This is the tax the thinned road (§11) avoids.

## 11 · Road two: thinning, the same target realized on the stream

Thinning reaches the identical rebalanced distribution from the other side: leave the loss
unweighted, and **admit** each candidate to the batch with a probability proportional to its
weight.

```
a(p̂) = w(p̂) / w_max ,     w_max = ( c_frac · p_ref )^{-a}
```

> **Matched mass.** With acceptance `a(p̂) = w/w_max`, the thinned measure `a·μ` over morphology
> equals the weighted measure `w·μ` up to the constant `1/w_max`. The two roads share the *same
> expected per-region gradient mass* and differ only in coverage versus leverage: weighting keeps
> every tile at unequal leverage; thinning keeps fewer, rarer tiles at equal leverage. Same tilt
> `a`, and the experiment isolates exactly that difference.

**The payoff.** Survivors are trained **unweighted**, so effective sample size is a full 1: the
rebalancing lives in *who is in the room*, and every tile pulls at full strength. Expected draws
per committed batch is `χ = 1/E[a]`; the pool is over-drawn by a factor above `χ` so `N` survivors
can be admitted. *Measured* in the two running thinned arms: `ESS` and mean weight sit at exactly
`1.000` on every logged step, and the accept rate settles at about `0.49` in both (`χ ≈ 2.04`), the
same value at `a = 0.5` with a 3× pool and at `a = 1.0` with a 6× pool.

**The ceiling must track the scale.** Because `w_max` cancels when the batch is finally selected,
its value sets only *how many* tiles survive, never *which*, provided it stays above every real
weight (`w` is maximal at `p̂ = 0`, so `w_max = (c_frac·p_ref)^{-a}` is the exact bound). Computing
it in closed form from the live `p_ref` each step keeps acceptance steady for the whole run. Freeze
it at an early scale and, as `p_ref` climbs, the normaliser goes stale, admissions starve, and the
batch can no longer fill itself: a silent failure cured only by restoring the scale-tracking, not
by a bigger number.

## 12 · The rule thinning must obey: the bank must see the true stream

The bank measures how rare each morphology is *in the incoming stream*. But thinning **flattens**
that stream. If the bank were updated on the thinned, committed batch, the rare tiles that thinning
concentrates would stop looking rare:

> `p̂` rises → `w → 1` → admission decays to uniform sampling. **The intervention would erase its
> own signal.** So in thinned mode the bank is updated **exactly once per step, on the un-thinned
> candidate pool**, and the committed training forward reads the bank strictly read-only.

**The scout.** `p̂` is normally a post-forward quantity, existing only after the fused student
pass, so there is no density by which to admit tiles *before* committing them. Thinning adds a
**scout**: a third global crop whose transform is `Resize + Normalize` and nothing else (no crop,
flip, rotation, jitter, or blur). Because the DINO objective learns representations *invariant* to
those augmentations, a tile's density is a property of its content and is read correctly from the
clean view. The scout is a single no-grad embed, one extra forward rather than an extra training
step, feeding the bank only; it never enters the losses and, being deterministic, leaves the
augmented crops' random state untouched, so the `weighted` and `off` arms stay byte-identical.

## 13 · Consensus, and a bias check

**Consensus.** Training runs data-parallel across GPUs. If each kept its own bank they would
disagree about which morphologies are rare and drift apart silently. So the bank is a **single
global estimator**: signatures are all-gathered across workers, every worker feeds its bank the
identical stream in the same order, and every update decision is deterministic (ties broken by
index). The banks stay byte-for-byte identical by construction, not by averaging, which is why
the bank's feed is far larger than any one worker's batch.

**The bias check.** A kernel sum at a finite radius carries a resolution-dependent bias. A
deliberately **coarse shadow bank** (the same estimator at about `2s` spacing, `M' = 2^{-d*}M`,
floored to 64) gives a second estimate, and their difference is a pointwise **Richardson** estimate
of the bias, `β̂ = u_2s − u_s`, with `u* = 2u_s − u_2s` the debiased density. Because the coarse
bank sits far below the working regime it is crude, so `β̂` is reported as an *indicator* and not
applied: the correction (`--thin_richardson_correct`) is implemented but off by default. It
measures the bias and reports it, rather than silently correcting for it.

## 14 · The unifying idea: scale-invariance runs through the whole design

Notice how often one idea recurred. The signature is calibrated by the bank's own spacing. The
hit-rate `λ̂ = p/g` divides out placement. The readout sums over a fixed volume so `g` cancels.
The weight's floor `c` rides `p_ref`. The acceptance ceiling `w_max` is recomputed from `p_ref`.
The headline health metric is a *ratio* of percentiles.

Every one of these makes a decision depend only on **relative** density, never absolute
magnitude, because the magnitudes have no fixed units and drift as the encoder trains. Multiply
every density in the system by a constant and nothing moves. That is not a stylistic preference; it
is what lets an unsupervised, self-tuning estimator stay correct across a long run whose scale
climbs tenfold. And when the invariance is broken anywhere, at a frozen ceiling, a fixed radius, or
a percentile that discards magnitude, the module does not crash. It simply stops rebalancing.

*Seen in the runs:* after the gate opens, `p_ref` climbs about 5× in the `a = 0.5` arm
(0.20 → 0.95) and about 10× in the `a = 1.0` arm (0.22 → 2.20), while the acceptance rate settles at
~0.49 in both, unmoved by the tilt or the scale. That steadiness is the scale-invariance doing its
job; the earlier analytic-ceiling fix was exactly the repair for a version where it had been lost
and admissions were starving.

> **The benefit has to survive the scale, or it was never real.**

## 15 · The dashboard: reading the bank's health

An unsupervised estimator isn't checked for correctness; it's watched for the specific ways it can
fail silently. Each signal guards one.

| Signal | What it is | Healthy | Guards against |
|---|---|---|---|
| `lam_spread` | density contrast, `q90/q10` of the hit-rates | **> 2.5** | the estimator going flat, losing rare-vs-common contrast, so the weight flattens and rebalancing becomes a no-op |
| `hit_frac` | share of tiles landing on an existing point | **> 0.5** | starved counters, tallies resting on too few observations |
| `turn_ratio` | point lifetime ÷ `n_eff` (~721) | **> 2** | eviction before a rate settles, so every density reads through counting noise |
| `underresolved` | stored-point spacing vs `L` | **0** | a memory budget coarser than the structure; the cure is more points |
| `thin_accept` | thinned: `N ÷ pool ≈ 1/χ` | steady, above fill line | a stale ceiling; a *decaying* rate is the miscalibration alarm |
| `thin_prof_err` | thinned: binned L1 of committed `p̂` vs target `a·μ` | low | admissions drifting off target, or the top-up fallback firing too often |
| `ess`, `w_mean` | weighted: effective batch and mean weight | ESS near intended | a few tiles owning the gradient (both pin to 1 when thinned) |

`lam_spread` is the master signal, a *ratio*, so it reads shape not scale, and it is the one to
watch above the others: when it collapses toward 1, the whole rebalancing flatlines.

*Live snapshot* (thinned `a = 0.5`, iter 93k): `lam_spread 3.1`, `hit_frac 0.90`, `turn_ratio 2.6`,
`underresolved 0`, `thin_accept 0.49`, `thin_prof_err 0.14`, all in range. The `a = 1.0` arm reads
`turn_ratio 1.0` instead: its 6× pool feeds the bank faster, so entries graduate and turn over more
often in step units. That is a feed-rate effect on the units, not instability, and the density
signals stay healthy.

## Where it stands: what the numbers say so far

The mechanism behaves as designed in the live runs: acceptance is scale-invariant, effective sample
size is exactly 1 under thinning against 0.67 to 0.89 under weighting, and the bank's health signals
sit in range. The downstream picture is honest and still forming. On twelve slide-level MIL
biomarker tasks, the two completed *weighted* arms land within ±3.5% of the untuned baseline, inside
the split-to-split spread on essentially every task and with no consistent direction: neutral, not
yet a win. The *thinned* arms are still training (93k and 70k of 125k steps) and have no downstream
evaluation yet. What is shown here is that the estimator does the thing it claims to do; whether
that reshaped diet improves the encoder is the open question the runs exist to answer.

## 16 · Making it run: the engineering underneath

The estimator has to keep pace with training. The thinned road's appetite, scouting a whole
gathered pool each step and folding it into the census, makes the bank's per-step update the single
heaviest cost, so it is vectorised rather than looped over candidates, the scout runs in half
precision, and augmentation happens on the GPU over only the survivors.

The most delicate cost is invisible in the maths. To let the backbone **compile** into fused
kernels while holding memory low through gradient checkpointing, the standard data-parallel
wrapper was replaced with a **hand-rolled gradient exchange**, because the off-the-shelf one
refuses to coexist with that memory trick. It reproduces the averaged gradient exactly, with one
careful detail: a parameter no worker touched on a step is left genuinely untouched, not zeroed,
so weight decay never nudges it. None of this changes what the curator *does*; it only lets the
whole thing run fast enough to be worth doing.

---

*The method decides **what** the model sees. The engineering only decides how quickly it gets to
see it. For the formal treatment, measured protocols, and references, see [`technique.md`](technique.md).*
