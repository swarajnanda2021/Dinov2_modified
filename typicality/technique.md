# The Typicality Curator: Online Redundancy Curation for Self-Supervised Pathology Foundation Models

*Technique section (paper draft). Self-contained: it carries the motivation, the abstract
curation problem, the empirical characterization of the training stream that fixes every
design constant, the estimator-theoretic derivation of the converged algorithm, the
pseudocode, the two loss-consumption variants (weighted-loss vs. adaptive-temperature),
and the honest resolution ceiling. All quantitative claims are inference-only measurements
on the baseline run `FMC_ViT-B_stab_baseline_rev3` and the `stab_bc_weightedloss_rev3`
checkpoint; nothing was retrained to produce them.*

---

## 1. Motivation: the data is the problem

Self-supervised recipes of the DINOv2 family assume a *curated* input distribution. This is
not incidental: the DINOv2 corpus LVD-142M was **built** by an explicit offline pipeline —
self-supervised copy-detection **de-duplication** followed by **retrieval-based
rebalancing** against curated seed sets [Oquab et al., 2023]. ImageNet-scale SSL works in
part because ImageNet is one-object-per-image, de-duplicated, and class-balanced. The recipe
presumes that curation already happened.

Computational histopathology violates every one of those assumptions at the tile level.
Whole-slide images (WSIs) are soft-curated at the **slide** level (a slide carries a
diagnosis), never at the **tile** level. Tiling a WSI produces thousands of tiles per slide,
dominated by redundant, near-duplicate morphology (stroma, adipose, background), with no
de-duplication and no balancing. Ported directly, a DINOv2 objective spends its capacity on
the redundant majority. Practitioners then blame the architecture or bolt on losses, without
asking whether the **input distribution** is the pathology.

The **typicality curator** supplies the missing curation *inline and in-domain*: one cannot
hand-de-duplicate 287M tiles by retrieval against a curated seed, so we estimate tile
redundancy online, in the model's own evolving morphology space, and convert it into a
per-tile loss modulation. It is the online, in-training analogue of the LVD-142M curation
step.

## 2. Notation and the score

The backbone (ViT-B, patch 16, register tokens) produces a pre-norm `[CLS]` token. A DINO
projection head maps it to an L2-normalized 256-d **bottleneck** `z`. A set of
**representative prototypes** `R ∈ ℝ^{256×256}` (row-normalized `R̂`) defines the **signature**

```
s = z · R̂ᵀ ∈ ℝ^256 ,        s_k = cos(z, R̂_k)          (raw cosine, no softmax)
```

so a signature is 256 cosines-to-prototypes. A signature is **peaky** when one component
dominates (`max_k s_k` large — the tile strongly resembles one morphology mode) and
**diffuse** when all components sit at the random-projection floor (`max_k s_k ≈ 0.16` for a
random unit vector through `R̂`). The curator maintains a bounded memory ("bank") of
signatures and, for each incoming tile, emits a **typicality score** `t(x) ∈ [0,1]` that is
monotone in the local density of the signature distribution — high for redundant morphology,
low for rare/novel morphology. `t` is then consumed by a loss modulation (§8).

## 3. The abstract curation problem

Strip the domain away and the curator is a **bounded-memory online density problem**. Let
`(ℝ^d, ρ)` be a metric space, `μ` an unknown measure with density `p` (heavy concentration,
a light outlier scatter), and `x₁, x₂, …` an i.i.d. stream processed in blocks of size `κ`
(the gathered batch). We maintain `B_t`, `|B_t| ≤ M`, updated by an **eviction kernel**, and
must emit `t(x)` from `B_t` alone. The objective is **importance weighting**: choose weights
so the reweighted stream behaves like a sample from a flattened target `ν ∝ μ^γ` — i.e.
estimate a monotone functional of `p(x)` online, in memory `M ≪ N`.

Two natural kernels give opposite behavior for the same distance statistic
`d(x;B)=min_{b∈B} ρ(x,b)`:

- **Density-faithful (F1).** If `emp(B) → μ`, then `d(x;B) ≍ (M·p(x))^{-1/d}` — monotone in
  `p`, but with `M ≪ N` a faithful sample barely represents light regions.
- **Repulsive / coverage (F2).** Evict-nearest relaxes `B` to an even covering of
  `supp(μ)`; `d(x;B) →` a constant packing radius, carrying no density information, and an
  isolated atom is never anyone's nearest neighbor, hence **never evicted** (an absorbing
  state — the estimator is non-ergodic).

The design problem is to choose a kernel + readout that reads density robustly across the
mass range without either failure. **The correct resolution — derived in §6 — separates the
two jobs: point positions do coverage; a per-atom counter does mass.**

## 4. Empirical characterization of the stream

Every constant below was measured; together they fix the entire design and — importantly —
show the operating regime is **mild**, so the curator is justified by principle and
robustness rather than by an extreme-skew necessity.

| quantity | symbol | measured value | how |
|---|---|---|---|
| intrinsic dimension of the signature support | `d*` | **≈ 9.5** (two-NN 9.2, linear PR 9.5; ~10–11 at 50k) | two-NN [Facco et al., 2017] + covariance participation ratio |
| prototype row-space rank (readout cap) | rank `R̂` | **≈ 44** (185 at init) | participation ratio of `R̂` spectrum |
| batch-level stationarity | VR, ρ(1) | **VR 1.7–2.0, ρ(1) ≈ −0.08** (IID); slide-grouped control VR 21.2, ρ(1) +0.48 | variance-ratio + lag-autocorrelation of batch-mean features |
| tile-scale burst factor | `b` | **≈ 1.2**; same-slide adjacency 3.7%, run-length ≈ 1 | signature autocorrelation + slide-run structure in stream order |
| autocorrelation length | `τ_ac` | **< 256 tiles** (< one block); cross-block ≈ 0 | signature autocorrelation vs lag |
| density dynamic range | `R` | **≈ 10³** (p99/p1), 10³·⁸ (p99.9/p0.1), 10⁴·⁸⁵ extreme | k-NN local-density ratio; cross-checked by log-density sill 2.2 → e^{2·2.33·√2.2} ≈ 10³·⁰ |
| mode structure | — | **continuum**: eff-#modes grows with resolution (30→240), no dominant mode | k-means sweep (equalizes mass, so used only for structure, not `R`) |
| one-off / artifact rate | `ρ_out` | **∼ N^{−0.8}, no floor → ≈ 10⁻⁵** at full scale | isolation rate vs sample size at coverage scale |
| log-density correlation length | `L` | **≈ 0.16** (‖∇log p‖⁻¹), 0.42 (decorrelation range); sill 2.2 | semivariogram of the k-NN log-density field |
| memory-bound cell scale at M=8192 | `s_M` | **≈ 0.14 ≈ L** | covering radius `(V/M)^{1/d*}` |
| drift horizon | `N_drift` | **≈ 6.7×10⁴ blocks** (0.1-CKA displacement); unit-scale ~10× longer | post-50k bottleneck CKA drift rate |

Four consequences drive the design:

**(a) When to start — representation stability (→ 50k late-fill).** The signature space is a
projection of the classhead bottleneck, not of the raw backbone, and the two settle on very
different schedules. Measuring linear CKA vs. the final (124k) model on a fixed 76,800-tile
probe:

- raw `[CLS]` (MIL feature) is only **55% CKA-converged at 50k** (0.547) — no knee there,
  ~44% of its total representational path lies after 50k;
- the **bottleneck is 94% converged at 50k** (0.944), velocity already ~10× below its peak.

A fixed-final-head decomposition (apply the frozen 124k head to every checkpoint's `[CLS]`)
gives **0.976 at 50k** — *more* converged than with the co-evolving head — proving the
stability is a genuine **early-forming stable subspace of the backbone**, not head
co-adaptation. So the signature the curator reads is essentially formed by 50k, which is why
the bank must be **filled only after ~50k** (the "late-fill" gate). Caveat retained: *local*
neighborhood structure is only 62% settled at 50k and finishes reorganizing by ~90k, so the
first ~40k iterations of curation run on globally-set / locally-still-settling signatures.

**(b) The regime is low-dimensional and mildly skewed.** `d* ≈ 9.5` (not hundreds) and
`R ≈ 10³` (not 10⁶). This makes a distance readout viable (spacing ratio `R^{1/d*} ≈ 2–3×`,
resolvable) and — because the mass ratio is modest and the support is a **continuum with no
isolated rare modes** — removes the reason to spread the bank toward coverage. The absorbing
outlier pathology of F2 retires to a corner case (`ρ_out ≈ 10⁻⁵`, with `N^{−0.8}` decay and
no artifact floor — a continuum with no atomic one-offs).

**(c) The stream is near-IID at the tile scale.** `b ≈ 1.2`, `τ_ac < κ`: the 16-way loader
interleave already whitens it. Per-block decay is fully whitened; the hard-window vs. decay
distinction is moot.

**(d) Counts never go stale.** The counter half-life (~250 blocks) is `t½/N_drift ≈ 3.7×10⁻³`
of the drift horizon — counts refresh ~270× faster than the encoder drifts.

## 5. Diagnosis of the coverage-only incumbent (why the design had to change)

The shipped bank (`rev3`) is the F2 kernel — admit-most-novel, **evict-nearest**, scored by
calibrated distance `t = 1 − Φ((d − µ)/σ)`, `k=1`. On the 124k checkpoint it fails
completely, and the failure is exactly what F2 predicts:

- **Seeded with junk.** `M = 8192`, per-rank batch 256 → the bank fills in ~32 steps, while
  the encoder is untrained and `R` is fresh; the `warmup=15000` gate governed only *loss
  modulation*, not *filling*. Seed signatures are diffuse: bank peak median **0.172**, i.e.
  the random-projection floor (0.159).
- **Never cleared.** Evict-nearest cannot expel isolated atoms, so the junk plateaus: the
  peak-histogram is **bimodal** — 78.5% in a spike at the random floor (0.15–0.20) and 21.5%
  genuinely peaky (median 0.715), with an empty valley between. Real freshly-encoded tiles
  are peaky (median 0.825, 0% diffuse), so the diffuseness is a property of the *stale bank*,
  not the data.
- **Inert signal.** Every real tile scores far from the junk bank (`d ≈ 3.7µ`, `t ≈ 0.08`),
  so the loss weight `w = 1 − β·t ≈ 0.96` is uniform — the modulation did nothing
  differential, unnoticed across ~45 recipes for lack of bank-health logging.
- **Structural flatness.** Even with a clean bank, a coverage net gives `d(x;B) ≈` const on
  support, so `Ψ((d−µ)/σ)` is a near-flat support-membership score — it cannot discriminate
  common from rare regardless of dimension. The coverage-only kernel is the **wrong end of
  the dial**.

Secondary: `R`'s effective rank collapses 185 → 44 over training, capping morphology
*resolution* (but not the cause of the all-rare failure).

## 6. Derivation of the converged curator

**6.1 Separate coverage from mass.** A bare point set forces geometry to carry both *where*
(coverage) and *how much* (mass), and no single placement does both across the mass range.
Attach to each atom a **decayed hit-counter** and a **decayed exposure**: positions do
coverage, counters do mass. This is a cache: the bank is a set of cached morphology anchors,
a "hit" is a tile landing in an anchor's cell, and the replacement policy is the design
object. Recency-eviction is **LRU**; count-eviction-with-aging is **LFU-with-aging**; the
analysis tool is the characteristic-time (Che) approximation [Che et al., 2002].

**6.2 The per-atom sufficient statistic `(S, E)`.** For a Poisson hit process under geometric
discounting, keep two decayed scalars per atom — decayed hits `S` and decayed exposure `E` —
and use the **rate** `λ̂ = S/E` for **both** eviction (evict `argmin λ̂`) and readout. This
subsumes the alternatives: recency `τ` is the degenerate compression sufficient only for the
binary "hit in window?" test (it discards magnitude, needed for the score); undecayed count
`c` discards time (a "zombie" with large historical count survives after `μ` moves on). The
extra scalar that protects newborns is **exposure `E`**, not recency: a fresh atom has small
`E`, so a single hit already gives it a healthy rate — graded protection, not a blunt window.

**6.3 The dial and its collapse on a continuum.** An eviction kernel that admits with
probability `∝ D^α` induces a stationary law `emp(B) ∝ p^γ` with `γ = d*/(d*+α)` (mean-field
balance; the same magnification exponent `ρ^{d/(d+r)}` as optimal quantization [Graf &
Luschgy]; `α=1` is Meyerson online facility location [Meyerson, 2001], `α=2` is the k-means++
D² rule [Arthur & Vassilvitskii, 2007]). The argument for `γ < 1` (coverage) runs entirely
through **discrete light modes**: with isolated rare modes, a faithful sample starves them.
On the measured **continuum** that cap is gone — a region of mass `m` gets `mM` points
regardless of density — so the readout error is monotone decreasing in `γ` with **no interior
optimum**, and the answer for the distance readout is `γ → 1` (plain reservoir + k-NN). For
the count readout the bias-variance optimum of the variable-bandwidth histogram gives
`s*(x) ∝ (p·N)^{-1/(d*+2)}`, i.e. `γ = d*/(d*+2) ≈ 0.83`. Both are **near-proportional**, a
world away from the coverage end; we place atoms at `γ ≈ 0.83–1` and note the counts make the
system nearly insensitive to the exact value.

**6.4 The readout: centered pooled rate.** Reading the nearest single atom's rate places the
query off-center in its cell — a first-order `∼ s/L` bias. Estimate instead a
**kernel-weighted, query-centered average of the `j ≈ 32–64` nearest atoms' `S/E`**; the
first-order gradient term cancels by symmetry, leaving curvature-order error without
systematic sign. Storage resolution `s` floors the *bandwidth*, not the readout *order*.

**6.5 Distance vs. counts, bandwidth-matched.** At `γ ≈ 1` both readouts count points in the
same ball around `x`: the net contributes `M`, the decayed stream contributes `N_eff`. The
count readout wins uniformly once `N_eff > M`, with sd advantage `√(N_eff/M)`. At
`η = 0.997/block`, `N_eff ≈ 330` blocks; matched (pool ~64 atoms) the crossover is
`N_eff ≈ M ≈ 8` blocks overall (~25 blocks in the bottom density decile). So the distance
readout is a **cold-start estimator for the first few dozen blocks** and thereafter a
permanent **consistency probe** (`log q̂_dist` should track `γ·log p̂_count`; the residual
field localizes drift or calibration error), not the primary estimator.

**6.6 Eviction timescale and scratch slack.** With `b ≈ 1.2`, `τ_ac < κ`, per-block decay is
whitened. The slowest cells need retention `T_need ≈ 3M·ln(N/3Mδ) ≈ 250` blocks; the
emergent memory-bound staleness threshold `T_C ≈ M·ln(1/ρ_adm)` is shorter, so under pressure
pure recency would churn the slow decile. A small **transient scratch pool** fixes it: reserve
`h·M` slots with `h ≥ ρ_out·T_need/M ≈ 0.3%` (well inside a 1% cap); transients then absorb
all evictions (their FIFO age exceeds any kept atom's staleness) and the explicit window never
binds — the operative timescale is the **emergent `T_C(M)`**. Set the decay half-life for the
*readout's* sake at `t½ ≈ T_need ≈ 250` blocks (`η ≈ 0.997`), which dwarfs both `κ` and
`τ_ac` and is `≪ N_drift` (§4d), so counts stay fresh.

**6.7 The resolution ceiling (stated honestly).** Since `s_M ≈ L` at `M = 8192`, **every**
`M`-bounded readout — distance, counts, any `γ` — has bandwidth `≥ s_M ≈ 0.88 L` and
estimates only the `L`-smoothed density `p * φ_L`. The sub-`L` component of the field (a third
to half the sill, ~0.8–1.0 log-units of sd) is invisible to any bounded summary; readouts
differ in variance and centering, not in what they can resolve. The only levers are memory
(`×3^{d*}` per refinement — infeasible) or **reducing `d*` upstream**. This is a property of
the regime, not of the algorithm.

## 7. The converged algorithm

```
ONLINE REDUNDANCY CURATOR  (converged form)

State (global, all-reduced & deterministic across data-parallel ranks):
  atoms B = { (b_i, S_i, E_i, born_i) }, |B| ≤ M          # position, decayed hits, decayed exposure, birth block
  M_scratch = ceil(h·M),  h ≈ 1%                          # FIFO transient pool (absorbs evictions)
  s ≈ L ≈ 0.15  (memory-bound s_M);  j ≈ 32–64 pooling;  η ≈ 0.997/block (t½ ≈ 250 blocks)
  T_warm = 50_000 iters;  warmup_blocks ≈ 30;  β (or temperature schedule) for §8

per training block  X = {x_1..x_κ}  (gathered signatures, one iteration):
  if iter < T_warm:                                       # (a) representation not yet settled
      return uniform weights                              #     — bank untouched (late-fill)

  # 1. decay (lazy, per-atom timestamp)
  for i: S_i *= η ; E_i *= η

  # 2. READOUT — score each incoming tile BEFORE updating
  for x in X:
      N ← j nearest atoms of x
      if iter < T_warm + warmup_blocks:                   # cold-start
          λ̂(x) ← calibrated k-NN distance readout(x, B)   #   (§6.5)
      else:
          λ̂(x) ← Σ_{i∈N} κ_h(x,b_i)·(S_i/E_i) / Σ κ_h    #   centered pooled rate (§6.4)
      t(x) ← calibrate(λ̂(x)) ∈ [0,1]                     #   monotone, high = common
      # consistency probe: assert log λ̂_dist(x) ≈ γ·log λ̂_count(x); residual → drift/calib alarm

  # 3. UPDATE counts (mass) — after scoring
  for x in X:
      b* ← nearest atom ; D ← ρ(x, b*)
      E_{cell(x)} += 1                                     # exposure of the region
      if D ≤ s:  S_{b*} += 1                               # hit
      else:      admit (b=x, S=1, E=1) into scratch
                 if |B| > M: evict argmin_i S_i/E_i  over atoms NOT in grace/scratch   # LFU-with-aging

  # 4. (optional) update ~10–100 macro-cells → calibration / drift monitor (not the estimator)

  return  w(x) = modulate(t(x))                            # §8: weighted-loss OR adaptive-temperature
```

Global determinism (all-gather of signatures; tie-breaks pinned to lowest index; all-reduce
of count increments) keeps the bank byte-identical across ranks and multi-node reproducible.

## 8. Consuming the score: weighted loss vs. adaptive temperature

The curator produces `t(x)`; two families consume it, differing in **what** they change.

**8.1 Weighted loss — modulates magnitude only.** `L = Σ_x (1 − β·t(x))·L_DINO+iBOT(x)`,
`β ∈ [0,1]`. A typical tile contributes a smaller-magnitude gradient; the *target* it is
trained toward (its teacher assignment) is unchanged. This is exactly the importance-weighting
of §3 — it reshapes the *effective sampling distribution* toward `ν ∝ μ^γ` (flattened
redundancy) while leaving each tile's learning signal intact. It is conservative, bounded,
and easy to reason about (a pure rescaling of per-tile contributions). This is the
`stab_bc_weightedloss` lineage.

**8.2 Adaptive temperature — modulates magnitude *and* distribution.** The DINO/iBOT loss is a
cross-entropy between temperature-sharpened softmax assignments; making the temperature a
function of typicality, `τ(x) = τ₀·f(t(x))`, changes the **shape** of the target distribution
for that tile — a softer (higher-entropy) target for a typical tile spreads probability mass
across prototypes rather than committing it, redistributing *what* the tile teaches, not just
*how much*. This is strictly more aggressive: it alters both the gradient magnitude and the
distributional target. It can, in principle, actively flatten over-represented modes in the
prototype assignment itself, but it couples the curator into the representation geometry (a
closed loop, §9) and is harder to bound. This is the `stab_bc_adaptivetemp` lineage.

The distinction matters for the paper's claim: **weighted loss is a de-biasing of the data
distribution; adaptive temperature is a de-biasing of the learning target.** The curator
(§7) is identical for both; only the consumer differs.

## 9. `rev4` as a stepping stone

`rev4` is a partial instance of the converged design — the **coverage substrate without the
counting layer** — and is best read as the last coverage-only variant before the dial was
walked to near-proportional:

- **Kept from rev4 (correct):** the **global all-gathered bank** (fixes rev3's per-rank
  fragmentation — N ranks held N independent, worse estimators); **deterministic churn**
  (lowest-index tie-breaks + cross-rank fingerprint check) so the global bank is
  byte-identical; the **late-fill gate at 50k** (`typicality_warmup_iters = 50_000`, now
  gating the *entire* bank block, not just modulation — this is the §4a fix and removes rev3's
  root cause); and **bank-health logging** (`diffuse_frac`, `t<0.1`).
- **Superseded (the wrong dial end):** rev4 retains **evict-nearest + a distance readout**,
  i.e. `γ ≈ 0` coverage with a structurally flat score (§5). The converged design replaces the
  eviction rule with **min-rate `S/E` (LFU-with-aging)**, the placement with
  **near-proportional `γ ≈ 0.83–1`**, and the readout with the **centered pooled rate**
  (distance demoted to cold-start + consistency probe), plus the **1% scratch pool** and the
  **`η ≈ 0.997` decay**.

So the ablation ladder for the paper is: rev3 (local, junk-seeded, inert) → rev4 (global,
late-filled, deterministic, but coverage-flat) → converged counted curator (near-proportional
placement, `(S,E)`-rate readout, LFU-aging eviction).

## 10. Limitations

- **Sub-`L` blindness (§6.7).** At `M = 8192`, `d* ≈ 9.5`, no `M`-bounded summary resolves
  below the density correlation length; the only real lever is reducing `d*` upstream.
- **Single-slice measurement.** `R`, `ρ_out`, and the density tail were measured on one
  76,800-tile LUAD-ish slice; the full 287M corpus may be more skewed (the counts make the
  system insensitive to the exact `γ`, which mitigates this).
- **Closed loop.** The score reweights the loss that updates the encoder that generates the
  signatures — `μ` is endogenous, outside the i.i.d. framework. The 50k late-start (turning
  curation on only after the signature space is ~settled, §4a) is the safeguard; formal
  stability of the loop is not established, and adaptive-temperature (§8.2) tightens the
  coupling relative to weighted-loss.

## References

- Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024. (LVD-142M curation: SSL dedup + retrieval rebalancing; KoLeo.)
- Caron et al. *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021.
- Zhou et al. *iBOT: Image BERT Pre-Training with Online Tokenizer.* ICLR 2022.
- Sablayrolles et al. *Spreading vectors for similarity search (KoLeo).* ICLR 2019.
- Facco et al. *Estimating the intrinsic dimension of datasets by a minimal neighborhood information.* Sci. Rep. 2017. (two-NN.)
- Levina & Bickel. *Maximum Likelihood Estimation of Intrinsic Dimension.* NeurIPS 2004.
- Kornblith et al. *Similarity of Neural Network Representations Revisited (linear CKA).* ICML 2019.
- Graf & Luschgy. *Foundations of Quantization for Probability Distributions.* Springer LNM 1730, 2000. (magnification law ρ^{d/(d+r)}.)
- Meyerson. *Online Facility Location.* FOCS 2001.
- Arthur & Vassilvitskii. *k-means++: The Advantages of Careful Seeding.* SODA 2007. (D² sampling.)
- Che, Tung & Wang. *Hierarchical Web Caching Systems: Modeling, Design and Experimental Results.* IEEE JSAC 2002. (characteristic-time approximation.)
- Einziger, Friedman & Manes. *TinyLFU: A Highly Efficient Cache Admission Policy.* ACM TOS 2017.
- Bifet & Gavaldà. *Learning from Time-Changing Data with Adaptive Windowing (ADWIN).* SDM 2007.
- Vitter. *Random Sampling with a Reservoir.* ACM TOMS 1985.
