# Behavior curves: the tilt `a` and the thinning mechanism

Two self-contained figures that explain how the curator behaves, in the physicist idiom of
plotting the governing equations across their regimes. Each figure regenerates from its script
with no inputs (`python a_tilt.py`, `python thinning_behavior.py`).

Everything rests on one weight law and one floor:

```
weight       w(p̂) = (p̂ + c)^(-a),        c = c_frac · p_ref   (c_frac = 0.25)
acceptance   A(p̂) = w / w_max = (1 + p̂/c)^(-a),   w_max = c^(-a)
```

`p̂` is the local density from the counted-coverage bank, `p_ref` is its running reference scale
(an EMA of the batch-median `p̂`), and `c` anchors the denominator to that scale, which is what
makes the whole thing scale-invariant. Two knobs with separate jobs: `c` sets *where* the tilt
turns on (the knee), `a` sets *how hard* it bites (the slope).

## 1. The tilt exponent `a`  (`a_regimes.png`)

Three views of the same equation.

- **(1) Weight, log-log.** Two regimes. For rare tiles (`p̂ ≪ c`) the weight saturates at a flat
  ceiling `w_max = c^(-a)`, so an empty region gets a finite weight, not infinity. For common tiles
  (`p̂ ≫ c`) the weight is a power law of slope `-a`. `a` is that log-slope: `a = 0` is flat (no
  rebalancing), and each step up in `a` tilts the common-suppression line steeper while lifting the
  rare ceiling.
- **(2) Acceptance.** The same curve normalized by the ceiling, read as a keep-probability. `a = 0`
  keeps everything (no thinning); higher `a` steepens the cliff so common tiles are rejected harder.
  The dots mark the median tile (`p̂ = p_ref`), kept about 45% of the time at `a = 0.5` and 20% at
  `a = 1.0`.
- **(3) Contrast.** Because `a` is an exponent, the rarest-to-commonest weight ratio grows as
  `R^a`. Doubling `a` squares the contrast: the measured gradient-mass ratios `2.77×` at `a = 0.5`
  and `7.68×` at `a = 1.0` satisfy `7.68 ≈ 2.77²`. The price rides along as falling effective sample
  size (ESS `0.89 → 0.67`), which is the subject of the second figure.

## 2. Thinning-mode behavior  (`thinning_behavior.png`)

Built on one illustrative heavy-tailed density (a lognormal, `sigma = 1.0`) that reproduces the
measured ESS almost exactly (`0.89 / 0.68`) and lands the accept rate and over-draw in range, so the
curves are quantitatively anchored, not just schematic.

- **(1) Matched mass.** The thinned committed distribution `A·μ / E[A]` is identical to the
  weighted effective distribution `w·μ / E[w]`, because `A = w / w_max`. The black markers
  (weighted effective) land exactly on the red thinned curve. Both roads reshape the *same* raw
  stream toward the rare tail; they are two mechanisms for one target diet.
- **(2) ESS is the weighted road's cost.** `ESS = (Σw)² / (B·Σw²)` falls as `a` grows, because a few
  rare tiles hoard the gradient. The thinned arm holds `ESS = 1` at every `a`, since every committed
  tile carries unit weight. The shaded gap is the effective batch the weighted arm gives up and the
  thinned arm keeps.
- **(3) Over-draw is the thinning road's cost.** To commit `N` tiles at acceptance `E[A]` you must
  draw `χ = 1/E[A]` candidates per commit, and `χ` rises with `a`. The weighted arm needs no
  over-draw. So the two methods trade the *same* tail emphasis for costs on *orthogonal* axes:
  weighting pays in ESS, thinning pays in pool size and scout compute.
- **(4) Fill statistics.** Survivors per step are roughly `Binomial(pool, E[A])`. At `a = 1.0` a 3×
  pool centers below `N = 256` and under-fills (the top-up fallback fires and matched mass breaks),
  which is why the `a = 1.0` arm runs at 6×. At `a = 0.5` a 3× pool clears `N` comfortably.

## The duality in one line

Weighting and thinning reach the identical committed diet (matched mass). Weighting keeps every tile
at unequal weight and pays in effective sample size. Thinning keeps fewer, rarer tiles at equal
weight and pays in over-draw. Same target distribution, orthogonal costs, which is the whole reason
both arms exist in the ablation.
