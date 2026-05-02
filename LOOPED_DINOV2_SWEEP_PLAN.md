# Looped DINOv2 — (L, T_max) Sweep: Experimental Plan

Planning addendum to the looped DINOv2 implementation branch. Specifies the
sweep matrix, the primary figure, and the postprocessing investigations
required for the paper. **This is not an implementation directive** — the
technique is fully specified in [`LOOPED_DINOV2.md`](LOOPED_DINOV2.md). This
document exists so the sweep is launched against a fixed plan and the
postprocessing produces the right artifacts for the paper.

## Theme

A controlled study of weight-tying intensity in DINOv2 backbones, evaluated
through the lens of mutation prediction. The sweep traverses the weight-tying
axis at matched effective depth — from canonical dense ViT-B to fully-tied
Universal-Transformer-style — and characterizes how mutation ROC-AUC and
learned halt-depth distributions vary along this axis. **The contribution is
the curve, not a horse-race verdict against dense.**

## Sweep matrix

At matched effective depth $L_{\text{eff}} = L \cdot T_{\max} = 12$:

$(L, T_{\max}) \in \{(12, 1),\; (6, 2),\; (4, 3),\; (3, 4),\; (2, 6),\; (1, 12)\}$

- $T_{\max} = 1$ reduces to the canonical dense ViT-B baseline (with the
  spec's degenerate-config caveat: sandwich LN and outer residual remain
  present, time embedding becomes a constant additive bias).
- $T_{\max} = 12$ is fully-tied Universal-Transformer-style.
- Intermediate points trace the weight-tying intensity axis.

All other hyperparameters are held at the spec defaults (LR, weight decay,
batch size, schedule, stochastic depth 0.4, LayerScale, KoLeo, iBOT and DINO
weights, embedding dimension 768, multi-crop config, augmentation pipeline).
The single varied axis across runs is $(L, T_{\max})$.

## Primary figure

One graph. X-axis: $(L, T_{\max})$ configurations, ordered by $T_{\max}$ from
1 to 12. Y-axis: mutation-prediction ROC-AUC. Five lines, one per mutation.
This is the headline result for the paper.

**Decisions to lock before launch:**

1. Which 5 mutations.
2. Which public benchmark(s) — internal benchmark may serve as appendix sanity
   check.
3. Probe protocol: tile-level linear probe, slide-level (ABMIL/TransMIL), or
   both.

## Required postprocessing investigations

For each $(L, T_{\max})$ configuration with $T_{\max} \ge 2$ (halting is
non-trivial). Both run on the trained checkpoints + per-tile halt
distributions saved during the sweep — no additional training.

**Investigation 1 — Halt-pattern qualitative study.** Sample 10k tiles. Group
by halt depth. Surface representative tiles per group, by random sampling
within group and by tiles nearest the group centroid in feature space. Visual
inspection: do early-halting and late-halting tiles look qualitatively
distinct? This is the necessary sanity check that halting is content-driven
rather than decorative. **Without it, the primary figure is uninterpretable.**

**Investigation 2 — Per-mutation halt-depth distributions.** For each of the
5 mutations, plot the halt-depth distribution over tiles labeled
mutation-positive vs mutation-negative. If positive tiles for a given
mutation systematically halt at greater depth than negatives, the looped
backbone is allocating compute toward that mutation's morphological signal.
A clean separation on any of the 5 mutations is a biological-interpretability
finding worth surfacing in the main paper alongside the primary figure.

## Out of scope (documented for follow-up work)

- Halting-on vs halting-off ablation at the chosen $(L, T_{\max})$.
- Architectural ablations of sandwich LN, time embeddings, input injection.
- Per-step linear probes on intermediate $z_1, \ldots, z_{T_{\max}}$.
- Probe quality conditioned on halt depth.
- Slide-level MIL uncertainty correlated with halt-depth statistics.

These are scientifically interesting but not part of the current sweep's
deliverable. They require either additional training runs or evaluations that
exceed the paper's scope.

---

## Implementation notes (link to the code)

The sweep matrix is launched through the existing CLI. For configuration
$(L, T_{\max})$:

```
python main_train.py \
    --use_looped_backbone=True \
    --shared_stack_L <L> \
    --recursion_T_max <T_max> \
    --vitdepth <L>          # informational; layer-LR-decay uses shared_stack_L
    --use_pathology_recipe=True \
    [other defaults]
```

For the $(L=12, T_{\max}=1)$ baseline run, the sandwich-LN / time-embedding /
input-injection structure of [`SharedStack`](models/vision_transformer/shared_stack.py)
is still applied, so the run is *not* literally identical to a dense ViT-B
forward pass. If a strict dense-ViT-B baseline is also wanted (no sandwich
LN, no time embedding, no input injection), launch with
`--use_looped_backbone=False --vitdepth 12` and treat that as a separate
reference point on the figure.

Halt distributions per tile are emitted at training time as part of the
metric-logger output (`h_step_{t}` averages and `ponder_mean_halt_step` per
iteration). For Investigation 1 / 2, augment the eval pipeline to dump the
per-tile $h_1, \ldots, h_{T_{\max}}$ vector at inference (single-tile, pool-
of-one CLS) so the halt-depth distribution can be reconstructed. This is a
one-line change at the inference driver and does not affect training.

## Schematic

The training method that this sweep evaluates is shown in
[`figures/looped_dinov2_training.svg`](figures/looped_dinov2_training.svg).
Source: [`figures/generate_looped_dinov2_schematic.py`](figures/generate_looped_dinov2_schematic.py).
