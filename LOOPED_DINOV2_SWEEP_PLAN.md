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

## Schematic of the training method

![Looped DINOv2 — training schematic with explicit residual pathway](figures/looped_dinov2_method_residual.png)

Vector originals: [`figures/looped_dinov2_method_residual.svg`](figures/looped_dinov2_method_residual.svg) (preferred for inclusion in the paper) and [`figures/looped_dinov2_method_residual.pdf`](figures/looped_dinov2_method_residual.pdf). Source: [`figures/generate_looped_dinov2_method_residual.py`](figures/generate_looped_dinov2_method_residual.py). Re-run with `python3 figures/generate_looped_dinov2_method_residual.py`.

**What the figure shows.**

The figure draws one full forward pass of the looped student backbone for `T_max = 4` recursion steps, organized as four horizontal lanes.

- **Recursion lane (top, gray).** The state circles `z_0, z_1, z_2, z_3, z_4` and the shared block `f_θ` are drawn in series: `z_0 → f_θ → ⊕ → z_1 → f_θ → ⊕ → z_2 → ...`. The same `f_θ` rectangle is repeated four times, signaling weight tying — emphasized by the purple brace and label `shared parameters f_θ (applied T_max times)` running across the entire lane. The arrow direction inside the lane (left to right) is the *time axis* of the recursion, not depth in the parameter sense.
- **Time-embedding inputs (top of the recursion lane).** Each `f_θ` invocation receives a per-step learnable bias `τ_t` from a small yellow circle directly above it. The four `τ_t` circles are distinct parameters, indexed by recursion step; they are what breaks the otherwise-identical behavior of the shared block across steps and let the recurrence carry out a different operation at each `t`.
- **Input-injection bypass (purple, between recursion and halt rows).** A horizontal purple highway taps `z_0` and feeds it back into the residual sum node `⊕` *between* every `f_θ` and its `z_t`. The `⊕` symbols make the residual structure `z_t = f_θ(...) + z_0` graphically explicit — the same convention used in ResNet / Transformer diagrams for skip connections. This visual emphasizes that the recurrence is anchored to the input at every step rather than drifting away.
- **Halt-head row (green, middle).** Each `z_t` is tapped (junction dots on the recursion lane) and feeds an image-pooled-CLS halt head `h_t = σ(W_halt · z̄_t)`, drawn as a small green circle. The four `h_t` circles together produce the PonderNet step-marginal distribution `{p_t}` — that's the bus that exits the row to the right.
- **Per-step loss row (orange, lower).** Each `z_t` also feeds a per-step loss head `L_t` (a rounded orange box). The row tag clarifies these are evaluated student-vs.-teacher at each recursion step (`student z_t` vs. `teacher z_t^T`); the teacher is intentionally not redrawn here — the figure's focus is the student recurrence and how step outputs route into the loss mixture.
- **Mixture / total-loss box (right, beige).** Both buses (`{p_t}` from the halt row, `{L_t}` from the loss row) flow into the total loss: `Σ_t p_t · L_t + β · KL(p ∥ Geom(λ_p))`. This is the single training objective; per-step student supervision is reweighted by the halting marginals and regularized by KL toward the truncated geometric prior.

**Why the residual pathway is drawn explicitly.** Earlier draft schematics treated the input injection as a side annotation on the shared-stack box. The bypass-and-`⊕` rendering used here is the same convention papers reach for when explaining a skip connection in a backbone, and it makes a content-rich claim of the design legible at a glance: the recurrence's stability is bought by the input injection, not by anything internal to the shared block.

**What is intentionally absent from the figure.** The teacher branch (run once at `T_max`, no halting head, EMA of student, gradients detached, Sinkhorn-Knopp targets shared across all student steps) is described in [`LOOPED_DINOV2.md`](LOOPED_DINOV2.md) §1 and [`training/looped_step.py`](training/looped_step.py). The packed multi-crop input (2 globals + 6 locals per image, BlockDiagonalMask) and the per-step decomposition of `L_t = L^DINO_t + w_iBOT · L^iBOT_t` plus the final-step KoLeo term are also documented there. The figure deliberately abstracts those into the single `L_t` box and the `z_0` input symbol so the recurrence + halt + mixture story stays the visual focus.
