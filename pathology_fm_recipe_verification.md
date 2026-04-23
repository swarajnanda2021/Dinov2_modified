# Pathology FM Recipe — Verification Notes

This file tracks the verification checklist from
`pathology_fm_standardization.md` Section "Testing and Verification".
Static checks were run in the implementation environment (no GPU).
Runtime items must be re-verified on an H100 node before the branch is
declared complete.

## Static checks (run now)

- [x] All edited modules parse cleanly:
      `python -c "import ast; ast.parse(open(path).read())"`
      for configs/config.py, losses/kde_loss.py, losses/__init__.py,
      data/transforms.py, data/datasets.py, utils.py,
      training/trainer.py, run_with_submitit.py.
- [x] Grep confirms no leftover hardcoded `num_register_tokens=4` or
      `qk_norm=False` at the ModernViT call site in trainer.py.
- [x] ModernViT already exposes `qk_norm` and `num_register_tokens`
      through `VisionTransformer.__init__`; no model code changes
      required.
- [x] `fp16_scaler` is forced to `None` and `bf16_mode=True` when
      `args.use_pathology_recipe=True`, so the GradScaler branch in the
      backward / optimizer step is skipped entirely.

## Runtime checks (to be run on H100 before merging)

1. **No-op check.** Launch with `--use_pathology_recipe=False` and
   confirm the first 10 iterations' loss values match the `consolidated`
   branch. Acceptance: identical numeric losses.

2. **ECT branching.** Temporarily add prints in
   `TMEDinoTransforms.__call__` to log branch selection. Acceptance:
   40x tiles route to ECT with ~40% frequency; 20x tiles always go to
   standard. Remove prints after verification.

3. **KDE numerical stability.** Feed `KDELoss` a toy batch where all
   feature vectors are identical clones. Acceptance: finite loss. For
   contrast, `KoLeoLoss` on the same batch returns inf / very large.

4. **Auto-gate trigger.** Launch with `embeddingdim=1280`,
   `use_pathology_recipe=True`. Acceptance: log contains
   `[pathology recipe auto-gate]`, shows `qk_norm=True`,
   `register_tokens=8`, and `StableAdamW active`.

5. **bf16 end-to-end.** With the recipe on, confirm no `GradScaler`
   operations appear in the training loop — inspect that
   `fp16_scaler is None` for the entire run.

6. **Single-epoch smoke test.** Full training launch for 100 iterations
   with `--use_pathology_recipe=True` on the ViT-L config. Acceptance:
   no NaN, no shape errors, losses within an order of magnitude of the
   `consolidated` baseline.

## Files changed

- `configs/config.py` — new CLI args.
- `losses/kde_loss.py` — new file (KDELoss).
- `losses/__init__.py` — export KDELoss.
- `data/transforms.py` — Random90Rotation, probabilistic ECT routing.
- `data/datasets.py` — recipe arg plumbing (three classes).
- `utils.py` — StableAdamW class.
- `training/trainer.py` — auto-gate, recipe overrides, KDE/KoLeo switch,
  bf16 mode, StableAdamW swap, teacher_temp override, ModernViT wiring.
- `run_with_submitit.py` — magnification table comment, commented-out
  toggle stanza.

## Out of scope (not in this branch)

DINOv2-LVD-142M initialization, multi-magnification sampling beyond
current data, Macenko / stain transfer, GPFM multi-teacher, PLUTO MAE
losses, Midnight HED color augmentation, and any changes to the
existing `--use_patch_prototype_clustering`, `--use_semantic_ibot`, or
`--use_typicality_dampening` toggles.
