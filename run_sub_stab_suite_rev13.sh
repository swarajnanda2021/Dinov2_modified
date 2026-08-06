#!/bin/bash
# run_sub_stab_suite_rev13.sh -- parametrized launcher for the "new-baseline" ViT-B ablations.
#
# REV13 = the MIDDLE M point of the resolution sweep (completes 8k / 16k / 32k, technique.md 4.1):
#   - bc_thinned_lo/hi go to M = 16384 (2x base). Same evidence-per-cell co-scaling as rev12,
#     halved because M only doubles here:
#       half-life 250->500, reserve 550->1100, reserve_residency 300->600 (all ~ M),
#       n_eff (effective hits) rises 721->1442.
#   - graduation_hits STAYS 2 here: the 2 ln M bound rounds to 2 through 16k and only reaches 3 at
#     32k, and holding it at 2 avoids the graduation-margin risk seen on the 32k lo arm.
#   - Oversample: lo 4x, hi 6x (STARTING values; tune if thin_accept under-fills N=256).
#   - This run exists to give the 3rd point so the s-vs-M and lam_spread-vs-M laws can be FIT
#     (slope -> effective d*), not just drawn through two points. Same lam_spread guardrail applies.
#
# REV12 = REV11 + BANK M-SCALING to M = 32768 (4x). Half-life 1000, reserve 2200, residency 1200,
#   graduation_hits 3, oversample lo 4x / hi 8x. (This rev13 is its 2x-M sibling.)
#
# REV11 = REV10 + ACCEPTANCE-TILT FIX for stream thinning:
#   - The thinned admission tilt now reads args.typicality_a / args.typicality_c_frac (was
#     hardcoded a_tilt=0.5), so bc_thinned_hi finally admits at its configured a=1.0 instead of
#     accepting like the lo arm. The a=0.5 lo arm is byte-identical (0.5 reduces to the old literal).
#   - bc_thinned_hi thin_oversample_factor 3.0 -> 6.0: a=1.0 roughly halves acceptance (accept
#     ~0.24-0.35, chi ~3-4), so the 3x pool would under-fill N=256; 6x is the right size. lo stays 3.0.
#   - Guard added below: FATALs on a stale clone where trainer.py still hardcodes a_tilt=0.5.
#
# REV10 = REV9 (hand-rolled DP + compile + loss tuning) + ANALYTIC w_max for stream thinning:
#   - Acceptance normalizer w_max is now the closed-form bound (c_frac*p_ref)^(-a), recomputed
#     each step, replacing the frozen 200-step running max. The frozen value went stale as p_ref
#     grew ~10x, inflating the normalizer ~3.3x and starving accept to ~0.15 (85% under-fill,
#     which broke the Bernoulli matched-mass guarantee). Analytic w_max >= max(w) always, so the
#     admitted DISTRIBUTION is unchanged (w_max cancels); only the accept RATE rises to ~0.49.
#   - thin_oversample_factor 6/12 -> 3.0 (pool 768; ~376 admits vs N=256, ~8sigma). Halves pool work.
#
#
# REV9 = REV8 recipe (thinning arms, fixed-radius bank) + PERFORMANCE TUNING:
#   - Hand-rolled data parallelism replaces DDP (training/hand_rolled_dp.py) so that
#     torch.compile + gradient checkpointing composes; compile_blocks is ON (~-22% compute,
#     lower memory). Manual grad all-reduce is bit-equivalent to DDP (verified on A100).
#   - Prototype loss: teacher path split (teacher_targets/student_prediction); the semantic
#     prototype call reuses g1's targets (dedup) and its teacher/koleo double-count is removed.
#   - Semantic-iBOT: the per-channel backbone loop is batched into one forward.
#
# REV8 (inherited): SAME FIXED ViT-B RECIPE + DATALOADER FIX + FIXED-RADIUS BANK AS REV7.
#
#       STREAM THINNING as an alternative realization of the same density rebalancing (section 3.9).
#       The rev7 weighted arm scales the DINO loss per tile by w = 1/(p_hat + c)^a. Thinning reaches
#       the SAME expected per-region gradient mass from the other side: it leaves the loss UNWEIGHTED
#       and instead ADMITS each tile to the batch with probability a(p_hat) = w(p_hat)/w_max. A third,
#       unaugmented (Resize+Normalize) scout crop is forwarded no-grad over an over-drawn candidate
#       pool to get p_hat BEFORE commit; the bank is updated on that un-thinned pool (the TRUE stream,
#       so rare stays rare), the pool is thinned to N survivors, and the normal fused forward runs on
#       the survivors only. Matched mass -> the thinned and weighted arms share their expected
#       gradient and differ only in effective sample size / coverage. Selected by --balance_mode.
#
#       The rev7->rev8 bc matrix is now {weighted, thinned} x {lo, hi} at matched (a, c_frac):
#
#         run key              balance_mode  a    c_frac  radius_mult  oversample   notes
#         bc_weightedloss_lo   weighted      0.5   0.25      1.5          --         rev7 arm, byte-unchanged
#         bc_weightedloss_hi   weighted      1.0   0.25      1.5          --         rev7 arm, byte-unchanged
#         bc_thinned_lo        thinned       0.5   0.25      1.5          6.0        NEW: matched to weightedloss_lo
#         bc_thinned_hi        thinned       1.0   0.25      1.5         12.0        NEW: matched to weightedloss_hi
#
#       THINNED arms now GPU-AUGMENT (kornia) only the committed survivors. The loader emits ONLY the
#       raw Resize(224) uint8 tile per sample (scout_pool_mode) -- NO CPU augmentation. The bank scouts
#       the normalized raw, density admits N survivors, and the 2 global + 8 local crops are generated
#       ON-GPU for those N only. The 5/6 of the over-drawn pool discarded by thinning is thus never
#       augmented, killing the ~6x CPU-augmentation cost (over-draw now = decode + one scout forward).
#       The bank stays at the BASE M=8192 / reserve=550 (the earlier 4x-M experiment diluted hits and
#       collapsed lam_spread to 0). REQUIRES `kornia` in the env (see PREREQUISITE).
#
#       Read each thinned arm AGAINST its weighted twin: same target distribution, the only
#       difference is thinning trades coverage for a rare-tile-concentrated batch (lower effective
#       sample size) while weighting keeps every tile at unequal leverage. bc_thinned_lo is the
#       primary controlled comparison (the thinning acceptance was matched to the lo settings).
#
#       COST: thinned mode OVER-DRAWS ceil(--thin_oversample_factor) loader batches per step and runs one
#       no-grad scout embed over the whole pool -- but the over-draw is now CHEAP (raw uint8, no CPU aug;
#       only the N survivors are GPU-augmented), so the per-step cost is ~decode + one scout forward, not
#       ~factor x the full augmentation. bc_thinned_hi (a=1.0) has a larger chi / pool. Watch thin_accept
#       (realized accepted/pool ~ 1/chi) and accepted (= thin_accept x thin_seen): if accepted grazes
#       N=256 (under-fills), RAISE --thin_oversample_factor. REV10: analytic w_max -> accept ~0.49, so factor 3 (pool 768) fills N=256 with ~8sigma margin (was 6/12 under the stale frozen w_max).
#
#       Carried over from REV7 (the fixed-radius absolute readout): the bank sums lambda_hat * K(d/R_rad)
#       over the established signatures within R_rad = radius_mult * s and returns the absolute local
#       density p_hat; the weight/acceptance is w = 1/(p_hat + c)^a, c = c_frac * p_ref. Percentile /
#       soft-rank machinery is gone; j is a resolution diagnostic. Both banks are inert until
#       typicality_warmup_iters (50k), so the bc arms warm-start from a prepared 50k checkpoint.
#
#       Carried over from REV3 (the dataloader fix, B1-B5, commit 862ffb7):
#         B1  worker_id never propagated -> all 10 workers emitted identical streams
#         B2  rank=args.gpu (local) paired with world_size=get_world_size() (global)
#         B3  shard collapsed to a bounding range -> every worker read ~96% of every zip
#         B4  per-tile rng.randint() over 287M tiles + PYTHONHASHSEED-dependent hash()
#         B5  zipfile.ZipFile() reopened for every single image
#       Carried over from REV2:
#         lr 2e-3 -> 2e-4  (Virchow2 ViT-B / DINOv2 vitl14)
#         drop_path 0.1 -> 0.4  (Virchow2 ViT-B)
#
# PREREQUISITE: the stream-thinning code (--balance_mode, scout_pool_mode, _scout_and_bank, the
#       Richardson bank, data/gpu_augment.py) must be on origin/pathology-fm-recipe, AND the conda env
#       must have `kornia` (thinned arms GPU-augment the survivors). The bc_thinned_* guard below FATALs
#       on a clone/env that predates either. Push the thinning commits, and:  pip install kornia
#       (weighted / off arms do NOT need kornia -- GPUCropAugment is imported only in thinned mode).
#
# Each run = the FIXED ViT-B baseline recipe + exactly ONE research ingredient
#            (or 'baseline' = vanilla DINOv2, no ingredient).
# Sets up the experiment dir and FORCES every toggle, then PRINTS the launch command.
# Does NOT submit. Run once per key; observe; launch the python yourself.
#
# Usage:  ./run_sub_stab_suite_rev8.sh <run_key>
#   run_key: baseline | semibot_1ch | semibot_3ch | protoclust_4096 | protoclust_4096_semproto
#            | bc_weightedloss_lo | bc_weightedloss_hi | bc_thinned_lo | bc_thinned_hi
#            | pathology_recipe
#
# Infra: 1 node x 4 GPU x bs256 = 1024 total -> applied LR = 2e-4 (sqrt factor 1.0).
#        num_workers=10 -> total_workers = world_size(4) x 10 = 40 shards.
#        partition gpu, constraint a100|h100 (hardcoded -> sed'd below).
#
# NOTE: pathology_recipe is a BUNDLE, not a clean ingredient. At ViT-B it flips
#       patch_size->14, KoLeo->KDE, fp16->bf16, koleo_weight 0.1->0.05, and the
#       augmentation to the ECT path (all at runtime in trainer.py). It is NOT a
#       /16 KoLeo run -- treat it as a separate recipe point, not an isolated mod.
set -e

RUN="${1:-}"
GITHUB_REPO="https://github.com/swarajnanda2021/Dinov2_modified.git"
BRANCH="pathology-fm-recipe-tuned"   # rev9 lives here (hand-rolled DP + compile + loss tuning)
BASE_DIR="/data1/vanderbc/test_dinov2_swaraj"

case "$RUN" in
  baseline|semibot_1ch|semibot_3ch|protoclust_4096|protoclust_4096_semproto|bc_weightedloss_lo|bc_weightedloss_hi|bc_thinned_lo|bc_thinned_hi|pathology_recipe) ;;
  *) echo "Usage: $0 <run_key>"
     echo "  run_key: baseline | semibot_1ch | semibot_3ch | protoclust_4096 | protoclust_4096_semproto | bc_weightedloss_lo | bc_weightedloss_hi | bc_thinned_lo | bc_thinned_hi | pathology_recipe"
     exit 1 ;;
esac

exp_name="FMC_ViT-B_stab_${RUN}_rev13"
exp_dir="$BASE_DIR/$exp_name"

echo "========================================"
echo "Setup: $exp_name  (branch: $BRANCH)"
echo "  ViT-B/16 fixed-loader recipe (== rev3/rev5/rev6/rev7) + ONE ingredient: $RUN"
echo "  (rev13 = 16k middle M point; H/reserve/residency halved vs rev12; grad stays 2; oversample lo 4x / hi 6x)"
echo "========================================"

[ -d "$exp_dir" ] && { echo "  Removing existing dir..."; rm -rf "$exp_dir"; }
mkdir -p "$exp_dir"; cd "$exp_dir"

echo "  Cloning ($BRANCH)..."
git clone -b "$BRANCH" "$GITHUB_REPO" .

# ---- Loader-fix guard: record the commit and refuse to proceed on a stale clone ----
echo "  Recording clone commit..."
git log -1 --oneline | tee "$exp_dir/CLONED_COMMIT.txt"

echo "  Verifying dataloader fix is present..."
if ! grep -q "get_worker_info" data/datasets.py; then
    echo "  FATAL: data/datasets.py has no get_worker_info() -- loader fix ABSENT. Aborting."
    exit 1
fi
if grep -q "_calculate_worker_shard" data/datasets.py; then
    echo "  FATAL: data/datasets.py still contains _calculate_worker_shard -- OLD shard code present. Aborting."
    exit 1
fi
if grep -q "rank=args.gpu" training/trainer.py; then
    echo "  FATAL: training/trainer.py still passes rank=args.gpu (local rank). Aborting."
    exit 1
fi
echo "    Loader fix verified (B1-B5 applied)."

# ---- Hand-rolled DP guard: rev9 replaces DDP with manual grad all-reduce so that
#      torch.compile + gradient checkpointing composes. Also verifies the rev9 loss
#      tuning (prototype teacher-path split / double-count removal). ----
echo "  Verifying rev13 hand-rolled DP + loss tuning + analytic w_max + thinned tilt fix is present..."
if grep -q "THIN_WMAX_WINDOW" training/trainer.py; then
    echo "  FATAL: training/trainer.py still has THIN_WMAX_WINDOW -- frozen w_max present (pre-rev10 clone). Aborting."
    exit 1
fi
if [ ! -f training/hand_rolled_dp.py ] || ! grep -q "class HandRolledDP" training/hand_rolled_dp.py; then
    echo "  FATAL: training/hand_rolled_dp.py / HandRolledDP missing -- hand-rolled DP ABSENT (pre-rev9 clone). Aborting."
    exit 1
fi
if ! grep -q "\.sync_grads()" training/trainer.py; then
    echo "  FATAL: training/trainer.py has no .sync_grads() call -- hand-rolled DP not wired. Aborting."
    exit 1
fi
if ! grep -q "def teacher_targets" losses/prototype_loss.py; then
    echo "  FATAL: losses/prototype_loss.py has no teacher_targets() -- prototype double-count fix ABSENT. Aborting."
    exit 1
fi
# ---- rev11 acceptance-tilt guard: the thinned admission must read the configured tilt, not a=0.5 ----
if grep -qE "a_tilt, *c_frac_tilt *= *0\.5, *0\.25" training/trainer.py; then
    echo "  FATAL: training/trainer.py still hardcodes 'a_tilt, c_frac_tilt = 0.5, 0.25' -- thinned tilt fix ABSENT (pre-rev11 clone). bc_thinned_hi would accept at a=0.5. Aborting."
    exit 1
fi
if ! grep -qE "a_tilt, *c_frac_tilt *= *args\.typicality_a, *args\.typicality_c_frac" training/trainer.py; then
    echo "  FATAL: training/trainer.py thinned admission does not read args.typicality_a/_c_frac -- rev11 tilt fix not wired. Aborting."
    exit 1
fi
echo "    rev13 hand-rolled DP + loss tuning + analytic w_max + thinned tilt fix verified."

# ---- Fixed-radius bank guard: every bc_* arm needs the rev7 readout ----
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi|bc_thinned_lo|bc_thinned_hi)
    echo "  Verifying fixed-radius counted-coverage bank code is present..."
    if [ ! -f typicality/counted_coverage_bank.py ]; then
        echo "  FATAL: typicality/counted_coverage_bank.py missing -- counted bank code ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "CountedCoverageBank" training/trainer.py; then
        echo "  FATAL: training/trainer.py does not reference CountedCoverageBank. Aborting."
        exit 1
    fi
    for a in typicality_a typicality_c_frac typicality_radius_mult; do
        if ! grep -q -- "--$a" configs/config.py; then
            echo "  FATAL: configs/config.py has no --$a -- clone predates the fixed-radius readout. Aborting."
            exit 1
        fi
    done
    if ! grep -q "p_ref" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py has no p_ref -- fixed-radius reference scale ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "absolute_weights" typicality/typicality_scorer.py; then
        echo "  FATAL: typicality_scorer.py has no absolute_weights -- fixed-radius weight ABSENT. Aborting."
        exit 1
    fi
    if grep -q "_soft_rank_score" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py still contains _soft_rank_score -- stale percentile clone. Aborting."
        exit 1
    fi
    if ! grep -q "_sweep_edge" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py has no _sweep_edge -- self-tuning s ABSENT (R_rad would be 0). Aborting."
        exit 1
    fi
    echo "    Fixed-radius bank verified (p_ref + absolute_weights present; percentile machinery gone; s sweep present)."
    ;;
esac

# ---- Stream-thinning guard: the thinned arms need the rev8 thinning code ----
case "$RUN" in
  bc_thinned_lo|bc_thinned_hi)
    echo "  Verifying rev8 stream-thinning code is present..."
    for a in balance_mode thin_oversample_factor thin_richardson_correct \
             typicality_bank_size typicality_halflife_steps typicality_reserve_size \
             typicality_reserve_residency typicality_graduation_hits; do
        if ! grep -q -- "--$a" configs/config.py; then
            echo "  FATAL: configs/config.py has no --$a -- clone predates stream thinning (rev8). Aborting."
            echo "         Push the thinning commits to origin/$BRANCH first."
            exit 1
        fi
    done
    if ! grep -q "scout_pool_mode" data/transforms.py; then
        echo "  FATAL: data/transforms.py has no scout_pool_mode -- raw-carry scout ABSENT (pre-GPU-aug clone). Aborting."
        exit 1
    fi
    if [ ! -f data/gpu_augment.py ] || ! grep -q "GPUCropAugment" data/gpu_augment.py; then
        echo "  FATAL: data/gpu_augment.py / GPUCropAugment missing -- GPU augmentation ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "def _scout_and_bank" training/trainer.py; then
        echo "  FATAL: training/trainer.py has no _scout_and_bank -- thinning loop ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "gpu_augment(survivor_raw" training/trainer.py; then
        echo "  FATAL: training/trainer.py does not GPU-augment survivors -- raw-carry path ABSENT. Aborting."
        exit 1
    fi
    if ! python -c "import kornia" 2>/dev/null; then
        echo "  FATAL: 'kornia' is not importable in this env -- thinned GPU augmentation needs it."
        echo "         Install it first:   pip install kornia"
        exit 1
    fi
    echo "    Stream-thinning + GPU-augment code verified (scout_pool_mode + GPUCropAugment + kornia present)."
    ;;
esac

echo "  Patching log path + GPU constraint..."
sed -i "s|p = Path(\".*\")|p = Path(\"$exp_dir/logs\")|" run_with_submitit.py
# h100-only -> a100 OR h100. Constraint is hardcoded (not a CLI arg). '@' delimiter: the value has a pipe.
sed -i "s@slurm_constraint='h100'@slurm_constraint='a100|h100'@" run_with_submitit.py

# ensure_arg: substitute "args.<k> = <v>" in place (single-space or '='-aligned); else append after patch_embed_lr_mult.
ensure_arg () {
    local key="args.$1"; local esc="args\\.$1"
    if grep -qE "^[[:space:]]*${esc}[[:space:]]*=" run_with_submitit.py; then
        sed -i -E "s|^([[:space:]]*)${esc}[[:space:]]*=.*|\\1${key} = $2|" run_with_submitit.py
    else
        sed -i -E "/^[[:space:]]*args\\.patch_embed_lr_mult[[:space:]]*=.*/a\\    ${key} = $2" run_with_submitit.py
    fi
}

echo "  Forcing new-baseline recipe + infra..."
ensure_arg vit_variant '"B"'
ensure_arg batch_size_per_gpu 256
ensure_arg lr                    2e-4
ensure_arg clip_grad             3.0
ensure_arg drop_path_rate        0.4
ensure_arg drop_path_uniform     True
ensure_arg momentum_teacher      0.992
ensure_arg total_iterations      125_001
ensure_arg n_standard_local_crops 8
ensure_arg lr_decay_rate         1.0
ensure_arg layerscale_init       1e-5
ensure_arg norm_last_layer       False
ensure_arg qk_norm               True
ensure_arg patch_embed_lr_mult   0.2
ensure_arg ffn_type              '"mlp"'
ensure_arg num_workers           10     # REV3+: num_workers now SETS the shard count (world_size x 10 = 40). Pin it.

echo "  Forcing ALL research ingredients OFF (per-run re-enables only what it needs)..."
ensure_arg use_pathology_recipe              False
ensure_arg use_looped_backbone               False
ensure_arg use_adversarial_mask_augmentation False
ensure_arg use_cellvit_augmentation          False
ensure_arg use_random_mask_augmentation      False
ensure_arg use_semantic_ibot                 False   # ships TRUE in main() -> off
ensure_arg use_semantic_prototypes           False   # ships TRUE -> off (never use; untested)
ensure_arg use_prototype_clustering          False
ensure_arg use_typicality_dampening          False
ensure_arg balance_mode                      '"weighted"'   # REV8: default rebalancing mode (thinned arms flip it)

# ---- SPEED (rev9): compile_blocks is ON. Per-block torch.compile is incompatible with DDP
#      (DDP wraps the autograd graph and double-counts the compiled node's saved tensors ->
#      CheckpointError), so rev9 replaces DDP with hand-rolled data parallelism
#      (training/hand_rolled_dp.py): a manual, bit-equivalent grad all-reduce after backward().
#      With DDP gone, compile + gradient checkpointing composes -- ~-22% compute at LOWER memory
#      (no Route-1 tradeoff). Verified bit-equal gradients on A100. ~80 s one-time compile warmup.
#      thinned arms also flip scout_amp_bf16 below.
ensure_arg compile_blocks                    True

echo "  Toggling ingredient: $RUN"
case "$RUN" in
  baseline)
    # vanilla DINOv2: nothing toggled -- all research arms already forced OFF above.
    ;;
  semibot_1ch)
    ensure_arg use_semantic_ibot            True
    ensure_arg num_masks                    3     # placeholder to build the 3ch mask model; weights from ckpt
    ensure_arg semantic_masks_per_iteration 1     # random 1-of-3 channel per iteration
    ;;
  semibot_3ch)
    ensure_arg use_semantic_ibot            True
    ensure_arg num_masks                    3
    ensure_arg semantic_masks_per_iteration 3     # all 3 channels every iteration (>= num_masks)
    ;;
  protoclust_4096)
    ensure_arg use_prototype_clustering     True
    ensure_arg num_prototypes               4096  # down from 16384 ship default (most collapse)
    ;;
  protoclust_4096_semproto)
    ensure_arg use_prototype_clustering     True
    ensure_arg num_prototypes               4096
    ensure_arg use_semantic_ibot            True  # REQUIRED: semantic proto consumes semantic-iBOT masks/tokens
    ensure_arg use_semantic_prototypes      True
    ensure_arg num_masks                    3
    ensure_arg semantic_masks_per_iteration 1
    ensure_arg semantic_ibot_weight         0.0   # Option A: isolate semantic-proto delta (zero the iBOT loss).
    ;;                                            #   Option B (full stack): set this to 1.0.

  # ---- typicality dampening: WEIGHTED arms (rev7, byte-unchanged) differ only in the tilt a ----
  bc_weightedloss_lo)
    ensure_arg use_typicality_dampening     True
    ensure_arg balance_mode                 '"weighted"'        # loss weighted by w = 1/(p_hat + c)^a
    ensure_arg typicality_modulation        '"weighted_loss"'
    ensure_arg typicality_a                 0.5                 # LO tilt: measured 2.77x, ESS 90.5%
    ensure_arg typicality_c_frac            0.25                # c = c_frac * p_ref
    ensure_arg typicality_radius_mult       1.5                 # R_rad = radius_mult * s
    ;;
  bc_weightedloss_hi)
    ensure_arg use_typicality_dampening     True
    ensure_arg balance_mode                 '"weighted"'
    ensure_arg typicality_modulation        '"weighted_loss"'
    ensure_arg typicality_a                 1.0                 # HI tilt: measured 7.68x, ESS 68.7%
    ensure_arg typicality_c_frac            0.25
    ensure_arg typicality_radius_mult       1.5
    ;;

  # ---- typicality dampening: THINNED arms (rev8) -- same (a, c_frac) as the weighted twin, but
  #      the loss is UNWEIGHTED and the batch is admitted by density (matched mass). ----
  bc_thinned_lo)
    ensure_arg use_typicality_dampening     True                # thinned reuses the bank / repr / R stack
    ensure_arg balance_mode                 '"thinned"'         # scout + density admission; UNWEIGHTED loss
    ensure_arg typicality_a                 0.5                 # LO tilt (FREE knob)
    ensure_arg typicality_c_frac            0.25
    ensure_arg typicality_radius_mult       1.5
    # ---- REV13 bank M-scaling: M=16384 (2x base); the five below are DERIVED (technique 3.10, halved vs rev12) ----
    ensure_arg typicality_bank_size         16384               # 2x M -> middle resolution point of the sweep
    ensure_arg typicality_halflife_steps    500                 # DERIVED ~M: restores per-cell hit-evidence, n_eff ~721->1442
    ensure_arg typicality_reserve_size      1100                # DERIVED ~M: absorbs 2x newborn inflow
    ensure_arg typicality_reserve_residency 600                 # DERIVED ~M: longer window to reach graduation
    ensure_arg typicality_graduation_hits   2                   # STAYS 2: 2 ln M rounds to 2 through 16k (steps to 3 only at 32k)
    ensure_arg thin_oversample_factor       4.0                 # DERIVED ~a, mild feed bump for 2x M. START; RAISE if thin_accept under-fills N=256.
    ensure_arg thin_richardson_correct      False               # two-scale bias probe is measure-only
    ensure_arg scout_amp_bf16               True                # bf16 scout fwd: ~-470 ms. Rounds p_hat -> shifts admissions (accepted for speed).
    # GUARDRAIL: watch lam_spread. Hold >= ~2.5 = co-scaling worked. Slide toward 1 = rare starving -> ABORT, lengthen half-life.
    ;;
  bc_thinned_hi)
    ensure_arg use_typicality_dampening     True
    ensure_arg balance_mode                 '"thinned"'
    ensure_arg typicality_a                 1.0                 # HI tilt (FREE knob)
    ensure_arg typicality_c_frac            0.25
    ensure_arg typicality_radius_mult       1.5
    # ---- REV13 bank M-scaling: identical DERIVED co-scale to bc_thinned_lo (M=16384); only a and oversample differ ----
    ensure_arg typicality_bank_size         16384
    ensure_arg typicality_halflife_steps    500
    ensure_arg typicality_reserve_size      1100
    ensure_arg typicality_reserve_residency 600
    ensure_arg typicality_graduation_hits   2
    ensure_arg thin_oversample_factor       6.0                 # a=1.0 halves accept -> 6x pool at 2x M. START; RAISE if admits < N=256.
    ensure_arg thin_richardson_correct      False
    ensure_arg scout_amp_bf16               True                # bf16 scout fwd: ~-470 ms. Rounds p_hat -> shifts admissions (accepted for speed).
    # GUARDRAIL: launch this only after bc_thinned_lo has held lam_spread >= ~2.5 for ~20-30k steps.
    ;;

  pathology_recipe)
    ensure_arg use_pathology_recipe         True
    # BUNDLE: at runtime trainer.py ALSO flips patch_size->14, KoLeo->KDE,
    # fp16->bf16, koleo_weight 0.1->0.05, and aug -> ECT path. qk_norm stays
    # True (explicit set wins; H/G auto-gate does not fire at ViT-B/768).
    ;;
esac

# ---- Counted-bank knob readout (shared by every bc_* arm): resolved from run_with_submitit.py ----
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi|bc_thinned_lo|bc_thinned_hi)
    echo "  Counted-coverage bank ON (fixed-radius absolute readout). ALL knobs are surfaced (grouped)"
    echo "  in run_with_submitit.py; a / c_frac / radius_mult are set per arm above. Bank M / reserve"
    echo "  stay at the base 8192 / 550 (the 4x-M experiment collapsed lam_spread). Resolved values:"
    for k in typicality_bank_size typicality_reserve_size typicality_K_prime \
             typicality_halflife_steps typicality_reserve_residency typicality_graduation_hits \
             typicality_pool_j typicality_a typicality_c_frac typicality_radius_mult \
             typicality_s_buffer_size typicality_s_sweep_interval typicality_s_grid_points \
             typicality_s_grid_span typicality_s_ema_alpha typicality_s_min_buffer typicality_s_headroom \
             typicality_pool_selftune typicality_pool_rse_target typicality_pool_max typicality_pool_ema; do
        v=$(grep -nE "^[[:space:]]*args\.${k}[[:space:]]*=" run_with_submitit.py | head -1)
        printf "    %-34s %s\n" "$k" "${v:-MISSING}"
    done
    ;;
esac
case "$RUN" in
  bc_thinned_lo|bc_thinned_hi)
    echo "  STREAM THINNING ON (--balance_mode thinned): the DINO loss is UNWEIGHTED; the batch is"
    echo "  admitted by density a(p_hat)=w/w_max on an over-drawn scout pool. Resolved thin knobs:"
    for k in balance_mode thin_oversample_factor thin_richardson_correct; do
        v=$(grep -nE "^[[:space:]]*args\.${k}[[:space:]]*=" run_with_submitit.py | head -1)
        printf "    %-34s %s\n" "$k" "${v:-MISSING}"
    done
    echo "  NB: the bank updates ONLY on the un-thinned scout pool (the true stream); the committed"
    echo "      batch is bank-read-only. p_ref / w_max / chi are re-measured on the UNAUGMENTED scout"
    echo "      basis during warmup -- they do NOT carry over from the weighted (augmented) arm."
    ;;
esac

mkdir -p "$exp_dir/logs"

echo ""
echo "  Verifying resolved settings ('MISSING' => fix before launch):"
for kv in \
    args.vit_variant args.batch_size_per_gpu args.lr args.clip_grad \
    args.drop_path_rate args.drop_path_uniform args.momentum_teacher args.total_iterations \
    args.n_standard_local_crops args.lr_decay_rate args.layerscale_init args.norm_last_layer \
    args.qk_norm args.patch_embed_lr_mult args.ffn_type args.patch_size args.num_workers \
    args.use_pathology_recipe args.use_looped_backbone args.use_adversarial_mask_augmentation \
    args.use_cellvit_augmentation args.use_random_mask_augmentation \
    args.use_semantic_ibot args.use_semantic_prototypes args.use_prototype_clustering \
    args.use_typicality_dampening args.num_masks args.semantic_masks_per_iteration \
    args.semantic_ibot_weight args.num_prototypes args.typicality_modulation \
    args.typicality_a args.typicality_c_frac args.typicality_radius_mult \
    args.typicality_bank_size args.typicality_reserve_size \
    args.balance_mode args.thin_oversample_factor args.thin_richardson_correct \
    args.mask_checkpoint args.mask_model_arch ; do
        hit=$(grep -nE "^[[:space:]]*${kv//./\\.}[[:space:]]*=" run_with_submitit.py | head -1)
        printf "    %-42s %s\n" "$kv" "${hit:-MISSING}"
done
echo "    GPU constraint -> $(grep -n "slurm_constraint" run_with_submitit.py | head -1)"
echo "    Clone commit   -> $(cat "$exp_dir/CLONED_COMMIT.txt")"
if [ "$RUN" = "pathology_recipe" ]; then
    echo "    NB: args.patch_size shows 16 here; trainer.py overrides it to 14 at runtime under the recipe."
fi

echo ""
echo "  NOT submitting."
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi|bc_thinned_lo|bc_thinned_hi)
    echo "  WARM-START (recommended): to skip the 0->50k warmup, place the PREPARED iteration-50k"
    echo "  checkpoint at:"
    echo "      $exp_dir/logs/checkpoint.pth"
    echo "  Use the SAME prepared checkpoint as the rev7 bc arms: the 'typicality_bank' key REMOVED,"
    echo "  EVERYTHING else retained (student, teacher, optimizer, repr_protos, R_optimizer,"
    echo "  dataset_position, iteration). The trainer auto-resumes at iteration 50000 = activation."
    case "$RUN" in
      bc_thinned_lo|bc_thinned_hi)
        echo "  THINNED note: the same checkpoint works unchanged -- the primary AND the coarse"
        echo "  Richardson bank both build fresh (the coarse bank is a new key, absent from the"
        echo "  prepared ckpt -> fresh, re-matures; it is measure-only). At 50k the resume data-skip"
        echo "  is exact (thinning is inert pre-warmup, so 0->50k ran at 1 batch/step). A LATER"
        echo "  mid-run requeue re-reads a harmless slice of the stream (over-draw vs iteration-based"
        echo "  skip) and re-warms w_max over ~500 steps."
        ;;
    esac
    echo "" ;;
esac
echo "  Launch manually with:"
echo "    cd $exp_dir && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
echo "      python run_with_submitit.py --nodes 1 --ngpus 4 --partition gpu"
echo ""
echo "  Post-warmup LR sanity (~iter 14k): expect peak ~2.0e-4 (sqrt factor = 1.0 at total batch 1024)."
echo ""
echo "  GO/NO-GO -- run this in the first minute of the job:"
echo "    grep -ho 'Worker [0-9]*/[0-9]*' $exp_dir/logs/*.out | sort -u"
echo "    EXPECT: ten DISTINCT ids, Worker 0/40 .. Worker 9/40  (only rank 0 prints)."
echo "    IF YOU SEE ten copies of 'Worker 0/40' -> KILL THE JOB. The fix did not land."
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi)
    W=$(grep -oE 'typicality_warmup_iters = [0-9_]+' run_with_submitit.py | awk '{print $NF}')
    if [ "$RUN" = "bc_weightedloss_lo" ]; then ESS_EXP="~90%  (a=0.5)"; else ESS_EXP="~69%  (a=1.0)"; fi
    echo ""
    echo "  WEIGHTED-ARM sanity (activation at iter $W; if warm-started, that is step 0):"
    echo "    grep -h '^typ |'  $exp_dir/logs/*.out    # compact line: p=<med>/ref<p_ref> w=<mean> ess=<pct>"
    echo "    # typ_p_ref -> non-zero within a few steps; typ_ess -> $ESS_EXP; typ_w_mean -> order 1-3."
    echo "    plot:  python3 plot_typicality.py $exp_dir -g score   (or -g health, or --status-only)"
    ;;
  bc_thinned_lo|bc_thinned_hi)
    W=$(grep -oE 'typicality_warmup_iters = [0-9_]+' run_with_submitit.py | awk '{print $NF}')
    OF=$(grep -oE 'args\.thin_oversample_factor = [0-9.]+' run_with_submitit.py | awk '{print $NF}')
    echo ""
    echo "  THINNED-ARM sanity (activation at iter $W; bank matures after that, then thinning starts):"
    echo "    grep -h 'counted-coverage bank' $exp_dir/logs/*.out   # EXPECT the bank was created"
    echo "    grep -h 'Richardson coarse bank' $exp_dir/logs/*.out  # EXPECT: M' = 64 (crude probe)"
    echo "    grep -h 'Balance mode' $exp_dir/logs/*.out            # EXPECT: Balance mode: thinned"
    echo "    # metric stream (log.txt), a few steps after the bank matures:"
    echo "    #   thin_seen   -> ~chi*N candidates scouted per step,"
    echo "    #   thin_accept -> realized accepted/pool ~ 1/chi;  accepted = thin_accept*thin_seen must"
    echo "    #                  stay > N=256. If it grazes/dips below (under-fills), RAISE --thin_oversample_factor,"
    echo "    #   thin_bias   -> mean|beta_hat| from the coarse probe (crude indicator only),"
    echo "    #   thin_prof_err -> L1 between the committed p_hat histogram and the target a*mu (small = good)."
    echo "    plot:  python3 plot_typicality.py $exp_dir -g health  (bank health carries over unchanged)"
    echo "    (compare bc_thinned_lo AGAINST bc_weightedloss_lo -- same target mass, different coverage/ESS)"
    ;;
esac
