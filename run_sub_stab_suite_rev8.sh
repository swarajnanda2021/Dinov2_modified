#!/bin/bash
# run_sub_stab_suite_rev8.sh -- parametrized launcher for the "new-baseline" ViT-B ablations.
#
# REV8: SAME FIXED ViT-B RECIPE + DATALOADER FIX + FIXED-RADIUS BANK AS REV7. The recipe block
#       below is byte-identical to run_sub_stab_suite_rev7.sh. ONE addition over REV7:
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
#       WARNING (cost): thinned mode OVER-DRAWS ceil(--thin_oversample_factor) loader batches per step,
#       runs one no-grad scout embed over the whole pool, AND (with the 4x bank) does a 4x-larger density
#       readout each step -- so it consumes ~factor x the data plus the scout + readout overhead.
#       bc_thinned_hi (a=1.0) has a larger chi and a larger pool -- the expensive arm. Watch thin_accept
#       (realized accepted/pool ~ 1/chi) and accepted (= thin_accept x thin_seen): if accepted grazes
#       N=256 (under-fills), RAISE --thin_oversample_factor. chi ~3.9 was measured for lo -> factor 6; hi -> 12.
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
BRANCH="pathology-fm-recipe"
BASE_DIR="/data1/vanderbc/test_dinov2_swaraj"

case "$RUN" in
  baseline|semibot_1ch|semibot_3ch|protoclust_4096|protoclust_4096_semproto|bc_weightedloss_lo|bc_weightedloss_hi|bc_thinned_lo|bc_thinned_hi|pathology_recipe) ;;
  *) echo "Usage: $0 <run_key>"
     echo "  run_key: baseline | semibot_1ch | semibot_3ch | protoclust_4096 | protoclust_4096_semproto | bc_weightedloss_lo | bc_weightedloss_hi | bc_thinned_lo | bc_thinned_hi | pathology_recipe"
     exit 1 ;;
esac

exp_name="FMC_ViT-B_stab_${RUN}_rev8"
exp_dir="$BASE_DIR/$exp_name"

echo "========================================"
echo "Setup: $exp_name  (branch: $BRANCH)"
echo "  ViT-B/16 fixed-loader recipe (== rev3/rev5/rev6/rev7) + ONE ingredient: $RUN"
echo "  (rev8 = rev7 + STREAM THINNING arms via --balance_mode)"
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
    for a in balance_mode thin_oversample_factor thin_richardson_correct; do
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
    ensure_arg typicality_a                 0.5                 # matched to bc_weightedloss_lo
    ensure_arg typicality_c_frac            0.25
    ensure_arg typicality_radius_mult       1.5
    ensure_arg thin_oversample_factor       6.0                 # pool 6x N; chi~3.9 measured -> 6x clears N=256. Over-draw is now CHEAP (raw uint8, no CPU aug).
    ensure_arg thin_richardson_correct      False               # two-scale bias probe is measure-only
    # NB: bank M / reserve stay at the BASE 8192 / 550 (the 4x-M experiment diluted hits -> hit_frac
    #     cratered and lam_spread collapsed to 0; the base bank ran healthy). typicality_modulation
    #     is unused in thinned mode (loss unweighted) -- left at its default.
    ;;
  bc_thinned_hi)
    ensure_arg use_typicality_dampening     True
    ensure_arg balance_mode                 '"thinned"'
    ensure_arg typicality_a                 1.0                 # matched to bc_weightedloss_hi
    ensure_arg typicality_c_frac            0.25
    ensure_arg typicality_radius_mult       1.5
    ensure_arg thin_oversample_factor       12.0                # a=1.0 -> larger chi -> larger pool (over-draw is cheap now: raw uint8, no CPU aug)
    ensure_arg thin_richardson_correct      False
    # bank M / reserve at base 8192 / 550 (see bc_thinned_lo).
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
    echo "  in run_with_submitit.py; a / c_frac / radius_mult are set per arm above, and THINNED arms"
    echo "  scale typicality_bank_size / typicality_reserve_size 4x. Resolved values:"
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
