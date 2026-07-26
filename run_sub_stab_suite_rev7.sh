#!/bin/bash
# run_sub_stab_suite_rev7.sh -- parametrized launcher for the "new-baseline" ViT-B ablations.
#
# REV7: SAME FIXED ViT-B RECIPE + DATALOADER FIX AS REV3/REV5/REV6. The recipe block below is
#       byte-identical to run_sub_stab_suite_rev6.sh. TWO changes from REV6:
#
#       (1) The typicality READOUT is now a FIXED-RADIUS ABSOLUTE scorer, not a fixed-COUNT
#           kernel sum. Same triweight kernel, same counters; it sums lambda_hat * K(d/R_rad)
#           over the established signatures within a FIXED radius R_rad = radius_mult * s, and
#           returns the absolute local density p_hat. The weight is w = 1/(p_hat + c)^a with
#           c = c_frac * p_ref (p_ref an EMA of the batch-median density). WHY: the old
#           fixed-count readout divided by the j-th nearest distance, which cancels the local
#           scale exactly -- it was measured (59,249 cached signatures, 4 checkpoints of the two
#           rev6 arms) to be bit-invariant to how crowded a neighbourhood is, and its agreement
#           with an offline kNN density fell 0.70 -> 0.29 between 52k (j=36) and 124k (j=7) as the
#           cover became near-uniform (measured total kernel weight 0.158 at j=7). The fixed-radius
#           form correlates 0.85 and is ~4x cheaper (no argsort). The percentile / soft-rank
#           machinery is deleted; j survives only as a resolution diagnostic.
#           NB: no AUROC effect is claimed -- every number above is offline gradient-mass arithmetic
#           on cached signatures; nothing has run inside training.
#
#       (2) The BANK AXIS IS GONE. The counted-coverage bank is the only implementation and the
#           distance bank (Alg.1) is retired, so there is no _counted / _distance suffix and no
#           --typicality_bank switch. The old 2x2 (modulation x bank) collapses to two arms that
#           differ ONLY in the tilt exponent a:
#
#             run key              a    c_frac  radius_mult   measured tilt   measured ESS
#             bc_weightedloss_lo   0.5   0.25      1.5           2.77x           90.5%
#             bc_weightedloss_hi   1.0   0.25      1.5           7.68x           68.7%
#
#           Both set typicality_modulation = "weighted_loss". a was chosen by MEASURED gradient
#           tilt (rarest:commonest gradient-mass ratio on the cached signatures), NOT by analogy
#           to the old beta values. The hi arm buys its extra tilt with 22 points of effective
#           sample size (68.7% vs 90.5%), so a difference between the two arms could be gradient
#           noise from the smaller ESS rather than the tilt itself -- read the pair together.
#           ADAPTIVE-TEMPERATURE arms are OUT OF SCOPE for rev7 (the fixed-radius readout returns
#           an absolute density, not a bounded score, so it does not feed adaptive_temperature).
#       Every run tagged _rev7.
#
#       WARM-START: because the counted bank is inert until typicality_warmup_iters (50k), the
#       weighted-loss rev7 runs resume from a PREPARED iteration-50k checkpoint instead of
#       retraining 0->50k. The prepared checkpoint has the 'typicality_bank' key REMOVED (the bank
#       is rebuilt fresh with the new buffer set -- p_ref/p_ref_init, no percentile buffers) and
#       EVERYTHING ELSE retained: student, teacher, optimizer state, repr_protos, R_optimizer,
#       dataset_position, iteration. Run this script to build the clone/config, place the prepared
#       checkpoint at  <exp_dir>/logs/checkpoint.pth  before launching -- the trainer auto-resumes
#       at iteration 50000, which lands exactly at typicality activation.
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
# Each run = the FIXED ViT-B baseline recipe + exactly ONE research ingredient
#            (or 'baseline' = vanilla DINOv2, no ingredient).
# Sets up the experiment dir and FORCES every toggle, then PRINTS the launch command.
# Does NOT submit. Run once per key; observe; launch the python yourself.
#
# Usage:  ./run_sub_stab_suite_rev7.sh <run_key>
#   run_key: baseline | semibot_1ch | semibot_3ch | protoclust_4096 | protoclust_4096_semproto
#            | bc_weightedloss_lo | bc_weightedloss_hi | pathology_recipe
#
# Infra: 1 node x 4 GPU x bs256 = 1024 total -> applied LR = 2e-4 (sqrt factor 1.0).
#        num_workers=10 -> total_workers = world_size(4) x 10 = 40 shards.
#        partition gpu, constraint a100|h100 (hardcoded -> sed'd below).
#
# NOTE: pathology_recipe is a BUNDLE, not a clean ingredient. At ViT-B it flips
#       patch_size->14, KoLeo->KDE, fp16->bf16, koleo_weight 0.1->0.05, and the
#       augmentation to the ECT path (all at runtime in trainer.py). It is NOT a
#       /16 KoLeo run -- treat it as a separate recipe point, not an isolated mod.
#
# NOTE: the bc_weightedloss_* arms need the rev7 FIXED-RADIUS bank (the fixed-radius
#       readout + p_ref + absolute_weights, percentile machinery deleted). The clone
#       guard below verifies it is present and that the retired percentile code is gone.
set -e

RUN="${1:-}"
GITHUB_REPO="https://github.com/swarajnanda2021/Dinov2_modified.git"
BRANCH="pathology-fm-recipe"
BASE_DIR="/data1/vanderbc/test_dinov2_swaraj"

case "$RUN" in
  baseline|semibot_1ch|semibot_3ch|protoclust_4096|protoclust_4096_semproto|bc_weightedloss_lo|bc_weightedloss_hi|pathology_recipe) ;;
  *) echo "Usage: $0 <run_key>"
     echo "  run_key: baseline | semibot_1ch | semibot_3ch | protoclust_4096 | protoclust_4096_semproto | bc_weightedloss_lo | bc_weightedloss_hi | pathology_recipe"
     exit 1 ;;
esac

exp_name="FMC_ViT-B_stab_${RUN}_rev7"
exp_dir="$BASE_DIR/$exp_name"

echo "========================================"
echo "Setup: $exp_name  (branch: $BRANCH)"
echo "  ViT-B/16 fixed-loader recipe (== rev3/rev5/rev6) + ONE ingredient: $RUN"
echo "  (rev7 = rev6 + FIXED-RADIUS absolute readout; bank axis dropped)"
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

# ---- Fixed-radius bank guard: the weighted-loss arms need the rev7 readout ----
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi)
    echo "  Verifying rev7 fixed-radius counted-coverage bank code is present..."
    if [ ! -f typicality/counted_coverage_bank.py ]; then
        echo "  FATAL: typicality/counted_coverage_bank.py missing -- counted bank code ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "CountedCoverageBank" training/trainer.py; then
        echo "  FATAL: training/trainer.py does not reference CountedCoverageBank. Aborting."
        exit 1
    fi
    # rev7-specific: the fixed-radius readout config knobs MUST exist (a stale rev6 clone would run
    # the retired fixed-count percentile readout instead).
    for a in typicality_a typicality_c_frac typicality_radius_mult; do
        if ! grep -q -- "--$a" configs/config.py; then
            echo "  FATAL: configs/config.py has no --$a -- clone predates the fixed-radius readout (rev7). Aborting."
            exit 1
        fi
    done
    # the p_ref reference scale and the absolute_weights scorer MUST be present...
    if ! grep -q "p_ref" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py has no p_ref -- fixed-radius reference scale ABSENT. Aborting."
        exit 1
    fi
    if ! grep -q "absolute_weights" typicality/typicality_scorer.py; then
        echo "  FATAL: typicality_scorer.py has no absolute_weights -- fixed-radius weight ABSENT. Aborting."
        exit 1
    fi
    # ...and the retired percentile machinery MUST be gone (a stale clone would still carry it).
    if grep -q "_soft_rank_score" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py still contains _soft_rank_score -- stale (pre-rev7) percentile clone. Aborting."
        exit 1
    fi
    # self-tuning s is still load-bearing (R_rad = radius_mult * s); verify the sweep is present.
    if ! grep -q "_sweep_edge" typicality/counted_coverage_bank.py; then
        echo "  FATAL: counted_coverage_bank.py has no _sweep_edge -- self-tuning s ABSENT (R_rad would be 0). Aborting."
        exit 1
    fi
    echo "    Fixed-radius bank verified (p_ref + absolute_weights present; percentile machinery gone; s sweep present)."
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
# REV7: no --typicality_bank switch any more (counted bank is the only implementation).

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

  # ---- typicality dampening: two weighted-loss arms differing ONLY in the tilt exponent a ----
  # a is the SOLE per-arm knob; c_frac (0.25) and radius_mult (1.5) live in run_with_submitit.py's
  # typicality block (the single source of truth) and are shared across both arms.
  bc_weightedloss_lo)
    ensure_arg use_typicality_dampening     True
    ensure_arg typicality_modulation        '"weighted_loss"'   # w = 1/(p_hat + c)^a
    ensure_arg typicality_a                 0.5                 # LO tilt: measured 2.77x, ESS 90.5%
    ;;
  bc_weightedloss_hi)
    ensure_arg use_typicality_dampening     True
    ensure_arg typicality_modulation        '"weighted_loss"'   # same modulation as _lo...
    ensure_arg typicality_a                 1.0                 # ...only a changes -> HI tilt: measured 7.68x, ESS 68.7%
    ;;

  pathology_recipe)
    ensure_arg use_pathology_recipe         True
    # BUNDLE: at runtime trainer.py ALSO flips patch_size->14, KoLeo->KDE,
    # fp16->bf16, koleo_weight 0.1->0.05, and aug -> ECT path. qk_norm stays
    # True (explicit set wins; H/G auto-gate does not fire at ViT-B/768).
    ;;
esac

# ---- Fixed-radius counted-bank knob readout (shared by both weighted-loss arms) ----
case "$RUN" in
  bc_weightedloss_lo|bc_weightedloss_hi)
    echo "  Counted-coverage bank ON (fixed-radius absolute readout). Knobs resolve from config.py"
    echo "  defaults except a / c_frac / radius_mult, set explicitly per arm above:"
    for k in typicality_a typicality_c_frac typicality_radius_mult \
             typicality_pool_j typicality_halflife_steps typicality_reserve_residency \
             typicality_reserve_size typicality_graduation_hits \
             typicality_s_buffer_size typicality_s_sweep_interval typicality_s_grid_points \
             typicality_s_ema_alpha typicality_s_min_buffer typicality_s_headroom \
             typicality_pool_selftune typicality_pool_rse_target typicality_pool_max typicality_pool_ema; do
        d=$(grep -F -- "'--$k'" configs/config.py | grep -oE "default=[^,]+" | head -1)
        printf "    %-34s %s\n" "$k" "${d:-MISSING}"
    done
    gs=$(grep -F -- "'--typicality_s_grid_span'" configs/config.py | grep -oE "default=\[[^]]+\]" | head -1)
    printf "    %-34s %s\n" "typicality_s_grid_span" "${gs:-MISSING}"
    echo "  NB: the readout radius is R_rad = radius_mult * s, and s is measured live (starts 0, first"
    echo "      sweep sets it ~13 at activation, drifts down). j is now a diagnostic only (it no longer"
    echo "      feeds the readout). To override a knob, add e.g.  ensure_arg typicality_a 0.75  in the arm."
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
  bc_weightedloss_lo|bc_weightedloss_hi)
    echo "  WARM-START (recommended): to skip the 0->50k warmup, place the PREPARED iteration-50k"
    echo "  checkpoint at:"
    echo "      $exp_dir/logs/checkpoint.pth"
    echo "  The prepared checkpoint has the 'typicality_bank' key REMOVED (rev7 builds a fresh"
    echo "  fixed-radius bank: p_ref/p_ref_init, no percentile buffers) and keeps EVERYTHING else --"
    echo "  student, teacher, optimizer, repr_protos, R_optimizer, dataset_position, iteration."
    echo "  The trainer auto-resumes at iteration 50000, which lands exactly at typicality activation."
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
    echo "  FIXED-RADIUS COUNTED-BANK sanity (activation at iter $W; if warm-started, that is step 0):"
    echo "    grep -h 'counted-coverage bank' $exp_dir/logs/*.out   # EXPECT: Created Typicality Dampening (counted-coverage bank)"
    echo "    grep -h '\[counted bank\]'       $exp_dir/logs/*.out   # each ckpt prints n_est/ s / j / p_ref"
    echo "    grep -h '^typ |'                $exp_dir/logs/*.out   # compact line: p=<med>/ref<p_ref> w=<mean> ess=<pct>"
    echo "    # metric stream (log.txt), a few steps after activation:"
    echo "    #   typ_p_ref  -> NON-ZERO within a few steps (initialised from the first matured batch median),"
    echo "    #   typ_ess    -> $ESS_EXP  (effective fraction of the batch; the health signal, replaces t_std),"
    echo "    #   typ_w_mean -> ORDER 1-3 (NOT below 1: absolute weights are not normalised to mean 1);"
    echo "    #   typ_p_median tracks typ_p_ref; the compact line flags !ess if ESS collapses below 50%."
    echo "    plot:  python3 plot_typicality.py $exp_dir -g score    # p_median, p_ref, w_mean, ess"
    echo "           python3 plot_typicality.py $exp_dir -g health   # evict_z, turn_ratio, ess, w_mean"
    echo "           python3 plot_typicality.py $exp_dir --status-only"
    echo "    (if typ_p_ref stays 0, or typ_w_mean is inf/nan -> p_ref never initialised; re-check that the"
    echo "     clone is rev7 (--typicality_a present) and that the prepared checkpoint dropped typicality_bank)"
    ;;
esac
