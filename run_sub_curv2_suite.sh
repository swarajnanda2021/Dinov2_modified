#!/bin/bash
# run_sub_curv2_suite.sh -- curator v2 + fp8 scout: 1 baseline + 6 v2 arms. ViT-B/16, 4 x H100, continual.
#
# Usage: ./run_sub_curv2_suite.sh <run_key>
#   curv2_base | curv2_a10_u01 | curv2_a10_u05 | curv2_a10_u10 | curv2_a15_u01 | curv2_a15_u05 | curv2_a15_u10
#
# Runs (all: --continual True, 400_001 iterations, seed 42, 4 GPU x bs 256, num_workers 10, compile_blocks True,
#       stochastic depth 0.4 uniform in TRAINING for all 7 runs; iteration checkpoints every 10_000):
#   curv2_base     no curation: use_typicality_dampening False, balance_mode off, use_gpu_augmentation True
#                  (the SAME raw-tile loader + GPU augmentation as the arms; only the curation differs).
#   curv2_a10_u01  keep_rule v2, a = 1.0, u0 = 0.01   \
#   curv2_a10_u05  keep_rule v2, a = 1.0, u0 = 0.05    |  each arm starts from a COPY of the baseline's
#   curv2_a10_u10  keep_rule v2, a = 1.0, u0 = 0.10    |  logs/checkpoint_iter_00050000.pth placed as
#   curv2_a15_u01  keep_rule v2, a = 1.5, u0 = 0.01    |  logs/checkpoint.pth; M = 8192; fp8 scout on;
#   curv2_a15_u05  keep_rule v2, a = 1.5, u0 = 0.05    |  scout in eval mode, chunks of N.
#   curv2_a15_u10  keep_rule v2, a = 1.5, u0 = 0.10   /
#
# Order: 1) ./run_sub_curv2_suite.sh curv2_base ; launch ; wait for logs/checkpoint_iter_00050000.pth
#        2) ./run_sub_curv2_suite.sh curv2_a10_u05 (and the other 5 arms) ; launch each.
# Based on run_sub_stab_suite_rev13.sh (same ensure_arg mechanism, same ViT-B recipe block). Differences:
#   - H100 only: run_with_submitit.py keeps slurm_constraint='h100' (fp8 needs it); no a100|h100 sed.
#   - A directory that holds checkpoints is never deleted or re-cloned.
#   - The log-path sed of rev13 is replaced by a check (logs already land under the clone = $exp_dir).
#   - Compiler caches go under $exp_dir/logs (nothing in the home directory).
#   - Arms copy the baseline checkpoint with cp. NEVER ln -s: torch.save follows a symlink and would
#     overwrite the baseline file.
# Does NOT submit. Prints the launch command. Every write goes under BASE_DIR.
set -e
RUN="${1:-}"
GITHUB_REPO="https://github.com/swarajnanda2021/Dinov2_modified.git"
BRANCH="pathology-fm-recipe-tuned"
BASE_DIR="/data1/vanderbc/test_dinov2_swaraj"          # every write goes under here

case "$RUN" in
  curv2_base|curv2_a10_u01|curv2_a10_u05|curv2_a10_u10|curv2_a15_u01|curv2_a15_u05|curv2_a15_u10) ;;
  *) echo "Usage: $0 <run_key>"
     echo "  run_key: curv2_base | curv2_a10_u01 | curv2_a10_u05 | curv2_a10_u10 | curv2_a15_u01 | curv2_a15_u05 | curv2_a15_u10"
     exit 1 ;;
esac
exp_dir="$BASE_DIR/FMC_ViT-B_${RUN}"
base_ckpt="$BASE_DIR/FMC_ViT-B_curv2_base/logs/checkpoint_iter_00050000.pth"

launch_line () {
  echo "  Launch:  cd $exp_dir && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\"
  echo "           python run_with_submitit.py --nodes 1 --ngpus 4 --partition gpu"
}

echo "========================================"
echo "Setup: FMC_ViT-B_${RUN}  (branch: $BRANCH)"
echo "========================================"

# ---- never delete a directory that holds checkpoints ----
if [ -d "$exp_dir" ] && ls "$exp_dir"/logs/checkpoint*.pth >/dev/null 2>&1; then
  echo "  $exp_dir holds checkpoints. Nothing is deleted or re-cloned. Resume with:"
  launch_line; exit 0
fi
[ -d "$exp_dir" ] && { echo "  Removing existing dir (no checkpoints in it)..."; rm -rf "$exp_dir"; }
mkdir -p "$exp_dir/logs"; cd "$exp_dir"
echo "  Cloning ($BRANCH)..."
git clone -b "$BRANCH" "$GITHUB_REPO" .
git log -1 --oneline | tee "$exp_dir/CLONED_COMMIT.txt"

# ---- guards: the rev13 loader/DP/bank/thinning guards that still apply, plus the v2 + fp8 code ----
grep -q "get_worker_info" data/datasets.py              || { echo "  FATAL: loader fix absent"; exit 1; }
grep -q "class HandRolledDP" training/hand_rolled_dp.py || { echo "  FATAL: HandRolledDP absent"; exit 1; }
grep -q "_sweep_edge" typicality/counted_coverage_bank.py || { echo "  FATAL: s sweep absent"; exit 1; }
grep -q "def _scout_and_bank" training/trainer.py      || { echo "  FATAL: thinning loop absent"; exit 1; }
grep -q "def teacher_targets" losses/prototype_loss.py || { echo "  FATAL: prototype double-count fix absent"; exit 1; }
for a in keep_rule keep_u0 keep_m_ema keep_f_margin keep_f_max fp8_scout scout_amp_bf16 continual \
         save_iter_checkpoint_freq use_gpu_augmentation; do
  grep -q -- "--$a" configs/config.py || { echo "  FATAL: configs/config.py has no --$a"; exit 1; }
done
for f in typicality/keep_rule_v2.py models/fp8_scout.py training/scout_mode.py data/gpu_augment_fast.py; do
  [ -f "$f" ] || { echo "  FATAL: $f missing"; exit 1; }
done
grep -q "thin_state" training/trainer.py                || { echo "  FATAL: v2 resume state absent"; exit 1; }
grep -q "replay_passes" data/datasets.py                || { echo "  FATAL: R1 loader fix absent"; exit 1; }
grep -q "FastCropAugment" training/trainer.py           || { echo "  FATAL: GPU augmenter absent"; exit 1; }
echo "    Code guards verified."

# ---- logs already land under $exp_dir/logs: run_with_submitit.py uses Path(__file__).parent / "logs" and the
#      clone IS $exp_dir. Check instead of sed. ----
grep -q 'p = script_dir / "logs"' run_with_submitit.py || { echo "  FATAL: log dir is not script_dir/logs"; exit 1; }
# ---- the GPU constraint stays 'h100' (fp8 needs it) ----
grep -q "slurm_constraint='h100'" run_with_submitit.py || { echo "  FATAL: constraint is not h100"; exit 1; }
# ---- compiler caches under exp_dir (nothing in the home directory; the triton default is ~/.triton) ----
sed -i "s|'ulimit -l unlimited',|'ulimit -l unlimited',\n            'export TORCHINDUCTOR_CACHE_DIR=$exp_dir/logs/inductor_cache',\n            'export TRITON_CACHE_DIR=$exp_dir/logs/triton_cache',|" run_with_submitit.py
grep -q "TRITON_CACHE_DIR=$exp_dir" run_with_submitit.py || { echo "  FATAL: cache-dir exports not inserted"; exit 1; }

# ensure_arg: substitute "args.<k> = <v>" in place (single-space or '='-aligned); else append after patch_embed_lr_mult.
ensure_arg () {
  local key="args.$1"; local esc="args\\.$1"
  if grep -qE "^[[:space:]]*${esc}[[:space:]]*=" run_with_submitit.py; then
    sed -i -E "s|^([[:space:]]*)${esc}[[:space:]]*=.*|\\1${key} = $2|" run_with_submitit.py
  else
    sed -i -E "/^[[:space:]]*args\\.patch_embed_lr_mult[[:space:]]*=.*/a\\    ${key} = $2" run_with_submitit.py
  fi
}

echo "  Forcing the shared recipe (rev13 ViT-B) + continual + run length..."
ensure_arg vit_variant '"B"';      ensure_arg batch_size_per_gpu 256;   ensure_arg lr 2e-4
ensure_arg clip_grad 3.0;          ensure_arg drop_path_rate 0.4;       ensure_arg drop_path_uniform True
ensure_arg momentum_teacher 0.992; ensure_arg n_standard_local_crops 8; ensure_arg lr_decay_rate 1.0
ensure_arg layerscale_init 1e-5;   ensure_arg norm_last_layer False;    ensure_arg qk_norm True
ensure_arg patch_embed_lr_mult 0.2; ensure_arg ffn_type '"mlp"';        ensure_arg num_workers 10
ensure_arg seed 42;                ensure_arg save_checkpoint_freq 2_000
ensure_arg save_iter_checkpoint_freq 10_000                            # checkpoint_iter_XXXXXXXX.pth every 10k (50k exists)
ensure_arg total_iterations 400_001
ensure_arg continual True;         ensure_arg lr_continual None;        ensure_arg warmup_iterations_continual None
ensure_arg weight_decay_continual 0.04; ensure_arg momentum_teacher_continual 0.994
ensure_arg compile_blocks True;    ensure_arg grad_checkpointing True
for k in use_pathology_recipe use_looped_backbone use_adversarial_mask_augmentation use_cellvit_augmentation \
         use_random_mask_augmentation use_semantic_ibot use_semantic_prototypes use_prototype_clustering; do
  ensure_arg $k False
done

v2_arm () {   # $1 = a, $2 = u0
  ensure_arg use_typicality_dampening True;  ensure_arg balance_mode '"thinned"'
  ensure_arg keep_rule '"v2"';              ensure_arg typicality_a $1;   ensure_arg keep_u0 $2
  ensure_arg keep_m_ema 0.99;                ensure_arg keep_f_margin 1.1
  # ensure_arg keep_f_max 16.0               # <- set after B1 from the loader rate (default 0 = cap 1/u0 only)
  ensure_arg thin_oversample_factor 4.0      # start pool of the first v2 step only; v2 then drives f
  ensure_arg typicality_bank_size 8192;      ensure_arg typicality_halflife_steps 250
  ensure_arg typicality_reserve_size 550;    ensure_arg typicality_reserve_residency 300
  ensure_arg typicality_graduation_hits 2;   ensure_arg typicality_c_frac 0.25
  ensure_arg typicality_radius_mult 1.5;     ensure_arg thin_richardson_correct False
  ensure_arg scout_amp_bf16 True;            ensure_arg fp8_scout True
  ensure_arg typicality_warmup_iters 52_000; ensure_arg use_gpu_augmentation False
}

echo "  Toggling run: $RUN"
case "$RUN" in
  curv2_base)
    ensure_arg use_typicality_dampening False; ensure_arg balance_mode '"off"'
    ensure_arg use_gpu_augmentation True       # same raw-tile loader + GPU augmentation as the arms
    ;;
  curv2_a10_u01) v2_arm 1.0 0.01 ;;  curv2_a10_u05) v2_arm 1.0 0.05 ;;  curv2_a10_u10) v2_arm 1.0 0.10 ;;
  curv2_a15_u01) v2_arm 1.5 0.01 ;;  curv2_a15_u05) v2_arm 1.5 0.05 ;;  curv2_a15_u10) v2_arm 1.5 0.10 ;;
esac

# ---- branch point: a real copy of the baseline 50k checkpoint (torch.save follows symlinks) ----
if [ "$RUN" != "curv2_base" ] && [ ! -f "$exp_dir/logs/checkpoint.pth" ]; then
  [ -f "$base_ckpt" ] || { echo "  FATAL: $base_ckpt does not exist; run the baseline past 50k first"; exit 1; }
  cp "$base_ckpt" "$exp_dir/logs/checkpoint.pth"
  echo "  Branch point: copied $base_ckpt -> logs/checkpoint.pth"
fi

echo ""
echo "  Resolved settings ('MISSING' => fix before launch):"
for kv in args.continual args.total_iterations args.seed args.num_workers args.compile_blocks \
          args.save_checkpoint_freq args.save_iter_checkpoint_freq \
          args.use_typicality_dampening args.balance_mode args.keep_rule args.typicality_a args.keep_u0 \
          args.keep_m_ema args.keep_f_margin args.thin_oversample_factor args.typicality_bank_size \
          args.typicality_halflife_steps args.typicality_reserve_size args.typicality_reserve_residency \
          args.fp8_scout args.scout_amp_bf16 args.typicality_warmup_iters args.use_gpu_augmentation; do
  hit=$(grep -nE "^[[:space:]]*${kv//./\\.}[[:space:]]*=" run_with_submitit.py | head -1)
  printf "    %-38s %s\n" "$kv" "${hit:-MISSING}"
done
echo "    GPU constraint -> $(grep -n "slurm_constraint" run_with_submitit.py | head -1)"
echo "    Clone commit   -> $(cat "$exp_dir/CLONED_COMMIT.txt")"
echo ""
echo "  NOT submitting."
launch_line
echo ""
echo "  GO/NO-GO -- run this in the first minute of the job:"
echo "    grep -ho 'Worker [0-9]*/[0-9]*' $exp_dir/logs/*.out | sort -u"
echo "    EXPECT: ten DISTINCT ids, Worker 0/40 .. Worker 9/40  (only rank 0 prints)."
case "$RUN" in
  curv2_base) ;;
  *) echo "    grep -h 'fp8_scout\|Keep rule\|gate OPEN\|thinned v2' $exp_dir/logs/*.out   # fp8 active, v2 settings, gate open near 52.4k" ;;
esac
