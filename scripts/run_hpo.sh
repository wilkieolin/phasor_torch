#!/usr/bin/env bash
# Single-node ytopt HPO study for ONE body (one study per body), using ytopt's
# `subprocess` evaluator. Run inside the `phasor_hpo` env (see setup_hpo_env.sh).
# ambs imports phasor_torch.hpo.Problem, which reads the PHASOR_HPO_* env vars
# set below.
#
# We use the subprocess evaluator (not Ray): Ray's ray.init() hangs on PBS
# nodes and its AF_UNIX plasma socket can't live on Lustre / overflows the
# 107-byte path limit. For multi-node / multi-tile parallelism use the
# libEnsemble launcher (scripts/run_hpo_libe.sh) instead.
#
# Examples:
#   # smoke (synthetic, fast, no audio dependency):
#   scripts/run_hpo.sh --body lca --source synthetic \
#       --max-evals 1 --epochs-min 1 --epochs-max 1 --device cpu \
#       --outdir /flare/EE-ECP/wolin/hpo_smoke
#   # single-node audio sweep:
#   scripts/run_hpo.sh --body lca --max-evals 50 --epochs-min 30 --epochs-max 80
#
# ambs writes its search-trajectory results.csv in the cwd; per-trial artifacts
# (config.json/history.json/checkpoint.h5) go under <outdir>/<body>/trial_*.
set -euo pipefail

# Run from the repo root so `phasor_torch` is importable.
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTHONUNBUFFERED=1

BODY=lca
MAX_EVALS=50
EPOCHS_MIN=30
EPOCHS_MAX=80
LEARNER=RF
SOURCE=audio
PATIENCE=6
NBLOCKS=1          # stacked (body -> dense) blocks; fixed per study
COSINE=""          # --cosine to enable cosine LR decay to LRMIN over each trial
LRMIN=1e-6
TRAIN=/flare/EE-ECP/wolin/mos2_oscillators/sound_data_raw.h5
TEST=/flare/EE-ECP/wolin/mos2_oscillators/sound_data_raw_test.h5
OUTDIR=hpo_runs
TRAIN_LIMIT=""
TEST_LIMIT=""
DEVICE=auto

while [[ $# -gt 0 ]]; do
  case "$1" in
    --body) BODY="$2"; shift 2;;
    --max-evals) MAX_EVALS="$2"; shift 2;;
    --epochs-min) EPOCHS_MIN="$2"; shift 2;;
    --epochs-max) EPOCHS_MAX="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
    --n-blocks) NBLOCKS="$2"; shift 2;;
    --cosine) COSINE=1; shift 1;;
    --lr-min) LRMIN="$2"; shift 2;;
    --source) SOURCE="$2"; shift 2;;
    --train-path) TRAIN="$2"; shift 2;;
    --test-path) TEST="$2"; shift 2;;
    --train-limit) TRAIN_LIMIT="$2"; shift 2;;
    --test-limit) TEST_LIMIT="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --outdir) OUTDIR="$2"; shift 2;;
    --learner) LEARNER="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

export PHASOR_HPO_BODY="$BODY"
export PHASOR_HPO_SOURCE="$SOURCE"
export PHASOR_HPO_TRAIN_PATH="$TRAIN"
export PHASOR_HPO_TEST_PATH="$TEST"
export PHASOR_HPO_EPOCHS_MIN="$EPOCHS_MIN"
export PHASOR_HPO_EPOCHS_MAX="$EPOCHS_MAX"
export PHASOR_HPO_PATIENCE="$PATIENCE"
export PHASOR_HPO_N_BLOCKS="$NBLOCKS"
export PHASOR_HPO_COSINE="$COSINE"
export PHASOR_HPO_LR_MIN="$LRMIN"
export PHASOR_HPO_DEVICE="$DEVICE"
export PHASOR_HPO_OUTDIR="$OUTDIR/$BODY"
[ -n "$TRAIN_LIMIT" ] && export PHASOR_HPO_TRAIN_LIMIT="$TRAIN_LIMIT"
[ -n "$TEST_LIMIT" ] && export PHASOR_HPO_TEST_LIMIT="$TEST_LIMIT"

mkdir -p "$PHASOR_HPO_OUTDIR"
echo "study: body=$BODY source=$SOURCE evaluator=subprocess max_evals=$MAX_EVALS epochs=[$EPOCHS_MIN,$EPOCHS_MAX] patience=$PATIENCE cosine=${COSINE:-0}(lr_min=$LRMIN) device=$DEVICE"
echo "artifacts: $PHASOR_HPO_OUTDIR   (ambs results.csv in $PWD)"

python -m ytopt.search.ambs \
  --evaluator subprocess \
  --problem phasor_torch.hpo.Problem \
  --max-evals "$MAX_EVALS" \
  --learner "$LEARNER"
