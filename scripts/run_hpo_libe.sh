#!/usr/bin/env bash
# Single-node libEnsemble HPO study (local comms) for ONE body. Run inside the
# `phasor_hpo` env. For multi-node / multi-tile use scripts/hpo_aurora.pbs.
#
# With --nworkers N: one worker hosts the persistent ytopt generator and N-1
# run trials concurrently. On a single Aurora node set N = tiles + 1 (e.g. 13)
# and use the PBS script for tile pinning; for a CPU/local smoke a few workers
# is plenty.
#
# Examples:
#   # smoke (synthetic, cpu, 3 concurrent trials):
#   scripts/run_hpo_libe.sh --body lca --source synthetic --nworkers 4 \
#       --max-evals 4 --epochs-min 1 --epochs-max 1 --device cpu \
#       --outdir /tmp/hpo_libe_smoke
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$PWD}"
export PYTHONUNBUFFERED=1

BODY=lca
NWORKERS=4
MAX_EVALS=50
EPOCHS_MIN=30
EPOCHS_MAX=80
LEARNER=RF
SOURCE=audio
PATIENCE=6
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
    --nworkers) NWORKERS="$2"; shift 2;;
    --max-evals) MAX_EVALS="$2"; shift 2;;
    --epochs-min) EPOCHS_MIN="$2"; shift 2;;
    --epochs-max) EPOCHS_MAX="$2"; shift 2;;
    --patience) PATIENCE="$2"; shift 2;;
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

# libEnsemble cds workers into per-worker dirs, so the artifact dir must be
# absolute (the objective writes from inside the worker).
[[ "$OUTDIR" = /* ]] || OUTDIR="$PWD/$OUTDIR"

export PHASOR_HPO_BODY="$BODY"
export PHASOR_HPO_SOURCE="$SOURCE"
export PHASOR_HPO_TRAIN_PATH="$TRAIN"
export PHASOR_HPO_TEST_PATH="$TEST"
export PHASOR_HPO_EPOCHS_MIN="$EPOCHS_MIN"
export PHASOR_HPO_EPOCHS_MAX="$EPOCHS_MAX"
export PHASOR_HPO_PATIENCE="$PATIENCE"
export PHASOR_HPO_COSINE="$COSINE"
export PHASOR_HPO_LR_MIN="$LRMIN"
export PHASOR_HPO_DEVICE="$DEVICE"
export PHASOR_HPO_OUTDIR="$OUTDIR/$BODY"
export PHASOR_HPO_ENSEMBLE_DIR="$PHASOR_HPO_OUTDIR/ensemble"
[ -n "$TRAIN_LIMIT" ] && export PHASOR_HPO_TRAIN_LIMIT="$TRAIN_LIMIT"
[ -n "$TEST_LIMIT" ] && export PHASOR_HPO_TEST_LIMIT="$TEST_LIMIT"

mkdir -p "$PHASOR_HPO_OUTDIR"
echo "libE study: body=$BODY source=$SOURCE nworkers=$NWORKERS (sims=$((NWORKERS-1))) max_evals=$MAX_EVALS epochs=[$EPOCHS_MIN,$EPOCHS_MAX] patience=$PATIENCE cosine=${COSINE:-0}(lr_min=$LRMIN) device=$DEVICE"
echo "artifacts: $PHASOR_HPO_OUTDIR   results: $PHASOR_HPO_OUTDIR/results.csv"

python -m phasor_torch.hpo_libe \
  --comms local --nworkers "$NWORKERS" \
  --learner "$LEARNER" --max-evals "$MAX_EVALS"
