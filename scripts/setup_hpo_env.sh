#!/usr/bin/env bash
# Create the dedicated `phasor_hpo` conda env (isolated from `nubun`) with the
# ytopt stack. Cloning nubun reuses the working torch build so trials can import
# phasor_torch + torch while ytopt orchestrates. (No Ray — it doesn't work on
# PBS nodes; libEnsemble, pulled in by ytopt, handles multi-node.)
#
# On Aurora prefer a venv on the frameworks module instead of cloning, then run
# the two pip blocks below + scripts/patch_skopt_imputer.py.
#
# Usage: scripts/setup_hpo_env.sh [env_name] [source_env]
#   env_name    target conda env to create   (default: phasor_hpo)
#   source_env  env to clone for torch deps  (default: nubun)
# The ytopt-team forks are git-only (not on PyPI); they're installed editable
# under $YTOPT_SRC_DIR (default ~/code/ytopt_src).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="${1:-phasor_hpo}"
SRC_ENV="${2:-nubun}"
WORK="${YTOPT_SRC_DIR:-$HOME/code/ytopt_src}"

echo ">> cloning conda env '$SRC_ENV' -> '$ENV_NAME'"
conda create -y --clone "$SRC_ENV" -n "$ENV_NAME"

PY="$(conda run -n "$ENV_NAME" which python)"
echo ">> using python: $PY"
"$PY" -m pip install --upgrade pip
# ConfigSpace for the search space; libEnsemble (multi-node launcher) is pulled
# in as a ytopt dependency below. No Ray — it doesn't work on PBS nodes.
"$PY" -m pip install "ConfigSpace"

mkdir -p "$WORK"; cd "$WORK"
echo ">> installing ytopt-team forks under $WORK"
# scikit-optimize fork (dh-scikit-optimize)
[ -d scikit-optimize ] || git clone https://github.com/ytopt-team/scikit-optimize.git
"$PY" -m pip install -e scikit-optimize
# autotune (version1 branch)
[ -d autotune ] || git clone -b version1 https://github.com/ytopt-team/autotune.git
"$PY" -m pip install -e autotune
# ytopt (main branch)
[ -d ytopt ] || git clone -b main https://github.com/ytopt-team/ytopt.git
"$PY" -m pip install -e ytopt

echo ">> patching dh-scikit-optimize imputers for modern numpy/sklearn"
"$PY" "$SCRIPT_DIR/patch_skopt_imputer.py"

echo ">> verifying imports"
"$PY" -c "import ytopt, autotune, ConfigSpace, skopt; print('hpo stack OK')"
echo ">> done. Activate with: conda activate $ENV_NAME"
