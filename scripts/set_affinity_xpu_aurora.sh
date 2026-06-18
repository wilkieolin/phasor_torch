#!/usr/bin/env bash
# Pin each MPI rank to a single Aurora XPU tile, then exec the command.
#
# An Aurora node has 6 PVC GPUs x 2 tiles = 12 tiles. We map the PALS local
# rank to a tile via ZE_AFFINITY_MASK="<gpu>.<subtile>" so each libEnsemble
# sim worker initializes torch on its own tile (avoids every worker grabbing
# xpu:0). The manager + generator ranks also get a mask but don't use the GPU,
# so any overlap is harmless.
#
# Usage (from mpiexec):
#   mpiexec -np <ranks> ... scripts/set_affinity_xpu_aurora.sh python -m phasor_torch.hpo_libe ...
set -euo pipefail

NTILES="${PHASOR_TILES_PER_NODE:-12}"
lr="${PALS_LOCAL_RANKID:-0}"
tile=$(( lr % NTILES ))
gpu=$(( tile / 2 ))
sub=$(( tile % 2 ))
export ZE_AFFINITY_MASK="${gpu}.${sub}"

# Set PHASOR_AFFINITY_VERBOSE=1 to print the rank->tile mapping (useful in the
# smoke to confirm workers land on distinct tiles).
if [[ -n "${PHASOR_AFFINITY_VERBOSE:-}" ]]; then
  echo "rank=${PALS_RANKID:-?} local=${lr} -> ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}" >&2
fi

exec "$@"
