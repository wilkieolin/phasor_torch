#!/usr/bin/env bash
# Pin each MPI rank to a single Aurora XPU tile, then exec the command.
#
# We pin by TORCH DEVICE INDEX (PHASOR_HPO_DEVICE=xpu:<tile>), NOT ZE_AFFINITY_MASK:
# under the frameworks module's ONEAPI_DEVICE_SELECTOR, setting ZE_AFFINITY_MASK
# makes torch report "No XPU devices are available." Instead torch sees all 12
# tiles and each rank uses its own index. An Aurora node has 6 PVC GPUs x 2
# tiles = 12 tiles; we map the PALS local rank to a tile. The manager + generator
# ranks also get an index but don't train, so any overlap is harmless.
#
# Usage (from mpiexec):
#   mpiexec -np <ranks> ... scripts/set_affinity_xpu_aurora.sh python -m phasor_torch.hpo_libe ...
set -eo pipefail

NTILES="${PHASOR_TILES_PER_NODE:-12}"
lr="${PALS_LOCAL_RANKID:-0}"
tile=$(( lr % NTILES ))

# Pin via torch device index; select_device() accepts "xpu:N" verbatim. This
# overrides any PHASOR_HPO_DEVICE set in the job script (per rank).
export PHASOR_HPO_DEVICE="xpu:${tile}"

# Set PHASOR_AFFINITY_VERBOSE=1 to print the rank->tile mapping (useful in the
# smoke to confirm workers land on distinct tiles).
if [[ -n "${PHASOR_AFFINITY_VERBOSE:-}" ]]; then
  echo "rank=${PALS_RANKID:-?} local=${lr} -> PHASOR_HPO_DEVICE=${PHASOR_HPO_DEVICE}" >&2
fi

exec "$@"
