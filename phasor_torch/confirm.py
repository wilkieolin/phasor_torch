"""Full-data confirmation of the top HPO configs from a sweep.

Reads a sweep's ``results.csv``, takes the top-K rows by objective (lower
``objective`` = better, since it is ``-test_acc``), and trains each at FULL data
(no train/test subset) to get its true accuracy. One config per MPI rank
(``PALS_RANKID``), each pinned to a tile by ``scripts/set_affinity_xpu_aurora.sh``,
so K configs train in parallel.

This is NOT a search — it just re-runs a fixed set of configs. Reuses
``hpo.point_to_runconfig`` so the index dims / mappings stay identical to the
sweep. Configure via the same ``PHASOR_HPO_*`` env vars (do NOT set
``PHASOR_HPO_TRAIN_LIMIT`` / ``TEST_LIMIT`` -> full data) plus:

  PHASOR_CONFIRM_RESULTS  path to the sweep's results.csv (required)
  PHASOR_CONFIRM_TOPK     number of top configs to confirm (default 8)
  PHASOR_CONFIRM_EPOCHS   optional override of each config's epoch count

Run (in the phasor_hpo env, under PBS): see scripts/confirm_top.pbs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os

from . import hpo
from .train import train


def read_top(results_path: str, k: int) -> list[dict]:
    """Return the k rows with the smallest (best) `objective`, best first."""
    with open(results_path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("objective") not in (None, "")]
    rows.sort(key=lambda r: float(r["objective"]))   # most negative -test_acc = best
    return rows[:k]


def _point_from_row(row: dict) -> dict:
    """A ConfigSpace point is the results row minus the objective column.

    Values stay as strings; point_to_runconfig coerces them (int()/float()).
    """
    return {k: v for k, v in row.items() if k != "objective"}


def main() -> int:
    results_path = os.environ["PHASOR_CONFIRM_RESULTS"]
    k = int(os.environ.get("PHASOR_CONFIRM_TOPK", "8"))
    epochs_override = os.environ.get("PHASOR_CONFIRM_EPOCHS")
    rank = int(os.environ.get("PALS_RANKID", os.environ.get("PMI_RANK", "0")))

    base = hpo.HpoBase.from_env()
    top = read_top(results_path, k)
    if rank >= len(top):
        print(f"[confirm rank {rank}] idle (only {len(top)} configs)", flush=True)
        return 0

    point = _point_from_row(top[rank])
    if epochs_override:
        point["epochs"] = epochs_override
    run = hpo.point_to_runconfig(point, base)

    h = hashlib.sha1(json.dumps(point, sort_keys=True).encode()).hexdigest()[:8]
    outdir = os.path.join(base.outdir, f"confirm_{rank:02d}_{h}")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "point.json"), "w") as fh:
        json.dump(point, fh, indent=2, sort_keys=True)

    result = train(run, save_path=os.path.join(outdir, "checkpoint.h5"))
    hist = result.get("history") or []
    best = max((float(r["test_acc"]) for r in hist), default=0.0)
    final = float((result.get("final") or {}).get("test_acc", 0.0))
    print(f"[confirm rank {rank}] explore_obj={top[rank]['objective']} "
          f"full_best_test_acc={best:.4f} full_final_test_acc={final:.4f} "
          f"epochs_run={len(hist)} dir={outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
