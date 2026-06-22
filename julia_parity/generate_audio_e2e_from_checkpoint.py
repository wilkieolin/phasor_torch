"""Generate an end-to-end audio-chain parity fixture from a trained checkpoint.

Finds the top-performing trial under a sweep dir (highest test_acc in
history.json), rebuilds its model from config.json, loads the trained weights
(best.h5), runs the full audio forward on a real audio batch, and saves the
weights + an (input, sims) IO pair so verify_audio_e2e.jl can confirm the Julia
chain produces the same outputs.

Usage:
  python julia_parity/generate_audio_e2e_from_checkpoint.py [run_dir] [ckpt_name]
    run_dir   sweep body dir (default: hpo_runs/lca)
    ckpt_name best.h5 (default) | checkpoint.h5 | ckpt_epoch70.h5 ...
"""

from __future__ import annotations

import dataclasses
import glob
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.config import ModelConfig
from phasor_torch.data.audio import load_audio
from phasor_torch.train import build_model, forward_model
from phasor_torch.weights import save_io_pair, save_state

AUDIO_TEST = "/home/wilkie/data/sound/sound_data_raw_test.h5"


def find_top_trial(run_dir: str) -> tuple[float, str]:
    best = None
    for d in glob.glob(os.path.join(run_dir, "trial_*")):
        hp = os.path.join(d, "history.json")
        if not os.path.exists(hp):
            continue
        hist = json.load(open(hp))
        if not hist:
            continue
        acc = max(r["test_acc"] for r in hist)
        if best is None or acc > best[0]:
            best = (acc, d)
    if best is None:
        raise SystemExit(f"no trials with history under {run_dir}")
    return best


def main() -> None:
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "hpo_runs/lca"
    ckpt_name = sys.argv[2] if len(sys.argv) > 2 else "best.h5"

    acc, trial = find_top_trial(run_dir)
    cfg_json = json.load(open(os.path.join(trial, "config.json")))
    fields = {f.name for f in dataclasses.fields(ModelConfig)}
    mc = {k: v for k, v in cfg_json["model"].items() if k in fields}
    cfg = ModelConfig(**mc)
    print(f"top trial: {trial}  explore_test_acc={acc:.4f}")
    print(f"  config: body={cfg.body} d_hidden={cfg.d_hidden} n_heads={cfg.n_heads} "
          f"n_anchors={cfg.n_anchors} n_freqs={cfg.n_freqs} readout_frac={cfg.readout_frac:.4f}")

    model, schema = build_model(cfg)
    from phasor_torch.weights import load_state
    load_state(os.path.join(trial, ckpt_name), schema)
    model.eval()

    # Real audio batch (a handful of held-out clips).
    x, y = load_audio(AUDIO_TEST, n_classes=cfg.n_classes, limit=32, seed=0)
    B = min(16, x.shape[2])
    x, y = x[:, :, :B].contiguous(), y[:B]
    ds = cfg.downsample_factor if cfg.frontend == "resonant" else 1
    with torch.no_grad():
        sims = forward_model(schema, x, ds)              # (n_classes, B)

    out = Path(__file__).resolve().parent / "fixtures"
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "trial": os.path.basename(trial), "ckpt": ckpt_name,
        "explore_test_acc": f"{acc:.6f}",
        "n_classes": str(cfg.n_classes), "d_hidden": str(cfg.d_hidden),
        "n_heads": str(cfg.n_heads), "n_anchors": str(cfg.n_anchors),
        "n_freqs": str(cfg.n_freqs), "downsample_factor": str(cfg.downsample_factor),
        "omega_lo": str(cfg.omega_lo), "omega_hi": str(cfg.omega_hi),
        "readout_frac": str(cfg.readout_frac), "init_mode": cfg.init_mode,
        "t_period": str(cfg.t_period), "body": cfg.body,
    }
    save_state(out / "audio_e2e_weights.h5", schema, metadata=meta)
    save_io_pair(out / "audio_e2e_io.h5",
                 inputs={"x": x, "labels": y.float()},
                 outputs={"sims": sims}, metadata=meta)
    preds = sims.argmax(dim=0)
    print(f"  batch B={B}  sims {tuple(sims.shape)}  PyTorch preds={preds.tolist()}")
    print(f"  wrote {out/'audio_e2e_weights.h5'} and {out/'audio_e2e_io.h5'}")


if __name__ == "__main__":
    main()
