"""Generate parity fixtures for Codebook + SSMReadout (PyTorch side).

Same pattern as generate_parity_phasor_dense.py: writes weight + IO HDF5
pairs that the corresponding julia_parity/verify_readouts.jl loads and
verifies. Both readouts are state-only (no trainable params); the `codes`
buffer is the thing that must round-trip.

Run:
  python julia_parity/generate_parity_readouts.py [out_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import Codebook, SSMReadout
from phasor_torch.weights import save_io_pair, save_state


# (case, layer_kind, d/h, n_or_K, init_mode, L (None for 2D), B, readout_frac, seed)
CASES = [
    ("codebook_random_small",   "codebook", 6,  4, "random",      None,  8,  None, 21),
    ("codebook_orthogonal_div", "codebook", 12, 4, "orthogonal",  None,  5,  None, 22),
    ("codebook_orthogonal_n1",  "codebook", 8,  1, "orthogonal",  None,  3,  None, 23),
    ("ssm_readout_frac25",      "ssm",      8,  5, None,          20,    3,  0.25, 31),
    ("ssm_readout_frac50_long", "ssm",      6,  4, None,          128,   2,  0.5,  32),
    ("ssm_readout_short_seq",   "ssm",      4,  3, None,          4,     2,  0.25, 33),
]


def _build(case):
    name, kind, d, K, init_mode, L, B, frac, seed = case
    g = torch.Generator().manual_seed(seed)
    if kind == "codebook":
        layer = Codebook(d, K, init_mode=init_mode, generator=g)
        x = (torch.rand(d, B, generator=g) * 2 - 1).float()
    else:  # ssm
        layer = SSMReadout(d, K, readout_frac=frac, generator=g)
        x = (torch.rand(d, L, B, generator=g) * 2 - 1).float()
    with torch.no_grad():
        y = layer(x)
    return layer, x, y


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        name, kind, d, K, init_mode, L, B, frac, seed = case
        layer, x, y = _build(case)

        meta = {
            "case": name,
            "kind": kind,
            "d": str(d),
            "K": str(K),
            "B": str(B),
        }
        if kind == "codebook":
            meta.update(init_mode=init_mode or "random")
        else:
            meta.update(L=str(L), readout_frac=str(frac))

        wpath = out_dir / f"{name}_weights.h5"
        ipath = out_dir / f"{name}_io.h5"
        layer_key = "codebook" if kind == "codebook" else "readout"
        save_state(wpath, {layer_key: layer}, metadata=meta)
        save_io_pair(ipath, inputs={"x": x}, outputs={"y": y}, metadata=meta)
        print(f"wrote {wpath.name} and {ipath.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
