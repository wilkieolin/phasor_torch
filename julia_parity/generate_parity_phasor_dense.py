"""Generate parity fixtures for PhasorDense (PyTorch side).

Writes two HDF5 files per test case:
  - <case>_weights.h5  — layer parameters keyed by Lux-compatible path
  - <case>_io.h5       — canned (input, output) pair for forward verification

Both 2D Phase mode and 3D Phase Dirac mode are exercised across init modes.
The companion julia_parity/verify_phasor_dense.jl loads each pair, rebuilds
the equivalent Lux PhasorDense, runs forward on the saved input, and
asserts the saved output matches within 1e-5.

Run:
  python julia_parity/generate_parity_phasor_dense.py [out_dir]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import torch

# Make `phasor_torch` importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import PhasorDense
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.primitives import normalize_to_unit_circle
from phasor_torch.weights import save_io_pair, save_state


# --------------------------------------------------------------------------
# Case definitions
# --------------------------------------------------------------------------


CASES = [
    # (case_name, mode, in_dims, out_dims, use_bias, init_mode, batch, seq_len, t_period, seed)
    ("dense_2d_default_bias",   "2d", 10, 5,  True,  "default", 8,   None, 1.0, 11),
    ("dense_2d_hippo_nobias",   "2d",  6, 12, False, "hippo",   4,   None, 1.0, 12),
    ("dense_3d_default_bias",   "3d",  8, 6,  True,  "default", 3,   16,   1.0, 13),
    ("dense_3d_hippo_nobias",   "3d",  4, 4,  False, "hippo",   2,   28,   1.0, 14),
    ("dense_3d_long_seq",       "3d",  6, 8,  False, "hippo",   2,   128,  1.0, 15),
]


def _build_layer(in_dims, out_dims, use_bias, init_mode, t_period, seed):
    g = torch.Generator().manual_seed(seed)
    layer = PhasorDense(
        in_dims, out_dims,
        activation=normalize_to_unit_circle,
        use_bias=use_bias,
        init_mode=init_mode,
        spk_args=SpikingArgs(t_period=t_period),
        generator=g,
    )
    # Make bias non-trivial so the test exercises it.
    if use_bias:
        with torch.no_grad():
            layer.bias_real.copy_(torch.linspace(-0.7, 0.5, out_dims))
            layer.bias_imag.copy_(torch.linspace(0.3, -0.9, out_dims))
    return layer


def _make_input(mode, in_dims, batch, seq_len, seed):
    g = torch.Generator().manual_seed(seed + 100)
    if mode == "2d":
        return (torch.rand(in_dims, batch, generator=g) * 2 - 1).float()
    if mode == "3d":
        return (torch.rand(in_dims, seq_len, batch, generator=g) * 2 - 1).float()
    raise ValueError(f"unknown mode {mode!r}")


def _generate_case(out_dir: Path, case):
    name, mode, in_d, out_d, use_bias, init_mode, batch, seq_len, t_period, seed = case
    layer = _build_layer(in_d, out_d, use_bias, init_mode, t_period, seed)
    x = _make_input(mode, in_d, batch, seq_len, seed)
    with torch.no_grad():
        y = layer(x)

    weights_path = out_dir / f"{name}_weights.h5"
    io_path = out_dir / f"{name}_io.h5"
    meta = {
        "case": name,
        "mode": mode,
        "in_dims": str(in_d),
        "out_dims": str(out_d),
        "use_bias": str(use_bias),
        "init_mode": init_mode,
        "t_period": str(t_period),
    }
    save_state(weights_path, {"dense": layer}, metadata=meta)
    save_io_pair(io_path, inputs={"x": x}, outputs={"y": y}, metadata=meta)
    return name, weights_path, io_path, meta


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        name, w, io, meta = _generate_case(out_dir, case)
        print(f"wrote {w.relative_to(out_dir.parent)} and {io.relative_to(out_dir.parent)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
