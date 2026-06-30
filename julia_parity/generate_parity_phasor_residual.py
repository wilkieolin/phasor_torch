"""Generate parity fixtures for the phasor transformer block (PyTorch side).

Covers PhasorTransformerBlock wrapping PhasorLSA or PhasorLCA, across both gate
modes, recenter on/off, 2D/3D, and a long (FFT-path) sequence. The Julia
verifier (verify_phasor_residual.jl) rebuilds the equivalent Lux block, injects
these weights, and asserts the forward outputs agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import PhasorLCA, PhasorLSA, PhasorTransformerBlock
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.weights import save_io_pair, save_state


CASES = [
    # (name, kind, D, n_heads, n_anchors, gate, alpha0, bis, recenter, d_ff,
    #  L, B, init_scale, t_period, seed)
    ("res_lsa_rezero",        "lsa", 16, 4,  0, "rezero", 0.3, 0.1, False, 0,  12, 3, 3.0, 1.0, 41),
    ("res_lca_rezero",        "lca", 16, 4,  8, "rezero", 0.2, 0.1, False, 0,  10, 3, 3.0, 1.0, 42),
    ("res_lsa_none",          "lsa",  8, 2,  0, "none",   0.0, 0.1, False, 0,   8, 2, 3.0, 1.0, 43),
    ("res_lsa_recenter",      "lsa", 16, 4,  0, "rezero", 0.25, 0.1, True, 0,   6, 3, 3.0, 1.0, 44),
    ("res_lca_recenter",      "lca", 16, 4,  8, "rezero", 0.15, 0.1, True, 0,   6, 3, 3.0, 1.0, 45),
    ("res_lsa_dff",           "lsa", 16, 4,  0, "rezero", 0.3, 0.2, False, 24, 10, 3, 3.0, 1.0, 46),
    ("res_lsa_2d",            "lsa", 12, 3,  0, "rezero", 0.3, 0.1, False, 0, None, 5, 3.0, 1.0, 47),
    ("res_lsa_long_seq",      "lsa", 12, 3,  0, "rezero", 0.3, 0.1, False, 0, 128, 2, 2.5, 1.0, 48),
]


def _make_attn(kind, D, n_heads, n_anchors, init_scale, t_period, g):
    spk = SpikingArgs(t_period=t_period)
    if kind == "lsa":
        return PhasorLSA(D, D, n_heads, init_scale=init_scale, spk_args=spk, generator=g)
    return PhasorLCA(D, D, n_heads, n_anchors, init_scale=init_scale,
                     spk_args=spk, generator=g)


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        (name, kind, D, n_heads, n_anchors, gate, alpha0, bis, recenter, d_ff,
         L, B, init_scale, t_period, seed) = case
        g = torch.Generator().manual_seed(seed)
        attn = _make_attn(kind, D, n_heads, n_anchors, init_scale, t_period, g)
        block = PhasorTransformerBlock(
            D, attn, d_ff=d_ff, gate=gate, alpha0=alpha0,
            branch_init_scale=bis, recenter=recenter,
            spk_args=SpikingArgs(t_period=t_period), generator=g,
        )

        if L is None:
            x = (torch.rand(D, B, generator=g) * 2 - 1).float()
        else:
            x = (torch.rand(D, L, B, generator=g) * 2 - 1).float()

        with torch.no_grad():
            y = block(x)

        meta = {
            "case": name,
            "kind": kind,
            "in_dims": str(D),
            "d_model": str(D),
            "n_heads": str(n_heads),
            "n_anchors": str(n_anchors),
            "gate": gate,
            "alpha0": str(alpha0),
            "branch_init_scale": str(bis),
            "recenter": "True" if recenter else "False",
            "d_ff": str(d_ff),
            "B": str(B),
            "init_scale": str(init_scale),
            "t_period": str(t_period),
            "mode": "2d" if L is None else "3d",
        }
        if L is not None:
            meta["L"] = str(L)

        wpath = out_dir / f"{name}_weights.h5"
        ipath = out_dir / f"{name}_io.h5"
        save_state(wpath, {"block": block}, metadata=meta)
        save_io_pair(ipath, inputs={"x": x}, outputs={"y": y}, metadata=meta)
        print(f"wrote {wpath.name} and {ipath.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
