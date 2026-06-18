"""Generate parity fixtures for PhasorLSA (PyTorch side)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import PhasorLSA
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.weights import save_io_pair, save_state


CASES = [
    # (case, in_dims, d_model, n_heads, init_mode, L, B, init_scale, t_period, seed)
    ("lsa_default_h2", 4,  8, 2, "default", 12,  3, 3.0, 1.0, 41),
    ("lsa_hippo_h4",   8, 16, 4, "hippo",   16,  3, 3.0, 1.0, 42),
    ("lsa_long_seq",   6, 12, 3, "hippo",   128, 2, 2.5, 1.0, 43),
    ("lsa_2d_input",   6, 12, 3, "hippo",   None, 5, 3.0, 1.0, 44),
]


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        name, in_d, d_model, n_heads, init_mode, L, B, init_scale, t_period, seed = case
        g = torch.Generator().manual_seed(seed)
        layer = PhasorLSA(
            in_d, d_model, n_heads,
            init_scale=init_scale,
            init_mode=init_mode,
            spk_args=SpikingArgs(t_period=t_period),
            generator=g,
        )
        if L is None:
            x = (torch.rand(in_d, B, generator=g) * 2 - 1).float()
        else:
            x = (torch.rand(in_d, L, B, generator=g) * 2 - 1).float()

        with torch.no_grad():
            y = layer(x)

        meta = {
            "case": name,
            "in_dims": str(in_d),
            "d_model": str(d_model),
            "n_heads": str(n_heads),
            "init_mode": init_mode,
            "B": str(B),
            "init_scale": str(init_scale),
            "t_period": str(t_period),
            "mode": "2d" if L is None else "3d",
        }
        if L is not None:
            meta["L"] = str(L)

        wpath = out_dir / f"{name}_weights.h5"
        ipath = out_dir / f"{name}_io.h5"
        save_state(wpath, {"attn": layer}, metadata=meta)
        save_io_pair(ipath, inputs={"x": x}, outputs={"y": y}, metadata=meta)
        print(f"wrote {wpath.name} and {ipath.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
