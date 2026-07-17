"""Generate BACKWARD-parity fixtures for PhasorLCA (PyTorch side).

Tests the attention backward path (similarity_outer_heads, anchor bundle, VSA
bind, scale) — the prime suspect for the linchpin attn-only training gap — against
Julia Zygote. Saves ybar (cotangent) + grads for every param and the input.

Run:  python julia_parity/generate_gradparity_phasor_lca.py [out_dir]
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import PhasorLCA
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.weights import save_io_pair, save_state

# (case, in, d_model, n_heads, n_anchors, init_mode, L, B, scale, period, seed)
CASES = [
    ("glca_default_h2_A3", 4,  8, 2, 3, "default", 12, 3, 3.0, 1.0, 51),
    ("glca_hippo_h4_A8",   8, 16, 4, 8, "hippo",   16, 3, 3.0, 1.0, 52),
    ("glca_long_seq",      6, 12, 3, 6, "hippo",   96, 2, 2.5, 1.0, 53),
]


def main(out_dir: str | None = None) -> None:
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, in_d, d_model, n_heads, n_anchors, init_mode, L, B, scale, period, seed in CASES:
        g = torch.Generator().manual_seed(seed)
        layer = PhasorLCA(in_d, d_model, n_heads, n_anchors, init_scale=scale,
                          init_mode=init_mode, spk_args=SpikingArgs(t_period=period),
                          generator=g)
        x = (torch.rand(in_d, L, B, generator=g) * 2 - 1).float().requires_grad_(True)
        y = layer(x)
        gy = torch.Generator().manual_seed(seed + 200)
        ybar = (torch.rand_like(y) * 2 - 1)
        (y * ybar).sum().backward()

        meta = {"case": name, "in_dims": str(in_d), "d_model": str(d_model),
                "n_heads": str(n_heads), "n_anchors": str(n_anchors),
                "init_mode": init_mode, "B": str(B), "init_scale": str(scale),
                "t_period": str(period), "mode": "3d", "L": str(L)}
        save_state(out_dir / f"{name}_weights.h5", {"attn": layer}, metadata=meta)
        save_io_pair(out_dir / f"{name}_io.h5",
                     inputs={"x": x.detach(), "ybar": ybar.detach()},
                     outputs={"y": y.detach()}, metadata=meta)
        with h5py.File(out_dir / f"{name}_grads.h5", "w") as f:
            f.create_dataset("kweight", data=layer.k_proj.weight.grad.cpu().numpy())
            f.create_dataset("klnl", data=layer.k_proj.log_neg_lambda.grad.cpu().numpy())
            f.create_dataset("vweight", data=layer.v_proj.weight.grad.cpu().numpy())
            f.create_dataset("vlnl", data=layer.v_proj.log_neg_lambda.grad.cpu().numpy())
            f.create_dataset("anchors", data=layer.anchors.grad.cpu().numpy())
            f.create_dataset("scale", data=layer.scale.grad.cpu().numpy())
            f.create_dataset("x", data=x.grad.cpu().numpy())
        print(f"wrote {name}: |gkW|={layer.k_proj.weight.grad.norm():.4g} "
              f"|ganch|={layer.anchors.grad.norm():.4g} |gscale|={layer.scale.grad.norm():.4g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
