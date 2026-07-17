"""Generate BACKWARD-parity fixtures for PhasorDense (PyTorch side).

Forward parity is already covered (generate_parity_phasor_dense.py). This adds the
gap the harness never checked: do PyTorch autograd gradients match Julia Zygote
gradients on the same weights + input + cotangent?

For each case we save, in addition to weights + IO:
  - ybar  : a fixed random cotangent (same shape as the output y)
  - grads : d/dp of  loss = sum(y * ybar)  for every parameter, and d/dx.
The companion verify_gradparity_phasor_dense.jl loads these, computes the SAME
scalar's Zygote gradient in Lux, and asserts agreement.

This specifically stresses the Julia `_exp_kdt` custom rrule + the causal-conv /
bias-kernel backward vs PyTorch autograd, and (in the near-origin case) the
`complex_to_angle` 1e-3 gradient gate that both sides claim to implement.

Run:  python julia_parity/generate_gradparity_phasor_dense.py [out_dir]
"""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import PhasorDense
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.primitives import normalize_to_unit_circle
from phasor_torch.weights import save_io_pair, save_state

# (name, in, out, use_bias, init_mode, batch, seq_len, t_period, seed)
CASES = [
    ("gdense_3d_default_bias", 8, 6, True,  "default", 3, 16,  1.0, 13),
    ("gdense_3d_hippo_nobias", 4, 4, False, "hippo",   2, 28,  1.0, 14),
    ("gdense_3d_hippo_long",   6, 8, False, "hippo",   2, 96,  1.0, 15),
    # near-origin stress: hippo (fast channels) + no bias tends to drive |z|->0,
    # exercising the complex_to_angle 1e-3 backward gate on both sides.
    ("gdense_3d_nearorigin",   6, 6, False, "hippo",   4, 32,  1.0, 16),
]


def _build_layer(in_d, out_d, use_bias, init_mode, t_period, seed):
    g = torch.Generator().manual_seed(seed)
    layer = PhasorDense(in_d, out_d, activation=normalize_to_unit_circle,
                        use_bias=use_bias, init_mode=init_mode,
                        spk_args=SpikingArgs(t_period=t_period), generator=g)
    if use_bias:
        with torch.no_grad():
            layer.bias_real.copy_(torch.linspace(-0.7, 0.5, out_d))
            layer.bias_imag.copy_(torch.linspace(0.3, -0.9, out_d))
    return layer


def _generate_case(out_dir: Path, case):
    name, in_d, out_d, use_bias, init_mode, batch, seq_len, t_period, seed = case
    layer = _build_layer(in_d, out_d, use_bias, init_mode, t_period, seed)
    gx = torch.Generator().manual_seed(seed + 100)
    x = (torch.rand(in_d, seq_len, batch, generator=gx) * 2 - 1).float().requires_grad_(True)
    y = layer(x)
    gy = torch.Generator().manual_seed(seed + 200)
    ybar = (torch.rand_like(y) * 2 - 1)                 # fixed cotangent
    loss = (y * ybar).sum()
    loss.backward()

    meta = {"case": name, "mode": "3d", "in_dims": str(in_d), "out_dims": str(out_d),
            "use_bias": str(use_bias), "init_mode": init_mode, "t_period": str(t_period)}
    save_state(out_dir / f"{name}_weights.h5", {"dense": layer}, metadata=meta)
    save_io_pair(out_dir / f"{name}_io.h5",
                 inputs={"x": x.detach(), "ybar": ybar.detach()},
                 outputs={"y": y.detach()}, metadata=meta)
    # grads (raw h5py; Julia reads with the usual reversed-dim convention)
    with h5py.File(out_dir / f"{name}_grads.h5", "w") as f:
        f.create_dataset("weight", data=layer.weight.grad.detach().cpu().numpy())
        f.create_dataset("log_neg_lambda", data=layer.log_neg_lambda.grad.detach().cpu().numpy())
        f.create_dataset("x", data=x.grad.detach().cpu().numpy())
        if use_bias:
            f.create_dataset("bias_real", data=layer.bias_real.grad.detach().cpu().numpy())
            f.create_dataset("bias_imag", data=layer.bias_imag.grad.detach().cpu().numpy())
    print(f"wrote {name}: |gW|={layer.weight.grad.norm():.4g} "
          f"|glnl|={layer.log_neg_lambda.grad.norm():.4g} |gx|={x.grad.norm():.4g}")


def main(out_dir: str | None = None):
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        _generate_case(out_dir, case)


if __name__ == "__main__":
    main(*sys.argv[1:])
