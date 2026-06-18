"""Generate parity fixtures for ResonantSTFT (PyTorch side).

Covers the 3D-Complex dispatch (the audio path): SLERP / normalize / identity
activations, bias on/off, and short (Toeplitz) vs long (FFT) sequences. Params
are perturbed off their deterministic init so the parity check is non-trivial
(it must match a learned per-channel omega, not just the linspace default).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.layers import ResonantSTFT, resolve_activation
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.weights import save_io_pair, save_state


CASES = [
    # (case, in_dims, n_freqs, activation, use_bias, L, B, t_period, seed)
    ("rstft_slerp_short",     1,  8, "slerp",     False, 24,  3, 1.0, 51),
    ("rstft_normalize_bias",  2, 12, "normalize", True,  20,  3, 1.0, 53),
    ("rstft_identity_short",  2, 10, "identity",  False, 16,  2, 1.0, 54),
    ("rstft_slerp_long",      1, 16, "slerp",     False, 128, 2, 1.0, 52),
    ("rstft_slerp_bias_long", 1, 12, "slerp",     True,  160, 2, 1.0, 55),
]


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for (name, in_d, n_freqs, act, use_bias, L, B, t_period, seed) in CASES:
        g = torch.Generator().manual_seed(seed)
        layer = ResonantSTFT(
            in_d, n_freqs, resolve_activation(act),
            use_bias=use_bias,
            spk_args=SpikingArgs(t_period=t_period),
            generator=g,
        )
        # Perturb params off their init for a stronger parity test.
        with torch.no_grad():
            layer.omega.add_(0.3 * torch.randn(n_freqs, generator=g))
            layer.log_neg_lambda.add_(0.2 * torch.randn(n_freqs, generator=g))
            if use_bias:
                layer.bias_real.add_(0.5 * torch.randn(n_freqs, generator=g))
                layer.bias_imag.add_(0.5 * torch.randn(n_freqs, generator=g))
            if act == "slerp":
                layer.log_r_lo.add_(0.2 * torch.randn(n_freqs, generator=g))
                layer.log_r_gap.add_(0.2 * torch.randn(n_freqs, generator=g))

        x = torch.complex(torch.randn(in_d, L, B, generator=g),
                          torch.randn(in_d, L, B, generator=g))

        with torch.no_grad():
            y = layer(x)

        meta = {
            "case": name,
            "in_dims": str(in_d),
            "n_freqs": str(n_freqs),
            "activation": act,
            "use_bias": str(bool(use_bias)).lower(),  # "true"/"false" for Julia parse
            "B": str(B),
            "L": str(L),
            "t_period": str(t_period),
            "mode": "3d_complex",
        }

        wpath = out_dir / f"{name}_weights.h5"
        ipath = out_dir / f"{name}_io.h5"
        save_state(wpath, {"stft": layer}, metadata=meta)
        save_io_pair(ipath, inputs={"x": x}, outputs={"y": y}, metadata=meta)
        print(f"wrote {wpath.name} and {ipath.name}")


if __name__ == "__main__":
    main(*sys.argv[1:])
