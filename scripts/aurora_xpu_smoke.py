"""Aurora (Intel PVC / xpu) pre-flight smoke for the ResonantSTFT audio frontend.

Isolates the one piece of the audio path that is unverified on xpu: the
`causal_conv` FFT branch running at the full audio length L=16000 **every step,
including the backward pass** (torch.fft.fft / ifft autograd on PVC). Uses a
synthetic input so it needs no staged data and fails in seconds if PVC's
torch.fft autograd misbehaves — run this before the multi-minute train.

Usage (on an Aurora compute node):
    module load frameworks
    cd phasor_torch
    PYTHONPATH=. python scripts/aurora_xpu_smoke.py [--device xpu] [--length 16000] [--batch 8]

Exit code 0 = all finite (PASS); 1 = NaN/inf detected or backend error (FAIL).
"""

from __future__ import annotations

import argparse
import sys

import torch

from phasor_torch.layers import (
    ResonantSTFT,
    downsample_time,
    encode_input,
    to_phase,
)


def _resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="xpu pre-flight for ResonantSTFT FFT-in-loop.")
    p.add_argument("--device", default="xpu",
                   help="torch device ('xpu' force-pin by default, or 'auto'/'cuda'/'cpu').")
    p.add_argument("--length", type=int, default=16000, help="waveform length L (FFT branch when >64).")
    p.add_argument("--batch", type=int, default=8, help="batch size B.")
    p.add_argument("--n-freqs", type=int, default=64, help="ResonantSTFT output channels.")
    p.add_argument("--downsample", type=int, default=32, help="time downsample factor.")
    ns = p.parse_args(argv)

    device = _resolve_device(ns.device)
    print(f"device: {device}")
    print(f"config: L={ns.length} B={ns.batch} n_freqs={ns.n_freqs} downsample={ns.downsample}")

    torch.manual_seed(0)
    frontend = ResonantSTFT(1, ns.n_freqs).to(device)   # SLERP default

    # Synthetic real waveform batch (1, L, B); requires_grad to exercise the
    # backward through the FFT-in-loop all the way to the input.
    x = torch.randn(1, ns.length, ns.batch, device=device, requires_grad=True)

    z = encode_input(x)                                 # complex (1, L, B)
    z = frontend(z)                                     # FFT branch (L >> 64): (n_freqs, L, B)
    fwd_finite = bool(torch.isfinite(z.real).all() and torch.isfinite(z.imag).all())
    print(f"forward (post-ResonantSTFT) finite: {fwd_finite}  shape={tuple(z.shape)}")

    phase = to_phase(downsample_time(z, ns.downsample)) # (n_freqs, L/ds, B) real phase
    pd = phase.detach()
    print(f"after downsample+to_phase: shape={tuple(phase.shape)} "
          f"range=[{float(pd.min()):.4f}, {float(pd.max()):.4f}]")

    # Backward: this is the part that has never run on PVC at L=16000.
    loss = phase.float().pow(2).mean()
    loss.backward()

    x_grad_finite = bool(x.grad is not None and torch.isfinite(x.grad).all())
    bad_params = [n for n, pp in frontend.named_parameters()
                  if pp.grad is None or not torch.isfinite(pp.grad).all()]
    print(f"loss: {float(loss):.6f}  input-grad finite: {x_grad_finite}")
    print(f"frontend params with missing/non-finite grad: {bad_params}")

    ok = fwd_finite and bool(torch.isfinite(phase).all()) and x_grad_finite and not bad_params
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
