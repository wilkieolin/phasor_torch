"""Codebook — fixed phasor codes + cosine similarity classifier.

Port of Julia src/network.jl:1270. Lux puts `codes` in state; we hold it as
a buffer (loadable from HDF5 for parity).

Forward: takes a 2D Phase tensor (d, B) and returns similarity scores against
each of the n codes. Output shape: (n, B). Mirrors the 2D real similarity_outer
dispatch in src/vsa.jl:465.

(For 3D Phase classification use SSMReadout instead — Codebook does not have
a 3D dispatch in the Julia codebase.)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..init import orthogonal_codes, random_symbols
from ..primitives import PI


class Codebook(nn.Module):
    """Fixed-codes classifier head.

    Args:
      d: feature dimension (must match the upstream layer's output channels).
      n: number of codes / classes.
      init_mode: 'random' or 'orthogonal'. The orthogonal init requires n <= d.
      generator: optional torch.Generator for deterministic codes.

    Buffers:
      codes: (d, n) float32, phase values in [-1, 1]. Loadable from HDF5.
    """

    def __init__(self, d: int, n: int, *,
                 init_mode: str = "random",
                 generator: torch.Generator | None = None):
        super().__init__()
        if init_mode not in ("random", "orthogonal"):
            raise ValueError(
                f"init_mode must be 'random' or 'orthogonal', got {init_mode!r}"
            )
        self.d = int(d)
        self.n = int(n)
        self.init_mode = init_mode
        if init_mode == "orthogonal":
            codes = orthogonal_codes(d, n, generator=generator)
        else:
            codes = random_symbols((d, n), generator=generator)
        self.register_buffer("codes", codes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise TypeError(
                f"Codebook expects 2D Phase input (d, B); got ndim={x.ndim}."
            )
        # diff[d, n, b] = x[d, b] - codes[d, n]
        # output[n, b] = mean_d cos(pi * diff[d, n, b])
        diff = x.unsqueeze(1) - self.codes.unsqueeze(2)        # (d, n, B)
        return torch.cos(PI * diff).mean(dim=0)                # (n, B)

    def parameter_dict(self) -> dict[str, Tensor]:
        # codes is the only persistent state — serialize for parity.
        return {"codes": self.codes.detach().cpu()}
