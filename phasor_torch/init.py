"""Initialization helpers — symbol/code generation matching Julia src/vsa.jl."""

from __future__ import annotations

import torch
from torch import Tensor


def random_symbols(shape: tuple[int, ...],
                   generator: torch.Generator | None = None) -> Tensor:
    """Uniform i.i.d. Phase samples in [-1, 1]. Mirrors Julia random_symbols."""
    u = torch.rand(*shape, dtype=torch.float32, generator=generator)
    return 2.0 * u - 1.0


def orthogonal_codes(d: int, n: int,
                     generator: torch.Generator | None = None) -> Tensor:
    """Construct n mutually (near-)orthogonal d-dim phasor codes via DFT shift.

    Mirrors Julia orthogonal_codes (src/vsa.jl:266). Output shape: (d, n).
    Requires n <= d (raises otherwise).
    """
    if n > d:
        raise ValueError(
            f"orthogonal_codes requires d >= n (cannot fit {n} orthogonal vectors in {d} dims)"
        )
    if n == 1:
        return random_symbols((d, 1), generator=generator)
    base = 2.0 * torch.rand(d, dtype=torch.float32, generator=generator) - 1.0
    # DFT shift pattern — Julia indexes 1:d so (k-1) % n is what 0..d-1 -> k % n gives.
    shift = torch.tensor(
        [2.0 * (k % n) / n for k in range(d)], dtype=torch.float32
    )
    codes = torch.empty(d, n, dtype=torch.float32)
    for i in range(n):
        codes[:, i] = torch.remainder(i * shift + base + 1.0, 2.0) - 1.0
    return codes
