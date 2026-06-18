"""SSMReadout — temporal-window codebook classifier for 3D Phase sequences.

Port of Julia src/ssm.jl:43. Takes a 3D Phase tensor (C, L, B), averages
similarity-to-codes over the last `readout_frac` of timesteps.

Codes live in a buffer (Lux puts them in state); loadable from HDF5 for parity.

Window selection (matches Julia ssm.jl:60-62 and ssm.jl:86-88):
    t0 = max(0, L - max(1, round(L * readout_frac)))    # 0-indexed start
    window  = x[:, t0:L, :]                              # length W = L - t0
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..init import random_symbols
from ..primitives import PI


def _readout_t0(L: int, readout_frac: float) -> int:
    """0-indexed start of the readout window. Mirrors the Julia formula."""
    window_len = max(1, round(L * float(readout_frac)))
    return max(0, L - window_len)


class SSMReadout(nn.Module):
    """Temporal-window similarity classifier for 3D Phase sequences.

    Args:
      hidden_dims: feature dim (must match upstream output channels).
      n_classes: number of target classes.
      readout_frac: fraction of the last timesteps to average (default 0.25).
      generator: optional torch.Generator for deterministic codes.

    Buffers:
      codes: (hidden_dims, n_classes) float32 phase values in [-1, 1].

    Forward: (C, L, B) Phase  ->  (n_classes, B) similarity-score logits.
    """

    def __init__(self, hidden_dims: int, n_classes: int, *,
                 readout_frac: float = 0.25,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.hidden_dims = int(hidden_dims)
        self.n_classes = int(n_classes)
        self.readout_frac = float(readout_frac)
        codes = random_symbols((hidden_dims, n_classes), generator=generator)
        self.register_buffer("codes", codes)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise TypeError(
                f"SSMReadout expects 3D Phase input (C, L, B); got ndim={x.ndim}."
            )
        if x.is_complex():
            # Match the Julia complex dispatch: normalize, take angle, then proceed.
            from ..primitives import complex_to_angle, normalize_to_unit_circle
            x = complex_to_angle(normalize_to_unit_circle(x))
        C, L, B = x.shape
        assert C == self.hidden_dims, (
            f"SSMReadout input channels {C} != hidden_dims {self.hidden_dims}"
        )

        t0 = _readout_t0(L, self.readout_frac)
        phases = x[:, t0:L, :]                              # (C, W, B)
        W = L - t0

        n_cls = self.n_classes
        # p: (C, 1, W, B);  c: (C, n_cls, 1, 1)
        p = phases.reshape(C, 1, W, B)
        c = self.codes.reshape(C, n_cls, 1, 1)
        cos_diff = torch.cos(PI * (p - c))                  # (C, n_cls, W, B)
        sims_per_step = cos_diff.mean(dim=0)                # (n_cls, W, B)
        sims_avg = sims_per_step.mean(dim=1)                # (n_cls, B)
        return sims_avg

    def parameter_dict(self) -> dict[str, Tensor]:
        return {"codes": self.codes.detach().cpu()}
