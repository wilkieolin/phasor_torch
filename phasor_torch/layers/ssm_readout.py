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
                 pool: str = "mean",
                 lse_kappa: float = 10.0,
                 learnable_codes: bool = False,
                 generator: torch.Generator | None = None):
        super().__init__()
        self.hidden_dims = int(hidden_dims)
        self.n_classes = int(n_classes)
        self.readout_frac = float(readout_frac)
        if pool not in ("mean", "logsumexp"):
            raise ValueError(f"pool must be 'mean' or 'logsumexp', got {pool!r}")
        self.pool = pool
        self.lse_kappa = float(lse_kappa)
        self.learnable_codes = bool(learnable_codes)
        codes = random_symbols((hidden_dims, n_classes), generator=generator)
        # Default (buffer) keeps forward-parity checkpoints unchanged. When
        # learnable_codes, the class prototypes co-adapt with the network.
        if learnable_codes:
            self.codes = nn.Parameter(codes)
        else:
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

        n_cls = self.n_classes
        if self.pool == "logsumexp":
            # "Is the class present at ANY timestep" — smooth max over the WHOLE
            # clip (the keyword-spotting inductive bias). Uses all L timesteps.
            p = x.reshape(C, 1, L, B)
            c = self.codes.reshape(C, n_cls, 1, 1)
            sims_per_step = torch.cos(PI * (p - c)).mean(dim=0)          # (n_cls, L, B)
            k = self.lse_kappa
            # (1/k)·(logsumexp_t(k·s) − log L) → max_t s as k→∞, mean_t s as k→0.
            return (torch.logsumexp(k * sims_per_step, dim=1) - math.log(L)) / k

        # default: mean over the last readout_frac window (parity-preserving path)
        t0 = _readout_t0(L, self.readout_frac)
        phases = x[:, t0:L, :]                              # (C, W, B)
        W = L - t0
        p = phases.reshape(C, 1, W, B)
        c = self.codes.reshape(C, n_cls, 1, 1)
        cos_diff = torch.cos(PI * (p - c))                  # (C, n_cls, W, B)
        sims_per_step = cos_diff.mean(dim=0)                # (n_cls, W, B)
        sims_avg = sims_per_step.mean(dim=1)                # (n_cls, B)
        return sims_avg

    def parameter_dict(self) -> dict[str, Tensor]:
        return {"codes": self.codes.detach().cpu()}
