"""Depth-robust phasor transformer blocks.

Port of the residual stack from Julia src/ssm.jl (`PhaseRecenter`,
`PhasorResidual`, `PhasorTransformerBlock`). These make stacked PhasorLSA /
PhasorLCA bodies trainable through many blocks: each attention sublayer is
wrapped in a residual with a ReZero gate (a learnable scalar alpha init ~= 0,
giving *exact identity at init*), which is what lets attention stack past
depth ~2 instead of collapsing.

Conventions follow the rest of the repo: Phase is a plain real tensor in
[-1, 1] (units of pi), feature-first layout `(C, L, B)` (or `(C, B)`), and the
channel axis is dim 0. The residual combine is `v_bind` (phase addition with a
detached / straight-through wrap), so the gradient is identity to both the
skip and the branch and a zero branch output passes the skip through unchanged.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn

from ..primitives import (
    angle_to_complex,
    complex_to_angle,
    normalize_to_unit_circle,
    v_bind,
)
from .phasor_dense import PhasorDense, SpikingArgs


class PhaseRecenter(nn.Module):
    """Parameter-free phase pre-norm: subtract the per-token circular mean.

    Mirrors Julia src/ssm.jl:991. For Phase input `x` of shape `(C, ...)`:

      z  = exp(i*pi*x)
      m  = angle(sum_c z) / pi          # circular mean angle over channel dim 0
      y  = v_bind(x, -m)                # = remap_phase(x - m)

    The circular mean is computed in the complex domain so it is well-defined
    under wraparound. Works on any rank with the channel axis at dim 0 (2D
    `(C, B)` or 3D `(C, L, B)`); the remaining axes broadcast.
    """

    def forward(self, x: Tensor) -> Tensor:
        z = angle_to_complex(x)
        m = complex_to_angle(z.sum(dim=0, keepdim=True))   # (1, ...) phase
        return v_bind(x, -m)

    def parameter_dict(self) -> dict[str, Tensor]:
        return {}


class PhasorResidual(nn.Module):
    """Identity-at-init residual wrapper around a shape-preserving phase layer.

    Mirrors Julia src/ssm.jl:1028. Computes `y = v_bind(x, g * sublayer(x))`
    where `g` is the gate:

      gate="none"   -> g = 1. Identity-at-init only if `sublayer` itself emits
                       ~0 output phase (e.g. a down-scaled PhasorDense branch).
      gate="rezero" -> g = alpha, a single learnable scalar (shape (1,)) init
                       to `alpha0`. With alpha0 = 0 the block is *exactly*
                       identity at init regardless of what `sublayer` computes
                       -- the right mechanism for attention sublayers.

    When `recenter` is given (a PhaseRecenter), it is applied to the branch
    input only (true pre-norm); the skip path `x` is untouched. This is
    mathematically identical to Julia wrapping `Chain(PhaseRecenter(), layer)`
    inside the residual, but keeps the parameter tree flat (recenter is
    parameter-free, so it never appears in `parameter_dict`).

    `sublayer` must map `(C, ...) -> (C, ...)` (in_dims == out_dims) for the
    skip to be well typed.
    """

    def __init__(
        self,
        sublayer: nn.Module,
        gate: str = "none",
        alpha0: float = 0.1,
        *,
        recenter: nn.Module | None = None,
    ):
        super().__init__()
        if gate not in ("none", "rezero"):
            raise ValueError(f"gate must be 'none' or 'rezero', got {gate!r}")
        self.gate = gate
        self.sublayer = sublayer
        self.recenter = recenter
        if gate == "rezero":
            self.alpha = nn.Parameter(torch.tensor([float(alpha0)], dtype=torch.float32))
        else:
            self.register_parameter("alpha", None)

    def forward(self, x: Tensor) -> Tensor:
        inp = self.recenter(x) if self.recenter is not None else x
        branch = self.sublayer(inp)
        if self.gate == "rezero":
            branch = self.alpha * branch
        return v_bind(x, branch)

    def parameter_dict(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for k, v in self.sublayer.parameter_dict().items():
            out[f"sublayer/{k}"] = v
        if self.gate == "rezero":
            out["alpha"] = self.alpha.detach().cpu()
        return out


class _PhasorFFN(nn.Module):
    """Two-layer PhasorDense MLP (`d_model -> d_ff -> d_model`).

    Mirrors the `ffn` Chain built inside Julia PhasorTransformerBlock
    (src/ssm.jl:1117). Both denses carry a bias and use the supplied phase
    activation; their weight init is down-scaled by `branch_init_scale` and
    their per-channel lambda init is set by `init_mode` (config B: 'hippo',
    the multi-timescale memory tape that belongs in the residual stream).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Callable[[Tensor], Tensor],
        *,
        branch_init_scale: float,
        spk_args: SpikingArgs,
        init_mode: str = "hippo",
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.fc1 = PhasorDense(
            d_model, d_ff, activation, use_bias=True, init_mode=init_mode,
            init_weight_scale=branch_init_scale, spk_args=spk_args,
            generator=generator,
        )
        self.fc2 = PhasorDense(
            d_ff, d_model, activation, use_bias=True, init_mode=init_mode,
            init_weight_scale=branch_init_scale, spk_args=spk_args,
            generator=generator,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x))

    def parameter_dict(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for sub_name, sub in (("fc1", self.fc1), ("fc2", self.fc2)):
            for k, v in sub.parameter_dict().items():
                out[f"{sub_name}/{k}"] = v
        return out


class PhasorTransformerBlock(nn.Module):
    """Pre-norm phasor transformer block: residual(attn) then residual(FFN).

    Mirrors Julia src/ssm.jl:1098. Forward:

        h = attn_res(x)        # v_bind(x, g * attn(recenter(x)))
        y = ffn_res(h)         # v_bind(h, g * ffn(recenter(h)))

    `attn` is any prebuilt shape-preserving `d_model -> d_model` phase
    attention layer (PhasorLSA / PhasorLCA), constructed by the caller. The FFN
    is a two-layer PhasorDense MLP whose weight init is down-scaled by
    `branch_init_scale`. The attention branch reaches identity-at-init only via
    the ReZero gate (`gate="rezero"`, `alpha0 -> 0`); down-scaling Q/K/V does
    not cleanly zero the attention output -- this is the load-bearing finding.

    Args:
      d_model: feature dim (must match attn in/out).
      attn:    prebuilt attention layer (PhasorLSA / PhasorLCA / any (C,..)->(C,..)).
      d_ff:    FFN hidden dim; <= 0 means d_ff = d_model.
      gate:    "none" | "rezero" (default "rezero").
      alpha0:  initial ReZero scalar (default 0.1).
      branch_init_scale: FFN weight-init down-scale (default 0.1; FFN-only).
      ffn_init_mode: per-channel lambda init of both FFN denses ("hippo" default,
                the multi-timescale memory tape that belongs in the residual
                stream; "default" = uniform lambda=-0.2). The attention
                projections' lambda init is set by the caller via the `attn`
                layer's own init_mode (config B: uniform read heads).
      recenter: prepend a PhaseRecenter (circular-mean pre-norm) to each
                residual branch, skip untouched (default False -- the recenter
                circular-mean complex_to_angle is a near-origin singularity /
                NaN source and is not helpful; see scripts/grad_diverge_probe.py).
      activation: phase activation for the FFN denses (default normalize).
      spk_args: oscillator config forwarded to the FFN denses.
      generator: optional torch.Generator for deterministic init.
    """

    def __init__(
        self,
        d_model: int,
        attn: nn.Module,
        *,
        d_ff: int = 0,
        gate: str = "rezero",
        alpha0: float = 0.1,
        branch_init_scale: float = 0.1,
        ffn_init_mode: str = "hippo",
        recenter: bool = False,
        activation: Callable[[Tensor], Tensor] = normalize_to_unit_circle,
        spk_args: Optional[SpikingArgs] = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        d_ff_eff = int(d_ff) if int(d_ff) > 0 else int(d_model)
        spk = spk_args or SpikingArgs()
        ffn = _PhasorFFN(
            d_model, d_ff_eff, activation,
            branch_init_scale=branch_init_scale, spk_args=spk,
            init_mode=ffn_init_mode, generator=generator,
        )
        attn_recenter = PhaseRecenter() if recenter else None
        ffn_recenter = PhaseRecenter() if recenter else None
        self.attn_res = PhasorResidual(attn, gate=gate, alpha0=alpha0,
                                       recenter=attn_recenter)
        self.ffn_res = PhasorResidual(ffn, gate=gate, alpha0=alpha0,
                                      recenter=ffn_recenter)

    def forward(self, x: Tensor) -> Tensor:
        return self.ffn_res(self.attn_res(x))

    def parameter_dict(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for sub_name, sub in (("attn_res", self.attn_res), ("ffn_res", self.ffn_res)):
            for k, v in sub.parameter_dict().items():
                out[f"{sub_name}/{k}"] = v
        return out
