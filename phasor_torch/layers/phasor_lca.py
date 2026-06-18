"""PhasorLCA — Local Cross-Attention (Hopfield-style anchor-based attention).

Port of Julia src/ssm.jl:584. Forward path on 3D Phase input:

  K = k_proj(x)                            # (D, L, B) Phase
  V = v_proj(x)
  Kh, Vh, Anchors_h = reshape into (Dh, H, *)            Dh = D / H
  scores  = similarity_outer_heads(Anchors_h, Kh)         # (A, H, L, B)
  weights = exp(scale * scores) / A                       # (A, H, L, B)
  Ac      = angle_to_complex(Anchors_h)                   # (Dh, H, A) complex
  Bundle  = _anchor_mix(Ac, weights)                      # (Dh, H, L, B) complex
  Vc      = angle_to_complex(Vh)                          # (Dh, H, L, B) complex
  Y       = Vc * Bundle                                   # element-wise (VSA bind)
  Y_phase = complex_to_angle(reshape(Y, (D, L, B)))
  return activation(Y_phase) if needed

Trainable parameters: k_proj weights, v_proj weights, anchors (d_model, n_anchors)
Phase, and a single scale Float32.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn

from ..primitives import (
    angle_to_complex,
    complex_to_angle,
    normalize_to_unit_circle,
    similarity_outer_heads,
)
from .phasor_dense import PhasorDense, SpikingArgs
from .phasor_lsa import _apply_phase_activation


def _anchor_mix(Ac: Tensor, w: Tensor) -> Tensor:
    """Bundle[d, h, l, b] = sum_a w[a, h, l, b] * Ac[d, h, a].

    Ac: (Dh, H, A) complex   — per-head anchor bank
    w:  (A,  H, L, B) real   — attention weights (broadcast as complex)
    Out: (Dh, H, L, B) complex.
    """
    return torch.einsum("dha,ahlb->dhlb", Ac, w.to(Ac.dtype))


class PhasorLCA(nn.Module):
    """Local Cross-Attention via anchor bank.

    Args:
      in_dims: input feature dim.
      d_model: output feature dim; must be divisible by n_heads.
      n_heads: number of attention heads.
      n_anchors: number of anchor patterns per head.
      activation: applied to the final Phase output (default normalize_to_unit_circle).
      init_scale: initial scale parameter (default 3.0).
      init_mode: PhasorDense init for k/v ('hippo' default).
      spk_args: SpikingArgs forwarded to k/v.
      generator: optional torch.Generator for deterministic init.
    """

    def __init__(
        self,
        in_dims: int,
        d_model: int,
        n_heads: int,
        n_anchors: int,
        activation: Optional[Callable[[Tensor], Tensor]] = normalize_to_unit_circle,
        *,
        init_scale: float = 3.0,
        init_mode: str = "hippo",
        spk_args: Optional[SpikingArgs] = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.in_dims = int(in_dims)
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_anchors = int(n_anchors)
        self.activation = activation
        spk = spk_args or SpikingArgs()
        kwargs = dict(use_bias=False, init_mode=init_mode, spk_args=spk)
        self.k_proj = PhasorDense(in_dims, d_model, generator=generator, **kwargs)
        self.v_proj = PhasorDense(in_dims, d_model, generator=generator, **kwargs)
        # Trainable anchor bank: (d_model, n_anchors) Phase, init uniform in [-1, 1].
        anchors_init = 2.0 * torch.rand(d_model, n_anchors, generator=generator) - 1.0
        self.anchors = nn.Parameter(anchors_init.float())
        self.scale = nn.Parameter(torch.tensor([float(init_scale)], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            x3 = x.unsqueeze(1)
            y3 = self.forward(x3)
            return y3.squeeze(1)
        if x.ndim != 3:
            raise TypeError(
                f"PhasorLCA: expected 2D or 3D Phase input; got ndim={x.ndim}."
            )
        if x.is_complex():
            y_phase = self.forward(complex_to_angle(x))
            return angle_to_complex(y_phase)

        K = self.k_proj(x)                                    # (D, L, B) Phase
        V = self.v_proj(x)

        D, L, B = K.shape
        H = self.n_heads
        Dh = self.d_model // H
        A = self.n_anchors

        # Same Julia col-major reshape convention as LSA (see PhasorLSA for the
        # rationale): head h owns the contiguous block [h*Dh : (h+1)*Dh] of D.
        Kh = K.reshape(H, Dh, L, B).transpose(0, 1).contiguous()
        Vh = V.reshape(H, Dh, L, B).transpose(0, 1).contiguous()
        Anchors_h = self.anchors.reshape(H, Dh, A).transpose(0, 1).contiguous()

        scores = similarity_outer_heads(Anchors_h, Kh)        # (A, H, L, B) real
        weights = torch.exp(self.scale * scores) / float(A)   # (A, H, L, B)

        Ac = angle_to_complex(Anchors_h)                       # (Dh, H, A) complex
        Bundle = _anchor_mix(Ac, weights)                      # (Dh, H, L, B) complex
        Vc = angle_to_complex(Vh)                              # (Dh, H, L, B) complex
        Y = Vc * Bundle                                        # element-wise VSA bind
        Y = Y.transpose(0, 1).contiguous().reshape(D, L, B)    # back to (D, L, B)
        Y_phase = complex_to_angle(Y)

        return _apply_phase_activation(self.activation, Y_phase)

    def parameter_dict(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for sub_name, sub_layer in (("k_proj", self.k_proj),
                                    ("v_proj", self.v_proj)):
            for k, v in sub_layer.parameter_dict().items():
                out[f"{sub_name}/{k}"] = v
        out["anchors"] = self.anchors.detach().cpu()
        out["scale"] = self.scale.detach().cpu()
        return out
