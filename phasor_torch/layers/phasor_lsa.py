"""PhasorLSA — Local Self-Attention via head-axis similarity.

Port of Julia src/ssm.jl:415. Forward path on 3D Phase input:

  Q = q_proj(x)                            # (D, L, B) Phase
  K = k_proj(x)
  V = v_proj(x)
  Qh, Kh, Vh = reshape(Q | K | V, (Dh, H, L, B))      Dh = D / H
  scores  = similarity_outer_heads(Qh, Kh)            # (H, H, L, B) real
  weights = exp(scale * scores) / H                   # (H, H, L, B) real
  Vc      = angle_to_complex(Vh)                      # (Dh, H, L, B) complex
  Y[:, h, :, :] = sum_h' weights[h, h', :, :] * Vc[:, h', :, :]
  Y_phase = complex_to_angle(reshape(Y, (D, L, B)))
  return activation(Y_phase) if activation else Y_phase

Trainable parameters: q_proj.weight, q_proj.log_neg_lambda, k_proj.*, v_proj.*,
and a single `scale` Float32. All three projections are bias-free PhasorDense.
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


def _apply_phase_activation(activation: Callable, y: Tensor) -> Tensor:
    """Mirror Julia _apply_phase_activation (src/ssm.jl:368).

    For pass-through activations (identity, normalize_to_unit_circle) the
    Phase array is already on the unit circle so we skip the round trip.
    Otherwise lift to complex, apply, drop back to phase.
    """
    if activation is None:
        return y
    if activation is normalize_to_unit_circle:
        return y
    # `is` check on the built-in identity placeholder. We accept either
    # `torch.nn.Identity()` instance or a callable named `identity`.
    if isinstance(activation, nn.Identity):
        return y
    return complex_to_angle(activation(angle_to_complex(y)))


def _head_mix(Vc: Tensor, weights: Tensor) -> Tensor:
    """Y[:, h_q, l, b] = sum_h_k weights[h_q, h_k, l, b] * Vc[:, h_k, l, b].

    Vc:      (Dh, H, L, B) complex      — second axis is the V head index (h_k)
    weights: (H, H, L, B) real          — weights[h_q, h_k, l, b] from
                                          similarity_outer_heads; first axis
                                          is the query head (output), second
                                          is the key head (summed).
    Out:     (Dh, H, L, B) complex      — second axis is the output head (h_q).

    Equivalent to Julia src/ssm.jl:463 _lsa_head_mix, which does the same
    contraction via permutedims + batched_mul.
    """
    return torch.einsum("ehlb,ihlb->eilb",
                        Vc, weights.to(Vc.dtype))


class PhasorLSA(nn.Module):
    """Local Self-Attention layer.

    Args:
      in_dims:   input feature dim (C_in of the q/k/v PhasorDense projections).
      d_model:   output feature dim; must be divisible by n_heads.
      n_heads:   number of attention heads.
      activation: applied to the final Phase output. Defaults to
                  normalize_to_unit_circle (which is a pass-through on Phase).
      init_scale: initial value of the scale parameter (default 3.0).
      init_mode:  PhasorDense lambda init for q/k/v ('default' or 'hippo';
                  default 'default' = uniform read heads, config B).
      spk_args:   SpikingArgs forwarded to q/k/v.
      generator:  optional torch.Generator for deterministic init.
    """

    def __init__(
        self,
        in_dims: int,
        d_model: int,
        n_heads: int,
        activation: Optional[Callable[[Tensor], Tensor]] = normalize_to_unit_circle,
        *,
        init_scale: float = 3.0,
        init_mode: str = "default",
        hippo_tau_max: float | None = None,
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
        self.activation = activation
        spk = spk_args or SpikingArgs()
        kwargs = dict(use_bias=False, init_mode=init_mode,
                      hippo_tau_max=hippo_tau_max, spk_args=spk)
        self.q_proj = PhasorDense(in_dims, d_model, generator=generator, **kwargs)
        self.k_proj = PhasorDense(in_dims, d_model, generator=generator, **kwargs)
        self.v_proj = PhasorDense(in_dims, d_model, generator=generator, **kwargs)
        # Single-element scale (matches Julia Vector{Float32}[init_scale]).
        self.scale = nn.Parameter(torch.tensor([float(init_scale)], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            # 2D Phase: wrap to 3D with L=1, run the 3D path, drop the time axis.
            x3 = x.unsqueeze(1)
            y3 = self.forward(x3)
            return y3.squeeze(1)
        if x.ndim != 3:
            raise TypeError(
                f"PhasorLSA: expected 2D or 3D Phase input; got ndim={x.ndim}."
            )
        if x.is_complex():
            y_phase = self.forward(complex_to_angle(x))
            return angle_to_complex(y_phase)

        Q = self.q_proj(x)                                    # (D, L, B) Phase
        K = self.k_proj(x)
        V = self.v_proj(x)

        D, L, B = Q.shape
        H = self.n_heads
        Dh = self.d_model // H
        # Head split must match Julia's column-major reshape: head `h` owns
        # the contiguous channel block [h*Dh : (h+1)*Dh] of D, NOT every
        # H-th channel (which is what PyTorch's row-major
        # `reshape(Dh, H, L, B)` would produce). Do (H, Dh, ...) then swap.
        Qh = Q.reshape(H, Dh, L, B).transpose(0, 1).contiguous()
        Kh = K.reshape(H, Dh, L, B).transpose(0, 1).contiguous()
        Vh = V.reshape(H, Dh, L, B).transpose(0, 1).contiguous()

        scores = similarity_outer_heads(Qh, Kh)               # (H, H, L, B) real
        weights = torch.exp(self.scale * scores) / float(H)   # (H, H, L, B)

        Vc = angle_to_complex(Vh)                              # (Dh, H, L, B) complex
        Y = _head_mix(Vc, weights)                             # (Dh, H, L, B) complex
        # Inverse of the head split: transpose then flatten to (D, L, B).
        Y = Y.transpose(0, 1).contiguous().reshape(D, L, B)
        Y_phase = complex_to_angle(Y)                          # (D, L, B) Phase

        return _apply_phase_activation(self.activation, Y_phase)

    def parameter_dict(self) -> dict[str, Tensor]:
        """Flat dict for HDF5; nested layers are namespaced with slashes."""
        out: dict[str, Tensor] = {}
        for sub_name, sub_layer in (("q_proj", self.q_proj),
                                    ("k_proj", self.k_proj),
                                    ("v_proj", self.v_proj)):
            for k, v in sub_layer.parameter_dict().items():
                out[f"{sub_name}/{k}"] = v
        out["scale"] = self.scale.detach().cpu()
        return out
