"""PhasorDense — Lux→PyTorch port of the 2D and 3D Phase forward paths.

Weight-compatible with Julia src/network.jl:288. Parameter layout:

  weight         (out, in)         float32   nn.Parameter
  log_neg_lambda (out,)            float32   nn.Parameter
  bias_real      (out,)            float32   nn.Parameter   (only if use_bias)
  bias_imag      (out,)            float32   nn.Parameter   (only if use_bias)

State (PyTorch buffers, derived from constructor config):

  omega          (out,)            float32   shared scalar 2*pi/t_period broadcast

Dispatched forward (`forward(x)` branches on input):

  x: real Tensor shape (C_in, B)         -> 2D Phase mode    -> (C_out, B) Phase
  x: real Tensor shape (C_in, L, B)      -> 3D Phase Dirac   -> (C_out, L, B) Phase
  x: complex Tensor shape (..., C_in, ?) -> 2D complex linear mode (matches Julia's
                                            (a::PhasorDense)(x::AbstractArray{<:Complex})
                                            — pre-activation, no normalization)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from ..kernels import bias_kernel_accumulation, causal_conv_dirac, hippo_legs_diagonal
from ..primitives import (
    PI,
    angle_to_complex,
    complex_to_angle,
    normalize_to_unit_circle,
)


@dataclass(frozen=True)
class SpikingArgs:
    """Subset of Julia's SpikingArgs needed for layer dynamics.

    Only the fields touched by the discrete (3D Phase) path are present —
    no solver, no spike kernel, no thresholds. The ODE path is out of
    scope for this port.
    """

    t_period: float = 1.0


def _default_log_neg_lambda(out_dims: int, init_mode: str,
                            hippo_tau_max: float | None = None) -> Tensor:
    """Match Julia's `_init_dynamics` for the discrete-path layers.

    `hippo_tau_max` sets the slowest (longest-memory) HiPPO timescale; None keeps
    the module default (kernels.HIPPO_TAU_MAX). Inert unless init_mode == 'hippo'.
    """
    if init_mode == "default":
        return torch.full((out_dims,), math.log(0.2), dtype=torch.float32)
    if init_mode == "hippo":
        lam, _omega = hippo_legs_diagonal(out_dims, tau_max=hippo_tau_max)
        return torch.log(-lam)
    raise ValueError(f"init_mode must be 'default' or 'hippo', got {init_mode!r}")


def _glorot_uniform(out_dims: int, in_dims: int, generator: torch.Generator | None
                    ) -> Tensor:
    """Lux-equivalent glorot_uniform init.

    Lux uses Float32 uniform in [-gain*scale, gain*scale] with
    gain = sqrt(6 / (fan_in + fan_out)) (default gain 1). Matches Julia's
    `glorot_uniform(rng, out, in)`.
    """
    scale = math.sqrt(6.0 / (in_dims + out_dims))
    w = torch.empty(out_dims, in_dims, dtype=torch.float32)
    with torch.no_grad():
        w.uniform_(-scale, scale, generator=generator)
    return w


class PhasorDense(nn.Module):
    """Phase-domain dense layer with shared-omega oscillator dynamics.

    Mirrors Julia src/network.jl:288. For the LSA/LCA training script we
    only exercise the 2D and 3D Phase forward paths; the SpikingCall /
    CurrentCall ODE paths are intentionally absent.

    Args:
      in_dims, out_dims: linear sizes.
      activation:  callable(complex Tensor) -> complex Tensor. Default
                   normalize_to_unit_circle. If the activation IS
                   normalize_to_unit_circle, the 3D path skips it (the
                   subsequent complex_to_angle is invariant to a positive
                   real scaling).
      use_bias: whether to learn complex bias as (bias_real, bias_imag).
      init_mode: 'default' (log(0.2) per channel) or 'hippo' (HiPPO-LegS).
      hippo_tau_max: longest (slowest) HiPPO timescale; None -> module default
                   (kernels.HIPPO_TAU_MAX). Only used when init_mode == 'hippo'.
      init_log_neg_lambda: optional per-channel override (float or 1-D tensor).
      init_weight_scale: post-glorot multiplier on `weight` only (bias
                   untouched). Mirrors Julia's branch_init_scale FFN lever.
                   Default 1.0 (no change).
      spk_args: oscillator config; only `t_period` is used (omega = 2pi/t_period).
      generator: optional torch.Generator for deterministic init.
    """

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        activation: Callable[[Tensor], Tensor] = normalize_to_unit_circle,
        *,
        use_bias: bool = True,
        init_mode: str = "default",
        hippo_tau_max: float | None = None,
        init_log_neg_lambda: Optional[float | Tensor] = None,
        init_weight_scale: float = 1.0,
        spk_args: Optional[SpikingArgs] = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        self.in_dims = int(in_dims)
        self.out_dims = int(out_dims)
        self.use_bias = bool(use_bias)
        self.activation = activation
        self.init_mode = init_mode
        self.spk_args = spk_args or SpikingArgs()

        # ---- parameters -----------------------------------------------
        weight = _glorot_uniform(self.out_dims, self.in_dims, generator)
        # Down-scale the glorot init toward a near-identity branch. Mirrors
        # Julia's `init_weight = branch_init_scale * glorot_uniform` (the FFN
        # branch lever in PhasorTransformerBlock, src/ssm.jl:1110). The bias is
        # NOT scaled — only the linear weight. Default 1.0 = no change.
        if init_weight_scale != 1.0:
            weight = weight * float(init_weight_scale)
        self.weight = nn.Parameter(weight)

        if init_log_neg_lambda is None:
            lnl = _default_log_neg_lambda(self.out_dims, init_mode, hippo_tau_max)
        elif isinstance(init_log_neg_lambda, (int, float)):
            lnl = torch.full((self.out_dims,), float(init_log_neg_lambda),
                             dtype=torch.float32)
        else:
            lnl = torch.as_tensor(init_log_neg_lambda, dtype=torch.float32).clone()
            assert lnl.shape == (self.out_dims,), \
                f"init_log_neg_lambda must have shape ({self.out_dims},), got {lnl.shape}"
        self.log_neg_lambda = nn.Parameter(lnl)

        if self.use_bias:
            # Default bias is `default_bias` in Julia: complex ones (1+0i),
            # split into real and imag. Initialize to match.
            self.bias_real = nn.Parameter(torch.ones(self.out_dims, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.out_dims, dtype=torch.float32))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        # ---- buffers (derived, not trained) ---------------------------
        # Shared per-layer carrier; broadcast over the (out_dims,) channel axis.
        omega_val = 2.0 * math.pi / float(self.spk_args.t_period)
        self.register_buffer(
            "omega", torch.full((self.out_dims,), omega_val, dtype=torch.float32)
        )

    # ----------------------------------------------------------------------
    # Forward dispatch
    # ----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if x.is_complex():
            # 2D complex: W*x + bias, no activation. Mirrors src/network.jl:385.
            return self._forward_complex_linear(x)
        if x.ndim == 2:
            return self._forward_phase_2d(x)
        if x.ndim == 3:
            return self._forward_phase_3d(x)
        raise TypeError(
            f"PhasorDense: unsupported input dtype={x.dtype} ndim={x.ndim}; "
            "expected real Phase tensor (C_in, B) or (C_in, L, B), or "
            "complex Tensor for the linear-mode dispatch."
        )

    # ----------------------------------------------------------------------
    # 2D complex linear mode
    # ----------------------------------------------------------------------

    def _forward_complex_linear(self, x: Tensor) -> Tensor:
        # x: (C_in, B) or any leading dims; matmul on the channel axis.
        # Julia: y_real = W*real(x); y_imag = W*imag(x); y = y_real + i*y_imag; y += bias.
        # In Python: W @ x.real is equivalent for the (C_in, B) layout.
        W = self.weight
        y_real = W @ x.real
        y_imag = W @ x.imag
        y = torch.complex(y_real, y_imag)
        if self.use_bias:
            bias_c = torch.complex(self.bias_real, self.bias_imag)
            # Broadcast bias on the channel axis; same layout as Julia (out,) onto (out, B).
            shape = (-1,) + (1,) * (y.ndim - 1)
            y = y + bias_c.reshape(shape)
        return y

    # ----------------------------------------------------------------------
    # 2D Phase: angle_to_complex -> linear -> activation -> complex_to_angle
    # ----------------------------------------------------------------------

    def _forward_phase_2d(self, x: Tensor) -> Tensor:
        xz = angle_to_complex(x)
        y = self._forward_complex_linear(xz)
        y_normalized = self.activation(y)
        return complex_to_angle(y_normalized)

    # ----------------------------------------------------------------------
    # 3D Phase Dirac path (the LSA/LCA workhorse).
    # ----------------------------------------------------------------------

    def _forward_phase_3d(self, x: Tensor) -> Tensor:
        lam = -torch.exp(self.log_neg_lambda)              # (out,)
        omega = self.omega                                  # (out,) shared 2*pi/T
        T = float(self.spk_args.t_period)
        L = x.shape[1]

        Z = causal_conv_dirac(x, self.weight, lam, omega, T)   # (out, L, B) complex

        if self.use_bias:
            bias_c = torch.complex(self.bias_real, self.bias_imag)      # (out,) complex
            bphase = torch.atan2(bias_c.imag, bias_c.real) / PI         # (out,)
            dt_b = T * (0.5 - bphase * 0.5)                              # (out,)
            k_c = torch.complex(lam, omega)                              # (out,) complex
            b_eff = bias_c.abs() * torch.exp(k_c * dt_b.to(torch.complex64))  # (out,) complex
            G = bias_kernel_accumulation(lam, omega, T, L)               # (out, L) complex
            # Broadcast: b_eff (out,1,1) * G (out,L,1)
            Z = Z + b_eff.reshape(-1, 1, 1) * G.unsqueeze(-1)

        # Frame correction (network.jl:466): -conj(z). With omega*T = 2*pi
        # (which is true by construction since omega = 2*pi/T), this is the
        # unrotate_solution at the integer sample times.
        Z = -Z.conj()

        if self.activation is normalize_to_unit_circle:
            return complex_to_angle(Z)
        return complex_to_angle(self.activation(Z))

    # ----------------------------------------------------------------------
    # Serialization helpers (for HDF5 round trip)
    # ----------------------------------------------------------------------

    def parameter_dict(self) -> dict[str, Tensor]:
        """Return the trainable parameters in a flat name -> tensor dict.

        Names match the Lux NamedTuple keys so the HDF5 schema can be
        traversed identically on both sides.
        """
        out = {
            "weight": self.weight.detach().cpu(),
            "log_neg_lambda": self.log_neg_lambda.detach().cpu(),
        }
        if self.use_bias:
            out["bias_real"] = self.bias_real.detach().cpu()
            out["bias_imag"] = self.bias_imag.detach().cpu()
        return out
