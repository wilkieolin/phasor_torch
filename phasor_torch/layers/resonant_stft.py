"""ResonantSTFT — Lux->PyTorch port of the trainable resonant filterbank.

Weight-compatible with Julia src/network.jl:823. This is the audio frontend:
a bank of `n_freqs` damped oscillators that turns a raw (single-channel,
complex-lifted) waveform into a phase-coded time-frequency representation the
PhasorLSA/LCA body can consume.

Parameter layout (matches the Lux NamedTuple key names):

  weight         (n_freqs, in_dims)  float32   nn.Parameter
  log_neg_lambda (n_freqs,)          float32   nn.Parameter
  omega          (n_freqs,)          float32   nn.Parameter   <-- see note
  bias_real      (n_freqs,)          float32   nn.Parameter   (only if use_bias)
  bias_imag      (n_freqs,)          float32   nn.Parameter   (only if use_bias)
  log_r_lo       (n_freqs,)          float32   nn.Parameter   (only if SLERP)
  log_r_gap      (n_freqs,)          float32   nn.Parameter   (only if SLERP)

** Per-channel omega: the documented exception to the shared-omega rule. **
Every other layer in this port keeps omega as a shared scalar `register_buffer`
(2*pi/t_period) so layers stay phase-locked for VSA ops. ResonantSTFT instead
carries a *trainable, per-channel* omega as an `nn.Parameter` — it is a learned
filterbank. It re-encodes its output back onto the shared downstream carrier
`omega_out = 2*pi/t_period` via `freq_shift` (src/network.jl:868) so downstream
layers resume phase-locked operation. This mirrors the only Julia layer that
breaks the shared-omega rule; do not "fix" omega into a buffer here.

Only the 3D-Complex dispatch (src/network.jl:928) is ported — that is the audio
path (`encode_input` lifts the real waveform to complex). The 3D-Phase Dirac
dispatch (src/network.jl:973, with the -conj frame correction) is intentionally
omitted; add it only if a parity case requires it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from torch import Tensor, nn

from ..kernels import bias_kernel_accumulation, causal_conv, phasor_kernel
from ..primitives import (
    complex_to_angle,
    freq_shift,
    normalize_to_unit_circle,
    soft_normalize_to_unit_circle,
)
from .phasor_dense import SpikingArgs, _glorot_uniform


def _identity(z: Tensor) -> Tensor:
    return z


# --------------------------------------------------------------------------
# Stateless frontend transforms (the glue around ResonantSTFT in the audio
# chain: encode_input -> ResonantSTFT -> downsample_time -> to_phase).
# Mirror scripts/audio_pipeline.jl encode_input / downsample_time / to_phase.
# --------------------------------------------------------------------------


def encode_input(x: Tensor) -> Tensor:
    """Lift a real waveform to complex with zero imaginary part.

    Real `(C, L, B)` -> complex `(C, L, B)`. For raw audio C = 1. Mirrors the
    Julia `encode_input` helper that feeds ResonantSTFT's 3D-Complex dispatch.
    """
    if x.is_complex():
        return x
    return torch.complex(x, torch.zeros_like(x))


def downsample_time(z: Tensor, ds: int) -> Tensor:
    """Mean-pool the time axis by a factor `ds`.

    `(C, L, B)` -> `(C, L // ds, B)`. If `L` is not divisible by `ds`, the
    trailing `L % ds` samples are dropped before pooling. Works on real or
    complex tensors (the audio chain pools the complex ResonantSTFT output).
    """
    ds = int(ds)
    if ds <= 1:
        return z
    C, L, B = z.shape
    L2 = L // ds
    z = z[:, : L2 * ds, :]
    return z.reshape(C, L2, ds, B).mean(dim=2)


def to_phase(z: Tensor) -> Tensor:
    """Complex `(C, L, B)` -> Phase `(C, L, B)` (real, [-1, 1] units of pi).

    `normalize_to_unit_circle` then `complex_to_angle`, so the phase-dispatching
    body can consume the downsampled ResonantSTFT output.
    """
    return complex_to_angle(normalize_to_unit_circle(z))


def resolve_activation(name: str) -> Optional[Callable[[Tensor], Tensor]]:
    """Map a config string to the ResonantSTFT `activation` argument.

    'slerp'     -> None (the trainable soft-normalize gate; the default),
    'normalize' -> normalize_to_unit_circle,
    'identity'  -> identity (pass complex through unchanged).
    """
    if name == "slerp":
        return None
    if name == "normalize":
        return normalize_to_unit_circle
    if name == "identity":
        return _identity
    raise ValueError(
        f"resonant_activation must be 'slerp'|'normalize'|'identity', got {name!r}"
    )


class ResonantSTFT(nn.Module):
    """Trainable resonant filterbank frontend (3D-Complex Phase forward).

    Args:
      in_dims:   input channels (1 for a raw waveform lifted to complex).
      n_freqs:   number of resonant filter channels (output dim).
      activation: None for the trainable-SLERP gate (default), or a callable
                  complex->complex (e.g. normalize_to_unit_circle, identity).
                  When None, the layer learns per-channel SLERP thresholds.
      use_bias:  learn a constant complex current per channel (default False).
      omega_lo, omega_hi: endpoints of the initial per-channel omega ramp.
      init_log_neg_lambda: per-channel decay init; lambda = -exp(.) (default log(0.1)).
      init_r_lo, init_r_hi: initial SLERP magnitude gate (0 < r_lo < r_hi).
      spk_args:  oscillator config; only t_period is used (omega_out = 2*pi/t_period).
      generator: optional torch.Generator for deterministic init.
    """

    def __init__(
        self,
        in_dims: int,
        n_freqs: int,
        activation: Optional[Callable[[Tensor], Tensor]] = None,
        *,
        use_bias: bool = False,
        omega_lo: float = 0.2,
        omega_hi: float = 2.5,
        init_log_neg_lambda: float = math.log(0.1),
        init_r_lo: float = 0.1,
        init_r_hi: float = 0.6,
        spk_args: Optional[SpikingArgs] = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if not (init_r_hi > init_r_lo > 0):
            raise ValueError(
                f"ResonantSTFT requires 0 < init_r_lo < init_r_hi "
                f"(got {init_r_lo}, {init_r_hi})"
            )
        self.in_dims = int(in_dims)
        self.n_freqs = int(n_freqs)
        self.use_bias = bool(use_bias)
        self.activation = activation
        self.spk_args = spk_args or SpikingArgs()

        # ---- parameters -----------------------------------------------
        self.weight = nn.Parameter(_glorot_uniform(self.n_freqs, self.in_dims, generator))
        self.log_neg_lambda = nn.Parameter(
            torch.full((self.n_freqs,), float(init_log_neg_lambda), dtype=torch.float32)
        )
        # Trainable, per-channel omega (NOT a shared buffer — see module docstring).
        omega_init = torch.linspace(float(omega_lo), float(omega_hi), self.n_freqs,
                                    dtype=torch.float32)
        self.omega = nn.Parameter(omega_init)

        if self.use_bias:
            # Julia default_bias is complex ones (1 + 0i), split real/imag.
            self.bias_real = nn.Parameter(torch.ones(self.n_freqs, dtype=torch.float32))
            self.bias_imag = nn.Parameter(torch.zeros(self.n_freqs, dtype=torch.float32))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        if self.activation is None:
            # Trainable SLERP thresholds, positivity-preserving:
            #   r_lo = exp(log_r_lo), r_hi = r_lo + exp(log_r_gap).
            self.log_r_lo = nn.Parameter(
                torch.full((self.n_freqs,), math.log(init_r_lo), dtype=torch.float32)
            )
            self.log_r_gap = nn.Parameter(
                torch.full((self.n_freqs,), math.log(init_r_hi - init_r_lo),
                           dtype=torch.float32)
            )
        else:
            self.register_parameter("log_r_lo", None)
            self.register_parameter("log_r_gap", None)

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if x.is_complex() and x.ndim == 3:
            return self._forward_complex_3d(x)
        raise TypeError(
            f"ResonantSTFT: expected complex (C_in, L, B) input, got "
            f"dtype={x.dtype} ndim={x.ndim}. Lift real waveforms with "
            "encode_input first."
        )

    def _forward_complex_3d(self, x: Tensor) -> Tensor:
        """3D-Complex dispatch. Mirrors Julia src/network.jl:928."""
        C_in, L, B = x.shape
        lam = -torch.exp(self.log_neg_lambda)                 # (n_freqs,)
        omega = self.omega                                    # (n_freqs,)
        T = float(self.spk_args.t_period)
        omega_out = 2.0 * math.pi / T

        K = phasor_kernel(lam, omega, T, L)                   # (n_freqs, L)

        # Weight mixing: project input channels to frequency channels, real and
        # imag separately (matches Julia complex.(W*real(x), W*imag(x))).
        xr = x.reshape(C_in, L * B)
        Hr = torch.complex(self.weight @ xr.real, self.weight @ xr.imag)
        H = Hr.reshape(self.n_freqs, L, B)

        Z = causal_conv(K, H)                                 # (n_freqs, L, B)

        if self.use_bias:
            bias_c = torch.complex(self.bias_real, self.bias_imag)   # (n_freqs,)
            k_c = torch.complex(lam, omega)                          # (n_freqs,)
            B_gain = (torch.exp(k_c * T) - 1.0) / k_c                # ZOH input gain
            b_eff = B_gain * bias_c                                  # (n_freqs,)
            G = bias_kernel_accumulation(lam, omega, T, L)          # (n_freqs, L)
            Z = Z + b_eff.reshape(-1, 1, 1) * G.unsqueeze(-1)

        # Re-encode per-channel omega onto the shared downstream carrier.
        Z = freq_shift(Z, omega, omega_out, T)

        if self.activation is None:
            r_lo = torch.exp(self.log_r_lo).reshape(-1, 1, 1)
            r_hi = r_lo + torch.exp(self.log_r_gap).reshape(-1, 1, 1)
            return soft_normalize_to_unit_circle(Z, r_lo, r_hi)
        return self.activation(Z)

    # ----------------------------------------------------------------------
    # Serialization (HDF5 round trip / Lux parity)
    # ----------------------------------------------------------------------

    def parameter_dict(self) -> dict[str, Tensor]:
        """Flat name -> tensor dict matching the Lux NamedTuple keys.

        Note `omega` is a trainable parameter here (unlike PhasorDense, where it
        is a derived buffer), so it is serialized alongside the weights.
        """
        out = {
            "weight": self.weight.detach().cpu(),
            "log_neg_lambda": self.log_neg_lambda.detach().cpu(),
            "omega": self.omega.detach().cpu(),
        }
        if self.use_bias:
            out["bias_real"] = self.bias_real.detach().cpu()
            out["bias_imag"] = self.bias_imag.detach().cpu()
        if self.activation is None:
            out["log_r_lo"] = self.log_r_lo.detach().cpu()
            out["log_r_gap"] = self.log_r_gap.detach().cpu()
        return out
