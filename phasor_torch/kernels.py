"""Discrete phasor SSM kernels and causal convolution.

Ports of:
  phasor_kernel             — src/kernels.jl:70
  causal_conv (hybrid)      — src/kernels.jl:139
  _causal_conv_toeplitz     — src/kernels.jl:157
  causal_conv_fft           — src/kernels.jl:196
  causal_conv_dirac         — src/kernels.jl:363
  hippo_legs_diagonal       — src/kernels.jl:442

The math: discretize the R&F ODE `dz/dt = k*z + I(t)` (k = lambda + i*omega)
into the recurrence `z[n+1] = A*z[n] + B*I[n]` with A = exp(k*dt), B = (A-1)/k.
Unrolling gives causal convolution z[n] = sum_j K[n-j] * I[j] where
K[n] = A^n * B — the kernel `phasor_kernel` returns.

The grouped-loop in causal_conv_dirac mirrors the Julia version (group_size=8).
We do NOT port the custom _exp_kdt rrule — PyTorch's autograd tape does not
suffer the Zygote Dual-lift problem, so the 3x memory cost the Julia rrule
dodges does not exist here. If memory profiling on the attention layers
demands it later, write a torch.autograd.Function with the closed-form
formula in src/kernels.jl:295.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# --------------------------------------------------------------------------
# 1. Discrete phasor kernel
# --------------------------------------------------------------------------


def _make_k(lam: Tensor, omega: Tensor) -> Tensor:
    """Build complex eigenvalue k = lambda + i*omega from real vectors."""
    return torch.complex(lam.float(), omega.float())


def phasor_kernel(lam: Tensor, omega: Tensor, dt: float, L: int) -> Tensor:
    """Causal impulse-response kernel for C damped oscillators.

    K[c, n] = A_c^n * B_c, where A_c = exp(k_c * dt), B_c = (A_c - 1) / k_c.

    Args:
      lam:   (C,) real, decay rates (must be negative for stability)
      omega: (C,) real, angular frequencies (rad/step)
      dt:    scalar time step
      L:     sequence length

    Returns:
      (C, L) complex64 kernel.
    """
    assert lam.ndim == 1 and omega.ndim == 1 and lam.shape == omega.shape
    k = _make_k(lam, omega)                                     # (C,)
    dt_f = float(dt)
    ns = torch.arange(L, dtype=torch.float32, device=k.device)  # (L,)
    # A_powers[c, n] = exp(k_c * dt * n)
    A_powers = torch.exp(k.unsqueeze(1) * dt_f * ns.unsqueeze(0))  # (C, L)
    A = torch.exp(k * dt_f)                                        # (C,)
    B_gain = ((A - 1.0) / k).unsqueeze(1)                          # (C, 1)
    return A_powers * B_gain                                       # (C, L)


def bias_kernel_accumulation(lam: Tensor, omega: Tensor,
                             T: float, L: int) -> Tensor:
    """Per-period constant-drive accumulation factor.

    G[c, m] = (1 - exp(k_c * (m+1) * T)) / (1 - exp(k_c * T))

    Mirrors Julia src/kernels.jl:99.
    """
    k = _make_k(lam, omega)                                       # (C,)
    T_f = float(T)
    A = torch.exp(k * T_f)                                        # (C,)
    ns = torch.arange(1, L + 1, dtype=torch.float32, device=k.device)  # (L,)
    A_pow = torch.exp(k.unsqueeze(1) * T_f * ns.unsqueeze(0))     # (C, L)
    return (1.0 - A_pow) / (1.0 - A).unsqueeze(1)


# --------------------------------------------------------------------------
# 2. Causal convolution (hybrid: Toeplitz for short, FFT for long)
# --------------------------------------------------------------------------


def causal_conv(K: Tensor, H: Tensor) -> Tensor:
    """Apply impulse-response kernel to a batch of input signals.

    Z[c, n, b] = sum_{j=0..n} K[c, n-j] * H[c, j, b].

    Selects FFT for L > 64, Toeplitz otherwise (matches Julia heuristic).

    Args:
      K: (C, L) complex
      H: (C, L, B) complex

    Returns:
      Z: (C, L, B) complex.
    """
    assert K.is_complex() and H.is_complex(), "complex inputs required"
    assert K.ndim == 2 and H.ndim == 3
    assert K.shape[0] == H.shape[0], f"channel mismatch K[{K.shape}] H[{H.shape}]"
    assert K.shape[1] == H.shape[1], f"length mismatch K[{K.shape}] H[{H.shape}]"
    L = H.shape[1]
    if L > 64:
        return causal_conv_fft(K, H)
    return _causal_conv_toeplitz(K, H)


def _causal_conv_toeplitz(K: Tensor, H: Tensor) -> Tensor:
    """Lower-triangular Toeplitz matmul. O(C * L^2 * B)."""
    C, L = K.shape
    _, _, B = H.shape

    # Pad kernel with one zero column so out-of-bounds indices map to 0.
    zero_col = torch.zeros(C, 1, dtype=K.dtype, device=K.device)
    K_pad = torch.cat([K, zero_col], dim=1)                          # (C, L+1)

    # Lower-triangular index matrix T[i, j] = i - j when i >= j else L (zero).
    # Build on K's device so the gather stays local.
    i_idx = torch.arange(L, device=K.device).unsqueeze(1)            # (L, 1)
    j_idx = torch.arange(L, device=K.device).unsqueeze(0)            # (1, L)
    diff = i_idx - j_idx                                              # (L, L)
    idx = torch.where(diff >= 0, diff, torch.full_like(diff, L))     # (L, L)

    # Gather to (C, L, L): T[c, i, j] = K_pad[c, idx[i, j]]
    T = K_pad[:, idx]                                                # (C, L, L)

    # bmm-friendly layout: (batch=C, L, L) @ (batch=C, L, B) -> (C, L, B)
    return torch.bmm(T, H)                                            # (C, L, B)


def causal_conv_fft(K: Tensor, H: Tensor) -> Tensor:
    """FFT-based causal convolution. O(C * L * log L * B)."""
    C, L = K.shape
    _, _, B = H.shape
    N = 2 * L

    # Zero-pad to 2L on the time axis (non-mutating, AD-friendly).
    K_pad = torch.cat([K, torch.zeros_like(K)], dim=1)                # (C, N)
    H_pad = torch.cat([H, torch.zeros_like(H)], dim=1)                # (C, N, B)

    K_f = torch.fft.fft(K_pad, dim=1)                                  # (C, N)
    H_f = torch.fft.fft(H_pad, dim=1)                                  # (C, N, B)

    Z_f = K_f.unsqueeze(2) * H_f                                       # (C, N, B)
    Z_full = torch.fft.ifft(Z_f, dim=1)                                # (C, N, B)
    Z = Z_full[:, :L, :]                                               # (C, L, B)

    # Make sure we don't accidentally hand back a complex128 from FFT promotion.
    if Z.dtype != torch.complex64:
        Z = Z.to(torch.complex64)
    return Z


# --------------------------------------------------------------------------
# 3. Dirac discretization (phase inputs)
# --------------------------------------------------------------------------


def dirac_encode(phases: Tensor, lam: Tensor, omega: Tensor, T: float) -> Tensor:
    """Per-channel Dirac response to phase-coded spikes.

    Each spike at phase theta arrives at time t_s = (theta/2 + 0.5)*T;
    the oscillator with eigenvalue k_c responds as exp(k_c * dt) where
    dt = T*(0.5 - theta/2) is the time remaining until the next sample.

    Args:
      phases: (C_in, L, B) real, phases in [-1, 1]
      lam:    (C_out,) real, per-output decay
      omega:  (C_out,) real, per-output angular frequency
      T:      scalar oscillation period

    Returns:
      (C_out, C_in, L, B) complex Dirac response.
    """
    k_c = _make_k(lam, omega)                                          # (C_out,)
    T_f = float(T)
    dt = T_f * (0.5 - phases * 0.5)                                    # (C_in, L, B)
    k_r = k_c.reshape(-1, 1, 1, 1)                                     # (C_out,1,1,1)
    dt_r = dt.unsqueeze(0).to(torch.complex64)                         # (1,C_in,L,B) complex
    return torch.exp(k_r * dt_r)                                       # (C_out,C_in,L,B)


def causal_conv_dirac(phases: Tensor, W: Tensor,
                      lam: Tensor, omega: Tensor, T: float,
                      group_size: int = 8) -> Tensor:
    """Causal convolution of phase-coded inputs (the 3D Phase forward path).

    Computes z_c[m] = sum_n K_c[m-n] * H_c[n], where
      K_c[n] = exp(k_c * n * T)
      H_c[n] = sum_j W[c, j] * exp(k_c * dt_j[n])
      dt_j[n] = T * (0.5 - phases[j, n] / 2)

    The dt expansion is factored from the K accumulation so the conv runs
    only on the (C_out, L, B) tensor; the per-input expansion stays in a
    grouped loop over output channels (group_size=8 by default) to bound
    peak memory.

    Args:
      phases:    (C_in, L, B) real, phases in [-1, 1]
      W:         (C_out, C_in) real, weight matrix
      lam:       (C_out,) real, per-output decay
      omega:     (C_out,) real, per-output angular frequency
      T:         scalar oscillation period
      group_size: number of output channels processed per inner iteration

    Returns:
      Z: (C_out, L, B) complex.
    """
    C_in, L, B = phases.shape
    C_out = lam.shape[0]
    assert W.shape == (C_out, C_in), f"W shape {W.shape} != ({C_out}, {C_in})"
    assert omega.shape == (C_out,)

    k_c = _make_k(lam, omega)                                          # (C_out,)
    T_f = float(T)

    # dt[j, n, b] = T * (0.5 - phases[j, n, b] / 2)
    dt = T_f * (0.5 - phases * 0.5)                                    # (C_in, L, B) real
    dt_flat = dt.reshape(1, C_in, L * B).to(torch.complex64)           # (1, C_in, L*B) complex

    W_c = W.to(torch.complex64)                                        # (C_out, C_in) complex
    G = min(int(group_size), C_out)

    H_groups = []
    for c_start in range(0, C_out, G):
        c_end = min(c_start + G, C_out)
        k_group = k_c[c_start:c_end].reshape(-1, 1, 1)                 # (g, 1, 1)
        enc = torch.exp(k_group * dt_flat)                              # (g, C_in, L*B)
        w_group = W_c[c_start:c_end, :].reshape(-1, C_in, 1)            # (g, C_in, 1)
        h = (w_group * enc).sum(dim=1)                                  # (g, L*B)
        H_groups.append(h.reshape(-1, L, B))                            # (g, L, B)
    H = torch.cat(H_groups, dim=0)                                      # (C_out, L, B)

    # Plain causal kernel: K_c[n] = exp(k_c * n * T)
    ns = torch.arange(L, dtype=torch.float32, device=k_c.device)
    K = torch.exp(k_c.unsqueeze(1) * T_f * ns.unsqueeze(0))             # (C_out, L)

    return causal_conv(K, H)


# --------------------------------------------------------------------------
# 4. HiPPO-LegS diagonal init
# --------------------------------------------------------------------------


def hippo_legs_diagonal(N: int, clip_decay: float | None = None
                        ) -> tuple[Tensor, Tensor]:
    """HiPPO-LegS diagonal initialization (the :hippo init mode).

    Returns (lam, omega) as length-N float32 vectors. Callers map lam
    to the log parameterization with log_neg_lambda = log(-lam).

    With clip_decay=None (default): log-spaced lam_mag from 0.5 to N-0.5
    across N channels. With clip_decay set: linear ns + 0.5 clipped above.

    Phase-locked layers (PhasorDense / PhasorConv / PhasorFixed /
    PhasorResonant) discard the omega vector and use a single shared
    omega = 2*pi to maintain HD-VSA carrier consistency. Only ResonantSTFT
    keeps the per-channel omega from this init.

    Mirrors Julia src/kernels.jl:442.
    """
    if clip_decay is None:
        log_lo = math.log(0.5)
        log_hi = math.log(N - 0.5)
        lam_mag = torch.exp(torch.linspace(log_lo, log_hi, N, dtype=torch.float32))
    else:
        ns = torch.arange(N, dtype=torch.float32)
        lam_mag = torch.clamp(ns + 0.5, max=float(clip_decay))
    lam = -lam_mag
    omega = math.pi * lam_mag
    return lam, omega
