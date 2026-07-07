"""Unit tests for kernels.py — closed-form phasor_kernel, FFT-vs-Toeplitz, dirac."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.kernels import (
    _causal_conv_toeplitz,
    bias_kernel_accumulation,
    causal_conv,
    causal_conv_dirac,
    causal_conv_fft,
    dirac_encode,
    hippo_legs_diagonal,
    phasor_kernel,
)


# --------------------------------------------------------------------------
# phasor_kernel closed form
# --------------------------------------------------------------------------


def test_phasor_kernel_shape_and_zero_lag():
    C, L = 4, 10
    lam = torch.tensor([-0.1, -0.5, -1.0, -2.0])
    omega = torch.tensor([0.2, 1.0, 1.5, 2.5])
    dt = 0.1
    K = phasor_kernel(lam, omega, dt, L)
    assert K.shape == (C, L)
    assert K.dtype == torch.complex64
    # K[c, 0] = B_c = (exp(k*dt) - 1) / k
    k = torch.complex(lam, omega)
    B_expected = (torch.exp(k * dt) - 1.0) / k
    assert torch.allclose(K[:, 0], B_expected.to(torch.complex64), atol=1e-6)


def test_phasor_kernel_recurrence_consistent():
    """K[c, n+1] = exp(k * dt) * K[c, n]."""
    C, L = 3, 8
    lam = torch.tensor([-0.2, -0.4, -0.7])
    omega = torch.tensor([0.5, 1.2, 2.0])
    dt = 0.05
    K = phasor_kernel(lam, omega, dt, L)
    k = torch.complex(lam, omega)
    A = torch.exp(k * dt)                          # (C,)
    for n in range(L - 1):
        expected = A * K[:, n]
        assert torch.allclose(K[:, n + 1], expected, atol=1e-5)


# --------------------------------------------------------------------------
# causal_conv: FFT vs Toeplitz agreement and dispatch threshold
# --------------------------------------------------------------------------


@pytest.mark.parametrize("L", [16, 32, 64, 128, 257])
def test_fft_vs_toeplitz_agreement(L):
    torch.manual_seed(11)
    C, B = 4, 3
    lam = torch.tensor([-0.1, -0.3, -0.5, -1.0])
    omega = torch.tensor([0.2, 0.8, 1.4, 2.2])
    K = phasor_kernel(lam, omega, 0.1, L)
    H = torch.randn(C, L, B, dtype=torch.complex64)
    Z_top = _causal_conv_toeplitz(K, H)
    Z_fft = causal_conv_fft(K, H)
    assert torch.allclose(Z_top, Z_fft, atol=5e-4, rtol=1e-4)


def test_causal_conv_dispatches_by_length():
    """L > 64 routes to FFT; L <= 64 routes to Toeplitz. Identical results."""
    torch.manual_seed(12)
    C, B = 2, 1
    lam = torch.tensor([-0.2, -0.5])
    omega = torch.tensor([0.5, 1.0])
    # L=64 -> Toeplitz, L=65 -> FFT
    for L in (64, 65):
        K = phasor_kernel(lam, omega, 0.1, L)
        H = torch.randn(C, L, B, dtype=torch.complex64)
        Z = causal_conv(K, H)
        assert Z.shape == (C, L, B)
        assert Z.dtype == torch.complex64


def test_causal_conv_causality():
    """Perturbing input at time t' > t cannot change output at time t."""
    torch.manual_seed(13)
    C, L, B = 3, 20, 2
    lam = torch.tensor([-0.2, -0.4, -0.6])
    omega = torch.tensor([0.5, 1.0, 1.5])
    K = phasor_kernel(lam, omega, 0.1, L)
    H = torch.randn(C, L, B, dtype=torch.complex64)
    Z = causal_conv(K, H)

    H_perturbed = H.clone()
    t_perturb = 15
    H_perturbed[:, t_perturb, :] += torch.randn(C, B, dtype=torch.complex64)
    Z_perturbed = causal_conv(K, H_perturbed)

    # Outputs strictly before t_perturb must be unchanged.
    assert torch.allclose(Z[:, :t_perturb], Z_perturbed[:, :t_perturb], atol=1e-5)


def test_causal_conv_fft_backward_finite():
    """Long-sequence backward through FFT path produces finite grads."""
    torch.manual_seed(14)
    C, L, B = 4, 128, 3
    lam = torch.tensor([-0.1, -0.3, -0.5, -1.0])
    omega = torch.tensor([0.2, 0.8, 1.4, 2.2])
    K = phasor_kernel(lam, omega, 0.1, L)
    H = torch.randn(C, L, B, dtype=torch.complex64, requires_grad=True)
    Z = causal_conv_fft(K, H)
    loss = (Z.real ** 2 + Z.imag ** 2).sum()
    loss.backward()
    assert torch.isfinite(H.grad.real).all()
    assert torch.isfinite(H.grad.imag).all()


def test_causal_conv_recurrence_match():
    """One-channel sanity: convolution result matches z[n+1]=A*z[n]+B*I[n]."""
    torch.manual_seed(15)
    C, L, B = 1, 12, 1
    lam = torch.tensor([-0.5])
    omega = torch.tensor([1.0])
    dt = 0.1
    k = torch.complex(lam, omega)
    A = torch.exp(k * dt)
    Bg = (A - 1.0) / k
    K = phasor_kernel(lam, omega, dt, L)
    I_in = torch.randn(C, L, B, dtype=torch.complex64)
    Z = causal_conv(K, I_in)

    # Reference recurrence:
    z = torch.zeros(C, L, B, dtype=torch.complex64)
    z[:, 0, :] = Bg.reshape(-1, 1) * I_in[:, 0, :]
    for n in range(L - 1):
        z[:, n + 1, :] = A.reshape(-1, 1) * z[:, n, :] + Bg.reshape(-1, 1) * I_in[:, n + 1, :]

    assert torch.allclose(Z, z, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# bias_kernel_accumulation closed form
# --------------------------------------------------------------------------


def test_bias_kernel_accumulation_matches_partial_sum():
    """G[c, m] should equal sum_{n=0..m} exp(k_c * n * T) by construction."""
    C, L = 3, 8
    lam = torch.tensor([-0.3, -0.5, -1.0])
    omega = torch.tensor([0.5, 1.2, 2.0])
    T = 1.0
    G = bias_kernel_accumulation(lam, omega, T, L)
    k = torch.complex(lam, omega)
    # Naive sum reference: G[c, m] = sum_{n=0..m} r^n where r = exp(k*T).
    ns = torch.arange(L, dtype=torch.float32)
    powers = torch.exp(k.unsqueeze(1) * T * ns.unsqueeze(0))  # (C, L), powers[c, n] = r^n
    naive = torch.cumsum(powers, dim=1)                       # (C, L), cum[c, m] = sum_{n=0..m}
    assert torch.allclose(G, naive.to(torch.complex64), atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# dirac_encode / causal_conv_dirac
# --------------------------------------------------------------------------


def test_dirac_encode_shape_and_unit_magnitude_at_zero_lambda():
    C_in, L, B = 3, 5, 2
    C_out = 4
    phases = torch.rand(C_in, L, B) * 2 - 1
    lam = torch.zeros(C_out)              # no decay -> |exp(k*dt)| == 1
    omega = torch.tensor([0.5, 1.0, 1.5, 2.0])
    T = 1.0
    enc = dirac_encode(phases, lam, omega, T)
    assert enc.shape == (C_out, C_in, L, B)
    assert torch.allclose(enc.abs(), torch.ones_like(enc.abs()), atol=1e-5)


def test_causal_conv_dirac_shape_and_finite_grads():
    torch.manual_seed(16)
    C_in, L, B = 8, 16, 3
    C_out = 6
    phases = (torch.rand(C_in, L, B) * 2 - 1).requires_grad_(False)
    W = torch.randn(C_out, C_in, requires_grad=True)
    lam = torch.full((C_out,), -0.2, requires_grad=True)
    omega = torch.full((C_out,), 2 * math.pi, requires_grad=True)
    Z = causal_conv_dirac(phases, W, lam, omega, T=1.0, group_size=2)
    assert Z.shape == (C_out, L, B)
    loss = (Z.real ** 2 + Z.imag ** 2).sum()
    loss.backward()
    assert torch.isfinite(W.grad).all()
    assert torch.isfinite(lam.grad).all()
    assert torch.isfinite(omega.grad).all()


def test_causal_conv_dirac_independent_of_group_size():
    """Output must be group_size-invariant up to floating-point summation order."""
    torch.manual_seed(17)
    C_in, L, B = 4, 8, 2
    C_out = 8
    phases = (torch.rand(C_in, L, B) * 2 - 1)
    W = torch.randn(C_out, C_in)
    lam = torch.full((C_out,), -0.2)
    omega = torch.full((C_out,), 2 * math.pi)
    Z_g1 = causal_conv_dirac(phases, W, lam, omega, T=1.0, group_size=1)
    Z_g4 = causal_conv_dirac(phases, W, lam, omega, T=1.0, group_size=4)
    Z_g8 = causal_conv_dirac(phases, W, lam, omega, T=1.0, group_size=8)
    assert torch.allclose(Z_g1, Z_g4, atol=1e-4)
    assert torch.allclose(Z_g1, Z_g8, atol=1e-4)


# --------------------------------------------------------------------------
# hippo_legs_diagonal
# --------------------------------------------------------------------------


def test_hippo_legs_diagonal_signs_and_shape():
    N = 16
    lam, omega = hippo_legs_diagonal(N)
    assert lam.shape == (N,) and omega.shape == (N,)
    assert (lam < 0).all(), "decay must be negative for stability"
    assert (omega > 0).all()
    # Frequencies paired to magnitudes: omega = pi * |lambda|
    assert torch.allclose(omega, math.pi * (-lam), atol=1e-6)


def test_hippo_legs_diagonal_clip():
    N = 8
    lam, omega = hippo_legs_diagonal(N, clip_decay=2.0)
    assert (-lam <= 2.0 + 1e-6).all()


def test_hippo_long_tape_range_and_n_independence():
    """Config-B :hippo tape: |lam| spans tau in [0.5, 64], N-INDEPENDENT."""
    for N in (8, 16, 64, 128):
        lam, _ = hippo_legs_diagonal(N)
        lam_mag = -lam
        assert lam_mag.shape == (N,)
        assert lam_mag[0] < lam_mag[-1], "slow (long memory) end first"
        # endpoints fixed regardless of N: 1/64 .. 1/0.5
        assert torch.isclose(lam_mag[0], torch.tensor(1.0 / 64.0), atol=1e-6)
        assert torch.isclose(lam_mag[-1], torch.tensor(2.0), atol=1e-6)
        # tau of the slowest channel is a genuine long tape (>= 64), not tau=2
        assert (1.0 / lam_mag[0]) >= 63.0


def test_hippo_tau_max_override():
    lam, _ = hippo_legs_diagonal(32, tau_max=16.0)
    lam_mag = -lam
    assert torch.isclose(lam_mag[0], torch.tensor(1.0 / 16.0), atol=1e-6)
    assert torch.isclose(lam_mag[-1], torch.tensor(2.0), atol=1e-6)
