"""Unit tests for primitives.py — round trips, NaN guards, gradcheck."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.primitives import (
    PI,
    _normalize_safe,
    _NormalizeHard,
    _ComplexToAngle,
    _similarity_outer_canonical_complex,
    angle_to_complex,
    complex_to_angle,
    normalize_to_unit_circle,
    remap_phase,
    similarity,
    similarity_outer_heads,
)


# --------------------------------------------------------------------------
# angle_to_complex / complex_to_angle
# --------------------------------------------------------------------------


def test_angle_complex_round_trip():
    torch.manual_seed(0)
    x = (torch.rand(20, 30, dtype=torch.float32) * 2.0 - 1.0)  # in (-1, 1)
    z = angle_to_complex(x)
    assert z.dtype == torch.complex64
    # |z| == 1
    mag = z.abs()
    assert torch.allclose(mag, torch.ones_like(mag), atol=1e-6)
    y = complex_to_angle(z)
    assert torch.allclose(y, x, atol=1e-6)


def test_complex_to_angle_zero_input_safe():
    z = torch.zeros(5, dtype=torch.complex64)
    y = complex_to_angle(z)
    assert torch.equal(y, torch.zeros_like(y))


def test_complex_to_angle_zero_grad_no_nan():
    """Cotangent at z=0 must be exactly zero, not NaN."""
    z = torch.zeros(4, dtype=torch.complex64, requires_grad=True)
    y = complex_to_angle(z)
    loss = y.sum()
    loss.backward()
    assert torch.isfinite(z.grad.real).all()
    assert torch.isfinite(z.grad.imag).all()
    assert torch.equal(z.grad.real, torch.zeros_like(z.grad.real))
    assert torch.equal(z.grad.imag, torch.zeros_like(z.grad.imag))


def test_complex_to_angle_gradcheck_supra_threshold():
    """gradcheck on inputs well above threshold (complex128 for precision)."""
    torch.manual_seed(1)
    z = (torch.randn(3, 4, dtype=torch.complex128, requires_grad=True) + 0.5)
    assert torch.autograd.gradcheck(
        lambda zz: _ComplexToAngle.apply(zz, 1e-10),
        (z,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-5,
    )


# --------------------------------------------------------------------------
# remap_phase
# --------------------------------------------------------------------------


def test_remap_phase_wraps_to_minus_one_one():
    x = torch.tensor([-2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 3.0])
    y = remap_phase(x)
    assert (y >= -1.0).all()
    assert (y <= 1.0).all()
    # angle equivalence check: cos(pi*x) should match cos(pi*y)
    assert torch.allclose(torch.cos(PI * x), torch.cos(PI * y), atol=1e-6)


def test_remap_phase_idempotent_in_band():
    torch.manual_seed(2)
    x = torch.rand(50, dtype=torch.float32) * 2 - 1
    assert torch.allclose(remap_phase(x), x, atol=1e-6)


# --------------------------------------------------------------------------
# normalize_to_unit_circle
# --------------------------------------------------------------------------


def test_normalize_safe_smooth_at_zero():
    z = torch.zeros(3, dtype=torch.complex64)
    y = normalize_to_unit_circle(z, eps=1e-8)
    # safe branch: y == 0 at z == 0
    assert torch.equal(y.abs(), torch.zeros_like(y.abs()))


def test_normalize_hard_zero_fallback():
    z = torch.complex(
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 2.0]),
    )
    y = normalize_to_unit_circle(z, eps=0.0)
    # z=0 -> 1+0i ; |y| == 1 elsewhere
    assert torch.allclose(y[0], torch.complex(torch.tensor(1.0), torch.tensor(0.0)))
    assert torch.allclose(y[1].abs(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(y[2].abs(), torch.tensor(1.0), atol=1e-6)


def test_normalize_hard_zero_grad_no_nan():
    z = torch.zeros(3, dtype=torch.complex64, requires_grad=True)
    y = normalize_to_unit_circle(z, eps=0.0)
    loss = (y.real ** 2 + y.imag ** 2).sum()
    loss.backward()
    assert torch.isfinite(z.grad.real).all()
    assert torch.isfinite(z.grad.imag).all()
    assert torch.equal(z.grad.real, torch.zeros_like(z.grad.real))
    assert torch.equal(z.grad.imag, torch.zeros_like(z.grad.imag))


def test_normalize_hard_gradcheck_supra_threshold():
    torch.manual_seed(3)
    z = (torch.randn(2, 3, dtype=torch.complex128, requires_grad=True) + 1.5)
    assert torch.autograd.gradcheck(
        lambda zz: _NormalizeHard.apply(zz, 1e-10),
        (z,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-5,
    )


# --------------------------------------------------------------------------
# similarity (cosine of phase diff)
# --------------------------------------------------------------------------


def test_similarity_diagonal_is_one():
    torch.manual_seed(4)
    x = torch.rand(8, 5, dtype=torch.float32) * 2 - 1
    s = similarity(x, x, dim=0)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)


def test_similarity_antipodal_is_minus_one():
    x = torch.zeros(4, 3)
    y = torch.ones_like(x)            # phase 1 = pi -> antipodal to 0
    s = similarity(x, y, dim=0)
    assert torch.allclose(s, -torch.ones_like(s), atol=1e-6)


# --------------------------------------------------------------------------
# similarity_outer_heads (LSA + LCA layouts)
# --------------------------------------------------------------------------


def test_similarity_outer_heads_lsa_self_diagonal():
    """For LSA on identical (q == k), s[h, h, l, b] == 1 across all l, b."""
    torch.manual_seed(5)
    Dh, H, L, B = 6, 4, 7, 3
    q = (torch.rand(Dh, H, L, B) * 2 - 1)
    s = similarity_outer_heads(q, q)
    assert s.shape == (H, H, L, B)
    diag = torch.stack([s[h, h] for h in range(H)], dim=0)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_similarity_outer_heads_lca_shape_and_range():
    torch.manual_seed(6)
    Dh, H, A, L, B = 5, 3, 4, 6, 2
    q = (torch.rand(Dh, H, A) * 2 - 1)
    k = (torch.rand(Dh, H, L, B) * 2 - 1)
    s = similarity_outer_heads(q, k)
    assert s.shape == (A, H, L, B)
    assert s.min() >= -1.0 - 1e-5 and s.max() <= 1.0 + 1e-5


# --------------------------------------------------------------------------
# _similarity_outer_canonical_complex: gradcheck + memory shape
# --------------------------------------------------------------------------


def test_similarity_outer_canonical_complex_gradcheck():
    torch.manual_seed(7)
    D, M, N, X = 3, 4, 5, 2
    A = torch.randn(D, M, X, dtype=torch.complex128, requires_grad=True)
    B = torch.randn(D, N, X, dtype=torch.complex128, requires_grad=True)
    assert torch.autograd.gradcheck(
        _similarity_outer_canonical_complex,
        (A, B), eps=1e-6, atol=1e-5, rtol=1e-5,
    )


def test_similarity_outer_canonical_complex_matches_naive():
    """Closed-form must match the naive (D, M, N, X) broadcast formula."""
    torch.manual_seed(8)
    D, M, N, X = 4, 3, 5, 2
    A = torch.randn(D, M, X, dtype=torch.complex64)
    B = torch.randn(D, N, X, dtype=torch.complex64)

    # Naive: s[m, n, x] = mean_d (|A[d,m,x] + B[d,n,x]|^2 / 2 - 1)
    sums = A.unsqueeze(2) + B.unsqueeze(1)        # (D, M, N, X)
    naive = (sums.abs() ** 2 * 0.5 - 1.0).mean(dim=0)  # (M, N, X)

    got = _similarity_outer_canonical_complex(A, B)
    assert torch.allclose(got, naive, atol=1e-4, rtol=1e-4)
