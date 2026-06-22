"""Tests for the ResonantSTFT support primitives.

Covers `soft_normalize_to_unit_circle` (trainable-SLERP gate) and
`freq_shift` (per-channel-omega -> shared-carrier re-encode). The reference
math is Julia src/activations.jl:208 and src/network.jl:868.
"""

import math

import torch

from phasor_torch.primitives import (
    freq_shift,
    remap_phase,
    soft_normalize_to_unit_circle,
    v_bind,
    v_bundle,
    v_unbind,
)


# --------------------------------------------------------------------------
# soft_normalize_to_unit_circle
# --------------------------------------------------------------------------


def _naive_soft_normalize(z, r_lo, r_hi):
    r = z.abs()
    mid = (r_lo + r_hi) / 2.0
    k = 6.0 / (r_hi - r_lo)
    blend = torch.sigmoid(k * (r - mid))
    out = torch.empty_like(z)
    for idx in range(z.numel()):
        zi = z.flatten()[idx]
        ri = max(abs(zi.item()), 1e-10)
        theta = math.atan2((zi / ri).imag.item(), (zi / ri).real.item())
        bi = blend.flatten()[idx].item()
        out.flatten()[idx] = complex(math.cos(bi * theta), math.sin(bi * theta))
    return out


def test_soft_normalize_value_matches_reference():
    g = torch.Generator().manual_seed(0)
    z = torch.complex(torch.randn(5, 4, 3, generator=g),
                      torch.randn(5, 4, 3, generator=g))
    y = soft_normalize_to_unit_circle(z, 0.1, 0.6)
    y_ref = _naive_soft_normalize(z, 0.1, 0.6)
    assert torch.allclose(y, y_ref, atol=1e-6)


def test_soft_normalize_output_is_unit_modulus():
    g = torch.Generator().manual_seed(1)
    z = torch.complex(torch.randn(8, 6, 2, generator=g),
                      torch.randn(8, 6, 2, generator=g)) * 3.0
    y = soft_normalize_to_unit_circle(z, 0.1, 0.6)
    assert torch.allclose(y.abs(), torch.ones_like(y.abs()), atol=1e-5)


def test_soft_normalize_subthreshold_collapses_to_one():
    # |z| << r_lo -> blend hits its floor sigmoid(-k*mid) ~ 0.015, so the
    # output stays within a small cap of 1 + 0i regardless of phase. (The
    # blend never reaches exactly 0 for finite mid; only z == 0 gives 1+0i
    # exactly. The cap is blend_floor * pi ~ 0.047.)
    z = 1e-3 * torch.exp(torch.complex(torch.zeros(7), torch.linspace(-3, 3, 7)))
    y = soft_normalize_to_unit_circle(z, 0.1, 0.6)
    assert (y.real > 0.99).all()
    assert (y.imag.abs() < 0.05).all()


def test_soft_normalize_suprathreshold_preserves_phase():
    # |z| >> r_hi -> blend ~ 1 -> output ~ z/|z|.
    angles = torch.linspace(-2.5, 2.5, 9)
    z = 10.0 * torch.exp(torch.complex(torch.zeros(9), angles))
    y = soft_normalize_to_unit_circle(z, 0.1, 0.6)
    y_angle = torch.atan2(y.imag, y.real)
    assert torch.allclose(y_angle, angles, atol=1e-3)


def test_soft_normalize_per_channel_thresholds_broadcast():
    z = torch.complex(torch.randn(4, 3, 2), torch.randn(4, 3, 2))
    r_lo = torch.tensor([0.05, 0.1, 0.2, 0.4]).reshape(4, 1, 1)
    r_hi = r_lo + 0.5
    y = soft_normalize_to_unit_circle(z, r_lo, r_hi)
    assert y.shape == z.shape
    assert torch.allclose(y.abs(), torch.ones_like(y.abs()), atol=1e-5)


def test_soft_normalize_grad_finite():
    g = torch.Generator().manual_seed(2)
    z = torch.complex(torch.randn(4, 3, 2, generator=g),
                      torch.randn(4, 3, 2, generator=g)).requires_grad_(True)
    r_lo = torch.full((4, 1, 1), 0.1, requires_grad=True)
    r_hi = r_lo.detach() + torch.full((4, 1, 1), 0.5, requires_grad=True)
    y = soft_normalize_to_unit_circle(z, r_lo, r_hi)
    (y.real.sum() + y.imag.sum()).backward()
    assert torch.isfinite(z.grad.real).all() and torch.isfinite(z.grad.imag).all()


def test_soft_normalize_grad_finite_at_zero():
    # Exact-zero input (silent clip): max(r, 1e-10) guard must keep grad finite.
    z = torch.zeros(3, 2, 2, dtype=torch.complex64, requires_grad=True)
    y = soft_normalize_to_unit_circle(z, 0.1, 0.6)
    (y.real.sum() + y.imag.sum()).backward()
    assert torch.isfinite(z.grad.real).all() and torch.isfinite(z.grad.imag).all()
    # Forward at z=0 collapses to 1 + 0i.
    assert torch.allclose(y.real, torch.ones_like(y.real), atol=1e-5)
    assert torch.allclose(y.imag, torch.zeros_like(y.imag), atol=1e-5)


# --------------------------------------------------------------------------
# freq_shift
# --------------------------------------------------------------------------


def test_freq_shift_zero_delta_is_identity():
    # If every channel already sits at omega_out, the shift is a no-op.
    L, B = 10, 3
    omega_out = 2.0 * math.pi
    omega = torch.full((4,), omega_out)
    Z = torch.complex(torch.randn(4, L, B), torch.randn(4, L, B))
    Y = freq_shift(Z, omega, omega_out, dt=1.0)
    assert torch.allclose(Y, Z, atol=1e-5)


def test_freq_shift_matches_formula():
    n_freqs, L, B = 3, 6, 2
    omega = torch.tensor([0.2, 1.0, 2.5])
    omega_out = 2.0 * math.pi
    dt = 0.7
    Z = torch.complex(torch.randn(n_freqs, L, B), torch.randn(n_freqs, L, B))
    Y = freq_shift(Z, omega, omega_out, dt)
    # Reference: Y[c,n,b] = Z[c,n,b] * exp(i*(omega_out-omega[c])*dt*n)
    d = omega_out - omega
    ns = torch.arange(L, dtype=torch.float32)
    phase = d.unsqueeze(1) * dt * ns.unsqueeze(0)
    shift = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    assert torch.allclose(Y, Z * shift.unsqueeze(-1), atol=1e-5)


def test_freq_shift_preserves_modulus():
    Z = torch.complex(torch.randn(4, 8, 2), torch.randn(4, 8, 2))
    omega = torch.linspace(0.2, 2.5, 4)
    Y = freq_shift(Z, omega, 2.0 * math.pi, dt=1.0)
    assert torch.allclose(Y.abs(), Z.abs(), atol=1e-5)


def test_freq_shift_grad_finite():
    Z = torch.complex(torch.randn(3, 5, 2), torch.randn(3, 5, 2)).requires_grad_(True)
    omega = torch.linspace(0.2, 2.5, 3, requires_grad=True)
    Y = freq_shift(Z, omega, 2.0 * math.pi, dt=1.0)
    (Y.real.sum() + Y.imag.sum()).backward()
    assert torch.isfinite(Z.grad.real).all() and torch.isfinite(Z.grad.imag).all()
    assert torch.isfinite(omega.grad).all()


# --------------------------------------------------------------------------
# VSA binding / bundling
# --------------------------------------------------------------------------


def test_v_bind_is_wrapped_phase_addition():
    x = torch.tensor([0.3, 0.8, -0.5, 0.9])
    y = torch.tensor([0.4, 0.5, -0.7, 0.6])
    out = v_bind(x, y)
    assert torch.allclose(out, remap_phase(x + y), atol=1e-6)
    assert (out >= -1.0).all() and (out <= 1.0).all()   # stays in phase range
    # 0.8 + 0.5 = 1.3 -> wraps to -0.7
    assert abs(float(out[1]) - (-0.7)) < 1e-5


def test_v_unbind_inverts_v_bind():
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(5, 3, 2, generator=g) * 2 - 1)
    y = (torch.rand(5, 3, 2, generator=g) * 2 - 1)
    recovered = v_unbind(v_bind(x, y), y)               # bind then unbind by y
    # equal up to phase wrapping
    diff = remap_phase(recovered - x)
    assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-5)


def test_v_bind_identity_gradient_highway():
    # The residual point: d(v_bind(x,y))/dx includes the identity (straight-through
    # wrap), so a unit upstream grad reaches x as ~1 (the gradient highway).
    x = torch.randn(4, 3, 2, requires_grad=True)
    y = torch.randn(4, 3, 2, requires_grad=True)
    v_bind(x, y).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad), atol=1e-6)
    assert torch.allclose(y.grad, torch.ones_like(y.grad), atol=1e-6)


def test_v_bundle_matches_complex_mean_direction():
    # Two opposite-ish phases bundle toward their circular mean; output in range.
    x = torch.tensor([[0.25], [-0.25]])      # +45deg, -45deg -> mean 0
    out = v_bundle(x, dim=0)                  # (1,)
    assert out.shape == (1,)
    assert abs(float(out[0])) < 1e-5          # bundles to phase 0


def test_v_bundle_grad_finite():
    x = (torch.rand(6, 4, 2) * 2 - 1).requires_grad_(True)
    v_bundle(x, dim=0).abs().sum().backward()
    assert torch.isfinite(x.grad).all()
