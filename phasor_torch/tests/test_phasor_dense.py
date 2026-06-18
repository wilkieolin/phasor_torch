"""Smoke tests for PhasorDense — shapes, range, finite gradients."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.layers import PhasorDense
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.primitives import normalize_to_unit_circle


def _identity(x):
    return x


# --------------------------------------------------------------------------
# Construction and parameter shapes
# --------------------------------------------------------------------------


def test_construction_default():
    layer = PhasorDense(10, 5)
    assert layer.weight.shape == (5, 10)
    assert layer.log_neg_lambda.shape == (5,)
    assert layer.bias_real.shape == (5,)
    assert layer.bias_imag.shape == (5,)
    assert layer.omega.shape == (5,)
    assert torch.allclose(layer.omega, torch.full_like(layer.omega, 2 * math.pi))
    # default mode: log(0.2) everywhere
    assert torch.allclose(layer.log_neg_lambda,
                          torch.full_like(layer.log_neg_lambda, math.log(0.2)))


def test_construction_no_bias():
    layer = PhasorDense(8, 4, use_bias=False)
    assert layer.bias_real is None
    assert layer.bias_imag is None


def test_construction_hippo_init():
    layer = PhasorDense(6, 16, init_mode="hippo")
    # log_neg_lambda elements should span the log of [0.5, N-0.5]
    log_lam = layer.log_neg_lambda
    assert log_lam.shape == (16,)
    # First lambda magnitude is exp(log(0.5)) = 0.5
    assert torch.isclose(torch.exp(log_lam[0]), torch.tensor(0.5), atol=1e-5)
    # Last lambda magnitude is exp(log(15.5)) = 15.5
    assert torch.isclose(torch.exp(log_lam[-1]), torch.tensor(15.5), atol=1e-4)


def test_construction_explicit_log_neg_lambda():
    custom = torch.linspace(-1.0, 1.0, 6)
    layer = PhasorDense(3, 6, init_log_neg_lambda=custom)
    assert torch.allclose(layer.log_neg_lambda, custom)


# --------------------------------------------------------------------------
# 2D Phase forward
# --------------------------------------------------------------------------


def test_forward_2d_phase_shape_and_range():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    layer = PhasorDense(10, 5, generator=g)
    x = (torch.rand(10, 32) * 2 - 1)
    y = layer(x)
    assert y.shape == (5, 32)
    assert y.dtype == torch.float32
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <= 1.0 + 1e-5


def test_forward_2d_phase_grads_finite():
    """2D mode is a pure linear layer; log_neg_lambda has no path so its grad stays None."""
    torch.manual_seed(1)
    layer = PhasorDense(8, 4)
    x = (torch.rand(8, 16) * 2 - 1).requires_grad_(False)
    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.bias_real.grad).all()
    assert torch.isfinite(layer.bias_imag.grad).all()
    # log_neg_lambda only flows in 3D path; in 2D it's correctly None.
    assert layer.log_neg_lambda.grad is None


# --------------------------------------------------------------------------
# 3D Phase forward (the LSA/LCA workhorse)
# --------------------------------------------------------------------------


def test_forward_3d_phase_shape_and_range():
    torch.manual_seed(2)
    g = torch.Generator().manual_seed(2)
    layer = PhasorDense(6, 8, generator=g)
    x = (torch.rand(6, 12, 4) * 2 - 1)
    y = layer(x)
    assert y.shape == (8, 12, 4)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <= 1.0 + 1e-5


def test_forward_3d_phase_grads_finite():
    torch.manual_seed(3)
    layer = PhasorDense(6, 8, init_mode="hippo")
    x = (torch.rand(6, 12, 4) * 2 - 1).requires_grad_(False)
    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(layer.log_neg_lambda.grad).all()
    assert torch.isfinite(layer.bias_real.grad).all()
    assert torch.isfinite(layer.bias_imag.grad).all()


def test_forward_3d_phase_no_bias():
    """Mirrors the reference LSA topology (use_bias=False)."""
    torch.manual_seed(4)
    layer = PhasorDense(4, 4, activation=normalize_to_unit_circle,
                        use_bias=False, init_mode="hippo")
    x = (torch.rand(4, 10, 3) * 2 - 1)
    y = layer(x)
    assert y.shape == (4, 10, 3)
    assert torch.isfinite(y).all()


def test_forward_3d_phase_identity_activation():
    """identity activation must produce finite output (path used by body layer)."""
    torch.manual_seed(5)
    layer = PhasorDense(5, 5, activation=_identity, use_bias=False, init_mode="hippo")
    x = (torch.rand(5, 8, 2) * 2 - 1)
    y = layer(x)
    assert y.shape == (5, 8, 2)
    assert torch.isfinite(y).all()


def test_forward_3d_phase_omega_t_invariant():
    """omega*T must equal 2*pi by construction (parity-critical)."""
    layer = PhasorDense(2, 3, spk_args=SpikingArgs(t_period=0.5))
    expected = 2 * math.pi / 0.5
    assert torch.allclose(layer.omega, torch.full_like(layer.omega, expected))


# --------------------------------------------------------------------------
# parameter_dict round-trip (serialization)
# --------------------------------------------------------------------------


def test_parameter_dict_keys():
    layer = PhasorDense(3, 4, use_bias=True)
    keys = set(layer.parameter_dict().keys())
    assert keys == {"weight", "log_neg_lambda", "bias_real", "bias_imag"}

    layer_nb = PhasorDense(3, 4, use_bias=False)
    keys_nb = set(layer_nb.parameter_dict().keys())
    assert keys_nb == {"weight", "log_neg_lambda"}


def test_save_load_round_trip(tmp_path):
    from phasor_torch.weights import save_state, load_state

    torch.manual_seed(7)
    src = PhasorDense(6, 4, init_mode="hippo")
    # Set bias to non-default so the round trip is unambiguous.
    with torch.no_grad():
        src.bias_real.copy_(torch.linspace(-1, 1, 4))
        src.bias_imag.copy_(torch.linspace(0.5, -0.5, 4))

    path = tmp_path / "dense.h5"
    save_state(path, {"dense": src})

    dst = PhasorDense(6, 4, init_mode="default")  # different init on purpose
    load_state(path, {"dense": dst})

    for name in ("weight", "log_neg_lambda", "bias_real", "bias_imag"):
        a = getattr(src, name)
        b = getattr(dst, name)
        assert torch.equal(a, b), f"mismatch on '{name}' after load"

    # Forwards must now match exactly.
    x = (torch.rand(6, 8, 2) * 2 - 1)
    assert torch.allclose(src(x), dst(x), atol=1e-7)
