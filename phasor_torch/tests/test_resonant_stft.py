"""Tests for the ResonantSTFT audio frontend layer.

Shape/dtype, grad-finiteness for every trainable parameter (including the
per-channel trainable omega), parameter_dict keys, and HDF5 round trip.
The canonical correctness gate is julia_parity/verify_resonant_stft.jl.
"""

import math
from pathlib import Path

import pytest
import torch

from phasor_torch.layers import ResonantSTFT, resolve_activation
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.primitives import normalize_to_unit_circle
from phasor_torch.weights import load_state, save_state


def _complex_input(in_dims, L, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(in_dims, L, B, generator=g),
                         torch.randn(in_dims, L, B, generator=g))


def test_construction_slerp_defaults():
    layer = ResonantSTFT(1, 8)
    assert layer.weight.shape == (8, 1)
    assert layer.omega.shape == (8,)
    assert isinstance(layer.omega, torch.nn.Parameter)  # trainable, not a buffer
    assert layer.log_r_lo is not None and layer.log_r_gap is not None
    assert layer.bias_real is None


def test_construction_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        ResonantSTFT(1, 4, init_r_lo=0.6, init_r_hi=0.1)


def test_omega_is_parameter_not_buffer():
    layer = ResonantSTFT(1, 16)
    names = dict(layer.named_parameters())
    assert "omega" in names
    assert "omega" not in dict(layer.named_buffers())


@pytest.mark.parametrize("L", [16, 128])  # Toeplitz and FFT branches
def test_forward_shape_and_dtype(L):
    layer = ResonantSTFT(1, 12)
    x = _complex_input(1, L, 3)
    y = layer(x)
    assert y.shape == (12, L, 3)
    assert y.dtype == torch.complex64
    assert torch.isfinite(y.real).all() and torch.isfinite(y.imag).all()


def test_slerp_output_unit_modulus():
    layer = ResonantSTFT(1, 10)  # SLERP gate -> |y| == 1
    y = layer(_complex_input(1, 24, 2))
    assert torch.allclose(y.abs(), torch.ones_like(y.abs()), atol=1e-4)


def test_identity_and_normalize_activations():
    x = _complex_input(2, 20, 2)
    y_id = ResonantSTFT(2, 6, activation=resolve_activation("identity"))(x)
    y_nm = ResonantSTFT(2, 6, activation=resolve_activation("normalize"))(x)
    assert y_id.shape == (6, 20, 2) and y_nm.shape == (6, 20, 2)
    # normalize_to_unit_circle pushes magnitude toward 1.
    assert torch.allclose(y_nm.abs(), torch.ones_like(y_nm.abs()), atol=1e-3)


def test_rejects_real_input():
    layer = ResonantSTFT(1, 4)
    with pytest.raises(TypeError):
        layer(torch.randn(1, 16, 2))


@pytest.mark.parametrize("use_bias", [False, True])
def test_grad_finite_all_params(use_bias):
    layer = ResonantSTFT(2, 8, use_bias=use_bias)
    x = _complex_input(2, 20, 3, seed=1)
    y = layer(x)
    (y.real.sum() + y.imag.sum()).backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_parameter_dict_keys():
    slerp = ResonantSTFT(1, 4, use_bias=True)
    assert set(slerp.parameter_dict()) == {
        "weight", "log_neg_lambda", "omega", "bias_real", "bias_imag",
        "log_r_lo", "log_r_gap",
    }
    nm = ResonantSTFT(1, 4, activation=normalize_to_unit_circle, use_bias=False)
    assert set(nm.parameter_dict()) == {"weight", "log_neg_lambda", "omega"}


def test_hdf5_round_trip(tmp_path: Path):
    spk = SpikingArgs(t_period=1.0)
    g = torch.Generator().manual_seed(7)
    src = ResonantSTFT(2, 8, use_bias=True, spk_args=spk, generator=g)
    # Perturb so params differ from a fresh init.
    with torch.no_grad():
        src.omega.add_(0.05)
        src.log_r_gap.add_(0.1)

    path = tmp_path / "rstft.h5"
    save_state(path, {"stft": src})

    g2 = torch.Generator().manual_seed(123)
    dst = ResonantSTFT(2, 8, use_bias=True, spk_args=spk, generator=g2)
    load_state(path, {"stft": dst})

    for name, p in src.named_parameters():
        assert torch.allclose(p, dict(dst.named_parameters())[name], atol=1e-6), name

    x = _complex_input(2, 32, 3, seed=9)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
