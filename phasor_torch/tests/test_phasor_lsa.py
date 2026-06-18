"""Smoke tests for PhasorLSA — shapes, gradients, save/load round trip."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.layers import PhasorLSA
from phasor_torch.layers.phasor_dense import SpikingArgs
from phasor_torch.primitives import normalize_to_unit_circle


def test_construction_default():
    layer = PhasorLSA(8, 16, n_heads=4)
    assert layer.in_dims == 8
    assert layer.d_model == 16
    assert layer.n_heads == 4
    assert layer.scale.shape == (1,)
    # All three projections are bias-free PhasorDense with d_model output.
    for sub in (layer.q_proj, layer.k_proj, layer.v_proj):
        assert sub.weight.shape == (16, 8)
        assert sub.use_bias is False


def test_d_model_must_divide_n_heads():
    with pytest.raises(ValueError):
        PhasorLSA(8, 15, n_heads=4)


def test_forward_3d_phase_shape_and_range():
    g = torch.Generator().manual_seed(0)
    layer = PhasorLSA(8, 16, n_heads=4, generator=g)
    x = (torch.rand(8, 12, 3) * 2 - 1)
    y = layer(x)
    assert y.shape == (16, 12, 3)
    assert y.dtype == torch.float32
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <= 1.0 + 1e-5


def test_forward_2d_phase_wrapping():
    """2D Phase input wraps to 3D with L=1 internally."""
    g = torch.Generator().manual_seed(1)
    layer = PhasorLSA(4, 8, n_heads=2, generator=g)
    x = (torch.rand(4, 5) * 2 - 1)
    y = layer(x)
    assert y.shape == (8, 5)


def test_forward_grads_finite():
    g = torch.Generator().manual_seed(2)
    layer = PhasorLSA(6, 12, n_heads=3, init_mode="hippo", generator=g)
    x = (torch.rand(6, 10, 4) * 2 - 1)
    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()
    for sub in (layer.q_proj, layer.k_proj, layer.v_proj):
        assert torch.isfinite(sub.weight.grad).all()
        assert torch.isfinite(sub.log_neg_lambda.grad).all()
    assert torch.isfinite(layer.scale.grad).all()


def test_parameter_dict_keys():
    layer = PhasorLSA(4, 8, n_heads=2)
    keys = set(layer.parameter_dict().keys())
    expected = {
        "q_proj/weight", "q_proj/log_neg_lambda",
        "k_proj/weight", "k_proj/log_neg_lambda",
        "v_proj/weight", "v_proj/log_neg_lambda",
        "scale",
    }
    assert keys == expected


def test_save_load_round_trip(tmp_path):
    from phasor_torch.weights import load_state, save_state

    src = PhasorLSA(6, 12, n_heads=3, init_mode="hippo",
                    generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        src.scale.fill_(2.5)

    path = tmp_path / "lsa.h5"
    save_state(path, {"attn": src})

    dst = PhasorLSA(6, 12, n_heads=3, init_mode="default",
                    generator=torch.Generator().manual_seed(4))
    load_state(path, {"attn": dst})

    for sub_name in ("q_proj", "k_proj", "v_proj"):
        src_sub = getattr(src, sub_name)
        dst_sub = getattr(dst, sub_name)
        assert torch.equal(src_sub.weight, dst_sub.weight), f"{sub_name}/weight mismatch"
        assert torch.equal(src_sub.log_neg_lambda, dst_sub.log_neg_lambda)
    assert torch.equal(src.scale, dst.scale)

    x = (torch.rand(6, 8, 2) * 2 - 1)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
