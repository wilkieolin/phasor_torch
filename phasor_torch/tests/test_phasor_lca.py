"""Smoke tests for PhasorLCA — shapes, gradients, save/load round trip."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.layers import PhasorLCA
from phasor_torch.layers.phasor_dense import SpikingArgs


def test_construction_default():
    layer = PhasorLCA(8, 12, n_heads=3, n_anchors=4)
    assert layer.in_dims == 8
    assert layer.d_model == 12
    assert layer.n_heads == 3
    assert layer.n_anchors == 4
    assert layer.scale.shape == (1,)
    assert layer.anchors.shape == (12, 4)
    for sub in (layer.k_proj, layer.v_proj):
        assert sub.weight.shape == (12, 8)
        assert sub.use_bias is False


def test_d_model_must_divide_n_heads():
    with pytest.raises(ValueError):
        PhasorLCA(6, 10, n_heads=3, n_anchors=4)


def test_anchors_range():
    g = torch.Generator().manual_seed(0)
    layer = PhasorLCA(4, 8, n_heads=2, n_anchors=5, generator=g)
    assert layer.anchors.min() >= -1.0
    assert layer.anchors.max() <= 1.0


def test_forward_3d_phase_shape_and_range():
    g = torch.Generator().manual_seed(1)
    layer = PhasorLCA(8, 16, n_heads=4, n_anchors=6, generator=g)
    x = (torch.rand(8, 10, 3) * 2 - 1)
    y = layer(x)
    assert y.shape == (16, 10, 3)
    assert torch.isfinite(y).all()
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <= 1.0 + 1e-5


def test_forward_2d_phase_wrapping():
    g = torch.Generator().manual_seed(2)
    layer = PhasorLCA(4, 8, n_heads=2, n_anchors=3, generator=g)
    x = (torch.rand(4, 5) * 2 - 1)
    y = layer(x)
    assert y.shape == (8, 5)


def test_forward_grads_finite():
    g = torch.Generator().manual_seed(3)
    layer = PhasorLCA(6, 12, n_heads=3, n_anchors=4, init_mode="hippo", generator=g)
    x = (torch.rand(6, 8, 2) * 2 - 1)
    y = layer(x)
    loss = (y ** 2).sum()
    loss.backward()
    for sub in (layer.k_proj, layer.v_proj):
        assert torch.isfinite(sub.weight.grad).all()
        assert torch.isfinite(sub.log_neg_lambda.grad).all()
    assert torch.isfinite(layer.anchors.grad).all()
    assert torch.isfinite(layer.scale.grad).all()


def test_parameter_dict_keys():
    layer = PhasorLCA(4, 8, n_heads=2, n_anchors=3)
    keys = set(layer.parameter_dict().keys())
    expected = {
        "k_proj/weight", "k_proj/log_neg_lambda",
        "v_proj/weight", "v_proj/log_neg_lambda",
        "anchors",
        "scale",
    }
    assert keys == expected


def test_save_load_round_trip(tmp_path):
    from phasor_torch.weights import load_state, save_state

    src = PhasorLCA(6, 12, n_heads=3, n_anchors=5, init_mode="hippo",
                    generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        src.scale.fill_(1.7)

    path = tmp_path / "lca.h5"
    save_state(path, {"attn": src})

    dst = PhasorLCA(6, 12, n_heads=3, n_anchors=5, init_mode="default",
                    generator=torch.Generator().manual_seed(5))
    load_state(path, {"attn": dst})

    for sub_name in ("k_proj", "v_proj"):
        src_sub = getattr(src, sub_name)
        dst_sub = getattr(dst, sub_name)
        assert torch.equal(src_sub.weight, dst_sub.weight)
        assert torch.equal(src_sub.log_neg_lambda, dst_sub.log_neg_lambda)
    assert torch.equal(src.anchors, dst.anchors)
    assert torch.equal(src.scale, dst.scale)

    x = (torch.rand(6, 7, 2) * 2 - 1)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
