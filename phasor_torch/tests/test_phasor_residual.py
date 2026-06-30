"""Tests for the phasor transformer block (PhaseRecenter / PhasorResidual /
PhasorTransformerBlock). Mirrors Julia test/test_transformer_block.jl.
"""

from __future__ import annotations

import pytest
import torch

from phasor_torch.layers import (
    PhaseRecenter,
    PhasorLCA,
    PhasorLSA,
    PhasorResidual,
    PhasorTransformerBlock,
)
from phasor_torch.primitives import angle_to_complex, complex_to_angle, remap_phase, v_bind
from phasor_torch.weights import load_state, save_state


def _phase(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(*shape, generator=g) * 2 - 1).float()


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


# --------------------------------------------------------------------------
# v_bind / remap_phase straight-through gradient
# --------------------------------------------------------------------------


def test_v_bind_identity_element():
    x = _phase(8, 5, 3)
    y = v_bind(x, torch.zeros_like(x))
    assert torch.allclose(y, x, atol=1e-6)


def test_remap_phase_straight_through_grad():
    x = _phase(8, 3).requires_grad_(True)
    remap_phase(x).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))


def test_v_bind_grad_to_both_operands():
    x = _phase(6, 4, seed=1).requires_grad_(True)
    y = _phase(6, 4, seed=2).requires_grad_(True)
    v_bind(x, y).sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x.grad))
    assert torch.allclose(y.grad, torch.ones_like(y.grad))


# --------------------------------------------------------------------------
# PhaseRecenter
# --------------------------------------------------------------------------


def test_recenter_parameter_free():
    assert PhaseRecenter().parameter_dict() == {}
    assert list(PhaseRecenter().parameters()) == []


@pytest.mark.parametrize("shape", [(8, 6, 3), (8, 3)])
def test_recenter_shape_range(shape):
    x = _phase(*shape)
    y = PhaseRecenter()(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y >= -1).all() and (y <= 1).all()


def test_recenter_circular_mean_zero():
    x = _phase(8, 6, 3)
    y = PhaseRecenter()(x)
    m = complex_to_angle(angle_to_complex(y).sum(dim=0, keepdim=True))
    assert m.abs().max() < 1e-3


# --------------------------------------------------------------------------
# PhasorResidual
# --------------------------------------------------------------------------


def test_residual_gate_none_no_alpha():
    res = PhasorResidual(PhasorLSA(16, 16, 4, generator=_gen()), gate="none")
    assert res.alpha is None
    assert "alpha" not in res.parameter_dict()


def test_residual_gate_rezero_adds_one_param():
    sub = PhasorLSA(16, 16, 4, generator=_gen())
    base = sum(p.numel() for p in sub.parameters())
    res = PhasorResidual(sub, gate="rezero", alpha0=0.1)
    assert res.alpha is not None and res.alpha.shape == (1,)
    assert sum(p.numel() for p in res.parameters()) == base + 1


def test_residual_bad_gate_raises():
    with pytest.raises(ValueError):
        PhasorResidual(PhasorLSA(16, 16, 4, generator=_gen()), gate="bogus")


def test_residual_rezero_alpha0_is_identity():
    x = _phase(16, 5, 3)
    res = PhasorResidual(PhasorLSA(16, 16, 4, generator=_gen()),
                         gate="rezero", alpha0=0.0)
    assert torch.allclose(res(x), x, atol=1e-5)


# --------------------------------------------------------------------------
# PhasorTransformerBlock
# --------------------------------------------------------------------------


def _make_attn(kind, D, H, A, seed=0):
    g = _gen(seed)
    if kind == "lsa":
        return PhasorLSA(D, D, H, generator=g)
    return PhasorLCA(D, D, H, A, generator=g)


@pytest.mark.parametrize("kind", ["lsa", "lca"])
@pytest.mark.parametrize("shape", [(16, 6, 3), (16, 3)])
def test_block_shape_range(kind, shape):
    D, H, A = 16, 4, 8
    blk = PhasorTransformerBlock(D, _make_attn(kind, D, H, A), generator=_gen())
    x = _phase(*shape)
    y = blk(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert (y >= -1).all() and (y <= 1).all()


@pytest.mark.parametrize("kind", ["lsa", "lca"])
def test_block_identity_at_init(kind):
    D, H, A = 16, 4, 8
    blk = PhasorTransformerBlock(D, _make_attn(kind, D, H, A),
                                 gate="rezero", alpha0=0.0, recenter=False,
                                 generator=_gen())
    for shape in [(D, 6, 3), (D, 3)]:
        x = _phase(*shape)
        assert torch.allclose(blk(x), x, atol=1e-5)


@pytest.mark.parametrize("kind", ["lsa", "lca"])
def test_block_recenter_runs(kind):
    D, H, A = 16, 4, 8
    blk = PhasorTransformerBlock(D, _make_attn(kind, D, H, A),
                                 recenter=True, generator=_gen())
    y = blk(_phase(D, 5, 3))
    assert y.shape == (D, 5, 3) and torch.isfinite(y).all()


def test_block_param_count_two_alphas():
    D, H = 16, 4
    blk = PhasorTransformerBlock(D, _make_attn("lsa", D, H, 8),
                                 gate="rezero", generator=_gen())
    alphas = [n for n, _ in blk.named_parameters() if n.endswith("alpha")]
    assert len(alphas) == 2


def test_block_depth4_finite_grads():
    D, H = 16, 4
    g = _gen()
    stack = torch.nn.Sequential(*[
        PhasorTransformerBlock(D, PhasorLSA(D, D, H, generator=g),
                               alpha0=0.1, generator=g)
        for _ in range(4)
    ])
    x = _phase(D, 6, 3).requires_grad_(True)
    loss = (angle_to_complex(stack(x)).real ** 2).sum()
    loss.backward()
    bad = [n for n, p in stack.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"


@pytest.mark.parametrize("kind", ["lsa", "lca"])
def test_block_hdf5_round_trip(kind, tmp_path):
    D, H, A = 16, 4, 8
    src = PhasorTransformerBlock(D, _make_attn(kind, D, H, A, seed=1),
                                 recenter=True, generator=_gen(1))
    dst = PhasorTransformerBlock(D, _make_attn(kind, D, H, A, seed=2),
                                 recenter=True, generator=_gen(2))
    path = tmp_path / "block.h5"
    save_state(path, {"block": src})
    load_state(path, {"block": dst})
    x = _phase(D, 7, 3, seed=5)
    assert torch.allclose(src(x), dst(x), atol=1e-6)
