"""Smoke tests for Codebook and SSMReadout — shapes, ranges, range invariants."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.init import orthogonal_codes, random_symbols
from phasor_torch.layers import Codebook, SSMReadout
from phasor_torch.layers.ssm_readout import _readout_t0


# --------------------------------------------------------------------------
# init helpers
# --------------------------------------------------------------------------


def test_random_symbols_range():
    g = torch.Generator().manual_seed(0)
    s = random_symbols((5, 8), generator=g)
    assert s.shape == (5, 8)
    assert s.min() >= -1.0
    assert s.max() <= 1.0


def test_orthogonal_codes_self_similarity_one():
    g = torch.Generator().manual_seed(1)
    d, n = 16, 4
    codes = orthogonal_codes(d, n, generator=g)
    assert codes.shape == (d, n)
    # Self-similarity diagonal must be exactly 1.
    for i in range(n):
        sim = torch.cos(math.pi * (codes[:, i] - codes[:, i])).mean()
        assert torch.isclose(sim, torch.tensor(1.0), atol=1e-6)


def test_orthogonal_codes_pairwise_zero_when_n_divides_d():
    g = torch.Generator().manual_seed(2)
    d, n = 12, 4   # n | d
    codes = orthogonal_codes(d, n, generator=g)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sim = torch.cos(math.pi * (codes[:, i] - codes[:, j])).mean()
            assert abs(sim.item()) < 1e-5


def test_orthogonal_codes_rejects_n_gt_d():
    with pytest.raises(ValueError):
        orthogonal_codes(4, 5)


# --------------------------------------------------------------------------
# Codebook
# --------------------------------------------------------------------------


def test_codebook_shape_and_self_diagonal():
    g = torch.Generator().manual_seed(3)
    d, n = 8, 4
    cb = Codebook(d, n, init_mode="orthogonal", generator=g)
    # Input is the codes themselves; output[i, i] should be 1 (self-similarity).
    x = cb.codes.clone()                                 # (d, n)
    sims = cb(x)
    assert sims.shape == (n, n)
    diag = torch.diag(sims)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)


def test_codebook_range():
    g = torch.Generator().manual_seed(4)
    cb = Codebook(6, 5, init_mode="random", generator=g)
    x = (torch.rand(6, 12) * 2 - 1)
    sims = cb(x)
    assert sims.shape == (5, 12)
    assert sims.min() >= -1.0 - 1e-5
    assert sims.max() <= 1.0 + 1e-5


def test_codebook_round_trip(tmp_path):
    from phasor_torch.weights import load_state, save_state

    src = Codebook(6, 4, init_mode="orthogonal",
                   generator=torch.Generator().manual_seed(5))
    path = tmp_path / "cb.h5"
    save_state(path, {"cb": src})

    dst = Codebook(6, 4, init_mode="random",      # different init on purpose
                   generator=torch.Generator().manual_seed(6))
    load_state(path, {"cb": dst})
    assert torch.equal(src.codes, dst.codes)


# --------------------------------------------------------------------------
# SSMReadout
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "L, frac, expected_t0",
    [
        (20, 0.25, 15),
        (20, 0.50, 10),
        (20, 1.0,   0),
        (20, 0.01,  19),    # ceil to 1
        (8,  0.25,  6),
        (4,  0.25,  3),
        (1,  0.25,  0),
    ],
)
def test_readout_t0(L, frac, expected_t0):
    assert _readout_t0(L, frac) == expected_t0


def test_ssm_readout_shape_and_range():
    g = torch.Generator().manual_seed(7)
    C, K = 8, 5
    L, B = 20, 4
    layer = SSMReadout(C, K, generator=g)
    x = (torch.rand(C, L, B) * 2 - 1)
    y = layer(x)
    assert y.shape == (K, B)
    assert y.min() >= -1.0 - 1e-5
    assert y.max() <= 1.0 + 1e-5


def test_ssm_readout_constant_phases_match_codes():
    """If x is constant equal to one of the codes, that class scores ~1."""
    g = torch.Generator().manual_seed(8)
    C, K, L, B = 6, 4, 10, 1
    layer = SSMReadout(C, K, generator=g)
    target_code = layer.codes[:, 2]                         # pick class 2
    x = target_code.reshape(C, 1, 1).expand(C, L, B).contiguous()
    y = layer(x)
    # Class 2 should score 1; others < 1.
    assert torch.isclose(y[2, 0], torch.tensor(1.0), atol=1e-5)
    others = torch.cat([y[:2, 0], y[3:, 0]])
    assert (others < 1.0 - 1e-3).all()


def test_ssm_readout_round_trip(tmp_path):
    from phasor_torch.weights import load_state, save_state

    src = SSMReadout(7, 3, readout_frac=0.5,
                     generator=torch.Generator().manual_seed(9))
    path = tmp_path / "ro.h5"
    save_state(path, {"ro": src})

    dst = SSMReadout(7, 3, readout_frac=0.5,
                     generator=torch.Generator().manual_seed(10))
    load_state(path, {"ro": dst})
    assert torch.equal(src.codes, dst.codes)
