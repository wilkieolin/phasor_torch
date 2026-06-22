"""Tests for stacked (body -> dense) blocks (ModelConfig.n_blocks)."""

import torch

from phasor_torch import hpo
from phasor_torch.config import ModelConfig
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.train import build_model, forward_model
from phasor_torch.weights import load_state, save_state


def _phase_input(C, L, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(C, L, B, generator=g) * 2 - 1).float()


def _cfg(body="lca", n_blocks=1, residual=False):
    return ModelConfig(frontend="none", body=body, d_hidden=32, n_heads=4,
                       n_anchors=8, in_dims=32, n_classes=10, n_blocks=n_blocks,
                       residual=residual)


def test_single_block_keys_unchanged():
    _, schema = build_model(_cfg("lca", n_blocks=1))
    assert list(schema.keys()) == ["input", "body", "dense", "readout"]


def test_stacked_block_keys_and_forward():
    _, schema = build_model(_cfg("lca", n_blocks=2))
    assert list(schema.keys()) == [
        "input", "body0", "dense0", "body1", "dense1", "readout"]
    sims = forward_model(schema, _phase_input(32, 12, 3))
    assert sims.shape == (10, 3)
    assert torch.isfinite(sims).all()


def test_stacked_grads_finite():
    model, schema = build_model(_cfg("lsa", n_blocks=2))
    x = _phase_input(32, 12, 3)
    y = torch.randint(0, 10, (3,))
    loss = similarity_loss(forward_model(schema, x), one_hot(y, 10))
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"


def test_stacked_hdf5_round_trip(tmp_path):
    cfg = _cfg("lca", n_blocks=2)
    _, src = build_model(cfg, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "stacked.h5"
    save_state(path, src)
    _, dst = build_model(cfg, generator=torch.Generator().manual_seed(2))
    load_state(path, dst)
    x = _phase_input(32, 16, 3, seed=5)
    assert torch.allclose(forward_model(src, x), forward_model(dst, x), atol=1e-6)


def test_n_blocks_passthrough():
    base = hpo.HpoBase(body="lsa", source="synthetic", n_blocks=2)
    point = {"lr": 3e-4, "d_hidden_i": 0, "n_heads_i": 1, "init_scale": 3.0,
             "readout_frac": 0.25, "weight_decay": 1e-8, "epochs": 1}
    run = hpo.point_to_runconfig(point, base)
    assert run.model.n_blocks == 2


# --- VSA-binding residual blocks (ModelConfig.residual) --------------------


def test_residual_schema_keys():
    _, s1 = build_model(_cfg("lca", n_blocks=1, residual=True))
    assert list(s1.keys()) == ["input", "block", "readout"]
    _, s2 = build_model(_cfg("lca", n_blocks=3, residual=True))
    assert list(s2.keys()) == ["input", "block0", "block1", "block2", "readout"]


def test_residual_forward_matches_hand_wired_vbind():
    from phasor_torch.primitives import v_bind
    _, schema = build_model(_cfg("lsa", n_blocks=1, residual=True),
                            generator=torch.Generator().manual_seed(3))
    x = _phase_input(32, 12, 4, seed=7)
    blk = schema["block"]
    # hand-wire the two binds
    expect = v_bind(v_bind(x, blk.body(x)), blk.dense(v_bind(x, blk.body(x))))
    got = forward_model({"block": blk}, x)   # block applied as one layer
    assert torch.allclose(got, expect, atol=1e-6)


def test_residual_grads_finite_deep():
    model, schema = build_model(_cfg("lca", n_blocks=4, residual=True))
    x = _phase_input(32, 12, 3)
    y = torch.randint(0, 10, (3,))
    loss = similarity_loss(forward_model(schema, x), one_hot(y, 10))
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"


def test_residual_hdf5_round_trip(tmp_path):
    cfg = _cfg("lca", n_blocks=2, residual=True)
    _, src = build_model(cfg, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "resid.h5"
    save_state(path, src)
    _, dst = build_model(cfg, generator=torch.Generator().manual_seed(2))
    load_state(path, dst)
    x = _phase_input(32, 16, 3, seed=5)
    assert torch.allclose(forward_model(src, x), forward_model(dst, x), atol=1e-6)


def test_residual_passthrough_in_hpo():
    base = hpo.HpoBase(body="lsa", source="synthetic")
    point = {"lr": 3e-4, "d_hidden_i": 0, "n_heads_i": 1, "init_scale": 3.0,
             "readout_frac": 0.25, "weight_decay": 1e-8, "epochs": 1}
    run = hpo.point_to_runconfig(point, base)
    assert run.model.residual is False    # default off
