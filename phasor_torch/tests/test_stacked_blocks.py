"""Tests for stacked blocks (ModelConfig.n_blocks): plain and rezero."""

import torch

from phasor_torch import hpo
from phasor_torch.config import ModelConfig, TrainConfig
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.train import build_model, build_optimizer, forward_model
from phasor_torch.weights import load_state, save_state


def _phase_input(C, L, B, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(C, L, B, generator=g) * 2 - 1).float()


def _cfg(body="lca", n_blocks=1, block_type="plain"):
    return ModelConfig(frontend="none", body=body, d_hidden=32, n_heads=4,
                       n_anchors=8, in_dims=32, n_classes=10, n_blocks=n_blocks,
                       block_type=block_type)


def test_single_block_keys_unchanged():
    _, schema = build_model(_cfg("lca", n_blocks=1))
    assert list(schema.keys()) == ["input", "body", "dense", "readout"]


def test_config_b_defaults_wired():
    """Config-B: uniform (lambda=-0.2) Q/K/V read heads, hippo FFN tape,
    recenter pre-norm on, hippo input embedding (long tape)."""
    import math
    cfg = _cfg("lca", n_blocks=1, block_type="rezero")
    # confirm the new defaults are the config-B values
    assert cfg.qkv_init_mode == "default"
    assert cfg.ffn_init_mode == "hippo"
    assert cfg.recenter is True
    _, schema = build_model(cfg)
    block = schema["block"]
    # attention K/V projections: uniform lambda init -> constant log(0.2)
    kproj = block.attn_res.sublayer.k_proj
    assert torch.allclose(kproj.log_neg_lambda,
                          torch.full_like(kproj.log_neg_lambda, math.log(0.2)),
                          atol=1e-6)
    # FFN fc1: hippo tape -> NOT constant, spans 1/64 .. 2
    fc1 = block.ffn_res.sublayer.fc1
    lam_mag = torch.exp(fc1.log_neg_lambda)
    assert lam_mag.std() > 0.0
    assert torch.isclose(lam_mag.min(), torch.tensor(1.0 / 64.0), atol=1e-4)
    assert torch.isclose(lam_mag.max(), torch.tensor(2.0), atol=1e-4)
    # recenter pre-norm present on both residual branches (skip untouched)
    assert block.attn_res.recenter is not None
    assert block.ffn_res.recenter is not None
    # input embedding stays hippo (long tape)
    inp_lam = torch.exp(schema["input"].log_neg_lambda)
    assert torch.isclose(inp_lam.min(), torch.tensor(1.0 / 64.0), atol=1e-4)


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


# --------------------------------------------------------------------------
# ReZero transformer blocks (block_type == "rezero")
# --------------------------------------------------------------------------


def test_rezero_block_keys_single_and_stacked():
    _, schema1 = build_model(_cfg("lsa", n_blocks=1, block_type="rezero"))
    assert list(schema1.keys()) == ["input", "block", "readout"]
    _, schema2 = build_model(_cfg("lca", n_blocks=3, block_type="rezero"))
    assert list(schema2.keys()) == [
        "input", "block0", "block1", "block2", "readout"]


def test_rezero_forward_and_grads():
    model, schema = build_model(_cfg("lsa", n_blocks=3, block_type="rezero"))
    x = _phase_input(32, 12, 3)
    sims = forward_model(schema, x)
    assert sims.shape == (10, 3) and torch.isfinite(sims).all()
    y = torch.randint(0, 10, (3,))
    loss = similarity_loss(sims, one_hot(y, 10))
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad}"


def test_rezero_hdf5_round_trip(tmp_path):
    cfg = _cfg("lca", n_blocks=2, block_type="rezero")
    _, src = build_model(cfg, generator=torch.Generator().manual_seed(1))
    path = tmp_path / "rezero.h5"
    save_state(path, src)
    _, dst = build_model(cfg, generator=torch.Generator().manual_seed(2))
    load_state(path, dst)
    x = _phase_input(32, 16, 3, seed=5)
    assert torch.allclose(forward_model(src, x), forward_model(dst, x), atol=1e-6)


def test_optimizer_alpha_param_group():
    model, _ = build_model(_cfg("lsa", n_blocks=2, block_type="rezero"))
    train = TrainConfig(lr=3e-4, alpha_lr_mult=5.0)
    opt = build_optimizer(model, train)
    assert len(opt.param_groups) == 2
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs == [3e-4, 3e-4 * 5.0]
    # 2 blocks x 2 residuals = 4 alpha scalars in the high-LR group.
    hi = max(opt.param_groups, key=lambda g: g["lr"])
    assert len(hi["params"]) == 4


def test_optimizer_single_group_when_plain():
    model, _ = build_model(_cfg("lsa", n_blocks=2, block_type="plain"))
    opt = build_optimizer(model, TrainConfig(lr=3e-4))
    assert len(opt.param_groups) == 1


def test_n_blocks_passthrough():
    base = hpo.HpoBase(body="lsa", source="synthetic", n_blocks=2)
    point = {"lr": 3e-4, "d_hidden_i": 0, "n_heads_i": 1, "init_scale": 3.0,
             "readout_frac": 0.25, "weight_decay": 1e-8, "epochs": 1}
    run = hpo.point_to_runconfig(point, base)
    assert run.model.n_blocks == 2
