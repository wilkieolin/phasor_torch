"""Tests for best.h5 + periodic checkpoint saving in train()."""

import torch

from phasor_torch.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from phasor_torch.train import train


def _run(tmp_path, **train_kwargs):
    run = RunConfig(
        model=ModelConfig(frontend="none", body="lsa", d_hidden=32, n_heads=4,
                          in_dims=32, n_classes=10, readout="ssm"),
        data=DataConfig(source="synthetic", task="copy", vocab_size=10,
                        max_length=12, num_train=64, num_test=32),
        train=TrainConfig(epochs=4, batch_size=16, device="cpu", seed=0,
                          **train_kwargs),
    )
    save = tmp_path / "checkpoint.h5"
    return train(run, save_path=str(save)), tmp_path


def test_best_and_periodic_off_by_default(tmp_path):
    _run(tmp_path)                       # save_best=False, checkpoint_every=0
    assert (tmp_path / "checkpoint.h5").exists()      # final still written
    assert not (tmp_path / "best.h5").exists()
    assert not list(tmp_path.glob("ckpt_epoch*.h5"))


def test_best_checkpoint_written(tmp_path):
    _run(tmp_path, save_best=True)
    assert (tmp_path / "best.h5").exists()
    assert (tmp_path / "checkpoint.h5").exists()      # final also written


def test_periodic_checkpoints_written(tmp_path):
    _run(tmp_path, checkpoint_every=2)   # epochs 2 and 4 -> two snapshots
    snaps = sorted(p.name for p in tmp_path.glob("ckpt_epoch*.h5"))
    assert snaps == ["ckpt_epoch2.h5", "ckpt_epoch4.h5"]


def test_no_dir_no_periodic(tmp_path):
    # Without a save target, best/periodic have nowhere to go and are skipped.
    run = RunConfig(
        model=ModelConfig(frontend="none", body="lsa", d_hidden=32, n_heads=4,
                          in_dims=32, n_classes=10),
        data=DataConfig(source="synthetic", vocab_size=10, max_length=12,
                        num_train=64, num_test=32),
        train=TrainConfig(epochs=2, batch_size=16, device="cpu",
                          save_best=True, checkpoint_every=1),
    )
    train(run)                            # no save_path -> nothing written
    assert not list(tmp_path.glob("*.h5"))


def test_hpo_checkpoint_defaults():
    from phasor_torch import hpo
    base = hpo.HpoBase(body="lsa", source="synthetic")
    point = {"lr": 3e-4, "d_hidden_i": 0, "n_heads_i": 1, "init_scale": 3.0,
             "readout_frac": 0.25, "weight_decay": 1e-8, "epochs": 1}
    run = hpo.point_to_runconfig(point, base)
    assert run.train.save_best is True        # best.h5 on by default for HPO trials
    assert run.train.checkpoint_every == 0    # periodic off by default
    assert run.train.early_stop_metric == "test_acc"   # acc-keyed early stop
    assert run.train.restore_best is True              # final == peak weights
    assert run.model.use_bias is False                 # bias off by default


def test_hpo_use_bias_threads_from_base():
    from phasor_torch import hpo
    point = {"lr": 3e-4, "d_hidden_i": 0, "n_heads_i": 1, "init_scale": 3.0,
             "readout_frac": 0.25, "weight_decay": 1e-8, "epochs": 1}
    on = hpo.point_to_runconfig(point, hpo.HpoBase(body="lsa", source="synthetic",
                                                   use_bias=True))
    assert on.model.use_bias is True
    off = hpo.point_to_runconfig(point, hpo.HpoBase(body="lsa", source="synthetic"))
    assert off.model.use_bias is False


def test_restore_best_makes_final_equal_best(tmp_path):
    """With restore_best, checkpoint.h5 (final) reloads the peak weights == best.h5."""
    import torch
    from phasor_torch.train import build_model, forward_model
    from phasor_torch.weights import load_state

    summary, _ = _run(tmp_path, save_best=True, restore_best=True)
    assert (tmp_path / "best.h5").exists()
    assert summary["best_epoch"] >= 1

    cfg = ModelConfig(frontend="none", body="lsa", d_hidden=32, n_heads=4,
                      in_dims=32, n_classes=10, readout="ssm")
    x = torch.rand(32, 12, 4) * 2 - 1
    _, sb = build_model(cfg, generator=torch.Generator().manual_seed(1))
    load_state(str(tmp_path / "best.h5"), sb)
    _, sf = build_model(cfg, generator=torch.Generator().manual_seed(2))
    load_state(str(tmp_path / "checkpoint.h5"), sf)
    assert torch.allclose(forward_model(sb, x), forward_model(sf, x), atol=1e-6)
