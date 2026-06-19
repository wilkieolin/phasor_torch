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
