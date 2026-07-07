"""Tests for the ytopt HPO driver (the parts that don't need the ytopt stack).

`point_to_runconfig` and `objective` are pure and run in the base env; the
ConfigSpace/autotune-dependent pieces are guarded with importorskip so they run
only in the dedicated `phasor_hpo` env.
"""

import json
from pathlib import Path

import pytest

from phasor_torch import hpo
from phasor_torch.config import RunConfig


def _full_point(body="lca"):
    # Discrete params are integer indices into hpo.DISCRETE_CHOICES:
    #   d_hidden_i=0 -> 64, n_heads_i=1 -> 4, n_anchors_i=1 -> 64
    p = {
        "lr": 5e-4,
        "d_hidden_i": 0,
        "n_heads_i": 1,
        "init_scale": 2.5,
        "readout_frac": 0.3,
        "weight_decay": 1e-6,
        "epochs": 1,
    }
    if body == "lca":
        p["n_anchors_i"] = 1
    return p


def test_point_to_runconfig_lca_audio():
    base = hpo.HpoBase(body="lca", source="audio",
                       train_path="/x/train.h5", test_path="/x/test.h5",
                       n_classes=30, train_limit=64, test_limit=32)
    run = hpo.point_to_runconfig(_full_point("lca"), base)
    assert isinstance(run, RunConfig)
    assert run.model.frontend == "resonant"
    assert run.model.body == "lca"
    assert run.model.n_anchors == 64          # n_anchors_i=1 -> 64
    assert run.model.n_classes == 30
    assert run.model.d_hidden == 64 and run.model.n_heads == 4   # indices 0,1
    assert run.data.source == "audio"
    assert run.data.train_path == "/x/train.h5"
    assert run.data.train_limit == 64
    assert run.train.lr == 5e-4 and run.train.epochs == 1
    assert run.train.weight_decay == 1e-6
    assert run.train.patience == 6        # HpoBase default early-stop patience


def test_point_to_runconfig_lsa_omits_anchors():
    base = hpo.HpoBase(body="lsa", source="audio",
                       train_path="/x/tr.h5", test_path="/x/te.h5")
    run = hpo.point_to_runconfig(_full_point("lsa"), base)
    assert run.model.body == "lsa"
    # n_anchors falls back to the ModelConfig default (unused for lsa); the
    # point carried none and point_to_runconfig must not require it.
    assert run.data.source == "audio"


def test_point_to_runconfig_cosine_passthrough():
    base = hpo.HpoBase(body="lsa", source="synthetic", cosine_schedule=True, lr_min=3e-5)
    run = hpo.point_to_runconfig(_full_point("lsa"), base)
    assert run.train.cosine_schedule is True
    assert run.train.lr_min == 3e-5
    # default base leaves cosine off
    run0 = hpo.point_to_runconfig(_full_point("lsa"), hpo.HpoBase(body="lsa", source="synthetic"))
    assert run0.train.cosine_schedule is False


def test_point_to_runconfig_block_type_default_plain():
    # Unchanged default: studies are 'plain' body->dense unless told otherwise.
    base = hpo.HpoBase(body="lsa", source="synthetic")
    run = hpo.point_to_runconfig(_full_point("lsa"), base)
    assert run.model.block_type == "plain"
    assert run.model.n_blocks == 1


def test_point_to_runconfig_rezero_depth_passthrough():
    # Deep study: rezero block + depth threaded through, knobs at the
    # recommended defaults, and the 5x alpha LR multiplier on the train side.
    base = hpo.HpoBase(body="lca", source="synthetic", n_blocks=3,
                       block_type="rezero", n_classes=10)
    run = hpo.point_to_runconfig(_full_point("lca"), base)
    assert run.model.block_type == "rezero"
    assert run.model.n_blocks == 3
    assert run.model.gate == "rezero"
    # config-B defaults threaded from HpoBase: recenter on, uniform Q/K/V read
    # heads, hippo FFN tape.
    assert run.model.recenter is True
    assert run.model.qkv_init_mode == "default"
    assert run.model.ffn_init_mode == "hippo"
    assert run.model.branch_init_scale == 0.1
    assert run.train.alpha_lr_mult == 5.0


def test_rezero_hpo_runconfig_builds_deep_model():
    # The threaded config must actually build a depth-stacked rezero model with
    # a dedicated alpha optimizer group.
    from phasor_torch.train import build_model, build_optimizer
    base = hpo.HpoBase(body="lsa", source="synthetic", n_blocks=2,
                       block_type="rezero", n_classes=10)
    run = hpo.point_to_runconfig(_full_point("lsa"), base)
    model, schema = build_model(run.model)
    assert [k for k in schema if k.startswith("block")] == ["block0", "block1"]
    opt = build_optimizer(model, run.train)
    assert len(opt.param_groups) == 2          # alpha group split out
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[1] == lrs[0] * 5.0              # alpha at 5x base lr


def test_point_to_runconfig_coerces_numpy_scalars():
    np = pytest.importorskip("numpy")
    base = hpo.HpoBase(body="lsa", source="synthetic", n_classes=10)
    point = {
        "lr": np.float64(3e-4),
        "d_hidden_i": np.int64(2),   # -> 256
        "n_heads_i": np.int64(2),    # -> 8
        "init_scale": np.float64(4.0),
        "readout_frac": np.float64(0.2),
        "weight_decay": np.float64(1e-7),
        "epochs": np.int64(1),
    }
    run = hpo.point_to_runconfig(point, base)
    assert run.model.d_hidden == 256 and isinstance(run.model.d_hidden, int)
    assert run.model.n_heads == 8 and isinstance(run.model.n_heads, int)
    assert isinstance(run.train.lr, float)


def test_objective_synthetic_end_to_end(tmp_path, monkeypatch):
    # Synthetic source = no audio-data dependency; epochs 1 + tiny sizes = fast.
    monkeypatch.setenv("PHASOR_HPO_SOURCE", "synthetic")
    monkeypatch.setenv("PHASOR_HPO_BODY", "lca")
    monkeypatch.setenv("PHASOR_HPO_N_CLASSES", "10")
    monkeypatch.setenv("PHASOR_HPO_BATCH", "8")
    monkeypatch.setenv("PHASOR_HPO_DEVICE", "cpu")
    monkeypatch.setenv("PHASOR_HPO_OUTDIR", str(tmp_path))

    val = hpo.objective(_full_point("lca"))
    assert isinstance(val, float)
    assert -1.0 <= val <= 0.0          # -test_acc

    trials = list(tmp_path.glob("trial_*"))
    assert len(trials) == 1
    t = trials[0]
    assert (t / "config.json").exists()
    assert (t / "history.json").exists()
    assert (t / "checkpoint.h5").exists()
    hist = json.loads((t / "history.json").read_text())
    assert isinstance(hist, list) and len(hist) == 1   # epochs=1


def test_objective_failure_returns_penalty(tmp_path, monkeypatch):
    # A bad config (synthetic with mismatched dims) must not abort the search;
    # it should record the error and return the +1.0 penalty.
    monkeypatch.setenv("PHASOR_HPO_SOURCE", "synthetic")
    monkeypatch.setenv("PHASOR_HPO_BODY", "lsa")
    monkeypatch.setenv("PHASOR_HPO_OUTDIR", str(tmp_path))
    bad = _full_point("lsa")
    bad["n_heads_i"] = 9          # out-of-range index -> resolve raises -> penalty
    val = hpo.objective(bad)
    assert val == 1.0
    trials = list(tmp_path.glob("trial_*"))
    assert (trials[0] / "error.txt").exists()


# --- ConfigSpace-dependent (run only in the phasor_hpo env) ----------------


def test_make_space_lca_has_anchors():
    pytest.importorskip("ConfigSpace")
    base = hpo.HpoBase(body="lca", epochs_min=30, epochs_max=80)
    cs = hpo.make_space(base)
    names = set(cs.keys()) if hasattr(cs, "keys") \
        else set(cs.get_hyperparameter_names())
    assert "n_anchors_i" in names
    assert {"lr", "d_hidden_i", "n_heads_i", "init_scale", "readout_frac",
            "weight_decay", "epochs"} <= names


def test_make_space_lsa_no_anchors():
    pytest.importorskip("ConfigSpace")
    base = hpo.HpoBase(body="lsa", epochs_min=1, epochs_max=1)   # epochs fixed -> omitted
    cs = hpo.make_space(base)
    names = set(cs.keys()) if hasattr(cs, "keys") \
        else set(cs.get_hyperparameter_names())
    assert "n_anchors_i" not in names
    assert "epochs" not in names   # fixed bounds -> not a swept dimension


def test_n_anchors_includes_256():
    assert hpo.DISCRETE_CHOICES["n_anchors"] == (32, 64, 128, 256)
    assert hpo._resolve_discrete({"n_anchors_i": 3}, "n_anchors") == 256


def test_widened_search_bounds():
    pytest.importorskip("ConfigSpace")
    cs = hpo.make_space(hpo.HpoBase(body="lca", epochs_min=30, epochs_max=80))
    assert cs["readout_frac"].upper == 1.0    # raised from 0.5
    assert cs["n_anchors_i"].upper == 3       # 4 choices -> index 0..3
    assert cs["lr"].upper == 1e-2            # widened aggressively (railed at 1e-3 before)


# --- early stopping --------------------------------------------------------


def test_early_stop_disabled_or_too_few():
    from phasor_torch.train import _early_stop
    assert _early_stop([1.0, 0.9, 0.8], patience=0, min_delta=0.0) is False   # disabled
    assert _early_stop([1.0, 0.9], patience=6, min_delta=0.0) is False        # too few epochs


def test_early_stop_triggers_on_plateau():
    from phasor_torch.train import _early_stop
    # best (0.80) was at epoch 3; last 3 never beat it -> stop.
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87]
    assert _early_stop(losses, patience=3, min_delta=0.0) is True


def test_early_stop_holds_while_improving():
    from phasor_torch.train import _early_stop
    losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]   # still decreasing
    assert _early_stop(losses, patience=3, min_delta=0.0) is False


def test_early_stop_min_delta():
    from phasor_torch.train import _early_stop
    # tiny improvements below min_delta count as no improvement -> stop.
    losses = [1.0, 0.9, 0.80, 0.799, 0.798, 0.797]
    assert _early_stop(losses, patience=3, min_delta=0.01) is True
    assert _early_stop(losses, patience=3, min_delta=0.0) is False


def test_early_stop_max_mode():
    from phasor_torch.train import _early_stop
    # test_acc (maximize): peak 0.80 at epoch 3, last 3 never beat it -> stop.
    accs = [0.50, 0.70, 0.80, 0.79, 0.78, 0.80]
    assert _early_stop(accs, patience=3, min_delta=0.0, mode="max") is True
    # still climbing -> hold.
    rising = [0.50, 0.70, 0.80, 0.81, 0.82, 0.83]
    assert _early_stop(rising, patience=3, min_delta=0.0, mode="max") is False
    # gains below min_delta count as no improvement -> stop.
    tiny = [0.50, 0.70, 0.80, 0.801, 0.802, 0.803]
    assert _early_stop(tiny, patience=3, min_delta=0.01, mode="max") is True
    assert _early_stop(tiny, patience=3, min_delta=0.0, mode="max") is False


# --- cosine LR schedule ----------------------------------------------------


def test_lr_scheduler_off_by_default():
    import torch
    from phasor_torch.config import TrainConfig
    from phasor_torch.train import _build_lr_scheduler
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    assert _build_lr_scheduler(opt, TrainConfig(cosine_schedule=False), 4) is None


def test_lr_scheduler_cosine_anneals_to_lr_min():
    import torch
    from phasor_torch.config import TrainConfig
    from phasor_torch.train import _build_lr_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=1e-3)
    cfg = TrainConfig(epochs=5, cosine_schedule=True, lr_min=1e-5)
    sched = _build_lr_scheduler(opt, cfg, steps_per_epoch=4)   # T_max = 20
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert abs(lrs[0] - 1e-3) < 1e-9            # starts at peak lr
    assert abs(lrs[-1] - 1e-5) < 1e-6           # ends at lr_min
    assert lrs[-1] < lrs[len(lrs) // 2] < lrs[0]  # monotonically decreasing


# --- libEnsemble launcher helpers ------------------------------------------


def test_libe_parse_user_args():
    from phasor_torch import hpo_libe
    args = hpo_libe._parse_user_args(["--learner", "RF", "--max-evals=10", "--comms", "local"])
    assert args["learner"] == "RF"
    assert args["max-evals"] == "10"
    assert args["comms"] == "local"


def test_libe_field_specs_types():
    pytest.importorskip("ConfigSpace")
    from phasor_torch import hpo_libe
    cs = hpo.make_space(hpo.HpoBase(body="lca", epochs_min=30, epochs_max=80))
    fields = dict(hpo_libe._field_specs(cs))
    assert fields["lr"] is float
    assert fields["d_hidden_i"] is int        # integer index dim
    assert fields["n_anchors_i"] is int
    assert fields["epochs"] is int
