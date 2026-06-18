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
    p = {
        "lr": 5e-4,
        "d_hidden": 64,
        "n_heads": 4,
        "init_scale": 2.5,
        "readout_frac": 0.3,
        "weight_decay": 1e-6,
        "epochs": 1,
    }
    if body == "lca":
        p["n_anchors"] = 64
    return p


def test_point_to_runconfig_lca_audio():
    base = hpo.HpoBase(body="lca", source="audio",
                       train_path="/x/train.h5", test_path="/x/test.h5",
                       n_classes=30, train_limit=64, test_limit=32)
    run = hpo.point_to_runconfig(_full_point("lca"), base)
    assert isinstance(run, RunConfig)
    assert run.model.frontend == "resonant"
    assert run.model.body == "lca"
    assert run.model.n_anchors == 64
    assert run.model.n_classes == 30
    assert run.model.d_hidden == 64 and run.model.n_heads == 4
    assert run.data.source == "audio"
    assert run.data.train_path == "/x/train.h5"
    assert run.data.train_limit == 64
    assert run.train.lr == 5e-4 and run.train.epochs == 1
    assert run.train.weight_decay == 1e-6


def test_point_to_runconfig_lsa_omits_anchors():
    base = hpo.HpoBase(body="lsa", source="audio",
                       train_path="/x/tr.h5", test_path="/x/te.h5")
    run = hpo.point_to_runconfig(_full_point("lsa"), base)
    assert run.model.body == "lsa"
    # n_anchors falls back to the ModelConfig default (unused for lsa); the
    # point carried none and point_to_runconfig must not require it.
    assert run.data.source == "audio"


def test_point_to_runconfig_coerces_numpy_scalars():
    np = pytest.importorskip("numpy")
    base = hpo.HpoBase(body="lsa", source="synthetic", n_classes=10)
    point = {
        "lr": np.float64(3e-4),
        "d_hidden": np.int64(128),
        "n_heads": np.int64(8),
        "init_scale": np.float64(4.0),
        "readout_frac": np.float64(0.2),
        "weight_decay": np.float64(1e-7),
        "epochs": np.int64(1),
    }
    run = hpo.point_to_runconfig(point, base)
    assert run.model.d_hidden == 128 and isinstance(run.model.d_hidden, int)
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
    bad["d_hidden"] = 65          # not divisible by n_heads=4 -> body build fails
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
    assert "n_anchors" in names
    assert {"lr", "d_hidden", "n_heads", "init_scale", "readout_frac",
            "weight_decay", "epochs"} <= names


def test_make_space_lsa_no_anchors():
    pytest.importorskip("ConfigSpace")
    base = hpo.HpoBase(body="lsa", epochs_min=1, epochs_max=1)   # Constant epochs
    cs = hpo.make_space(base)
    names = set(cs.keys()) if hasattr(cs, "keys") \
        else set(cs.get_hyperparameter_names())
    assert "n_anchors" not in names
