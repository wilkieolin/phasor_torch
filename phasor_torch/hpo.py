"""Ray + ytopt hyperparameter-optimization driver for the audio LCA/LSA archs.

ytopt drives the search by importing a module-level ``Problem`` object and
calling its objective; it MINIMIZES the objective, so we return ``-test_acc``.
Launch (in the dedicated `phasor_hpo` env):

    PHASOR_HPO_BODY=lca PHASOR_HPO_TRAIN_PATH=... PHASOR_HPO_TEST_PATH=... \\
    python -m ytopt.search.ambs --evaluator ray \\
        --problem phasor_torch.hpo.Problem --max-evals 50 --learner RF

ambs imports ``phasor_torch.hpo.Problem`` with no arguments, so all per-run
configuration arrives through ``PHASOR_HPO_*`` environment variables (see
``HpoBase.from_env``). One study per body (``PHASOR_HPO_BODY=lca|lsa``).

Import layout: the heavy optional deps (``ConfigSpace``, ``autotune``) are
imported lazily inside ``make_space`` / ``build_problem``, and ``Problem`` is
constructed lazily via module ``__getattr__``. So ``import phasor_torch.hpo``
(and unit-testing ``point_to_runconfig`` / ``objective``) works in the base
``nubun`` env that has torch but not the ytopt stack.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from . import config
from .train import train


# --------------------------------------------------------------------------
# Base (fixed) settings, read from the environment for ambs.
# --------------------------------------------------------------------------


@dataclass
class HpoBase:
    """Fixed settings for a study; the search space varies the rest.

    Built from ``PHASOR_HPO_*`` env vars by :meth:`from_env` (ambs cannot pass
    arguments to the imported ``Problem``).
    """

    body: str = "lca"                 # 'lca' | 'lsa' (one study per body)
    n_blocks: int = 1                 # stacked blocks (fixed per study; depth = 1/2/3)
    # Block topology, fixed per study. 'plain' = the original (body -> dense)
    # stacking (collapses past depth ~2). 'rezero' = depth-robust
    # PhasorTransformerBlock; use this for n_blocks > 1. The remaining knobs are
    # the recommended-default regime from results/lsa_lca_residual and are not
    # part of the search space.
    block_type: str = "plain"         # 'plain' | 'rezero'
    gate: str = "rezero"              # 'none' | 'rezero' (only when block_type == 'rezero')
    recenter: bool = False            # pre-norm; off is best (Julia findings)
    branch_init_scale: float = 0.1    # FFN-only weight-init down-scale
    d_ff: int = 0                     # FFN hidden dim; 0 -> d_ff = d_hidden
    alpha_lr_mult: float = 5.0        # ReZero alpha gates train at lr * this
    source: str = "audio"             # 'audio' | 'synthetic' (synthetic = plumbing tests)
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    n_classes: int = 30
    n_freqs: int = 64
    downsample_factor: int = 32
    batch_size: int = 8               # FLOPS-bound; batching up doesn't help (project memory)
    device: str = "auto"
    train_limit: Optional[int] = None
    test_limit: Optional[int] = None
    epochs_min: int = 30              # epochs is a swept dim; bounds come from env
    epochs_max: int = 80
    patience: int = 6                 # early-stop a trial after N epochs w/o metric improvement
    min_delta: float = 0.0
    early_stop_metric: str = "test_acc"   # "test_acc" (maximize) | "test_loss" (minimize)
    cosine_schedule: bool = False     # cosine LR decay (lr -> lr_min) over each trial
    lr_min: float = 1e-6
    save_best: bool = True            # write best.h5 per trial (cheap; matches reported best test_acc)
    checkpoint_every: int = 0         # periodic ckpt_epoch{N}.h5 per trial (0 = off)
    restore_best: bool = True         # final checkpoint.h5 == peak test_acc weights, not last epoch
    seed: int = 0
    outdir: str = "hpo_runs"
    # synthetic-only fallbacks (used when source == 'synthetic', for fast tests)
    in_dims: int = 64
    max_length: int = 16
    num_train: int = 64
    num_test: int = 32

    @classmethod
    def from_env(cls) -> "HpoBase":
        e = os.environ.get

        def _i(key: str, default: int) -> int:
            v = e(key)
            return int(v) if v not in (None, "") else default

        def _oi(key: str) -> Optional[int]:
            v = e(key)
            return int(v) if v not in (None, "") else None

        return cls(
            body=e("PHASOR_HPO_BODY", "lca"),
            n_blocks=_i("PHASOR_HPO_N_BLOCKS", 1),
            block_type=e("PHASOR_HPO_BLOCK_TYPE", "plain"),
            gate=e("PHASOR_HPO_GATE", "rezero"),
            recenter=(e("PHASOR_HPO_RECENTER", "").lower() in ("1", "true", "yes")),
            branch_init_scale=float(e("PHASOR_HPO_BRANCH_INIT_SCALE") or 0.1),
            d_ff=_i("PHASOR_HPO_D_FF", 0),
            alpha_lr_mult=float(e("PHASOR_HPO_ALPHA_LR_MULT") or 5.0),
            source=e("PHASOR_HPO_SOURCE", "audio"),
            train_path=e("PHASOR_HPO_TRAIN_PATH") or None,
            test_path=e("PHASOR_HPO_TEST_PATH") or None,
            n_classes=_i("PHASOR_HPO_N_CLASSES", 30),
            n_freqs=_i("PHASOR_HPO_N_FREQS", 64),
            downsample_factor=_i("PHASOR_HPO_DOWNSAMPLE", 32),
            batch_size=_i("PHASOR_HPO_BATCH", 8),
            device=e("PHASOR_HPO_DEVICE", "auto"),
            train_limit=_oi("PHASOR_HPO_TRAIN_LIMIT"),
            test_limit=_oi("PHASOR_HPO_TEST_LIMIT"),
            epochs_min=_i("PHASOR_HPO_EPOCHS_MIN", 30),
            epochs_max=_i("PHASOR_HPO_EPOCHS_MAX", 80),
            patience=_i("PHASOR_HPO_PATIENCE", 6),
            min_delta=float(e("PHASOR_HPO_MIN_DELTA") or 0.0),
            early_stop_metric=e("PHASOR_HPO_EARLY_STOP_METRIC", "test_acc"),
            cosine_schedule=(e("PHASOR_HPO_COSINE", "").lower() in ("1", "true", "yes")),
            lr_min=float(e("PHASOR_HPO_LR_MIN") or 1e-6),
            save_best=(e("PHASOR_HPO_SAVE_BEST", "1").lower() not in ("0", "false", "no")),
            checkpoint_every=_i("PHASOR_HPO_CHECKPOINT_EVERY", 0),
            restore_best=(e("PHASOR_HPO_RESTORE_BEST", "1").lower() not in ("0", "false", "no")),
            seed=_i("PHASOR_HPO_SEED", 0),
            outdir=e("PHASOR_HPO_OUTDIR", "hpo_runs"),
        )


# --------------------------------------------------------------------------
# Search space (ConfigSpace).
# --------------------------------------------------------------------------


# Ordered discrete params are searched as INTEGER INDEX dimensions ("<name>_i"),
# NOT skopt Categoricals: dh-scikit-optimize's Categorical inverse_transform
# breaks against modern numpy/sklearn ("argmax of an empty sequence"). Integer
# dims never hit that path, and these sizes are genuinely ordinal. The index is
# mapped back to the real value in point_to_runconfig.
DISCRETE_CHOICES: dict[str, tuple[int, ...]] = {
    "d_hidden": (64, 128, 256),
    "n_heads": (2, 4, 8),
    "n_anchors": (32, 64, 128, 256),
}
DISCRETE_DEFAULT_IDX = {"d_hidden": 0, "n_heads": 1, "n_anchors": 0}  # 64, 4, 32


def make_space(base: HpoBase):
    """Build the per-study ConfigSpace. Imports ConfigSpace lazily.

    Discrete choices are Integer index dims (`d_hidden_i`/`n_heads_i`/
    `n_anchors_i`); `lr`/`init_scale`/`readout_frac`/`weight_decay` are Float;
    `epochs` is a swept Integer (omitted when its bounds are equal -> falls back
    to `epochs_min`; ytopt's skopt fork has no Constant support). `n_anchors_i`
    is added only for the LCA body. Supports new + old ConfigSpace APIs.
    """
    import ConfigSpace as CS

    lo_e, hi_e = sorted((int(base.epochs_min), int(base.epochs_max)))
    cs = CS.ConfigurationSpace(seed=base.seed)
    sweep_epochs = lo_e < hi_e

    def _idx_hi(name: str) -> int:
        return len(DISCRETE_CHOICES[name]) - 1

    try:  # new convenience API (ConfigSpace >= 1.0)
        from ConfigSpace import Float, Integer

        params = [
            Float("lr", bounds=(1e-4, 1e-2), log=True, default=3e-4),
            Integer("d_hidden_i", bounds=(0, _idx_hi("d_hidden")),
                    default=DISCRETE_DEFAULT_IDX["d_hidden"]),
            Integer("n_heads_i", bounds=(0, _idx_hi("n_heads")),
                    default=DISCRETE_DEFAULT_IDX["n_heads"]),
            Float("init_scale", bounds=(1.0, 5.0), default=3.0),
            Float("readout_frac", bounds=(0.1, 1.0), default=0.25),
            Float("weight_decay", bounds=(1e-8, 1e-3), log=True, default=1e-8),
        ]
        if sweep_epochs:
            params.append(Integer("epochs", bounds=(lo_e, hi_e), default=lo_e))
        if base.body == "lca":
            params.append(Integer("n_anchors_i", bounds=(0, _idx_hi("n_anchors")),
                                  default=DISCRETE_DEFAULT_IDX["n_anchors"]))
    except ImportError:  # old API
        import ConfigSpace.hyperparameters as CSH

        params = [
            CSH.UniformFloatHyperparameter("lr", lower=1e-4, upper=1e-2, log=True,
                                           default_value=3e-4),
            CSH.UniformIntegerHyperparameter("d_hidden_i", lower=0,
                                             upper=_idx_hi("d_hidden"),
                                             default_value=DISCRETE_DEFAULT_IDX["d_hidden"]),
            CSH.UniformIntegerHyperparameter("n_heads_i", lower=0,
                                             upper=_idx_hi("n_heads"),
                                             default_value=DISCRETE_DEFAULT_IDX["n_heads"]),
            CSH.UniformFloatHyperparameter("init_scale", lower=1.0, upper=5.0,
                                           default_value=3.0),
            CSH.UniformFloatHyperparameter("readout_frac", lower=0.1, upper=1.0,
                                           default_value=0.25),
            CSH.UniformFloatHyperparameter("weight_decay", lower=1e-8, upper=1e-3,
                                           log=True, default_value=1e-8),
        ]
        if sweep_epochs:
            params.append(CSH.UniformIntegerHyperparameter("epochs", lower=lo_e,
                                                           upper=hi_e, default_value=lo_e))
        if base.body == "lca":
            params.append(CSH.UniformIntegerHyperparameter("n_anchors_i", lower=0,
                                                           upper=_idx_hi("n_anchors"),
                                                           default_value=DISCRETE_DEFAULT_IDX["n_anchors"]))

    try:
        cs.add(params)               # ConfigSpace >= 1.0
    except (AttributeError, TypeError):
        cs.add_hyperparameters(params)
    return cs


# --------------------------------------------------------------------------
# Point -> RunConfig and the objective.
# --------------------------------------------------------------------------


def _scalar(v: Any) -> Any:
    """Coerce numpy scalars (and stringified numbers) to plain Python."""
    if hasattr(v, "item"):
        return v.item()
    return v


def _resolve_discrete(p: dict, name: str) -> int:
    """Map an `<name>_i` index in the point to its real value in DISCRETE_CHOICES."""
    return int(DISCRETE_CHOICES[name][int(p[f"{name}_i"])])


def point_to_runconfig(point: dict, base: HpoBase) -> config.RunConfig:
    """Merge a sampled ConfigSpace point with the fixed base into a RunConfig.

    Discrete params arrive as integer indices (`d_hidden_i` etc.) and are mapped
    back to their real values via DISCRETE_CHOICES.
    """
    p = {k: _scalar(v) for k, v in point.items()}

    model: dict[str, Any] = {
        "frontend": "resonant" if base.source == "audio" else "none",
        "body": base.body,
        "n_blocks": int(base.n_blocks),
        # Block topology is fixed per study (depth-robust rezero for n_blocks > 1);
        # the knobs are inert when block_type == 'plain'. Not searched.
        "block_type": base.block_type,
        "gate": base.gate,
        "recenter": bool(base.recenter),
        "branch_init_scale": float(base.branch_init_scale),
        "d_ff": int(base.d_ff),
        "n_classes": int(base.n_classes),
        "n_freqs": int(base.n_freqs),
        "downsample_factor": int(base.downsample_factor),
        "d_hidden": _resolve_discrete(p, "d_hidden"),
        "n_heads": _resolve_discrete(p, "n_heads"),
        "init_scale": float(p["init_scale"]),
        "readout": "ssm",
        "readout_frac": float(p["readout_frac"]),
    }
    if base.body == "lca":
        model["n_anchors"] = _resolve_discrete(p, "n_anchors")

    # epochs may be swept (in the point) or fixed (omitted -> base.epochs_min).
    epochs = int(p["epochs"]) if "epochs" in p else int(base.epochs_min)
    train_cfg: dict[str, Any] = {
        "batch_size": int(base.batch_size),
        "lr": float(p["lr"]),
        "alpha_lr_mult": float(base.alpha_lr_mult),
        "weight_decay": float(p["weight_decay"]),
        "epochs": epochs,
        "device": base.device,
        "seed": int(base.seed),
        "patience": int(base.patience),
        "min_delta": float(base.min_delta),
        "early_stop_metric": str(base.early_stop_metric),
        "cosine_schedule": bool(base.cosine_schedule),
        "lr_min": float(base.lr_min),
        "save_best": bool(base.save_best),
        "checkpoint_every": int(base.checkpoint_every),
        "restore_best": bool(base.restore_best),
    }

    if base.source == "audio":
        data: dict[str, Any] = {
            "source": "audio",
            "train_path": base.train_path,
            "test_path": base.test_path,
            "sample_rate": 16000,
            "train_limit": base.train_limit,
            "test_limit": base.test_limit,
        }
    else:  # synthetic — fast plumbing path, no audio data needed
        model["in_dims"] = int(base.in_dims)
        data = {
            "source": "synthetic",
            "task": "copy",
            "vocab_size": int(base.n_classes),
            "max_length": int(base.max_length),
            "num_train": int(base.num_train),
            "num_test": int(base.num_test),
        }

    return config.from_dict({"model": model, "data": data, "train": train_cfg})


def _trial_dir(outdir: str, point: dict) -> Path:
    """Stable per-trial directory keyed by a short hash of the point.

    ambs does not pass a trial id to the objective, so we derive one from the
    sampled point's contents.
    """
    key = json.dumps({k: _scalar(point[k]) for k in sorted(point)}, sort_keys=True)
    h = hashlib.sha1(key.encode()).hexdigest()[:10]
    return Path(outdir) / f"trial_{h}"


def objective(point: dict) -> float:
    """ytopt objective: train one config, return -test_acc (ytopt minimizes).

    Writes per-trial artifacts (config.json, history.json, checkpoint.h5) under
    ``PHASOR_HPO_OUTDIR``. A crashed trial returns a +1.0 penalty (worse than
    any real -test_acc in [-1, 0]) so the search continues rather than aborting.
    """
    base = HpoBase.from_env()
    trial = _trial_dir(base.outdir, point)
    trial.mkdir(parents=True, exist_ok=True)
    try:
        run = point_to_runconfig(point, base)
        result = train(run, save_path=str(trial / "checkpoint.h5"))
        history = result.get("history") or []
        final = result.get("final") or {}
        # Best test_acc over the run (robust to the early-stop plateau tail).
        test_acc = max((float(r.get("test_acc", 0.0)) for r in history),
                       default=float(final.get("test_acc", 0.0)))
        # Record both the raw sampled point (indices) and the resolved config
        # (real values) so the trial dir is human-readable.
        (trial / "config.json").write_text(json.dumps({
            "point": {k: _scalar(v) for k, v in point.items()},
            "model": asdict(run.model),
            "train": asdict(run.train),
            "data": asdict(run.data),
        }, indent=2, sort_keys=True))
        (trial / "history.json").write_text(json.dumps(result["history"], indent=2))
        return -test_acc
    except Exception as exc:  # keep the search alive; record the failure
        (trial / "error.txt").write_text(repr(exc))
        return 1.0


# --------------------------------------------------------------------------
# ytopt Problem (lazy — only needs autotune at access time).
# --------------------------------------------------------------------------


def build_problem(base: Optional[HpoBase] = None):
    """Construct the autotune.TuningProblem for ambs. Imports autotune lazily."""
    base = base or HpoBase.from_env()
    from autotune import TuningProblem
    from autotune.space import Real, Space

    input_space = make_space(base)
    output_space = Space([Real(float("-inf"), float("inf"), name="objective")])
    return TuningProblem(
        task_space=None,
        input_space=input_space,
        output_space=output_space,
        objective=objective,
        constraints=None,
        model=None,
    )


def __getattr__(name: str):
    # PEP 562: build Problem lazily so `import phasor_torch.hpo` works without
    # the ytopt stack; ambs resolves `phasor_torch.hpo.Problem` in the hpo env.
    if name == "Problem":
        return build_problem()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
