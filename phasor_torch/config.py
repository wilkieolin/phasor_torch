"""Config dataclasses for the LSA/LCA training script.

Pluggable encoder + body + readout per the user's "config-driven topology"
preference. The body selects between PhasorLSA, PhasorLCA, or none (the
last giving a pure PhasorDense chain — useful as a baseline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


# --------------------------------------------------------------------------
# Model topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Topology of the trainable network."""

    # Input embedding: a PhasorDense from C_in to d_hidden.
    in_dims: int = 64           # C_in (= codebook n_hd)
    d_hidden: int = 64
    init_mode: Literal["default", "hippo"] = "hippo"

    # Body: 'none' | 'lsa' | 'lca'.
    body: Literal["none", "lsa", "lca"] = "lsa"
    n_heads: int = 4
    n_anchors: int = 32         # only used when body == 'lca'
    init_scale: float = 3.0

    # Readout: 'codebook' | 'ssm'.
    readout: Literal["codebook", "ssm"] = "ssm"
    n_classes: int = 10
    readout_frac: float = 0.25  # only used when readout == 'ssm'
    codebook_init_mode: Literal["random", "orthogonal"] = "random"

    # Oscillator config.
    t_period: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    """Synthetic sequence dataset config."""

    task: Literal["copy", "reversal", "retrieval", "sorting"] = "copy"
    vocab_size: int = 10                  # equals model.n_classes for first-token-recall
    max_length: int = 32
    num_train: int = 1024
    num_test: int = 256
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Optimization + bookkeeping config."""

    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.0
    log_every: int = 0                    # 0 disables intra-epoch logging
    device: str = "auto"                  # 'auto' | 'cpu' | 'cuda' | 'xpu'
    seed: int = 0
    checkpoint_path: Optional[str] = None  # HDF5 path; saved per epoch if set


@dataclass(frozen=True)
class RunConfig:
    """Bundle of the three configs the training script consumes."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# --------------------------------------------------------------------------
# YAML / dict loading (lightweight; we keep YAML optional).
# --------------------------------------------------------------------------


def from_dict(d: dict) -> RunConfig:
    """Build a RunConfig from a nested dict. Missing keys take defaults."""
    return RunConfig(
        model=ModelConfig(**(d.get("model") or {})),
        data=DataConfig(**(d.get("data") or {})),
        train=TrainConfig(**(d.get("train") or {})),
    )


def from_yaml(path: str) -> RunConfig:
    import yaml
    with open(path, "r") as f:
        return from_dict(yaml.safe_load(f) or {})
