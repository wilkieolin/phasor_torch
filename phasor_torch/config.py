"""Config dataclasses for the LSA/LCA training script.

Pluggable encoder + body + readout per the user's "config-driven topology"
preference. The body selects between PhasorLSA, PhasorLCA, or none (the
last giving a pure PhasorDense chain — useful as a baseline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional


# --------------------------------------------------------------------------
# Model topology
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    """Topology of the trainable network."""

    # Input embedding: a PhasorDense from C_in to d_hidden.
    in_dims: int = 64           # C_in (= codebook n_hd); ignored in audio mode
    d_hidden: int = 64
    # Input-embedding PhasorDense lambda init. UNIFORM (:default, tau=5) by
    # default: the input embedding is a "read" layer, so it does not need the
    # multi-timescale tape (that belongs in the FFN residual stream). hippo's
    # fast channels (tau as low as 0.5, < the resonant period) drove the input
    # embedding's SSM sum to the origin (|z|->1e-9) and were the dominant NaN
    # source once QKV went uniform -- see scripts/grad_diverge_probe.py. The
    # input embedding also uses a complex bias (build_model) to keep |z| off the
    # origin, the same reason the (bias-carrying) FFN stays healthy.
    init_mode: Literal["default", "hippo"] = "default"
    # Body PhasorDense recurrence preset. None keeps init_mode's default lambda;
    # the audio archs use RNN_KW = log(0.1) (per-neuron trainable decay). When
    # frontend == "resonant" and this is None, build_model applies RNN_KW.
    init_log_neg_lambda: Optional[float] = None

    # Audio frontend: 'none' (synthetic Phase input) | 'resonant' (ResonantSTFT).
    frontend: Literal["none", "resonant"] = "none"
    n_freqs: int = 64
    omega_lo: float = 0.2
    omega_hi: float = 2.5
    downsample_factor: int = 32
    resonant_init_log_neg_lambda: float = math.log(0.1)
    init_r_lo: float = 0.1
    init_r_hi: float = 0.6
    resonant_activation: Literal["slerp", "identity", "normalize"] = "slerp"

    # Body: 'none' | 'lsa' | 'lca'.
    body: Literal["none", "lsa", "lca"] = "lsa"
    n_heads: int = 4
    n_anchors: int = 32         # only used when body == 'lca'
    init_scale: float = 3.0
    # Attention-block lambda-init placement (the "config B" default from the
    # PhasorNetworks.jl MQAR ablation): the Q/K/V read-head projections stay
    # uniform (:default, tau=5) so they sharply localize the current token,
    # while the residual-stream FFN carries the multi-timescale memory tape
    # (:hippo). HiPPO in the read heads actively hurts long-range routing.
    qkv_init_mode: Literal["default", "hippo"] = "default"
    ffn_init_mode: Literal["default", "hippo"] = "hippo"
    # Number of stacked blocks between the input embedding and readout. 1 = the
    # canonical single-block chain (unchanged behavior).
    n_blocks: int = 1

    # Block topology:
    #   'plain'  -> the original (body -> dense) stacking (unchanged default).
    #   'rezero' -> each block is a PhasorTransformerBlock (residual-wrapped
    #               attention + FFN with a ReZero gate), making deep stacks
    #               trainable. Requires body in ('lsa', 'lca').
    block_type: Literal["plain", "rezero"] = "plain"
    # ReZero block knobs (only used when block_type == 'rezero').
    gate: Literal["none", "rezero"] = "rezero"
    # PhaseRecenter pre-norm: OFF by default. The phasor_torch NaN investigation
    # (scripts/grad_diverge_probe.py) traced the config-B rezero blow-ups to the
    # recenter circular-mean complex_to_angle (min|z|->~2e-5); the on/off sweep and
    # the local ablation both show it is not helpful and potentially harmful. Leave
    # off unless a workload specifically benefits.
    recenter: bool = False
    # Include the FFN residual sublayer in each rezero block (default True).
    # False -> each block is a single ReZero-gated attention residual with NO
    # FFN (the depth-enabling residual is retained). Used to test whether
    # stacking attention blocks without the intermediate FFN scales with depth.
    use_ffn: bool = True
    branch_init_scale: float = 0.1   # FFN-only weight-init down-scale
    d_ff: int = 0                    # FFN hidden dim; 0 -> d_ff = d_hidden

    # Readout: 'codebook' | 'ssm'.
    readout: Literal["codebook", "ssm"] = "ssm"
    n_classes: int = 10
    readout_frac: float = 0.25  # only used when readout == 'ssm', pool == 'mean'
    codebook_init_mode: Literal["random", "orthogonal"] = "random"
    # SSMReadout temporal aggregation. 'mean' = average similarity over the last
    # readout_frac window (default, parity-preserving). 'logsumexp' = smooth max
    # over the WHOLE clip ("is the keyword present anywhere?" — the KWS inductive
    # bias; the local TIR readout ablation showed it flips the FFN redundant and
    # lifts accuracy). learnable_codes makes the class prototypes trainable.
    readout_pool: Literal["mean", "logsumexp"] = "mean"
    logsumexp_kappa: float = 10.0
    learnable_codes: bool = False

    # Oscillator config.
    t_period: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    """Dataset config — synthetic sequence tasks or raw keyword-spotting audio."""

    source: Literal["synthetic", "audio"] = "synthetic"

    # --- synthetic-sequence knobs (source == "synthetic") ---------------
    task: Literal["copy", "reversal", "retrieval", "sorting"] = "copy"
    vocab_size: int = 10                  # equals model.n_classes for first-token-recall
    max_length: int = 32
    num_train: int = 1024
    num_test: int = 256
    seed: int = 0

    # --- audio knobs (source == "audio") -------------------------------
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    sample_rate: int = 16000
    # Optional caps on clip count for local smoke runs (None = use all clips).
    train_limit: Optional[int] = None
    test_limit: Optional[int] = None


@dataclass(frozen=True)
class TrainConfig:
    """Optimization + bookkeeping config."""

    batch_size: int = 32
    epochs: int = 5
    lr: float = 3e-4
    # Classification loss. 'similarity' = the phase-distance regression toward the
    # true prototype (default, matches history). 'softmax_ce' = contrastive
    # cross-entropy over beta*sims logits — explicitly optimizes the class margin;
    # the biggest single accuracy lever in the local TIR readout ablation.
    loss: Literal["similarity", "softmax_ce"] = "similarity"
    ce_beta: float = 10.0                  # softmax temperature (loss == 'softmax_ce')
    # ReZero alpha gates warm up from ~0 and benefit from a higher LR than the
    # rest of the network. When the model has any `alpha` params, they train at
    # `lr * alpha_lr_mult`; otherwise this is inert. Matches the Julia 5x rule.
    alpha_lr_mult: float = 5.0
    weight_decay: float = 0.0
    log_every: int = 0                    # 0 disables intra-epoch logging
    device: str = "auto"                  # 'auto' | 'cpu' | 'cuda' | 'xpu'
    seed: int = 0
    checkpoint_path: Optional[str] = None  # HDF5 path for the FINAL weights (written once at end)
    # Early stopping: stop if the monitored metric hasn't improved (by > min_delta)
    # over the last `patience` epochs. 0 disables (the trainer runs all epochs).
    #   early_stop_metric -> "test_acc" (maximize; default) or "test_loss" (minimize).
    # test_acc is preferred: test_loss plateaus AFTER test_acc peaks and starts
    # falling, so a loss-keyed stop wastes epochs past the accuracy peak (verified
    # on the audio confirm runs — see results / memory notes).
    patience: int = 0
    min_delta: float = 0.0
    early_stop_metric: str = "test_acc"
    # Cosine LR decay over the whole run, annealing from `lr` to `lr_min`
    # (per optimizer step). Mirrors Julia Args.cosine_schedule / lr_min.
    cosine_schedule: bool = False
    lr_min: float = 1e-6
    # Checkpointing (only active when a save target / checkpoint dir exists):
    #   save_best       -> write best.h5 whenever test_acc improves (matches the
    #                      HPO objective's reported best, unlike the final weights).
    #   checkpoint_every -> write ckpt_epoch{N}.h5 every N epochs (0 = off) for
    #                      weight-trajectory analysis / restart points.
    #   restore_best    -> after training, reload the best (peak test_acc) weights
    #                      into the model before the FINAL save, so checkpoint.h5 /
    #                      the returned model equal the peak, not the last epoch.
    save_best: bool = False
    checkpoint_every: int = 0
    restore_best: bool = True


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
