"""Training loop for LSA/LCA phasor SSM networks (PyTorch).

Builds the configured topology (input PhasorDense -> {LSA | LCA | none}
-> body PhasorDense -> {SSMReadout | Codebook}), feeds embedded
synthetic-sequence batches through it, trains with Adam against
similarity_loss, logs per-epoch accuracy, and optionally saves
checkpoints in the HDF5 schema that Lux can load via julia_parity.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from typing import Optional

import torch
from torch import Tensor, nn

from .config import DataConfig, ModelConfig, RunConfig, TrainConfig
from .data import (
    SequenceTaskConfig,
    build_codebook,
    first_token_classification,
    make_dataloader,
)
import math

from .layers import (
    Codebook,
    PhasorDense,
    PhasorLCA,
    PhasorLSA,
    ResonantSTFT,
    SSMReadout,
    downsample_time,
    encode_input,
    resolve_activation,
    to_phase,
)
from .layers.phasor_dense import SpikingArgs
from .losses import accuracy, codebook_loss, one_hot, similarity_loss
from .primitives import normalize_to_unit_circle


# --------------------------------------------------------------------------
# Model building
# --------------------------------------------------------------------------


def select_device(name: str) -> torch.device:
    """Resolve 'auto' to xpu > cuda > cpu in that order. Matches Aurora docs."""
    if name != "auto":
        return torch.device(name)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(cfg: ModelConfig, generator: torch.Generator | None = None
                ) -> tuple[nn.Module, OrderedDict[str, nn.Module]]:
    """Construct the full chain as a nn.Sequential plus a labeled schema
    dict that names each layer for HDF5 round trip.

    The schema preserves the order used by Lux Chain, so the
    saved HDF5 layout maps cleanly to a corresponding Lux chain in Julia.
    """
    spk = SpikingArgs(t_period=cfg.t_period)

    # Audio frontend (ResonantSTFT) and the RNN_KW recurrence preset for the
    # surrounding PhasorDense layers. In audio mode the input embedding consumes
    # the n_freqs frequency channels instead of cfg.in_dims.
    frontend: Optional[nn.Module] = None
    embed_in = cfg.in_dims
    eff_lnl = cfg.init_log_neg_lambda
    if cfg.frontend == "resonant":
        frontend = ResonantSTFT(
            1, cfg.n_freqs, resolve_activation(cfg.resonant_activation),
            omega_lo=cfg.omega_lo, omega_hi=cfg.omega_hi,
            init_log_neg_lambda=cfg.resonant_init_log_neg_lambda,
            init_r_lo=cfg.init_r_lo, init_r_hi=cfg.init_r_hi,
            spk_args=spk, generator=generator,
        )
        embed_in = cfg.n_freqs
        if eff_lnl is None:
            eff_lnl = math.log(0.1)        # RNN_KW preset for the body PhasorDense layers
    elif cfg.frontend != "none":
        raise ValueError(f"unknown frontend kind {cfg.frontend!r}")

    input_layer = PhasorDense(
        embed_in, cfg.d_hidden, normalize_to_unit_circle,
        use_bias=False, init_mode=cfg.init_mode, init_log_neg_lambda=eff_lnl,
        spk_args=spk, generator=generator,
    )
    body: Optional[nn.Module]
    if cfg.body == "none":
        body = None
    elif cfg.body == "lsa":
        body = PhasorLSA(
            cfg.d_hidden, cfg.d_hidden, n_heads=cfg.n_heads,
            init_scale=cfg.init_scale, init_mode=cfg.init_mode,
            spk_args=spk, generator=generator,
        )
    elif cfg.body == "lca":
        body = PhasorLCA(
            cfg.d_hidden, cfg.d_hidden, n_heads=cfg.n_heads,
            n_anchors=cfg.n_anchors, init_scale=cfg.init_scale,
            init_mode=cfg.init_mode, spk_args=spk, generator=generator,
        )
    else:
        raise ValueError(f"unknown body kind {cfg.body!r}")

    body_dense = PhasorDense(
        cfg.d_hidden, cfg.d_hidden, activation=nn.Identity(),
        use_bias=False, init_mode=cfg.init_mode, init_log_neg_lambda=eff_lnl,
        spk_args=spk, generator=generator,
    )

    if cfg.readout == "ssm":
        readout = SSMReadout(
            cfg.d_hidden, cfg.n_classes,
            readout_frac=cfg.readout_frac, generator=generator,
        )
    elif cfg.readout == "codebook":
        readout = Codebook(
            cfg.d_hidden, cfg.n_classes,
            init_mode=cfg.codebook_init_mode, generator=generator,
        )
    else:
        raise ValueError(f"unknown readout kind {cfg.readout!r}")

    # Build the ordered schema (used both as the forward chain and as
    # the HDF5-save mapping).
    schema: OrderedDict[str, nn.Module] = OrderedDict()
    if frontend is not None:
        schema["frontend"] = frontend
    schema["input"] = input_layer
    if body is not None:
        schema["body"] = body
    schema["dense"] = body_dense
    schema["readout"] = readout

    model = nn.Sequential(*schema.values())
    return model, schema


# --------------------------------------------------------------------------
# Codebook-readout adapter: collapse (C, L, B) -> (C, B) for 2D Codebook.
# --------------------------------------------------------------------------


def _maybe_collapse_for_codebook(x: Tensor, readout: nn.Module) -> Tensor:
    """If the readout is a 2D-only Codebook, take the temporal mean of x."""
    if isinstance(readout, Codebook) and x.ndim == 3:
        # Use the last 25% of timesteps' angular mean to mirror SSMReadout's
        # window choice; simple temporal pooling on phases isn't well-defined,
        # so use the complex-mean trick: phase -> complex -> mean -> phase.
        from .primitives import angle_to_complex, complex_to_angle
        L = x.shape[1]
        t0 = max(0, L - max(1, round(L * 0.25)))
        z = angle_to_complex(x[:, t0:L, :])
        z_mean = z.mean(dim=1)
        return complex_to_angle(z_mean)
    return x


def forward_model(schema: OrderedDict[str, nn.Module], x: Tensor,
                  downsample_factor: int = 1) -> Tensor:
    """Custom forward that wires the audio frontend glue and the readout adapter.

    The "frontend" (ResonantSTFT) layer is wrapped with the stateless transforms
    that are not nn.Modules: `encode_input` lifts the real waveform to complex
    before it, and `downsample_time` -> `to_phase` follow it so the
    phase-dispatching body can consume the result.
    """
    out = x
    for name, layer in schema.items():
        if name == "frontend":
            out = encode_input(out)
            out = layer(out)
            out = to_phase(downsample_time(out, downsample_factor))
            continue
        if name == "readout":
            out = _maybe_collapse_for_codebook(out, layer)
        out = layer(out)
    return out


# --------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(schema: OrderedDict[str, nn.Module], loader, device: torch.device,
             n_classes: int, downsample_factor: int = 1) -> tuple[float, float]:
    """Return (mean_loss, accuracy) over a loader."""
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        sims = forward_model(schema, x, downsample_factor)
        oh = one_hot(y, n_classes)
        loss = similarity_loss(sims, oh)
        preds = sims.argmax(dim=0)
        total_loss += float(loss.item())
        total_correct += int((preds == y).sum().item())
        total_samples += int(y.numel())
        n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def train(run: RunConfig, *, save_path: Optional[str] = None) -> dict:
    """End-to-end training run. Returns a small metrics summary dict.

    If `save_path` (or run.train.checkpoint_path) is set, the final model
    weights are written to that path in the HDF5 schema that Lux can load.
    """
    device = select_device(run.train.device)
    g = torch.Generator(device="cpu").manual_seed(run.train.seed)

    # --- Data ---------------------------------------------------------
    if run.data.source == "audio":
        if run.model.frontend != "resonant":
            raise ValueError(
                "data.source == 'audio' requires model.frontend == 'resonant'"
            )
        if not run.data.train_path or not run.data.test_path:
            raise ValueError(
                "data.source == 'audio' requires data.train_path and data.test_path"
            )
        from .data import make_audio_dataloaders
        train_loader, test_loader = make_audio_dataloaders(
            run.data.train_path, run.data.test_path, run.model.n_classes,
            run.train.batch_size,
            train_limit=run.data.train_limit, test_limit=run.data.test_limit,
            seed=run.data.seed, generator=g,
        )
    else:
        codebook_g = torch.Generator().manual_seed(run.data.seed)
        codebook = build_codebook(run.data.vocab_size, run.model.in_dims,
                                  generator=codebook_g)
        train_cfg = SequenceTaskConfig(
            task=run.data.task, num_samples=run.data.num_train,
            max_length=run.data.max_length, vocab_size=run.data.vocab_size,
            n_hd=run.model.in_dims, seed=run.data.seed,
        )
        test_cfg = SequenceTaskConfig(
            task=run.data.task, num_samples=run.data.num_test,
            max_length=run.data.max_length, vocab_size=run.data.vocab_size,
            n_hd=run.model.in_dims, seed=run.data.seed + 9999,
        )
        x_tr, y_tr = first_token_classification(train_cfg, codebook)
        x_te, y_te = first_token_classification(test_cfg,  codebook)

        train_loader = make_dataloader(x_tr, y_tr, run.train.batch_size,
                                       shuffle=True, generator=g)
        test_loader = make_dataloader(x_te, y_te, run.train.batch_size,
                                      shuffle=False, drop_last=False)

    # --- Model --------------------------------------------------------
    model, schema = build_model(run.model, generator=g)
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=run.train.lr,
                           weight_decay=run.train.weight_decay)

    ds = run.model.downsample_factor if run.model.frontend == "resonant" else 1

    history: list[dict] = []
    for epoch in range(1, run.train.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        n_batches = 0
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            sims = forward_model(schema, x, ds)
            oh = one_hot(y, run.model.n_classes)
            loss = similarity_loss(sims, oh)
            loss.backward()
            opt.step()
            preds = sims.argmax(dim=0)
            epoch_loss += float(loss.item())
            epoch_correct += int((preds == y).sum().item())
            epoch_samples += int(y.numel())
            n_batches += 1
            if run.train.log_every and (step + 1) % run.train.log_every == 0:
                print(f"epoch {epoch} step {step + 1}: loss {loss.item():.4f}")
        train_loss = epoch_loss / max(1, n_batches)
        train_acc = epoch_correct / max(1, epoch_samples)

        model.eval()
        test_loss, test_acc = evaluate(schema, test_loader, device,
                                       run.model.n_classes, ds)
        row = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss,  "test_acc": test_acc,
        }
        history.append(row)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
              f" | test_loss={test_loss:.4f} test_acc={test_acc:.3f}")

    final_save = save_path or run.train.checkpoint_path
    if final_save is not None:
        from .weights import save_state
        meta = {f"cfg.{k}": str(v) for k, v in asdict(run).items()}
        save_state(final_save, schema, metadata=meta)
        print(f"saved checkpoint to {final_save}")

    return {"history": history, "final": history[-1] if history else None}


# --------------------------------------------------------------------------
# CLI entry
# --------------------------------------------------------------------------


def _parse_argv(argv: list[str]) -> RunConfig:
    """Minimal CLI: `python -m phasor_torch.train --config path.yaml`.
    Without --config falls back to all defaults.
    """
    import argparse
    p = argparse.ArgumentParser(description="Train LSA/LCA phasor SSM (PyTorch).")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (optional).")
    p.add_argument("--save", type=str, default=None,
                   help="HDF5 path to write final weights (optional).")
    ns = p.parse_args(argv)
    if ns.config:
        from .config import from_yaml
        run = from_yaml(ns.config)
    else:
        run = RunConfig()
    if ns.save:
        return run, ns.save
    return run, None


def main(argv: list[str] | None = None) -> dict:
    import sys
    run, save = _parse_argv(argv if argv is not None else sys.argv[1:])
    return train(run, save_path=save)


if __name__ == "__main__":
    main()
