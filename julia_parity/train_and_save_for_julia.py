"""Train an LSA chain in PyTorch, save its weights + the test fixtures
that the companion julia_parity/verify_end_to_end.jl needs to verify the
PyTorch -> Julia weight transfer end-to-end.

Outputs:
  fixtures/e2e_chain_weights.h5    — full Chain weights in the HDF5 schema.
  fixtures/e2e_chain_io.h5         — test-set inputs + PyTorch outputs +
                                     codebook + labels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phasor_torch.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from phasor_torch.data import (
    SequenceTaskConfig,
    build_codebook,
    first_token_classification,
    make_dataloader,
)
from phasor_torch.losses import similarity_loss, one_hot
from phasor_torch.train import build_model, evaluate, forward_model, train
from phasor_torch.weights import save_io_pair, save_state


def main(out_dir: str | None = None) -> None:
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train a small LSA chain.
    run = RunConfig(
        model=ModelConfig(in_dims=32, d_hidden=32, body="lsa", n_heads=4,
                          init_mode="hippo", readout="ssm", n_classes=6,
                          readout_frac=0.25, t_period=1.0),
        data=DataConfig(task="copy", vocab_size=6, max_length=12,
                        num_train=512, num_test=128, seed=0),
        train=TrainConfig(batch_size=32, epochs=4, lr=3e-3, device="cpu", seed=0),
    )
    weights_path = out_dir / "e2e_chain_weights.h5"
    io_path = out_dir / "e2e_chain_io.h5"

    # Train and save.
    summary = train(run, save_path=str(weights_path))
    final = summary["final"]
    print(f"final test acc: {final['test_acc']:.3f}")

    # Recreate test set, codebook, and PyTorch eval outputs (no torch RNG drift
    # — first_token_classification is deterministic from cfg.seed).
    cb_g = torch.Generator().manual_seed(run.data.seed)
    codebook = build_codebook(run.data.vocab_size, run.model.in_dims, generator=cb_g)
    test_cfg = SequenceTaskConfig(
        task=run.data.task, num_samples=run.data.num_test,
        max_length=run.data.max_length, vocab_size=run.data.vocab_size,
        n_hd=run.model.in_dims, seed=run.data.seed + 9999,
    )
    x_te, y_te = first_token_classification(test_cfg, codebook)   # x: (C, L, N), y: (N,)

    # Build and reload the saved chain (clean instance, same weights).
    fresh_g = torch.Generator().manual_seed(42)
    model_b, schema_b = build_model(run.model, generator=fresh_g)
    from phasor_torch.weights import load_state
    load_state(str(weights_path), schema_b)

    # Run PyTorch forward (in eval mode) to capture the canonical outputs.
    model_b.eval()
    with torch.no_grad():
        y_sims_te = forward_model(schema_b, x_te)             # (n_classes, N)
        preds = y_sims_te.argmax(dim=0)
        pt_acc = float((preds == y_te).float().mean().item())
        oh = one_hot(y_te, run.model.n_classes)
        pt_loss = float(similarity_loss(y_sims_te, oh).item())
    print(f"saved PyTorch test acc: {pt_acc:.3f} loss: {pt_loss:.4f}")

    # Save IO pair: test inputs, PyTorch outputs (sims), labels.
    meta = {
        "task": "e2e_lsa_copy_first_token",
        "n_classes": str(run.model.n_classes),
        "d_hidden": str(run.model.d_hidden),
        "n_heads": str(run.model.n_heads),
        "in_dims": str(run.model.in_dims),
        "max_length": str(run.data.max_length),
        "vocab_size": str(run.data.vocab_size),
        "t_period": str(run.model.t_period),
        "init_mode": run.model.init_mode,
        "init_scale": str(run.model.init_scale),
        "readout_frac": str(run.model.readout_frac),
        "pt_test_acc": str(pt_acc),
        "pt_test_loss": str(pt_loss),
    }
    save_io_pair(io_path,
                 inputs={"x": x_te, "labels": y_te.float(), "codebook": codebook},
                 outputs={"sims": y_sims_te},
                 metadata=meta)
    print(f"saved {weights_path.name} and {io_path.name}")


if __name__ == "__main__":
    main()
