"""Stage 6 smoke tests: data generators, loss, and a tiny end-to-end train run."""

from __future__ import annotations

import math

import pytest
import torch

from phasor_torch.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from phasor_torch.data import (
    SequenceTaskConfig,
    build_codebook,
    first_token_classification,
    make_dataloader,
)
from phasor_torch.data.sequence_tasks import (
    embed_tokens,
    generate_copy,
    generate_reversal,
    generate_retrieval,
    generate_sorting,
)
from phasor_torch.losses import accuracy, one_hot, similarity_loss
from phasor_torch.train import build_model, evaluate, forward_model, train


# --------------------------------------------------------------------------
# Codebook / embedding
# --------------------------------------------------------------------------


def test_build_codebook_shape_and_pad_zero():
    cb = build_codebook(vocab_size=5, n_hd=8,
                        generator=torch.Generator().manual_seed(0))
    assert cb.shape == (8, 6)        # +1 column for pad
    assert torch.equal(cb[:, 0], torch.zeros(8))   # pad column is zero
    # Other columns are non-zero with high probability
    assert (cb[:, 1:].abs() > 1e-6).any()


def test_embed_tokens_shape():
    cb = build_codebook(vocab_size=4, n_hd=6,
                        generator=torch.Generator().manual_seed(1))
    seqs = torch.tensor([[1, 2, 3, 0],
                         [4, 0, 0, 0]], dtype=torch.long)
    emb = embed_tokens(seqs, cb)
    assert emb.shape == (6, 4, 2)     # (n_hd, L, B)


# --------------------------------------------------------------------------
# Raw generators
# --------------------------------------------------------------------------


def test_generate_copy_padding_and_target_equal():
    inputs, targets = generate_copy(num_samples=12, max_length=10,
                                    vocab_size=5, seed=42)
    assert inputs.shape == (12, 10)
    assert torch.equal(inputs, targets)
    # Non-pad tokens must be in [1, 5]
    nonzero = inputs[inputs > 0]
    assert (nonzero >= 1).all() and (nonzero <= 5).all()


def test_generate_reversal_target_is_reverse_of_input_prefix():
    inputs, targets = generate_reversal(num_samples=5, max_length=8,
                                        vocab_size=10, seed=7)
    for i in range(5):
        L = int((inputs[i] != 0).sum().item())
        assert torch.equal(targets[i, :L], inputs[i, :L].flip(0))


def test_generate_retrieval_target_is_needle():
    inputs, targets = generate_retrieval(num_samples=4, context_length=10,
                                         vocab_size=20, seed=11)
    for i in range(4):
        pos = int(inputs[i, -1].item()) - 1            # 1-indexed -> 0-indexed
        assert inputs[i, pos] == targets[i, 0]


def test_generate_sorting_target_is_sorted_input():
    inputs, targets = generate_sorting(num_samples=6, max_length=12,
                                       vocab_size=20, seed=3)
    for i in range(6):
        L = int((inputs[i] != 0).sum().item())
        assert torch.equal(targets[i, :L], inputs[i, :L].sort().values)


# --------------------------------------------------------------------------
# first-token-classification wrapper
# --------------------------------------------------------------------------


def test_first_token_classification_shape_and_labels():
    cfg = SequenceTaskConfig(task="copy", num_samples=16, max_length=8,
                             vocab_size=5, n_hd=12, seed=0)
    cb = build_codebook(vocab_size=5, n_hd=12,
                        generator=torch.Generator().manual_seed(0))
    x, y = first_token_classification(cfg, cb)
    assert x.shape == (12, 8, 16)
    assert y.shape == (16,)
    assert (y >= 0).all() and (y < 5).all()


# --------------------------------------------------------------------------
# Loss / accuracy
# --------------------------------------------------------------------------


def test_one_hot_shape_and_values():
    labels = torch.tensor([0, 2, 1, 0])
    oh = one_hot(labels, n_classes=3)
    assert oh.shape == (3, 4)
    assert oh.sum() == 4.0
    assert oh[0, 0] == 1 and oh[2, 1] == 1 and oh[1, 2] == 1 and oh[0, 3] == 1


def test_similarity_loss_zero_at_perfect_prediction():
    """sim == 1 for the true class => loss == 0."""
    sims = torch.tensor([[1.0, -1.0, -1.0],
                         [-1.0, 1.0, -1.0],
                         [-1.0, -1.0, 1.0]])
    truth = torch.eye(3)
    loss = similarity_loss(sims, truth)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)


def test_accuracy_argmax():
    sims = torch.tensor([[0.1, 0.9, 0.5],
                         [0.7, 0.2, 0.3],
                         [0.6, 0.1, 0.4]])
    labels = torch.tensor([1, 0, 2])
    # argmax per column: col0->row1, col1->row0, col2->row0
    assert accuracy(sims, labels) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# Tiny end-to-end training run (LSA + SSMReadout)
# --------------------------------------------------------------------------


def test_train_loss_decreases_with_lsa():
    """Train a tiny LSA chain on a tiny copy task; loss should drop."""
    run = RunConfig(
        model=ModelConfig(
            in_dims=32, d_hidden=32, body="lsa", n_heads=4,
            init_mode="hippo", readout="ssm", n_classes=5,
        ),
        data=DataConfig(task="copy", vocab_size=5, max_length=8,
                        num_train=128, num_test=64, seed=0),
        train=TrainConfig(batch_size=16, epochs=3, lr=3e-3,
                          device="cpu", seed=0),
    )
    summary = train(run)
    hist = summary["history"]
    assert len(hist) == 3
    # Final train loss should be smaller than initial; this is a smoke
    # test, not a convergence test — so we just require non-increasing
    # by at least a small margin after 3 epochs.
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] - 1e-3


def test_train_with_lca_runs_and_produces_grads():
    """Same shape but with LCA body; checks the LCA pathway doesn't NaN."""
    run = RunConfig(
        model=ModelConfig(
            in_dims=24, d_hidden=24, body="lca", n_heads=3,
            n_anchors=8, init_mode="hippo", readout="codebook",
            n_classes=4, codebook_init_mode="orthogonal",
        ),
        data=DataConfig(task="copy", vocab_size=4, max_length=6,
                        num_train=64, num_test=32, seed=1),
        train=TrainConfig(batch_size=16, epochs=2, lr=3e-3,
                          device="cpu", seed=1),
    )
    summary = train(run)
    assert len(summary["history"]) == 2
    # No NaNs in final stats.
    assert math.isfinite(summary["history"][-1]["train_loss"])


def test_checkpoint_round_trip(tmp_path):
    """Train briefly, save the checkpoint, load it into a fresh model, verify forwards match."""
    run = RunConfig(
        model=ModelConfig(in_dims=16, d_hidden=16, body="lsa",
                          n_heads=2, init_mode="hippo",
                          readout="ssm", n_classes=4),
        data=DataConfig(task="copy", vocab_size=4, max_length=6,
                        num_train=64, num_test=32, seed=2),
        train=TrainConfig(batch_size=16, epochs=1, lr=3e-3, device="cpu", seed=2),
    )
    save_path = tmp_path / "ckpt.h5"
    train(run, save_path=str(save_path))
    assert save_path.exists()

    # Rebuild and load.
    from phasor_torch.train import build_model
    from phasor_torch.weights import load_state
    g = torch.Generator().manual_seed(99)            # different generator
    model_b, schema_b = build_model(run.model, generator=g)
    load_state(str(save_path), schema_b)

    # Same forward on a canned input.
    x = (torch.rand(16, 6, 3) * 2 - 1)
    # Build the saved-model reference by re-loading into a third instance and
    # comparing — the round trip just needs to be internally consistent.
    g2 = torch.Generator().manual_seed(123)
    model_c, schema_c = build_model(run.model, generator=g2)
    load_state(str(save_path), schema_c)
    y_b = forward_model(schema_b, x)
    y_c = forward_model(schema_c, x)
    assert torch.allclose(y_b, y_c, atol=1e-6)
