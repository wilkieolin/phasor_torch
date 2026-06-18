"""Sequence task data generators + symbol-to-phase codebook embedding.

Ports the Julia generators in test/data.jl (copy / reversal / retrieval /
sorting / pattern), then wraps them into a single-class supervised dataset
("first-token recall") that fits our LSA/LCA classification topology.

Embedding layout: (C_in, L, B) Phase tensors.
- C_in = n_hd     (codebook dimensionality)
- L    = max_length     (padded sequence length)
- B    = batch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from ..init import random_symbols


# --------------------------------------------------------------------------
# Symbol codebook (token id -> n_hd Phase vector)
# --------------------------------------------------------------------------


def build_codebook(vocab_size: int, n_hd: int, *,
                   generator: torch.Generator | None = None) -> Tensor:
    """Build a (n_hd, vocab_size + 1) Phase codebook.

    Column 0 is the zero/pad symbol (all zeros, the silent reference);
    columns 1..vocab_size are random unit-modulus phase symbols. Mirrors
    Julia generate_codebook (test/data.jl:64) but with the pad-as-column-0
    convention used throughout this trainer.
    """
    cols = random_symbols((n_hd, vocab_size), generator=generator)   # (n_hd, vocab)
    pad = torch.zeros(n_hd, 1, dtype=torch.float32)
    return torch.cat([pad, cols], dim=1)                              # (n_hd, vocab + 1)


def embed_tokens(seqs: Tensor, codebook: Tensor) -> Tensor:
    """Map a batch of token-id sequences to Phase (C_in, L, B) layout.

    Args:
      seqs:      (B, L) long, token ids in [0, vocab_size] (0 = pad).
      codebook:  (n_hd, vocab_size + 1) Phase tensor (column 0 = zero pad).
    Returns:
      (n_hd, L, B) Phase float32.
    """
    assert seqs.ndim == 2, f"expected (B, L), got {tuple(seqs.shape)}"
    # codebook[:, seqs] -> (n_hd, B, L); permute to (n_hd, L, B)
    embedded = codebook[:, seqs.long()]                                # (n_hd, B, L)
    return embedded.permute(0, 2, 1).contiguous()                      # (n_hd, L, B)


# --------------------------------------------------------------------------
# Raw sequence generators (port of test/data.jl)
# --------------------------------------------------------------------------


def _rng(seed: int | None) -> torch.Generator:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return g


def generate_copy(num_samples: int, max_length: int, vocab_size: int,
                  min_length: int = 5, seed: int | None = None
                  ) -> tuple[Tensor, Tensor]:
    """Copy task: target == input. Returns (inputs, targets) of shape (N, L)."""
    g = _rng(seed)
    inputs = torch.zeros(num_samples, max_length, dtype=torch.long)
    lengths = torch.randint(min_length, max_length + 1, (num_samples,), generator=g)
    for i in range(num_samples):
        L = int(lengths[i].item())
        inputs[i, :L] = torch.randint(1, vocab_size + 1, (L,), generator=g)
    return inputs, inputs.clone()


def generate_reversal(num_samples: int, max_length: int, vocab_size: int,
                      min_length: int = 5, seed: int | None = None
                      ) -> tuple[Tensor, Tensor]:
    g = _rng(seed)
    inputs = torch.zeros(num_samples, max_length, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    lengths = torch.randint(min_length, max_length + 1, (num_samples,), generator=g)
    for i in range(num_samples):
        L = int(lengths[i].item())
        seq = torch.randint(1, vocab_size + 1, (L,), generator=g)
        inputs[i, :L] = seq
        targets[i, :L] = seq.flip(0)
    return inputs, targets


def generate_retrieval(num_samples: int, context_length: int, vocab_size: int,
                       special_token: int = 999, seed: int | None = None
                       ) -> tuple[Tensor, Tensor]:
    """Retrieval: context...needle...query special_token + position -> needle.

    Returns (inputs, targets). Inputs have shape (N, context_length + 2);
    targets are single-token (N, 1) values.
    """
    g = _rng(seed)
    L_total = context_length + 2
    inputs = torch.zeros(num_samples, L_total, dtype=torch.long)
    targets = torch.zeros(num_samples, 1, dtype=torch.long)
    for i in range(num_samples):
        haystack = torch.randint(1, vocab_size + 1, (context_length - 1,), generator=g)
        pos = int(torch.randint(0, context_length - 1, (1,), generator=g).item())
        needle = int(torch.randint(1, vocab_size + 1, (1,), generator=g).item())
        # Insert needle at position `pos` (0-indexed).
        haystack = torch.cat([haystack[:pos],
                              torch.tensor([needle]),
                              haystack[pos:]])
        inputs[i, :context_length] = haystack
        inputs[i, context_length] = special_token
        inputs[i, context_length + 1] = pos + 1
        targets[i, 0] = needle
    return inputs, targets


def generate_sorting(num_samples: int, max_length: int, vocab_size: int,
                     min_length: int = 5, seed: int | None = None
                     ) -> tuple[Tensor, Tensor]:
    g = _rng(seed)
    inputs = torch.zeros(num_samples, max_length, dtype=torch.long)
    targets = torch.zeros_like(inputs)
    lengths = torch.randint(min_length, max_length + 1, (num_samples,), generator=g)
    for i in range(num_samples):
        L = int(lengths[i].item())
        seq = torch.randint(1, vocab_size + 1, (L,), generator=g)
        inputs[i, :L] = seq
        sorted_seq, _ = seq.sort()
        targets[i, :L] = sorted_seq
    return inputs, targets


# --------------------------------------------------------------------------
# Classification-shaped wrappers (compatible with SSMReadout / Codebook)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceTaskConfig:
    task: str            # 'copy' | 'reversal' | 'retrieval' | 'sorting'
    num_samples: int
    max_length: int
    vocab_size: int
    n_hd: int            # codebook dimensionality (== model input C_in)
    seed: int = 0


def _generate_raw(cfg: SequenceTaskConfig) -> tuple[Tensor, Tensor]:
    g = cfg.seed
    if cfg.task == "copy":
        return generate_copy(cfg.num_samples, cfg.max_length, cfg.vocab_size, seed=g)
    if cfg.task == "reversal":
        return generate_reversal(cfg.num_samples, cfg.max_length, cfg.vocab_size, seed=g)
    if cfg.task == "retrieval":
        # context_length and vocab_size are interpreted as: max_length is context_length.
        return generate_retrieval(cfg.num_samples, cfg.max_length, cfg.vocab_size, seed=g)
    if cfg.task == "sorting":
        return generate_sorting(cfg.num_samples, cfg.max_length, cfg.vocab_size, seed=g)
    raise ValueError(f"unknown task {cfg.task!r}")


def first_token_classification(cfg: SequenceTaskConfig, codebook: Tensor
                                ) -> tuple[Tensor, Tensor]:
    """Wrap a raw generator as a classification dataset.

    Label = the first token of each generated sequence (a value in
    [1, vocab_size]). The full sequence is embedded; the model must learn
    to recall the first token from its representation. Mirrors the
    `first_element_recall` task in scripts/long_range_ssm.jl.

    Returns:
      x: (n_hd, L, N) Phase float32 - embedded sequences
      y: (N,) long    - class labels in [0, vocab_size - 1]
    """
    inputs, _targets = _generate_raw(cfg)                  # (N, L) long
    embedded = embed_tokens(inputs, codebook)              # (n_hd, L, N)
    # First token is in column 0; subtract 1 to make class IDs 0..vocab_size-1.
    labels = (inputs[:, 0] - 1).long()
    return embedded, labels


# --------------------------------------------------------------------------
# DataLoader helper
# --------------------------------------------------------------------------


def make_dataloader(x: Tensor, y: Tensor, batch_size: int,
                    shuffle: bool = True, drop_last: bool = True,
                    generator: torch.Generator | None = None) -> DataLoader:
    """Wrap (x, y) into a TensorDataset DataLoader.

    Note: x is shape (C, L, N) — the SAMPLE axis is the LAST one to match
    the (C, L, B) layout used by every layer. We work around DataLoader's
    "sample axis = 0" assumption by stuffing the samples into the first
    axis here and then transposing back in the collate step. To keep things
    simple we use a custom collate that splits along the N axis.
    """
    C, L, N = x.shape
    assert y.shape == (N,), f"y shape {tuple(y.shape)} != ({N},)"
    # Permute to (N, C, L) so DataLoader sees N as the sample axis.
    x_nfirst = x.permute(2, 0, 1).contiguous()
    dataset = TensorDataset(x_nfirst, y)

    def _collate(samples):
        xs = torch.stack([s[0] for s in samples], dim=0)   # (B, C, L)
        ys = torch.stack([s[1] for s in samples], dim=0)
        # Back to (C, L, B) for the model.
        return xs.permute(1, 2, 0).contiguous(), ys

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, collate_fn=_collate,
        generator=generator,
    )
