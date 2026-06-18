"""Loss + accuracy helpers for similarity-based classification.

Ports of similarity_loss (src/metrics.jl), specialized to the readout
output shape (n_classes, B) produced by SSMReadout and Codebook.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def similarity_loss(sims: Tensor, targets_onehot: Tensor) -> Tensor:
    """`2 * sin^2(pi/4 * dist)` where dist = sum(|1 - sim| * truth) per sample.

    Mirrors Julia src/metrics.jl similarity_loss. Reduces to a per-sample
    nonneg loss; averaged over the batch dim.

    Args:
      sims:           (n_classes, B) real, similarity scores in [-1, 1].
      targets_onehot: (n_classes, B) one-hot float.

    Returns:
      Scalar mean loss.
    """
    assert sims.shape == targets_onehot.shape
    dist = ((1.0 - sims).abs() * targets_onehot).sum(dim=0)     # (B,)
    return (2.0 * torch.sin(math.pi * 0.25 * dist) ** 2).mean()


def codebook_loss(sims: Tensor, targets_onehot: Tensor) -> Tensor:
    """Alias for similarity_loss — Codebook + SSMReadout both use the same loss."""
    return similarity_loss(sims, targets_onehot)


def one_hot(labels: Tensor, n_classes: int) -> Tensor:
    """labels: (B,) long -> (n_classes, B) float one-hot."""
    onehot = torch.zeros(n_classes, labels.shape[0],
                         dtype=torch.float32, device=labels.device)
    onehot[labels, torch.arange(labels.shape[0], device=labels.device)] = 1.0
    return onehot


def accuracy(sims: Tensor, labels: Tensor) -> float:
    """argmax over classes vs integer labels. Returns fraction correct."""
    preds = sims.argmax(dim=0)                                  # (B,)
    return (preds == labels).float().mean().item()
