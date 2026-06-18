"""Raw keyword-spotting audio loader.

Reads `sound_data_raw.h5` (and its test mirror) and applies the same
preprocessing as the Julia `load_audio` (scripts/train_audio_ssm_attention.jl):
per-clip RMS normalize, silence skip, clamp + rescale, OOD-label drop. Output
follows this repo's `(C, L, B)` convention with the sample axis LAST, so the
existing `make_dataloader` collate trick applies unchanged.

HDF5 layout (verified on /home/wilkie/data/sound):
  "audio"  : (N, L)  float32   raw waveforms, L = 16000 (fs = 16 kHz, 1 s clips)
  "labels" : (N,)    int       keyword-class id; the test mirror carries an
                                extra OOD "unknown" id == n_classes (dropped).

NOTE the layout is (N, L) — N clips of length L — NOT (L, N) as an early draft
of the feature spec stated. RMS is taken over the L axis (axis=1).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .sequence_tasks import make_dataloader

RMS_FLOOR = 1e-3   # clips quieter than this are skipped (silent -> stays silent)
CLAMP = 5.0        # clamp to +-CLAMP then rescale to +-1


def load_audio(path: str | Path, n_classes: int, *,
               limit: int | None = None, seed: int = 0) -> tuple[Tensor, Tensor]:
    """Load and preprocess raw audio into `(1, L, N)` Phase-ready waveforms.

    Args:
      path:      HDF5 file with "audio" (N, L) and "labels" (N,).
      n_classes: keep only labels in [0, n_classes - 1]; drop the rest (OOD).
      limit:     optional cap on number of clips (random subset; for local smoke
                 runs). Subselection happens before preprocessing/filtering.
      seed:      RNG seed for the `limit` subset.

    Returns:
      x: (1, L, N) float32 — raw single-channel waveforms, sample axis LAST.
      y: (N,) long — class labels in [0, n_classes - 1].
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        labels_all = np.asarray(f["labels"][...])
        N = labels_all.shape[0]
        if limit is not None and limit < N:
            g = np.random.default_rng(seed)
            idx = np.sort(g.choice(N, size=int(limit), replace=False))
            audio = np.asarray(f["audio"][idx, :], dtype=np.float32)
            labels = labels_all[idx]
        else:
            audio = np.asarray(f["audio"][...], dtype=np.float32)
            labels = labels_all

    audio_t = torch.from_numpy(audio).float()                  # (M, L)
    labels_t = torch.from_numpy(np.asarray(labels)).long()     # (M,)

    # Per-clip RMS over the L axis.
    rms = torch.sqrt(torch.mean(audio_t * audio_t, dim=1))     # (M,)

    # Keep audible, in-distribution clips.
    keep = (rms > RMS_FLOOR) & (labels_t >= 0) & (labels_t < int(n_classes))
    audio_t = audio_t[keep]
    labels_t = labels_t[keep]
    rms = rms[keep]

    # RMS normalize, clamp, rescale to [-1, 1].
    audio_t = audio_t / rms.unsqueeze(1)
    audio_t = torch.clamp(audio_t, -CLAMP, CLAMP) / CLAMP

    # (M, L) -> (1, L, M): single channel, sample axis last.
    x = audio_t.transpose(0, 1).unsqueeze(0).contiguous()      # (1, L, M)
    return x, labels_t


def make_audio_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    n_classes: int,
    batch_size: int,
    *,
    train_limit: int | None = None,
    test_limit: int | None = None,
    seed: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, test_loader) from the raw-audio HDF5 files."""
    x_tr, y_tr = load_audio(train_path, n_classes, limit=train_limit, seed=seed)
    x_te, y_te = load_audio(test_path, n_classes, limit=test_limit, seed=seed + 1)
    train_loader = make_dataloader(x_tr, y_tr, batch_size,
                                   shuffle=True, generator=generator)
    test_loader = make_dataloader(x_te, y_te, batch_size,
                                  shuffle=False, drop_last=False)
    return train_loader, test_loader
