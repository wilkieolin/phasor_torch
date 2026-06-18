"""Tests for the raw-audio loader preprocessing.

Uses a tiny synthetic HDF5 file (the real /home/wilkie/data/sound files are
~1.5 GB and not appropriate for pytest). Covers: (N, L) layout, per-clip RMS
normalize, silence skip, clamp/rescale, OOD-label drop, and output shape.
"""

import h5py
import numpy as np
import torch

from phasor_torch.data.audio import load_audio, make_audio_dataloaders


def _write_synthetic(path, audio: np.ndarray, labels: np.ndarray) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("audio", data=audio.astype("float32"))
        f.create_dataset("labels", data=labels.astype("int64"))


def test_load_audio_basic_shape_and_range(tmp_path):
    N, L = 6, 64
    rng = np.random.default_rng(0)
    audio = rng.standard_normal((N, L)).astype("float32")
    labels = np.array([0, 1, 2, 0, 1, 2])
    p = tmp_path / "a.h5"
    _write_synthetic(p, audio, labels)

    x, y = load_audio(p, n_classes=3)
    assert x.shape == (1, L, N)          # (C=1, L, N) sample axis last
    assert y.shape == (N,)
    assert x.dtype == torch.float32
    assert float(x.min()) >= -1.0 and float(x.max()) <= 1.0
    assert torch.isfinite(x).all()


def test_load_audio_skips_silent_clips(tmp_path):
    N, L = 5, 32
    rng = np.random.default_rng(1)
    audio = rng.standard_normal((N, L)).astype("float32")
    audio[2] = 0.0                       # exact silence -> RMS 0 -> skipped
    audio[4] = 1e-5                      # near silence -> RMS < 1e-3 -> skipped
    labels = np.arange(N) % 3
    p = tmp_path / "b.h5"
    _write_synthetic(p, audio, labels)

    x, y = load_audio(p, n_classes=3)
    assert x.shape[2] == 3              # 2 silent clips dropped
    assert torch.isfinite(x).all()


def test_load_audio_drops_ood_labels(tmp_path):
    N, L = 6, 32
    rng = np.random.default_rng(2)
    audio = rng.standard_normal((N, L)).astype("float32")
    labels = np.array([0, 1, 2, 3, 4, 30])   # n_classes=3 -> keep only 0,1,2
    p = tmp_path / "c.h5"
    _write_synthetic(p, audio, labels)

    x, y = load_audio(p, n_classes=3)
    assert x.shape[2] == 3
    assert int(y.max()) < 3 and int(y.min()) >= 0


def test_load_audio_rms_normalized(tmp_path):
    # A loud and a quiet clip should both come out at comparable scale
    # (post-RMS-normalize, pre-clamp the RMS is 1; clamp rarely bites Gaussians).
    N, L = 2, 256
    rng = np.random.default_rng(3)
    base = rng.standard_normal((1, L)).astype("float32")
    audio = np.concatenate([base * 0.01, base * 10.0], axis=0)  # same shape, diff gain
    labels = np.array([0, 1])
    p = tmp_path / "d.h5"
    _write_synthetic(p, audio, labels)

    x, _ = load_audio(p, n_classes=2)
    # After RMS normalize the two clips are identical up to clamp; check their
    # std ratio is ~1 (was 1000x apart before).
    c0 = x[0, :, 0]
    c1 = x[0, :, 1]
    ratio = float(c1.std() / c0.std())
    assert 0.5 < ratio < 2.0


def test_make_audio_dataloaders(tmp_path):
    N, L = 16, 64
    rng = np.random.default_rng(4)
    audio = rng.standard_normal((N, L)).astype("float32")
    labels = np.arange(N) % 4
    tr = tmp_path / "tr.h5"
    te = tmp_path / "te.h5"
    _write_synthetic(tr, audio, labels)
    _write_synthetic(te, audio[:8], labels[:8])

    train_loader, test_loader = make_audio_dataloaders(
        tr, te, n_classes=4, batch_size=4)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (1, L, 4)        # (C, L, B)
    assert yb.shape == (4,)
