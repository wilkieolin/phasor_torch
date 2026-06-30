# phasor_torch

A PyTorch trainer for phasor SSM networks with **Local Self-Attention (LSA)** and **Local Cross-Attention (LCA)** bodies, built so that weights trained on accelerators with broad PyTorch support (e.g. **Intel PVC / `xpu` on Aurora**) can be loaded back into the canonical Julia reference [PhasorNetworks.jl](https://github.com/wilkieolin/PhasorNetworks.jl) for deployment or further analysis.

The Julia package is the source of truth for the math. This repo is a **training-only port** of the discrete (3D Phase / 3D Complex) forward path through the LSA/LCA topology — no ODE solver, no spiking dispatch, no EP/hEP. It covers the core attention bodies, a **depth-robust ReZero residual stack**, a trainable **ResonantSTFT audio frontend**, a real keyword-spotting **audio data path**, and an **Aurora HPO harness** — all gated by a Julia parity verifier.

## Layout

```
phasor_torch/                # Python package
├── primitives.py            # angle_to_complex, complex_to_angle, normalize_to_unit_circle,
│                            #   soft_normalize_to_unit_circle, remap_phase, v_bind, freq_shift,
│                            #   similarity, similarity_outer_heads
│                            #   + autograd.Function shims for NaN guards and memory-efficient sim
├── kernels.py               # phasor_kernel, causal_conv (Toeplitz / FFT), causal_conv_dirac,
│                            #   bias_kernel_accumulation, HiPPO
├── init.py                  # random_symbols, orthogonal_codes
├── layers/
│   ├── phasor_dense.py      # 2D + 3D Phase forward (the projection workhorse)
│   ├── codebook.py          # Fixed-codes 2D classifier
│   ├── ssm_readout.py       # Temporal-window 3D classifier
│   ├── phasor_lsa.py        # Local Self-Attention (head-axis similarity)
│   ├── phasor_lca.py        # Local Cross-Attention (anchor bank, Hopfield-style)
│   ├── phasor_residual.py   # PhaseRecenter / PhasorResidual / PhasorTransformerBlock (ReZero)
│   └── resonant_stft.py     # Trainable resonant filterbank — audio frontend (per-channel ω)
├── data/
│   ├── sequence_tasks.py    # Copy / reversal / retrieval / sorting + first-token-recall wrapper
│   └── audio.py             # Raw-waveform loader (RMS-norm, clamp, OOD drop) for sound_data_raw.h5
├── losses.py                # similarity_loss, codebook_loss, accuracy
├── config.py                # ModelConfig / DataConfig / TrainConfig / RunConfig (+ YAML loader)
├── train.py                 # Adam loop, evaluate, early-stop, checkpoint, CLI
├── confirm.py               # Full-data top-K confirmation runner for HPO winners
├── hpo.py                   # ytopt HPO driver (Ray evaluator) — search space + objective
├── hpo_libe.py              # libEnsemble launcher for multi-node / multi-tile Aurora scale-out
├── weights.py               # HDF5 round trip (nested layer paths supported)
└── tests/                   # 16 test files, 160+ pytest cases

julia_parity/                # Verification harness — PyTorch → Julia round trip
├── load_pytorch.jl          # HDF5 → Lux NamedTuple
├── generate_parity_*.py     # PyTorch-side fixture generators (one per layer)
├── verify_*.jl              # Per-layer + ReZero residual + ResonantSTFT + end-to-end + audio-e2e
└── Project.toml             # PhasorNetworks referenced via sibling path

configs/                     # YAML run configs (audio LCA/LSA small + Aurora, long_fft)
scripts/                     # Aurora PBS jobs (HPO, confirm, bench), XPU smoke + affinity helpers
```

## Quickstart (local CPU / CUDA)

```bash
# 1. PyTorch env. Stock PyTorch >= 2.10 works on any CPU/CUDA backend.
#    On Aurora compute nodes:  module load frameworks
pip install -e .[dev]

# 2. Run the PyTorch test suite.
pytest

# 3. End-to-end LSA training on the synthetic copy task (Adam, CPU; tweak via config).
python -m phasor_torch.train

# 4. Train + save a checkpoint Lux can read.
python -m phasor_torch.train --save model.h5

# 5. Train from a YAML config (audio or sequence task).
python -m phasor_torch.train --config configs/audio_lca_small.yaml --save run.h5
```

## Model bodies

The canonical chain is `input PhasorDense → body → body PhasorDense → readout → similarity_loss`,
wired config-driven in `train.build_model`. Three things are worth calling out beyond the base
LSA/LCA bodies:

### Depth-robust ReZero residual blocks

Stacked phase attention only trains at depth when each attention sublayer is wrapped in a residual
with a **ReZero gate** (a learnable scalar `alpha` init ≈ 0 → exact identity at init).
`PhasorTransformerBlock` = `PhasorResidual(attn) → PhasorResidual(FFN)`, combined via `v_bind`
(phase addition with a straight-through wrap). The residual combine — *not* weight down-scaling — is
what lets attention stack past depth ~2. Enable with `ModelConfig.block_type="rezero"` (coexists with
the default `"plain"` `body→dense` stacking). ReZero `alpha` params train at `lr * alpha_lr_mult`
(5× by default) via a dedicated optimizer group. Ported from Julia `src/ssm.jl`.

### ResonantSTFT audio frontend

`ResonantSTFT` is a bank of trainable damped oscillators that turns a raw waveform into a phase-coded
time-frequency representation the LSA/LCA body can consume. It is the **one documented exception** to
the port's shared-ω rule: its `omega` is a *trainable, per-channel* `nn.Parameter`, and it re-encodes
its output back onto the shared downstream carrier `ω_out = 2π/t_period` via `freq_shift` so
downstream layers resume phase-locked operation. The audio chain is
`encode_input → ResonantSTFT → downsample_time → to_phase → body → readout`. Enable with
`ModelConfig.frontend="resonant"` and `DataConfig.source="audio"`. Ported from Julia
`src/network.jl` (3D-Complex dispatch only).

## Audio data + HPO

Real keyword-spotting audio lives in `sound_data_raw.h5` (`"audio"` `(L, N)` float32, 16 kHz / 1 s
clips; `"labels"` int; 30 in-distribution classes + an OOD "unknown" label in the test set).
`data/audio.py` matches the Julia `load_audio` preprocessing (per-clip RMS-normalize, skip silent
clips, clamp ±5 → rescale to ±1, drop OOD labels) and emits `(C, L, B)` batches.

HPO is driven by **ytopt** (not a daemon-based sweeper): `hpo.py` exposes a `Problem` object + search
space and objective (maximize `final["test_acc"]`); `hpo_libe.py` wraps it in a **libEnsemble**
persistent-generator launcher for embarrassingly-parallel trials — **one trial per MPI rank, pinned
to one XPU tile** — across Aurora nodes. (The Ray evaluator path works locally; libEnsemble is the
PBS/multi-node path.) `confirm.py` re-runs the top-K winners on the full dataset. See
`AURORA_HPO_FEATURES.md` for the full design and `scripts/hpo_aurora*.pbs` for launch templates.

## Aurora (Intel PVC / xpu)

Per the [ALCF PyTorch docs](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/) the
`frameworks/2025.3.1` module provides PyTorch `2.10.0a` with XPU support upstreamed (no
`intel_extension_for_pytorch` import required). On a compute node:

```bash
module load frameworks
cd /path/to/phasor_torch
PYTHONPATH=. python -m phasor_torch.train   # train.select_device('auto') resolves to xpu
```

`train.select_device` resolves `"auto"` → `xpu` if `torch.xpu.is_available()`, else `cuda`, else
`cpu`. For HPO scale-out, the libEnsemble driver pins each rank to an XPU tile via
`PALS_LOCAL_RANKID` (the ALCF PALS launcher) — see `scripts/set_affinity_xpu_aurora.sh` and
`scripts/aurora_xpu_smoke.py`. Single-model multi-tile DDP (init with `backend='xccl'`,
wrap-then-move-to-XPU) is **not** wired in — the HPO path is embarrassingly parallel and doesn't need
it.

## Weight compatibility (PyTorch → Julia)

The HDF5 schema mirrors Lux's `Chain` NamedTuple layout — one HDF5 group per layer, one dataset per
parameter:

```
/input/weight              (D_hidden, C_in)        float32
/input/log_neg_lambda      (D_hidden,)             float32
/body/q_proj/weight        (D_hidden, D_hidden)    float32
/body/q_proj/log_neg_lambda
/body/k_proj/... v_proj/...
/body/scale                ()                       float32
/body/anchors              (D_hidden, n_anchors)    float32  Phase semantics (LCA only)
/dense/weight              ...
/readout/codes             (D_hidden, n_classes)    float32  Phase semantics
```

To verify a model trained in PyTorch reproduces the same outputs in Julia:

```bash
# PyTorch side: train + save fixtures.
python julia_parity/train_and_save_for_julia.py

# Julia side: instantiate (one-time) and verify.
julia --project=julia_parity -e 'using Pkg; Pkg.instantiate()'
julia --project=julia_parity julia_parity/verify_end_to_end.jl
```

The verifier rebuilds the equivalent Lux `Chain`, loads the PyTorch HDF5 weights, runs forward on the
saved test inputs, and asserts the similarity scores and argmax accuracy match (≤5e-3 score
tolerance, ≤1 pp accuracy tolerance).

## Validation status

All Julia parity scripts pass at float32 tolerance:

| Layer / chain | Cases | Tolerance |
|---|---|---|
| PhasorDense | 5 | 1e-5 |
| Codebook + SSMReadout | 6 | 1e-5 |
| PhasorLSA | 4 | 5e-4 long (FFT) / 1e-5 short (Toeplitz) |
| PhasorLCA | 4 | 5e-4 long / 1e-5 short |
| PhasorResidual / PhasorTransformerBlock | ✓ | 5e-4 long / 1e-5 short |
| ResonantSTFT | ✓ | 5e-4 long / 1e-5 short |
| End-to-end chain | 1 | 5e-3 score / 1 pp accuracy |
| Audio end-to-end (from checkpoint) | 1 | 5e-3 score / 1 pp accuracy |

(`verify_*.jl` are the canonical correctness gate; internal pytest consistency is necessary but not
sufficient.)

## What's intentionally NOT in this port

- ODE solver and spiking path (`SpikingCall`, `CurrentCall`, `oscillator_bank`) — discrete 3D Phase /
  3D Complex is sufficient for the LSA/LCA/audio archs. Only `ResonantSTFT`'s discrete dispatches are
  ported, not its spiking return-types.
- Equilibrium Propagation / Holomorphic EP (`src/ep.jl`, `src/hep.jl`).
- `AttractorPhasorSSM`, `SSMCrossAttention`, `SSMSelfAttention`, `PhasorAttention`, `PhasorResonant`,
  `PhasorConv`, `PhasorFixed`, `ComplexBias`.
- The `precompute_stft.jl` complex-STFT cache path and librosa/MFCC frontends — the trainable
  `ResonantSTFT` is the frontend the LCA/LSA audio archs require.
- Multi-backend abstraction (`select_device(:cuda|:cpu|:oneapi)`) — PyTorch handles devices natively.
- Single-model multi-tile DDP — the HPO path is embarrassingly parallel.
- Custom CUDA / Triton / KernelAbstractions kernels — vectorized PyTorch ops suffice in this regime.

If you need any of those in PyTorch, this repo is a good starting point but expect to fork in new
layers under `layers/` and add matching parity scripts under `julia_parity/`.
