# phasor_torch

A PyTorch trainer for phasor SSM networks with **Local Self-Attention (LSA)** and **Local Cross-Attention (LCA)** bodies, built so that weights trained on accelerators with broad PyTorch support (e.g. **Intel PVC / `xpu` on Aurora**) can be loaded back into the canonical Julia reference [PhasorNetworks.jl](https://github.com/wilkieolin/PhasorNetworks.jl) for deployment or further analysis.

The Julia package is the source of truth for the math. This repo is a **training-only port** of the discrete (3D Phase) forward path through the LSA/LCA topology — no ODE solver, no spiking dispatch, no EP/hEP — covering ~2300 LOC across the PyTorch package and the Julia parity harness.

## Layout

```
phasor_torch/                # Python package
├── primitives.py            # angle_to_complex, complex_to_angle, normalize_to_unit_circle,
│                            #   remap_phase, similarity, similarity_outer_heads
│                            #   + autograd.Function shims for NaN guards and memory-efficient sim
├── kernels.py               # phasor_kernel, causal_conv (Toeplitz / FFT), causal_conv_dirac, HiPPO
├── init.py                  # random_symbols, orthogonal_codes
├── layers/
│   ├── phasor_dense.py      # 2D + 3D Phase forward (the projection workhorse)
│   ├── codebook.py          # Fixed-codes 2D classifier
│   ├── ssm_readout.py       # Temporal-window 3D classifier
│   ├── phasor_lsa.py        # Local Self-Attention (head-axis similarity)
│   └── phasor_lca.py        # Local Cross-Attention (anchor bank, Hopfield-style)
├── data/sequence_tasks.py   # Copy / reversal / retrieval / sorting + first-token-recall wrapper
├── losses.py                # similarity_loss, codebook_loss, accuracy
├── config.py                # ModelConfig / DataConfig / TrainConfig (+ YAML loader)
├── train.py                 # Adam loop, evaluate, checkpoint, CLI
├── weights.py               # HDF5 round trip (nested layer paths supported)
└── tests/                   # 91 pytest cases

julia_parity/                # Verification harness — PyTorch → Julia round trip
├── load_pytorch.jl          # HDF5 → Lux NamedTuple
├── generate_parity_*.py     # PyTorch-side fixture generators (one per layer)
├── verify_*.jl              # Julia-side verifiers (5 + 6 + 4 + 4 + end-to-end)
└── Project.toml             # PhasorNetworks referenced via sibling path
```

## Quickstart (local CPU / CUDA)

```bash
# 1. PyTorch env. Stock PyTorch >= 2.10 works on any CPU/CUDA backend.
#    On Aurora compute nodes:  module load frameworks
pip install -e .[dev]

# 2. Run the PyTorch test suite.
pytest                                       # 91 tests

# 3. End-to-end LSA training (Adam, CPU; tweak via config).
python -m phasor_torch.train

# 4. Train + save a checkpoint Lux can read.
python -m phasor_torch.train --save model.h5
```

## Aurora (Intel PVC / xpu)

Per the [ALCF PyTorch docs](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/) the `frameworks/2025.3.1` module provides PyTorch `2.10.0a` with XPU support upstreamed (no `intel_extension_for_pytorch` import required). On a compute node:

```bash
module load frameworks
cd /path/to/phasor_torch
PYTHONPATH=. python -m phasor_torch.train   # train.select_device('auto') resolves to xpu
```

`train.select_device` resolves `"auto"` → `xpu` if `torch.xpu.is_available()`, else `cuda`, else `cpu`. Multi-tile / multi-rank DDP (init with `backend='xccl'`, pin via `torch.xpu.set_device(LOCAL_RANK)`) is not wired into the trainer yet — straightforward extension if you need it.

## Weight compatibility (PyTorch → Julia)

The HDF5 schema mirrors Lux's `Chain` NamedTuple layout — one HDF5 group per layer, one dataset per parameter:

```
/input/weight              (D_hidden, C_in)        float32
/input/log_neg_lambda      (D_hidden,)             float32
/body/q_proj/weight        (D_hidden, D_hidden)    float32
/body/q_proj/log_neg_lambda
/body/k_proj/... v_proj/...
/body/scale                ()                       float32
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

The verifier rebuilds the equivalent Lux `Chain`, loads the PyTorch HDF5 weights, runs forward on the saved test inputs, and asserts the similarity scores and argmax accuracy match (≤5e-3 score tolerance, ≤1 pp accuracy tolerance).

## Validation status

All Julia parity scripts pass at float32 tolerance:

| Layer | Cases | Max error | Tolerance |
|---|---|---|---|
| PhasorDense | 5 | 9e-6 | 1e-5 |
| Codebook + SSMReadout | 6 | 1e-7 | 1e-5 |
| PhasorLSA | 4 | 1e-4 (long-seq FFT) | 5e-4 long / 1e-5 short |
| PhasorLCA | 4 | 9e-5 (long-seq FFT) | 5e-4 long / 1e-5 short |
| End-to-end chain | 1 | 2.6e-6 on scores; 0 pp accuracy gap | 5e-3 / 1 pp |

## What's intentionally NOT in this port

- ODE solver and spiking path (`SpikingCall`, `CurrentCall`, `oscillator_bank`) — discrete 3D Phase is sufficient for LSA/LCA training.
- Equilibrium Propagation / Holomorphic EP (`src/ep.jl`, `src/hep.jl`).
- `AttractorPhasorSSM`, `SSMCrossAttention`, `SSMSelfAttention`, `PhasorAttention`, `PhasorResonant`, `ResonantSTFT`, `PhasorConv`, `PhasorFixed`, `ComplexBias`.
- Multi-backend abstraction (`select_device(:cuda|:cpu|:oneapi)`) — PyTorch handles devices natively.
- Custom CUDA / Triton / KernelAbstractions kernels — vectorized PyTorch ops suffice in this regime.

If you need any of those in PyTorch, this repo is a good starting point but expect to fork in new layers under `layers/` and add matching parity scripts under `julia_parity/`.
