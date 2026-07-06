# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`phasor_torch` is a **PyTorch trainer** for phasor SSM networks with **Local Self-Attention (PhasorLSA)** and **Local Cross-Attention (PhasorLCA)** bodies. The canonical reference for the math is the Julia package [PhasorNetworks.jl](../PhasorNetworks.jl) — this repo is a **training-only port** of the discrete (3D Phase) forward path. It exists so that the model can be trained on accelerators with broad PyTorch support (specifically **Intel PVC / `xpu` on Aurora**, where the JuliaGPU stack is poorly supported), with the resulting weights **loadable back into Lux for deployment / evaluation**.

The single load-bearing requirement is that PyTorch's forward semantics match Julia's exactly. Weights trained in PyTorch must produce equivalent outputs when loaded into the Julia chain; that's the whole point of the repo.

## Common Commands

```bash
# PyTorch tests (91 cases — primitives, kernels, layers, train loop, HDF5 round trip)
pytest                                       # uses ./pyproject.toml's testpaths

# A specific test file
pytest phasor_torch/tests/test_phasor_lsa.py -v

# Train end-to-end with defaults (CPU; Adam; copy-task first-token-recall)
python -m phasor_torch.train

# Train and save a Lux-loadable checkpoint
python -m phasor_torch.train --save model.h5

# Train with a YAML config
python -m phasor_torch.train --config configs/lca_small.yaml --save run.h5

# Regenerate parity fixtures (PyTorch side)
python julia_parity/generate_parity_phasor_dense.py
python julia_parity/generate_parity_readouts.py
python julia_parity/generate_parity_phasor_lsa.py
python julia_parity/generate_parity_phasor_lca.py
python julia_parity/generate_parity_phasor_residual.py   # PhasorTransformerBlock
python julia_parity/train_and_save_for_julia.py     # Stage 7 end-to-end fixture

# Run Julia parity verifiers (one-time: instantiate)
julia --project=julia_parity -e 'using Pkg; Pkg.instantiate()'
julia --project=julia_parity julia_parity/verify_phasor_dense.jl
julia --project=julia_parity julia_parity/verify_readouts.jl
julia --project=julia_parity julia_parity/verify_phasor_lsa.jl
julia --project=julia_parity julia_parity/verify_phasor_lca.jl
julia --project=julia_parity julia_parity/verify_phasor_residual.jl
julia --project=julia_parity julia_parity/verify_end_to_end.jl
```

No linter / formatter is configured. PyTorch ≥ 2.10 is required (matches Aurora's `frameworks/2025.3.1` module).

## Architecture

### Discrete-mode SSM forward

Every layer in this repo implements the discrete (3D Phase) forward path of `dz/dt = k*z + W*I(t)` via causal convolution:

```
K[c, n]    = exp(k_c * n * Δt) * (exp(k_c * Δt) - 1) / k_c       # phasor_kernel
z[c, n, b] = Σ_{j ≤ n} K[c, n - j] * H[c, j, b]                   # causal_conv (Toeplitz / FFT)
```

with `k = λ + iω`, `λ = -exp(log_neg_lambda)` (per-channel trainable), `ω = 2π / t_period` (per-layer **shared scalar** — see "per-channel ω rule" below).

For phase inputs, the kernel is built **without** the `B = (A-1)/k` gain term and the per-spike `exp(k·dt)` factor is folded into the channel-mixed input `H` via `causal_conv_dirac`. Don't add `B` back — see `phasor_torch/kernels.py:causal_conv_dirac` and the docstring there for the derivation.

### Layer inventory

| Layer | File | Forward paths |
|---|---|---|
| `PhasorDense` | `phasor_torch/layers/phasor_dense.py` | 2D Phase, 3D Phase Dirac, 2D Complex linear |
| `Codebook` | `phasor_torch/layers/codebook.py` | 2D Phase → similarity vs fixed codes |
| `SSMReadout` | `phasor_torch/layers/ssm_readout.py` | 3D Phase → temporal-window similarity |
| `PhasorLSA` | `phasor_torch/layers/phasor_lsa.py` | 2D / 3D Phase via head-axis similarity_outer_heads |
| `PhasorLCA` | `phasor_torch/layers/phasor_lca.py` | 2D / 3D Phase via anchor bank (Hopfield-style) |
| `PhaseRecenter` / `PhasorResidual` / `PhasorTransformerBlock` | `phasor_torch/layers/phasor_residual.py` | depth-robust ReZero residual stack around LSA/LCA + FFN |

LSA/LCA use bias-free `PhasorDense` projections internally (`q_proj`, `k_proj`, `v_proj` for LSA; `k_proj`, `v_proj` for LCA). LCA additionally carries a trainable `anchors` `(d_model, n_anchors)` Phase parameter.

#### Depth-robust stacking (ReZero blocks)

Stacked phase attention only trains at depth when each attention sublayer is wrapped in a residual with a **ReZero gate** (learnable scalar `alpha` init ≈ 0 → exact identity at init). `PhasorTransformerBlock` = `PhasorResidual(attn) → PhasorResidual(FFN)`, combined via `v_bind` (phase addition with a straight-through wrap; see `primitives.v_bind`/`remap_phase`). The residual combine, *not* weight down-scaling, is what makes attention stack past depth ~2. Enable via `ModelConfig.block_type="rezero"` (coexists with the default `"plain"` `body→dense` stacking); knobs: `gate`, `recenter`, `branch_init_scale` (FFN-only), `d_ff`, `qkv_init_mode`, `ffn_init_mode`. ReZero `alpha` params train at `lr * TrainConfig.alpha_lr_mult` (5×) via a dedicated optimizer group (`train.build_optimizer`). Ported from Julia `src/ssm.jl`; see `results/lsa_lca_residual/PHASOR_TORCH_PORT.md`.

**Config-B defaults (from the PhasorNetworks.jl MQAR / init-placement ablation).** The recommended attention-block config, now the package default:

| axis | Q/K/V read heads | FFN (residual stream) | ModelConfig field |
|---|---|---|---|
| λ init | `default` (uniform, τ=5) | `hippo` (long tape, τ∈[0.5,64]) | `qkv_init_mode="default"`, `ffn_init_mode="hippo"` |
| complex bias | `use_bias=False` (gate handles origin) | `use_bias=True` | (fixed in layers) |
| ReZero | `gate="rezero"`, α₀=0.1, α-lr ×5 | same | `gate`, `alpha_lr_mult` |
| phase recenter | `recenter=True` (pre-norm, skip untouched) | same | `recenter=True` |

Two enabling mechanisms make this coherent: (1) the near-origin **gradient gate** in `complex_to_angle` (backward gated at `|z|<1e-3`; see autograd note below) makes hippo/uniform projections trainable without the `1/|z|²` NaN blow-up regardless of origin proximity; (2) the **`:hippo` redefinition** to a genuine long tape (`kernels.HIPPO_TAU_MIN=0.5`, `HIPPO_TAU_MAX=64`; λ log-spaced over τ∈[0.5,64], N-independent) makes the FFN memory tape actually reach. HiPPO in the read heads *hurts* long-range routing — keep them uniform. The input embedding stays `hippo` (long tape) and is exempt from the audio RNN_KW preset (only an explicit `init_log_neg_lambda` overrides it).

### Data flow (reference topology, mirrors `scripts/local_attention_compare.jl:245`)

```
Phase input (C_in, L, B)
  → PhasorDense(C_in → D_hidden, normalize_to_unit_circle, use_bias=False, hippo init)    # input embedding
  → PhasorLSA(D → D, n_heads) | PhasorLCA(D → D, n_heads, n_anchors)                       # body
  → PhasorDense(D → D, identity, use_bias=False, hippo init)                               # body dense
  → SSMReadout(D → n_classes) | Codebook(D → n_classes)                                    # readout
→ similarity scores (n_classes, B)
→ similarity_loss vs one-hot target
```

The training script wires this in `phasor_torch/train.py:build_model` via a config-driven `ModelConfig` (see `phasor_torch/config.py`).

### Source file responsibilities

| File | Role |
|---|---|
| `phasor_torch/primitives.py` | `angle_to_complex`, `complex_to_angle`, `remap_phase`, `normalize_to_unit_circle`, `similarity`, `similarity_outer_heads`, plus three `torch.autograd.Function` shims (NaN guards + memory-efficient pairwise sim) |
| `phasor_torch/kernels.py` | `phasor_kernel`, `causal_conv` (Toeplitz/FFT hybrid), `causal_conv_dirac`, `bias_kernel_accumulation`, `hippo_legs_diagonal` |
| `phasor_torch/init.py` | `random_symbols`, `orthogonal_codes` (port of Julia VSA code generators) |
| `phasor_torch/layers/` | nn.Module layers — see table above |
| `phasor_torch/losses.py` | `similarity_loss`, `codebook_loss`, `accuracy`, `one_hot` |
| `phasor_torch/data/sequence_tasks.py` | Copy / reversal / retrieval / sorting generators + `first_token_classification` wrapper |
| `phasor_torch/config.py` | `ModelConfig`, `DataConfig`, `TrainConfig`, `RunConfig`, YAML loader |
| `phasor_torch/train.py` | Adam loop, eval, checkpointing, `select_device('auto')`, CLI |
| `phasor_torch/weights.py` | HDF5 save/load with nested-path resolution (`save_state`, `load_state`, `save_io_pair`) |
| `julia_parity/load_pytorch.jl` | HDF5 → Lux NamedTuple helper (`load_params`, `load_io_pair`, `load_meta`) |
| `julia_parity/verify_*.jl` | Per-layer parity verifiers + end-to-end (Stage 7) check |

## Critical Conventions (load-bearing)

### Phase is a plain real `torch.Tensor`, NOT a wrapper class

Values are float32 in `[-1, 1]` (units of π). NaN is allowed (sub-threshold sentinel). The Julia `Phase <: Real` wrapper exists for dispatch; we don't need it here because dispatch is via explicit `isinstance` / `ndim` checks. Document the invariant at function boundaries; don't enforce at runtime.

### Weight-flow direction: PyTorch → Julia (strictest)

The PyTorch `forward()` of any layer **must** be a bit-faithful port of the Julia Lux equivalent, because the whole point is that PyTorch-trained weights produce equivalent outputs in Julia. Silent semantic drift (sign convention, normalization, frame rotation) corrupts the weights from Julia's perspective and won't show up in pytest alone.

**Always** add a Julia parity script for any new layer. Pattern:
1. `julia_parity/generate_parity_<layer>.py` — initialize random weights in PyTorch, save weights + IO pair to HDF5.
2. `julia_parity/verify_<layer>.jl` — build the equivalent Lux layer, inject the saved weights, run forward on the saved input, assert agreement.
3. Target tolerance: **1e-5** for Toeplitz path, **5e-4** for long-sequence FFT path.

### Head split: column-major reshape semantics

The reference Lux code reshapes `(D, L, B) → (Dh, H, L, B)` in column-major order, so head `h` owns the **contiguous channel block `[h*Dh : (h+1)*Dh]`** of D. PyTorch's default `reshape(Dh, H, L, B)` is row-major and interleaves heads (head `h` gets every `H`-th channel) — this is a **silent failure mode**. The correct PyTorch idiom is:

```python
Qh = Q.reshape(H, Dh, L, B).transpose(0, 1).contiguous()    # match Julia col-major
```

Same for the inverse reshape on the way back. See `phasor_torch/layers/phasor_lsa.py` and `phasor_torch/layers/phasor_lca.py` for the existing applications. This bug cost us a debugging round in Stage 4 — don't repeat it.

### Per-channel ω rule (inherited from PhasorNetworks.jl)

Most layers in this port use a **single shared `ω = 2π / t_period`** across channels. It lives as a `register_buffer` named `omega`, not a `nn.Parameter`. This maintains phase-locked communication between layers for downstream VSA operations. The **one documented exception** is `ResonantSTFT` (the audio frontend): it carries a per-channel *trainable* ω as an `nn.Parameter` and re-encodes its output back onto the shared downstream carrier via `freq_shift` so later layers resume phase-locked operation. Do not "fix" ResonantSTFT's ω into a buffer — see `phasor_torch/layers/resonant_stft.py`.

### HDF5 schema and the row-major / column-major dance

HDF5 stores arrays in row-major order. h5py (Python) writes them directly; Julia's `HDF5.jl` reads them and produces arrays whose dimensions appear **reversed** because Julia is column-major. The parity loaders compensate by `permutedims` on every 2D parameter:

```julia
weight = permutedims(raw.weight, (2, 1))    # PyTorch (out, in) -> Julia (out, in)
```

For 3D IO tensors saved as `(C, L, B)` in PyTorch, the Julia loader reads them as `(B, L, C)` and applies `permutedims(x, (3, 2, 1))`. Already done in all existing `verify_*.jl`; any new verifier must do the same.

### Custom `torch.autograd.Function` — when to add one

PyTorch's autograd is generally fine; only add a custom Function in two cases:

1. **NaN guards + near-origin gradient gate**: PyTorch's native autograd for `torch.angle` / `1/|z|` produces NaN at `z=0` and poisons sibling cotangents via `0*NaN = NaN`. Already handled in `_ComplexToAngle` and `_NormalizeHard`. `_ComplexToAngle` additionally **gates its backward** at `|z| < grad_threshold` (default `1e-3`): the `dz = ȳ·i·z/(π·|z|²)` singularity blows up when an SSM/attention phasor sum cancels to `|z|~1e-9` (the depth-2 LCA NaN blow-up — traced to the first block's K/V projections). A collapsed phasor carries no useful phase, so its cotangent is zeroed, capping `|dz|` at `~|ȳ|/(π·grad_threshold)`. The **forward is unchanged** (only `|z|<1e-10` zeroed, matching Julia) → parity-safe. Mirrors Julia `src/domains.jl:65` (commit `ae00ded`). `grad_threshold` and `threshold` are decoupled kwargs. Add a similar shim if you port any other phase-domain primitive with a singularity.
2. **Shape blowup**: if a forward implementation would materialize an `O(D·M·N·B)` intermediate that the closed-form rrule can avoid. Already handled in `_SimilarityOuterCanonicalComplex` (the LSA/LCA inner product). Profile first; only port a closed-form rrule if memory is actually a problem.

Skip the autograd.Function for everything else. PyTorch's tape doesn't suffer Zygote's `ForwardDiff.Dual` blowup, so most of the Julia rrules (`_exp_kdt`, `angle_to_complex`, soft normalize, `Phase` constructor) don't need PyTorch equivalents.

### Parameter / buffer conventions

- **`nn.Parameter`**: trainable. `weight`, `log_neg_lambda`, `bias_real`, `bias_imag`, `scale`, `anchors`.
- **`register_buffer`**: derived const tensor that must move with `.to(device)` and persist in `state_dict`. `omega`, `codes` (Codebook / SSMReadout).
- **Plain `self.attr`**: Python primitives that don't need device migration or serialization. `spk_args`, `activation` callable, `init_mode`.

### Weight shape and bias

- `weight` is `(out_dims, in_dims)`, **same as both Lux and PyTorch's `nn.Linear`**. No transpose at use-time.
- Bias is split into `bias_real` and `bias_imag` (two `nn.Parameter`s). **Do not** fuse into a single complex `nn.Parameter` — some optimizers misbehave on complex params, and the Lux side expects the split.
- `log_neg_lambda` parameterization: `λ = -torch.exp(self.log_neg_lambda)`. Identical to Julia.

### Float32 everywhere

Use `torch.float32` and `torch.complex64`. Don't promote to float64/complex128 except inside `torch.autograd.gradcheck` (which needs the precision). Aurora's `xpu` complex64 support is what we're shipping for; complex128 may not be available on every backend.

## Testing

PyTorch tests live under `phasor_torch/tests/`. Run all with `pytest`. Per-layer test files cover:

- shape, dtype, value range (`[-1, 1]` for Phase outputs)
- gradient finiteness for all `nn.Parameter`s
- `gradcheck` (complex128) for autograd.Function shims
- HDF5 `save_state` → `load_state` round trip

The Julia parity verifiers are the **canonical correctness gate**. If a parity script fails, the PyTorch implementation is wrong (regardless of what pytest says). Internal pytest consistency is necessary but not sufficient.

Tolerances:
- Toeplitz path / short sequences: `1e-5`
- FFT path (L > 64) / chained-layer end-to-end: `5e-4` (compounded float32 error)

## Aurora (Intel PVC / xpu)

```bash
module load frameworks                       # provides PyTorch 2.10 + XPU support
cd /path/to/phasor_torch
PYTHONPATH=. python -m phasor_torch.train    # device='auto' resolves to xpu if available
```

`train.select_device('auto')` resolves `xpu > cuda > cpu`. Use `"xpu"` directly to force-pin (e.g. on a node with both CUDA and XPU visible).

**Not yet wired:**
- Multi-tile / multi-rank DDP. Per the ALCF docs, init with `backend='xccl'`, pin via `torch.xpu.set_device(int(os.environ['PALS_LOCAL_RANKID']))`, wrap with `DDP(model)`, and move to XPU **after** DDP wrap when using multiple CCSs per tile.
- Mixed-precision / `torch.compile`. Both should "just work" but haven't been profile-tested on PVC.

Local testing on DGX Spark (`nubun` conda env) uses PyTorch 2.11.0+cu130, which is a strict superset of Aurora's 2.10 for the APIs in use. XPU-specific bits (complex64 op coverage on PVC, FFT correctness) are verified by running on an actual compute node — not locally.

## Workflow: adding a new layer

1. Read the Julia source carefully — note every `permutedims`, `−conj`, `unrotate_solution`-style frame correction. Each is load-bearing.
2. Implement the PyTorch layer with `forward()` matching the Julia 3D Phase dispatch exactly. Skip non-3D-Phase paths unless the LSA/LCA chain needs them.
3. Expose params via `parameter_dict()` with flat (slash-separated) names matching the Lux NamedTuple. Nested sub-layers use `"sub/weight"` keys — `weights.py` walks these via attribute traversal.
4. Add a `phasor_torch/tests/test_<layer>.py` covering shape, range, finite grads, and HDF5 round trip.
5. Add `julia_parity/generate_parity_<layer>.py` — initialize 3–5 cases varying `init_mode`, sequence length, bias on/off; save weights + IO pairs.
6. Add `julia_parity/verify_<layer>.jl` — load fixtures, inject weights into a freshly-built Lux layer, run forward, compare. Apply `permutedims(weight, (2, 1))` on load.
7. Verify parity passes at the standard tolerances before considering the port done.

## What's intentionally out of scope

Adding any of these requires the Julia code to grow a corresponding feature first, and the trainer's "training-only" charter to be revisited. Don't add them without explicit ask:

- ODE solver / spiking path (`SpikingCall`, `CurrentCall`, `oscillator_bank`, `torchdiffeq` integration).
- Equilibrium Propagation / Holomorphic EP (`src/ep.jl`, `src/hep.jl`).
- `AttractorPhasorSSM`, `SSMCrossAttention`, `SSMSelfAttention`, `PhasorAttention`, `PhasorResonant`, `PhasorConv`, `PhasorFixed`, `ComplexBias`. (`ResonantSTFT` *is* now in scope — it's the audio frontend the LCA/LSA archs require; only its discrete 3D-Complex dispatch is ported, not its spiking return-types.)
- A `Phase` wrapper class. The raw-tensor + documented-invariant approach is intentional.
- Custom CUDA / Triton / Numba kernels. Vectorized PyTorch ops + `torch.compile` should be enough for this regime; profile before reaching for hand-written kernels.

## Code Style

- 4-space indentation
- `snake_case` for functions, `PascalCase` for types / classes
- Type hints throughout (Python 3.10+ `X | Y` syntax is fine)
- Docstrings: one-line summary, then Args/Returns sections for non-trivial APIs
- When porting from Julia, **link the source file:line in the docstring** (e.g. `"Mirrors Julia src/network.jl:288"`) so a reader can compare implementations side by side
