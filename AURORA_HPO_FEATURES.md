# Feature Overview: Audio LCA/LSA Hyperparameter Optimization on Aurora

**Audience:** an agent working inside the `phasor_torch` repo.
**Goal:** run a hyperparameter optimization (HPO) sweep of the **PhasorLCA** and **PhasorLSA**
architectures on the **real keyword-spotting audio dataset**, on **Aurora (Intel PVC / `xpu`)**.

This document describes *what to build*. It does not prescribe exact code. Read the existing
`train.py`, `config.py`, and `data/sequence_tasks.py` first — the model core already exists and is
parity-verified; the gaps are **porting the `ResonantSTFT` frontend, raw-audio loading, HPO
orchestration, and Aurora scale-out**.

---

## 0. Context: what already works (do not rebuild)

- `PhasorLSA` / `PhasorLCA` bodies are ported and Julia-parity-verified (≤5e-4 FFT path).
- `train.build_model` already wires the canonical topology:
  `input PhasorDense → {lsa|lca} → body PhasorDense → {ssm|codebook} readout → similarity_loss`.
- `train.train(run, save_path)` runs an Adam loop and **returns** `{"history":[...], "final":{...}}`
  — the natural HPO objective hook.
- `select_device('auto')` resolves `xpu > cuda > cpu`.
- `PhasorDense.__init__` already accepts `init_log_neg_lambda` (per-channel λ override) — this is how
  the Julia `RNN_KW` / `FF_KW` recurrence presets are expressed (see §3).
- HDF5 checkpoint save/load is Lux-compatible (`weights.py`).

The **only data wired to the model today is synthetic** (copy/reversal/etc. → first-token recall).
Replacing that with real audio is the central gap.

---

## 1. Data source decision (settled)

**Read the raw-waveform file `sound_data_raw.h5` and feed it through a ported `ResonantSTFT` layer
inside the trainer.** The LCA/LSA archs we want to sweep are *defined* as
`raw audio → ResonantSTFT → downsample → to_phase → body → readout`; the trainable resonant
filterbank IS part of the architecture, so it must be ported, not substituted.

- **Do NOT** use `precompute_stft.jl` / the complex-STFT cache — flaky/unconfirmed on Aurora, and a
  frozen cache can't train ω.
- **Do NOT** substitute `torch.stft` — that's a different, non-learned frontend; results wouldn't map
  onto the Julia ResonantSTFT archs.

`sound_data_raw.h5` layout (HDF5):
- key **`"audio"`** — `(L, N)` Float32 raw waveforms; **L = 16000 samples, fs = 16 kHz** (1 s clips).
- key **`"labels"`** — Int, one keyword-class id per clip.
- A matching test file exists (mirror the project's `*_raw.h5` / `*_raw_test.h5` convention).

---

## 2. Port `ResonantSTFT` to PyTorch (the central new layer)

Reference Julia source: `PhasorNetworks.jl/src/network.jl:823-1015` (struct + both dispatches) and
`src/activations.jl:142-211` (`soft_normalize_to_unit_circle`). The audio archs use the
**3D-Complex dispatch** (the `encode_input` helper lifts the real waveform to complex `(1, L, B)`).

### Good news: ~80% of the dependencies are already ported and parity-verified
Reuse from `phasor_torch/kernels.py`: **`phasor_kernel`, `causal_conv` (Toeplitz+FFT),
`causal_conv_dirac`, `bias_kernel_accumulation`**. From `primitives.py`: `normalize_to_unit_circle`,
`angle_to_complex`, `complex_to_angle`. Do not reimplement these.

### What to build
1. **`soft_normalize_to_unit_circle`** in `primitives.py` (trainable-SLERP form,
   `src/activations.jl:208`). ~10 lines, pure math, plain autograd (no custom `Function`):
   `r=|z|; blend=sigmoid(k·(r−mid))` with `k=6/(r_hi−r_lo)`, `mid=(r_lo+r_hi)/2`; `θ=angle(z/max(r,1e-10))`;
   return `cos(blend·θ)+i·sin(blend·θ)`. `r_lo`/`r_hi` broadcast as `(n_freqs,1,1)` for per-channel gating.
2. **`_freq_shift`** helper (`network.jl:868`): `Δω = ω_out − ω`; multiply `Z` by
   `exp(i·Δω·Δt·n)` over `n=0…L-1` → re-encodes per-channel ω to the shared downstream
   `ω_out = 2π/t_period`.
3. **`ResonantSTFT(nn.Module)`** in `phasor_torch/layers/resonant_stft.py`:
   - **Parameters:** `weight (n_freqs, in_dims)`, `log_neg_lambda (n_freqs,)` init `log(0.1)`, and
     **`omega (n_freqs,)` as a trainable `nn.Parameter`** init `linspace(omega_lo, omega_hi, n_freqs)`
     (defaults `omega_lo=0.2`, `omega_hi=2.5`). Optional `bias_real`/`bias_imag`. When SLERP is
     active (activation=None, the default): `log_r_lo`, `log_r_gap` `(n_freqs,)` with
     `r_lo=exp(log_r_lo)`, `r_hi=r_lo+exp(log_r_gap)` (positivity-preserving), init from
     `init_r_lo=0.1`, `init_r_hi=0.6`.
   - ⚠️ **Per-channel ω deviation:** `omega` must be a trainable `nn.Parameter` here, NOT a shared
     `register_buffer`. This intentionally breaks the port's shared-ω rule — `ResonantSTFT` is the
     one documented exception (it re-encodes to shared `ω_out` via `_freq_shift` so downstream layers
     resume phase-locked). Note this prominently in the layer docstring and `parameter_dict`.
   - **3D-Complex forward** (`network.jl:928`): `λ=-exp(log_neg_lambda)`; `K=phasor_kernel(λ,ω,T,L)`;
     `H = weight @ x` (real & imag separately) → `(n_freqs,L,B)`; `Z_sig=causal_conv(K,H)`;
     optional ZOH bias via `bias_kernel_accumulation` with gain `B=(exp(k·T)−1)/k`;
     `Z=_freq_shift(Z_sig,ω,ω_out,T)`; `Y = soft_normalize_to_unit_circle(Z, r_lo, r_hi)` (SLERP) or
     `activation(Z)`. Returns complex `(n_freqs,L,B)`.
   - (Optional) the **3D-Phase Dirac** dispatch (`network.jl:973`) for completeness — uses
     `causal_conv_dirac` + the `−conj(·)` frame correction + `complex_to_angle`. Not needed for the
     complex-input audio path; port only if a parity case needs it.
4. **`encode_input`** (real `(L,B)` → complex `(1,L,B)`, zero imag) and **`downsample_time`**
   (mean-pool time axis by `ds`) helpers — `scripts/audio_pipeline.jl:18-46`.
5. **`to_phase`** = `normalize_to_unit_circle` → `complex_to_angle` (already available) — applied
   after downsample so the Phase-dispatching body can consume the output.

### Parity gate (non-negotiable — this is where per-channel-ω / frame bugs surface)
Add `julia_parity/generate_parity_resonant_stft.py` + `verify_resonant_stft.jl` following the
existing pattern: init random weights+ω in PyTorch, save weights + a complex IO pair to HDF5; in
Julia build the equivalent `ResonantSTFT`, inject weights (`permutedims(weight,(2,1))`; 3D IO via
`permutedims(x,(3,2,1))`), run forward, assert agreement. Target **1e-5** short / **5e-4** long-seq
FFT. Cover: trainable-SLERP vs fixed `identity`/`normalize` activation; bias on/off; a short L
(Toeplitz) and a long L (FFT path).

### FFT-in-loop risk (flag, don't pre-optimize)
At full `L=16000`, `causal_conv` takes its **FFT path** (`causal_conv_fft`, uses `torch.fft`) every
step — native PyTorch FFT on `xpu`, same unconfirmed-on-Aurora risk class as before but more likely
solid than the Julia oneAPI path. This is also the per-step cost the Julia side avoided by caching —
but with trainable ω you *want* it live. Add an `xpu` smoke test of the full
`encode_input→ResonantSTFT→downsample→to_phase` frontend (finite output, no NaN) before the sweep.

### Raw-waveform preprocessing — match the Julia `load_audio` exactly
(`scripts/train_audio_ssm_attention.jl:213-249`) in a new `phasor_torch/data/audio.py`:
1. Load `"audio"` `(L, N)` + `"labels"`.
2. **Per-clip RMS normalize:** divide each clip by its own RMS; **skip clips with RMS ≤ 1e-3**
   (silent input stays silent — do not divide by ~0).
3. **Clamp ±5 then rescale to ±1** (`clamp(-5,5)/5`).
4. **Drop OOD labels** outside `[0, n_classes-1]` (test set carries an "unknown" class).
5. Sample axis **last** (`(C,L,B)` convention); reuse the permute/collate trick from
   `data/sequence_tasks.make_dataloader`.

> **Silence / NaN check (project memory):** silent clips → exact-zero conv outputs → `z=0` in
> normalize/SLERP. The `max(r,1e-10)` guard + NaN-guarded primitives handle this, but verify on
> **real audio** (not random Gaussian) that no NaN reaches the loss. This has bitten the project before.

---

## 3. Config + `build_model` wiring for the audio topology

`ModelConfig`/`DataConfig`/`TrainConfig` cover most knobs. Extend to match the Julia `stft_phasor_lsa`
/ `stft_phasor_lca` archs:

- **`DataConfig`**: add `source: "synthetic"|"audio"`, `train_path`, `test_path`,
  `sample_rate=16000`, `n_classes ≈ 30` (keyword count).
- **`ModelConfig`**: add ResonantSTFT frontend knobs — `n_freqs` (default 64; conv path used 128),
  `omega_lo=0.2`, `omega_hi=2.5`, `downsample_factor=32`, `resonant_init_log_neg_lambda=log(0.1)`,
  `init_r_lo=0.1`, `init_r_hi=0.6`, `resonant_activation="slerp"|"identity"|"normalize"`. In audio
  mode the body `in_dims = n_freqs`.
- **`ModelConfig`**: also expose **`init_log_neg_lambda`** for the body `PhasorDense` layers (Julia presets):
  - `FF_KW`  = `init_mode="default", init_log_neg_lambda=log(10)`  (λ=-10, no recurrence / "leakage 0")
  - `RNN_KW` = `init_mode="default", init_log_neg_lambda=log(0.1)` (λ=-0.1, trainable per-neuron decay)
  The audio LCA/LSA archs use **RNN_KW** on the surrounding PhasorDense layers — `build_model` must
  apply this (today it leaves them at the `hippo` default).
- **`train.train` / `build_model`**: branch on `DataConfig.source`. In audio mode prepend the
  frontend `encode_input → ResonantSTFT → downsample_time → to_phase` to the chain.
- Target chain (must match Julia `train_audio_ssm_attention.jl:472-505`):
  `encode_input → ResonantSTFT(1→n_freqs, slerp) → downsample_time(ds) → to_phase →
   PhasorDense(n_freqs→H, RNN_KW) → PhasorL{S,C}A(H→H, n_heads[, n_anchors]) →
   PhasorDense(H→H, RNN_KW) → SSMReadout(H→n_classes, readout_frac)`.

### Suggested HPO search space (ladder defaults + project memory)
- `lr` log-uniform ~1e-4 … 1e-3 (project memory: lr is the dominant knob)
- `d_hidden`/`H` ∈ {64,128,256} — must be divisible by `n_heads`
- `n_heads` ∈ {2,4,8}; `n_anchors` (LCA only) ∈ {32,64,128}
- `init_scale` ∈ ~1–5; `readout_frac` ∈ 0.1–0.5
- `batch_size`: memory says phasor archs are **FLOPS-bound — batch=8 is the safe pick**, batching up
  doesn't improve throughput; `weight_decay`; per-trial `seed`
- STFT geometry (`n_freqs`, `hop_length`) optionally swept
- `body ∈ {lca, lsa}` (one study each, or one combined study)

**Expected ballpark** (project memory, ~50 epochs): LCA ≈ 55% test, LSA ≈ 47% test. The 25-epoch
ladder badly understated phasor performance — **budget ≥ 50 epochs per trial**.

---

## 4. HPO harness (does not exist — build it)

No sweep/Optuna/Ray infra today. Add a self-contained driver.

- **Library:** **Optuna** (lightweight, no daemon, RDB storage for resumable parallel studies). Add
  to `pyproject.toml` `[project.optional-dependencies]`.
- **Objective:** wrap `train.train(run)`; maximize `final["test_acc"]`. Sample a `RunConfig` via
  `config.from_dict`.
- **Persistence:** Optuna RDBStorage (`sqlite:///study.db` on shared FS, or Postgres) so many ranks
  feed **one** study concurrently.
- **Per-trial artifacts:** capture the returned `history` → JSON/CSV per trial + final HDF5
  checkpoint in a per-trial dir keyed by trial number (`train()` only prints history today).
- **Pruning:** optional `MedianPruner` on per-epoch `test_acc`.
- **CLI:** `python -m phasor_torch.hpo --study-name … --storage … --n-trials N --body lca|lsa
  --train-path sound_data_raw.h5 --test-path sound_data_raw_test.h5`.

---

## 5. Aurora scale-out (not wired)

`select_device` resolves a single `xpu`; nothing maps trials onto ranks/tiles.

- **Parallelism model — embarrassingly-parallel trials, not DDP.** Run **one trial per MPI rank**,
  each pinned to one XPU tile, all writing the **same Optuna RDB study**. Simpler than DDP and ideal
  for HPO. (DDP — `backend='xccl'`, `torch.xpu.set_device(PALS_LOCAL_RANKID)`, wrap-then-move-to-XPU
  — is only needed if a *single* model must span tiles; not required here.)
- **Device pinning:** read `PALS_LOCAL_RANKID` (ALCF PALS launcher), set the trial's device to that
  XPU, pass through `TrainConfig.device`.
- **PBS job script** (`scripts/` or `configs/`; none exists in this repo). Model it on the parent
  repo's `scripts/dispatch_audio_aurora.sh` / `smoke_test_archs_aurora.sh`. It must:
  - `module load frameworks` (PyTorch 2.10 + XPU; no `intel_extension_for_pytorch` import needed),
  - `mpiexec` across nodes×tiles, one HPO worker per rank,
  - point every worker at the shared Optuna storage + the staged raw-audio files,
  - `device="auto"` (or force `"xpu"`), pinned via `PALS_LOCAL_RANKID`.
- **Data staging:** copy `sound_data_raw.h5` (+ test) to node-local/fast storage before the sweep;
  ranks read read-only.
- **Logging:** set `PYTHONUNBUFFERED=1` / `flush=True` so long Aurora runs aren't silent.

---

## 6. Suggested build order

1. **`soft_normalize_to_unit_circle` + `_freq_shift`** primitives, with unit tests.
2. **`ResonantSTFT` layer** (`layers/resonant_stft.py`) reusing the existing kernels; **parity
   scripts** (`generate_parity_resonant_stft.py` + `verify_resonant_stft.jl`) green at 1e-5/5e-4.
   This is the gate — do not proceed until parity passes.
3. **Raw-audio loader** (`data/audio.py`) matching Julia preprocessing; silence/NaN test on a real
   slice; `xpu` smoke test of the full frontend (FFT-path `causal_conv` + ResonantSTFT).
4. **Config + `build_model` wiring**: `source` switch, ResonantSTFT knobs, `init_log_neg_lambda` (RNN_KW).
5. **Single end-to-end audio run** locally (CPU/CUDA): loss descends; sanity-check accuracy reaches
   the LCA/LSA ballpark above.
6. **HPO driver** (`hpo.py`) with Optuna + RDB + per-trial artifacts.
7. **Aurora PBS + rank-pinned worker**; smoke-test 2 ranks × few trials before the full sweep.

## 7. Out of scope (do not add)
`PhasorConv`, the `precompute_stft.jl` cache path, librosa/MFCC, ODE/spiking path (`SpikingCall`/
`CurrentCall` — port only ResonantSTFT's discrete complex/Dirac dispatches, not its spiking
return-types), EP/hEP, custom XPU kernels. `ResonantSTFT` itself is now **in scope** (it was the one
frontend the LCA/LSA archs require).

---

### Cross-references
**Layer port (canonical math — PhasorNetworks.jl dev clone at `/home/wilkie/code/PhasorNetworks.jl`):**
- `src/network.jl:823-1015` — `ResonantSTFT` struct, `initialparameters`, `_freq_shift`, and the
  3D-Complex (`:928`) + 3D-Phase (`:973`) forward dispatches incl. the `−conj` frame correction.
- `src/activations.jl:142-211` — `soft_normalize_to_unit_circle` (kwargs + trainable positional form).

**Reference topology / data (parent `mos2_oscillators` repo):**
- `scripts/train_audio_ssm_attention.jl` — `stft_phasor_lsa`/`stft_phasor_lca` topology (~472–505);
  `load_audio` preprocessing (213–249); ResonantSTFT band/geometry + defaults (5–62).
- `scripts/audio_pipeline.jl` — `encode_input` / `downsample_time` / `to_phase` / `FF_KW` / `RNN_KW`.
- `scripts/dispatch_audio_aurora.sh`, `scripts/smoke_test_archs_aurora.sh` — Aurora launch templates.
