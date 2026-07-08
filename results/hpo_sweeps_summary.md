# Aurora HPO sweeps — summary

Running log of the audio (30-class keyword-spotting) HPO sweeps on Aurora.
Update the results table as new sweeps complete.

**Common setup (all sweeps):** 200 evals (libEnsemble / RF surrogate), audio
source, **16k-clip train subset** exploration pass (test 1024), epochs 30–80
(swept), batch 8, one trial per XPU tile. Reported `peak`/`median` are the
**subset exploration objective (test_acc)** — noisy, NOT full-data confirm.
`dead` = peak test_acc ≤ 12% (stuck near chance); `NaN` = a NaN epoch appeared
in history (restore_best still keeps the pre-blowup peak, so these count toward
peak/median); `healthy` = 200 − dead − NaN.

Regenerate the results table with:

    python scripts/analyze_d1.py                 # stats for the studies in STUDIES
    python scripts/find_nan_trials.py <study>    # per-NaN-trial conditions

`cb` = "config-B / current defaults" generation (uniform QKV, hippo FFN tape
τ∈[0.5,64], **uniform+bias input embedding**, recenter OFF, complex_to_angle
backward gate 1e-3). `pre-cfgB` = older generation (short hippo τ≤2, no gate,
hippo QKV, RNN_KW λ=−0.1 input, recenter off).

## Conditions

| study (`hpo_runs/…`) | body · depth · block | input embedding | QKV λ | FFN | recenter / ReZero | gate · hippo-τ · gen |
|---|---|---|---|---|---|---|
| `lca` | LCA · 1 · plain | λ=−0.1 RNN_KW (τ=10), no-bias | hippo, no-bias | **none** (single `dense`: identity, λ=−0.1, no-bias) | none / none | ✗ (1e-10) · ≤2 · pre-cfgB |
| `lsa` | LSA · 1 · plain | λ=−0.1 RNN_KW (τ=10), no-bias | hippo, no-bias | **none** (single `dense`: identity, λ=−0.1, no-bias) | none / none | ✗ · ≤2 · pre-cfgB |
| `lca_d2_rezero` | LCA · 2 · rezero | λ=−0.1 RNN_KW (τ=10), no-bias | hippo, no-bias | uniform λ, bias-on (2-layer) | off / on (α₀=0.1, ×5) | ✗ · ≤2 · pre-cfgB |
| `lca_d1_rezero` | LCA · 1 · rezero | hippo (τ≤64), no-bias | uniform, no-bias | hippo (τ≤64), bias-on | **on** / on | ✓ (1e-3) · ≤64 · cfgB |
| `lca_d1_rezero_norecenter` | LCA · 1 · rezero | hippo (τ≤64), no-bias | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB |
| `lca_d1_rezero_cb` *(pending)* | LCA · 1 · rezero | **uniform τ=5 + bias** | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB (current) |
| `lca_d2_rezero_cb` *(pending)* | LCA · 2 · rezero | **uniform τ=5 + bias** | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB (current) |

Bias summary: Q/K/V projections bias-free in every sweep; FFN denses bias-on
(rezero only); input embedding bias-free in all **completed** sweeps, bias-**on**
only in the pending `_cb` runs. FFN (2-layer MLP + ReZero) exists only in the
`rezero` sweeps — the `plain` runs have a single post-attention `dense`, not an FFN.

## Results (of 200 trials each)

| study | peak acc* | median | dead (≤12%) | NaN blow-ups | healthy | dispatch script |
|---|---|---|---|---|---|---|
| `lca` (d1 plain) | 66.3% | 21.7% | 58 | 0 | 142 | `hpo_aurora.pbs` (body=lca) |
| `lsa` (d1 plain) | 63.9% | 27.6% | 26 | 0 | 174 | `hpo_aurora.pbs` (body=lsa) |
| `lca_d2_rezero` (d2, pre-cfgB) | 62.3% | 22.2% | 76 | 16 | 108 | (pre-cfgB `hpo_aurora_d2.pbs`) |
| `lca_d1_rezero` (rezero, recenter ON) | 54.0% | 29.1% | 31 | 30 | 139 | (cfgB, recenter default was on) |
| `lca_d1_rezero_norecenter` (rezero, recenter OFF) | 49.5% | 22.5% | 51 | 32 | 117 | (removed `_norecenter` script) |
| `lca_d1_rezero_cb` *(pending)* | TBD | TBD | TBD | TBD | TBD | `hpo_aurora_d1_rezero.pbs` |
| `lca_d2_rezero_cb` *(pending)* | TBD | TBD | TBD | TBD | TBD | `hpo_aurora_d2.pbs` |

\*single best trial's subset test_acc.

## Notes / caveats

- **Confounded across generations.** The three pre-cfgB rows ran with old code
  (short hippo, no gate, hippo QKV, RNN_KW input); the cfgB rows ran with new
  code. Only `lca_d1_rezero` vs `…_norecenter` is a clean single-variable A/B
  (recenter).
- **Key findings so far:**
  - Plain d1 has the best subset peak (66/64%) and 0 NaN, but is depth-1 only.
  - NaN blow-ups appear only in rezero blocks, only at high lr (5–10e-3).
  - recenter is NOT the NaN cause (removing it left NaN ~unchanged, 30→32) and
    was net-helpful on audio (dead 31 vs 51) — so it was kept as a knob but the
    real fix targeted the collapse layer.
  - The NaN singularity lives in hippo-init, bias-free layers (|z|→0 via SSM
    cancellation): first the LCA K/V projections (pre-cfgB), then the input
    embedding once QKV went uniform. Fixed by making the input embedding uniform
    + bias (validated locally: input min|z| ~0.5 vs ~1e-9).
- **The `_cb` pending runs are the first with the current defaults** (uniform+bias
  input); they also give the first clean d1-vs-d2 depth comparison under matched
  settings.
- Numbers are subset-exploration objective; consider a full-data confirm of each
  study's top-K for true accuracy.
