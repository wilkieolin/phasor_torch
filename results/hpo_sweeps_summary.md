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
| `lca_d1_rezero_cb` | LCA · 1 · rezero | **uniform τ=5 + bias** | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB (current) |
| `lca_d2_rezero_cb` *(pending)* | LCA · 2 · rezero | **uniform τ=5 + bias** | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB (current) |

Bias summary: Q/K/V projections bias-free in every sweep; FFN denses bias-on
(rezero only); input embedding bias-free in all **completed** sweeps, bias-**on**
only in the pending `_cb` runs. FFN (2-layer MLP + ReZero) exists only in the
`rezero` sweeps — the `plain` runs have a single post-attention `dense`, not an FFN.

## Results (of 200 trials each)

| study | peak acc* | **confirm (best full)**† | median | dead (≤12%) | NaN blow-ups | healthy | dispatch script |
|---|---|---|---|---|---|---|---|
| `lca` (d1 plain) | 66.3% | **78.1%** | 21.7% | 58 | 0 | 142 | `hpo_aurora.pbs` (body=lca) |
| `lsa` (d1 plain) | 63.9% | 63.3% | 27.6% | 26 | 0 | 174 | `hpo_aurora.pbs` (body=lsa) |
| `lca_d2_rezero` (d2, pre-cfgB) | 62.3% | 61.7% | 22.2% | 76 | 16 | 108 | (pre-cfgB `hpo_aurora_d2.pbs`) |
| `lca_d1_rezero` (rezero, recenter ON) | 54.0% | — (not run) | 29.1% | 31 | 30 | 139 | (cfgB, recenter default was on) |
| `lca_d1_rezero_norecenter` (rezero, recenter OFF) | 49.5% | — (not run) | 22.5% | 51 | 32 | 117 | (removed `_norecenter` script) |
| `lca_d1_rezero_cb` (rezero, current defaults) | 62.6% | *pending* | 22.2% | 49 | 19 | 132 | `hpo_aurora_d1_rezero.pbs` |
| `lca_d2_rezero_cb` *(pending)* | TBD | *pending* | TBD | TBD | TBD | TBD | `hpo_aurora_d2.pbs` |

\*single best trial's subset (16k) test_acc — noisy exploration objective.
†best of the **top-8 full-data confirm** (`confirm.py` re-trains at full ~51k
data, no subset; `best.h5` = restored peak weights). **This is the real
headline metric** — full data lifts every config well above the subset proxy.
Confirm results live in PBS stdout (`phasor_confirm*.o*`), not `history.json`
(those runs predate that field); see `scripts/eval_confirm_curves.py`.

### Full-data confirmation detail (top-8)

Three studies confirmed; the two `plain` (no-FFN) plus the depth-2 ReZero.

| study | best full | incumbent (subset-#1) full | key finding |
|---|---|---|---|
| `lca` (plain, no FFN) | **78.1%** (subset-6th) | 73.7% | subset rank ≠ full rank — the full winner was the subset-**6th** config; subset-#1 fell to mid-pack |
| `lsa` (plain, no FFN) | 63.3% (subset-#1) | 63.3% | full-data LCA≫LSA gap blows out to **~15 pt** (78.1 vs 63.3), vs ~2.4 pt on the subset; lr anti-correlates with full acc (low lr wins) |
| `lca_d2_rezero` (d2 FFN+ReZero) | 61.7% (subset-#4) | 60.7% best / **35.8% final** | depth-2 does **not** help even at full data (61.7 vs d1-plain 78.1, even below d1-LSA 63.3); severe peak→final collapse (rank0 60.7→35.8, −25 pt) — instability persists at depth |

**The best result overall is 78.1% — vanilla no-FFN LCA, full-data confirmed.**
Depth-2 (`lca_d2_rezero`) confirms *below* both plain d1 studies, so the FFN +
ReZero depth stack has not paid off yet. Whether the **d1** FFN block (`_cb`)
closes the gap is still **unknown until `confirm_d1_rezero_cb.pbs` runs**. Never
rank on the subset proxy alone — confirm top-K, not top-1.

> **Reconstruction caveat (d2):** `confirm_lca_d2.pbs` rebuilds each top-K point
> under the *current* `HpoBase` defaults (config-B-ish), **not** the exact
> pre-cfgB net the `lca_d2_rezero` sweep trained (that was hippo body, τ_max=2,
> no grad-gate). So 61.7% is the config-B reconstruction of those HP points, not
> a pure re-run of the original sweep. The `_cb` confirm has no such caveat (its
> sweep already ran on current defaults).

## Notes / caveats

- **Confounded across generations.** The three pre-cfgB rows ran with old code
  (short hippo, no gate, hippo QKV, RNN_KW input); the cfgB rows ran with new
  code. Two clean single-variable A/Bs exist within the cfgB generation:
  `lca_d1_rezero` vs `…_norecenter` isolates **recenter** (both hippo no-bias
  input); `…_norecenter` vs `…_cb` isolates the **input embedding**
  (uniform τ=5 + bias vs hippo τ≤64 no-bias), both recenter OFF.
- **Key findings so far:**
  - **Best confirmed result overall: 78.1% full-data (vanilla no-FFN LCA `lca`).**
    Full data lifts the plain LCA from 66.3% subset → 78.1%; LSA from 63.9 → 63.3
    (~flat), so the LCA≫LSA advantage is a full-data phenomenon (~15 pt gap).
  - **Depth-2 confirmed at 61.7% full-data — below both plain d1 studies.** The
    FFN + ReZero depth stack has not helped even at scale (78.1 plain > 63.3 LSA
    > 61.7 d2), and its confirms show severe peak→final collapse (−25 pt). The
    d1 FFN block (`_cb`) is the remaining unknown.
  - Plain d1 has the best subset peak (66/64%) and 0 NaN, but is depth-1 only.
  - NaN blow-ups appear only in rezero blocks, only at high lr (5–10e-3).
  - recenter is NOT the NaN cause (removing it left NaN ~unchanged, 30→32) and
    was net-helpful on audio (dead 31 vs 51) — so it was kept as a knob but the
    real fix targeted the collapse layer.
  - The NaN singularity lives in hippo-init, bias-free layers (|z|→0 via SSM
    cancellation): first the LCA K/V projections (pre-cfgB), then the input
    embedding once QKV went uniform. Fixed by making the input embedding uniform
    + bias (validated locally: input min|z| ~0.5 vs ~1e-9).
  - **The input-embedding fix lands (`…_cb` vs `…_norecenter`, both recenter
    OFF, single-variable A/B):** peak 49.5% → **62.6%** (+13 pts, back to
    plain-baseline territory of 66/64%), NaN 32 → **19** (−40%), healthy
    117 → **132**; dead ~unchanged (51 → 49). Confirms the collapse site was the
    hippo no-bias input embedding, and uniform+bias is the right default.
  - Within recenter-OFF cfgB, `…_cb` now beats both prior cfgB rows on peak
    (62.6 vs 54.0 recenter-ON / 49.5 recenter-OFF), so the input fix outweighs
    whatever recenter bought (dead 49 vs 31 recenter-ON — recenter still trims
    dead trials, but is not worth the NaN/peak cost given the input fix).
- **`lca_d1_rezero_cb` is the first sweep at the current package defaults**
  (uniform+bias input). `lca_d2_rezero_cb` (pending) will give the first clean
  d1-vs-d2 depth comparison under matched current settings.
- Subset `peak acc` is a noisy proxy; the `confirm (best full)` column is the
  metric to trust. Confirmed: `lca`, `lsa`, `lca_d2_rezero`. Still pending: the
  d1 FFN block (`confirm_d1_rezero_cb.pbs`, ready) and `lca_d2_rezero_cb`.
