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
| `lca_d2_rezero_cb` | LCA · 2 · rezero | **uniform τ=5 + bias** | uniform, no-bias | hippo (τ≤64), bias-on | off / on | ✓ · ≤64 · cfgB (current) |
| `lca_plain_cb` | LCA · 1 · plain | **uniform τ=5 + bias** | uniform, no-bias | **none** (single `dense`: identity, RNN_KW λ=−0.1, no-bias) | none / none | ✓ · ≤64 · cfgB (current) |
| `lca_attn_d1` | LCA · 1 · rezero | **uniform τ=5 + bias** | uniform, no-bias | **none** (`use_ffn=0`) | off / on | ✓ · ≤64 · cfgB (current) |
| `lca_attn_d2` | LCA · 2 · rezero | **uniform τ=5 + bias** | uniform, no-bias | **none** (`use_ffn=0`) | off / on | ✓ · ≤64 · cfgB (current) |

Bias summary: Q/K/V projections bias-free in every sweep; FFN denses bias-on
(rezero+FFN only); input embedding bias-free in all pre-`_cb` sweeps, bias-**on**
in every `_cb`/current-defaults run. FFN (2-layer MLP + ReZero) exists only in
the `rezero`+FFN sweeps; `plain` has a single post-attention `dense` (not an FFN)
and `lca_attn_d{1,2}` are ReZero attention residuals with the FFN removed
(`use_ffn=0`) — the depth-enabling residual kept, the FFN memory tape dropped.

## Results (of 200 trials each)

| study | peak acc* | **confirm (best full)**† | median | dead (≤12%) | NaN blow-ups | healthy | dispatch script |
|---|---|---|---|---|---|---|---|
| `lca` (d1 plain) | 66.3% | **78.1%** | 21.7% | 58 | 0 | 142 | `hpo_aurora.pbs` (body=lca) |
| `lsa` (d1 plain) | 63.9% | 63.3% | 27.6% | 26 | 0 | 174 | `hpo_aurora.pbs` (body=lsa) |
| `lca_d2_rezero` (d2, pre-cfgB) | 62.3% | 61.7% | 22.2% | 76 | 16 | 108 | (pre-cfgB `hpo_aurora_d2.pbs`) |
| `lca_d1_rezero` (rezero, recenter ON) | 54.0% | — (not run) | 29.1% | 31 | 30 | 139 | (cfgB, recenter default was on) |
| `lca_d1_rezero_norecenter` (rezero, recenter OFF) | 49.5% | — (not run) | 22.5% | 51 | 32 | 117 | (removed `_norecenter` script) |
| `lca_d1_rezero_cb` (rezero+FFN, current defaults) | 62.6% | 63.7% | 22.2% | 49 | 19 | 132 | `hpo_aurora_d1_rezero.pbs` |
| `lca_d2_rezero_cb` (rezero+FFN d2, current defaults) | 67.2% | *pending* | 30.0% | 69 | 13 | 118 | `hpo_aurora_d2.pbs` |
| **`lca_plain_cb`** (d1 **plain**, current defaults) | **74.8%** | **79.3%** | **56.4%** | 37 | 0 | 163 | `hpo_aurora_lca_plain_cb.pbs` |
| `lca_attn_d1` (attn-only, **no FFN**, d1) | 46.5% | *pending* | 25.5% | 54 | 0 | 146 | `hpo_aurora_lca_attn_d1.pbs` |
| `lca_attn_d2` (attn-only, **no FFN**, d2) | 65.6% | *pending* | 27.2% | 44 | 0 | 156 | `hpo_aurora_lca_attn_d2.pbs` |

\*single best trial's subset (16k) test_acc — noisy exploration objective.
†best of the **top-8 full-data confirm** (`confirm.py` re-trains at full ~51k
data, no subset; `best.h5` = restored peak weights). **This is the real
headline metric** — full data lifts every config well above the subset proxy.
Confirm results live in PBS stdout (`phasor_confirm*.o*`), not `history.json`
(those runs predate that field); see `scripts/eval_confirm_curves.py`.

### Full-data confirmation detail (top-8)

Five studies confirmed: the two original `plain`, the pre-cfgB depth-2 ReZero,
and the two current-defaults d1 runs (`lca_plain_cb`, `lca_d1_rezero_cb`).

| study | best full | incumbent (subset-#1) full | key finding |
|---|---|---|---|
| **`lca_plain_cb`** (plain, current defaults) | **79.3%** (subset-#2) | 71.0% | **NEW BEST** — current defaults edge past the old plain LCA (78.1→79.3); subset rank ≠ full rank *again* (winner was subset-#2, incumbent → 71.0%) |
| `lca` (plain, pre-cfgB) | 78.1% (subset-6th) | 73.7% | subset rank ≠ full rank — the full winner was the subset-**6th** config; subset-#1 fell to mid-pack |
| `lsa` (plain, pre-cfgB) | 63.3% (subset-#1) | 63.3% | full-data LCA≫LSA gap blows out to **~15 pt** (78.1 vs 63.3), vs ~2.4 pt on the subset; lr anti-correlates with full acc (low lr wins) |
| `lca_d1_rezero_cb` (d1 FFN+ReZero, current) | 63.7% (subset-#1) | 63.7% | the FFN d1 block confirms **~15.6 pt below plain** (63.7 vs 79.3); many configs collapse peak→final (r1 44→9, r3 36→0.4, r5 48→9) — FFN instability, restore_best saves the peak |
| `lca_d2_rezero` (d2 FFN+ReZero, pre-cfgB) | 61.7% (subset-#4) | 60.7% best / **35.8% final** | depth-2 does **not** help even at full data (61.7 vs plain 79.3, below LSA 63.3); severe peak→final collapse (−25 pt) — instability persists at depth |

**The best result overall is now 79.3% — vanilla no-FFN LCA at current defaults
(`lca_plain_cb`), a new high past the old 78.1%.** Every FFN+ReZero variant
confirmed so far sits ~15 pt below vanilla plain (63.7 d1, 61.7 d2), so the FFN
depth stack still has not paid off. Whether **attn-only depth** (`lca_attn_d2`,
which led d1 by +19 pt on the subset with 0 NaN) closes any of that gap is the
key remaining question — pending `confirm_lca_attn_d{1,2}.pbs`. Never rank on the
subset proxy alone — confirm top-K, not top-1 (held again here: plain winner was
subset-#2, not #1).

> **Reconstruction caveat (d2):** `confirm_lca_d2.pbs` rebuilds each top-K point
> under the *current* `HpoBase` defaults (config-B-ish), **not** the exact
> pre-cfgB net the `lca_d2_rezero` sweep trained (that was hippo body, τ_max=2,
> no grad-gate). So 61.7% is the config-B reconstruction of those HP points, not
> a pure re-run of the original sweep. The `_cb` confirm has no such caveat (its
> sweep already ran on current defaults).

### Current-defaults sweep round — subset results

Five sweeps at the current package defaults (uniform+bias input, grad-gate,
hippo τ≤64). Two are now full-data confirmed (`lca_plain_cb` → **79.3%**,
`lca_d1_rezero_cb` → 63.7%; see the confirmation-detail table above); the other
three are subset-only. The subset proxy has repeatedly mis-ranked vs full data,
so read the un-confirmed rows as leaderboards of *candidate pools*, not final
accuracy.

| study | peak | median | ≥50% | dead | NaN | vs. its predecessor |
|---|---|---|---|---|---|---|
| **`lca_plain_cb`** (plain d1) | **74.8%** | **56.4%** | 116 | 37 | 0 | vs `lca` (pre-cfgB): peak 66.3→74.8, **median 21.7→56.4**, ≥50% 20→116 |
| `lca_attn_d2` (attn-only, no FFN) | 65.6% | 27.2% | 3 | 44 | 0 | depth **d1→d2: 46.5→65.6** (+19 pt) |
| `lca_attn_d1` (attn-only, no FFN) | 46.5% | 25.5% | 0 | 54 | 0 | d1 anchor for the depth ladder |
| `lca_d2_rezero_cb` (rezero+FFN d2) | 67.2% | 30.0% | 28 | 13 | — | vs `lca_d1_rezero_cb`: peak 62.6→67.2 |
| `lca_d1_rezero_cb` (rezero+FFN d1) | 62.6% | 22.2% | 13 | 19 | — | (input-embedding fix row) |

**Answering the two questions (on the subset proxy):**

- **(a) Is plain LCA still the leader? Emphatically yes — and it CONFIRMED to a
  new best.** `lca_plain_cb`: 74.8% peak / **56.4% median** / 0 NaN on the subset
  (old `lca`: 66.3 / 21.7, ≥50% count 20→116), and **full-data confirm = 79.3%**,
  edging past the old 78.1% record. The uniform+bias input + grad-gate turned
  plain LCA from "high ceiling, mostly-dead pool" into a broad robust basin *and*
  nudged the ceiling up. It dominates every other topology at both scales.
- **(b) Does stacking LCA blocks *without* the FFN scale with depth? Yes.**
  Attn-only ReZero: **d1 46.5% → d2 65.6% peak** (+19 pt), median 25.5→27.2.
  Depth clearly helps once the FFN is removed — and, notably, **attn-only has 0
  NaN at both depths**, versus 13–32 NaN in every rezero+FFN sweep. Removing the
  FFN eliminated the blow-ups. The small residual gap to `lca_d2_rezero_cb`
  (67.2 peak) shows the FFN buys ~1.6 pt of peak at the cost of instability
  (13 NaN, 69 dead vs 44). Neither depth-2 variant reaches plain d1 (74.8) on
  the subset.

Confirmed: `lca_plain_cb` (79.3%), `lca_d1_rezero_cb` (63.7%). Still to confirm:
`confirm_lca_attn_d{1,2}.pbs` (the depth-without-FFN verdict) and
`confirm_lca_d2_rezero_cb.pbs`.

## Notes / caveats

- **Confounded across generations.** The three pre-cfgB rows ran with old code
  (short hippo, no gate, hippo QKV, RNN_KW input); the cfgB rows ran with new
  code. Two clean single-variable A/Bs exist within the cfgB generation:
  `lca_d1_rezero` vs `…_norecenter` isolates **recenter** (both hippo no-bias
  input); `…_norecenter` vs `…_cb` isolates the **input embedding**
  (uniform τ=5 + bias vs hippo τ≤64 no-bias), both recenter OFF.
- **Key findings so far:**
  - **Best confirmed result overall: 79.3% full-data (vanilla no-FFN LCA at
    current defaults, `lca_plain_cb`)** — a new high past the old `lca` 78.1%.
    Current defaults preserved the leader and nudged the ceiling up. LSA sits at
    63.3%, so the LCA≫LSA advantage (~16 pt) is a full-data phenomenon.
  - **Every FFN+ReZero variant confirms ~15 pt below vanilla plain:** d1
    `lca_d1_rezero_cb` = 63.7%, d2 `lca_d2_rezero` = 61.7% — the FFN depth stack
    has not paid off at any depth or data scale, and its confirms show severe
    peak→final collapse (d1 configs 44→9 / 36→0.4; d2 −25 pt). restore_best saves
    the peak, but the instability is real. Attn-only depth (no FFN) is the
    remaining hope — pending `confirm_lca_attn_d{1,2}.pbs`.
  - Plain d1 dominates the subset (leader 74.8% at current defaults, `lca_plain_cb`)
    with 0 NaN, but is depth-1 only. See the current-defaults subset round above.
  - **Depth scales without the FFN (subset):** attn-only ReZero d1→d2 = 46.5→65.6%.
    Removing the FFN also removes the NaN (0 vs 13–32 in every rezero+FFN sweep) —
    the FFN, not the ReZero stack, is the blow-up source at depth.
  - NaN blow-ups appear only in rezero **+ FFN** blocks, only at high lr (5–10e-3).
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

## Next direction — Tier-1 readout (`lca_plain_tier1`, dispatch READY)

Driven by the local synthetic study + a local audio A/B (see
`../../results/temporal_scaling/FINDINGS_knobs.md` §Tier-1 readout ablation,
`findings_report.html`, and `results/LINCHPIN_FINDINGS.md`). The reconciliation
established the **readout/loss is a bigger lever than any body knob**, and the
biggest movers are:
- **softmax-CE** (contrastive) instead of the non-contrastive `similarity_loss`
  (which only pulls toward the true prototype) — biggest general accuracy lever.
- **logsumexp-over-time** pooling instead of mean-of-last-25% — smooth max over
  the whole clip ("is the keyword present *anywhere*", the KWS inductive bias).

**Local audio A/B (plain LCA, 8k subset, 20 ep, DGX):** Tier-1 trains robustly to
~0.15 at both lr∈{1e-4,3e-4}, while the similarity+mean baseline **dies (~0.01,
below chance) at both lrs** — i.e. the readout, not lr, is the plain-LCA
fragility source. Making Tier-1 the default should raise the ceiling AND remove
the ~18% dead-trial waste (37/200 dead in `lca_plain_cb`).

Dispatch: `scripts/hpo_aurora_lca_plain_tier1.pbs` (plain LCA, no FFN, Tier-1
readout fixed; search space unchanged — width d_hidden{64,128,256}, n_heads{2,4,8},
n_anchors, lr, epochs). New env knobs (read by `HpoBase.from_env`):
`PHASOR_HPO_LOSS`, `PHASOR_HPO_CE_BETA`, `PHASOR_HPO_READOUT_POOL`,
`PHASOR_HPO_LSE_KAPPA`, `PHASOR_HPO_LEARNABLE_CODES`. (Note: `readout_frac` is
inert under logsumexp.) Predicted to beat the current 79.3% leader; confirm
top-K at full data as usual (never rank on the subset proxy alone).
