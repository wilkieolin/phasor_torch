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
| `lca_plain_tier1` | LCA · 1 · plain | **uniform τ=5 + bias** | uniform, no-bias | **none** (single `dense`) | none / none | ✓ · ≤64 · cfgB + **Tier-1 readout** |

**Tier-1 readout** (`lca_plain_tier1` only): identical to `lca_plain_cb` except
the readout uses **softmax-CE loss** (contrastive, `PHASOR_HPO_LOSS=softmax_ce`)
+ **logsumexp pooling** (smooth max over the whole clip = the "keyword present
anywhere" KWS bias, `PHASOR_HPO_READOUT_POOL=logsumexp`) instead of the
non-contrastive similarity-regression + mean pool. `readout_frac` is inert under
logsumexp. All other studies use the similarity+mean readout.

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
| `lca_d2_rezero_cb` (rezero+FFN d2, current defaults) | 67.2% | 69.3% | 30.0% | 69 | 13 | 118 | `hpo_aurora_d2.pbs` |
| **`lca_plain_cb`** (d1 **plain**, current defaults) | 74.8% | 79.3% | 56.4% | 37 | 0 | 163 | `hpo_aurora_lca_plain_cb.pbs` |
| 🥇 **`lca_plain_tier1`** (d1 plain, **Tier-1 readout**) | **81.9%** | **82.3%** | **68.3%** | **11** | 0 | **189** | `hpo_aurora_lca_plain_tier1.pbs` |
| `lca_attn_d1` (attn-only, **no FFN**, d1) | 46.5% | 47.1% | 25.5% | 54 | 0 | 146 | `hpo_aurora_lca_attn_d1.pbs` |
| `lca_attn_d2` (attn-only, **no FFN**, d2) | 65.6% | 47.7% | 27.2% | 44 | 0 | 156 | `hpo_aurora_lca_attn_d2.pbs` |

\*single best trial's subset (16k) test_acc — noisy exploration objective.
†best of the **top-8 full-data confirm** (`confirm.py` re-trains at full ~51k
data, no subset; `best.h5` = restored peak weights). **This is the real
headline metric** — full data lifts every config well above the subset proxy.
Confirm results live in PBS stdout (`phasor_confirm*.o*`), not `history.json`
(those runs predate that field); see `scripts/eval_confirm_curves.py`.

### Full-data confirmation detail (top-8)

Nine studies confirmed (all sweeps except pre-cfgB `lca_d1_rezero*` variants).

| study | best full | incumbent (subset-#1) full | key finding |
|---|---|---|---|
| 🥇 **`lca_plain_tier1`** (plain, **Tier-1 readout**) | **82.3%** (subset-#2) | 76.5% | **NEW BEST overall.** Readout-only swap (softmax-CE + logsumexp) beats the similarity-readout leader (79.3→82.3); also *retains* the peak far better (full_final 79–80% for top configs vs the 5–9 pt drops elsewhere) |
| `lca_plain_cb` (plain, similarity readout) | 79.3% (subset-#2) | 71.0% | current defaults edged past the old plain LCA (78.1→79.3); subset≠full rank (winner subset-#2) |
| `lca` (plain, pre-cfgB) | 78.1% (subset-6th) | 73.7% | subset rank ≠ full rank — full winner was subset-**6th** |
| `lca_d2_rezero_cb` (d2 FFN+ReZero, current) | 69.3% (subset-#5) | 58.1% | **FFN + depth helps at full data: d1 63.7 → d2 69.3 (+5.6 pt)** — best of the attention-stack variants, but still ~13 pt below plain-tier1 |
| `lca_d1_rezero_cb` (d1 FFN+ReZero, current) | 63.7% (subset-#1) | 63.7% | FFN d1 confirms ~15.6 pt below plain-cb; several configs collapse peak→final (44→9, 36→0.4) — FFN instability, restore_best saves the peak |
| `lsa` (plain, pre-cfgB) | 63.3% (subset-#1) | 63.3% | full-data LCA≫LSA gap ~19 pt (82.3 vs 63.3); lr anti-correlates with full acc |
| `lca_d2_rezero` (d2 FFN+ReZero, pre-cfgB) | 61.7% (subset-#4) | 60.7% / 35.8% final | depth-2 (pre-cfgB) ≈ its cfgB cousin minus the tuning; instability persists (−25 pt tail) |
| `lca_attn_d2` (attn-only, **no FFN**, d2) | 47.7% (subset-#4) | **35.4%** | ⚠️ **depth-without-FFN was a subset mirage** — the subset d1→d2 +19 pt gain VANISHES at full data (d2 47.7 ≈ d1 47.1); the subset-#1 config collapsed to 35.4% |
| `lca_attn_d1` (attn-only, **no FFN**, d1) | 47.1% (subset-#4) | 40.0% | attn-only is the **worst** family (~47 pt), ~16 pt below even the FFN d1 — the FFN is doing real work |

**The best result overall is now 82.3% — plain LCA with the Tier-1 readout
(`lca_plain_tier1`).** Two clean full-data lessons from this round:

1. **Readout > architecture.** Swapping only the readout (similarity+mean →
   softmax-CE + logsumexp) on the *same* plain body added +3 pt over the prior
   record (79.3→82.3) and dramatically improved peak retention. It is the single
   biggest lever found, and it holds at full data (subset 81.9 → full 82.3).
2. **The FFN is load-bearing; depth only helps *with* it.** At full data:
   attn-only (no FFN) d1 47.1 ≈ d2 47.7 (depth gives ~nothing) and is the worst
   family; add the FFN and it jumps to 63.7 (d1) → 69.3 (d2), where depth finally
   pays +5.6 pt. So the earlier subset reading — "depth scales without the FFN,
   +19 pt" — was a **proxy artifact** that the full-data confirm reversed. Even
   so, the best attention-stack (d2 FFN, 69.3) trails plain-tier1 by ~13 pt.

Confirm top-K, not top-1: held again everywhere (plain-tier1 winner subset-#2;
attn-d2 subset-#1 → 35.4%).

> **Reconstruction caveat (d2):** `confirm_lca_d2.pbs` rebuilds each top-K point
> under the *current* `HpoBase` defaults (config-B-ish), **not** the exact
> pre-cfgB net the `lca_d2_rezero` sweep trained (that was hippo body, τ_max=2,
> no grad-gate). So 61.7% is the config-B reconstruction of those HP points, not
> a pure re-run of the original sweep. The `_cb` confirm has no such caveat (its
> sweep already ran on current defaults).

### Current-defaults sweep round — subset results

Six sweeps at the current package defaults (uniform+bias input, grad-gate,
hippo τ≤64). **All six are now full-data confirmed** — see the confirmation-
detail table above for the numbers that matter. The subset columns below are
kept for the record and to show where the proxy misled (notably `lca_attn_d2`).

| study | subset peak | subset median | dead | NaN | **full confirm** | subset→full |
|---|---|---|---|---|---|---|
| 🥇 **`lca_plain_tier1`** (plain, Tier-1 readout) | **81.9%** | **68.3%** | **11** | 0 | **82.3%** | ✓ tracked |
| **`lca_plain_cb`** (plain) | 74.8% | 56.4% | 37 | 0 | 79.3% | ✓ tracked |
| `lca_d2_rezero_cb` (rezero+FFN d2) | 67.2% | 30.0% | 69 | 13 | 69.3% | ✓ tracked |
| `lca_attn_d2` (attn-only, no FFN) | 65.6% | 27.2% | 44 | 0 | **47.7%** | ✗ **proxy lied** |
| `lca_d1_rezero_cb` (rezero+FFN d1) | 62.6% | 22.2% | 49 | 19 | 63.7% | ✓ tracked |
| `lca_attn_d1` (attn-only, no FFN) | 46.5% | 25.5% | 54 | 0 | 47.1% | ✓ tracked |

**Findings (now with full-data confirms):**

- **(0) The Tier-1 readout is the biggest lever — and it holds at full data.**
  Readout-only swap (similarity+mean → **softmax-CE + logsumexp**) on the
  plain-LCA leader: subset peak 74.8→**81.9%**, median 56.4→**68.3%**, dead
  **37→11**, 0 NaN — and **full-data confirm 82.3%, the new record** (subset
  81.9 → full 82.3, the proxy tracked cleanly). Its incumbent also grows to a
  **bigger model** (d256/h8 vs plain_cb's d128/h2): the contrastive+KWS readout
  lets width/heads pay off, and it retains the peak far better (top configs hold
  full_final ~79–80%). **Readout choice beat every architecture change tried.**

- **(a) Is plain LCA still the leader? Emphatically yes — and it CONFIRMED to a
  new best.** `lca_plain_cb`: 74.8% peak / **56.4% median** / 0 NaN on the subset
  (old `lca`: 66.3 / 21.7, ≥50% count 20→116), and **full-data confirm = 79.3%**,
  edging past the old 78.1% record — later beaten by the Tier-1 readout (82.3%).
- **(b) Does stacking LCA blocks *without* the FFN scale with depth? NO — the
  subset said yes, full data reversed it.** Subset: attn-only d1 46.5 → d2 65.6
  (+19 pt) looked like clean depth scaling. **Full-data confirm kills it: d1 47.1
  ≈ d2 47.7** — depth gives essentially nothing, and the subset-#1 d2 config
  collapsed to 35.4%. Worse, attn-only is the **worst family full-data** (~47 pt,
  ~16 pt below even the FFN d1). The FFN is load-bearing: adding it lifts d1 to
  63.7 and d2 to 69.3, and **depth only pays off *with* the FFN** (+5.6 pt d1→d2).
  The 0-NaN advantage of removing the FFN was real but bought nothing in accuracy.
  Net: no attention-stack variant reaches plain (best d2-FFN 69.3 vs plain-tier1
  82.3). **A single plain LCA block with the right readout beats every stack.**

All six current-defaults sweeps confirmed. Remaining un-confirmed sweeps are only
the older pre-cfgB `lca_d1_rezero` / `…_norecenter` (superseded, low priority).

## Notes / caveats

- **Confounded across generations.** The three pre-cfgB rows ran with old code
  (short hippo, no gate, hippo QKV, RNN_KW input); the cfgB rows ran with new
  code. Two clean single-variable A/Bs exist within the cfgB generation:
  `lca_d1_rezero` vs `…_norecenter` isolates **recenter** (both hippo no-bias
  input); `…_norecenter` vs `…_cb` isolates the **input embedding**
  (uniform τ=5 + bias vs hippo τ≤64 no-bias), both recenter OFF.
- **Key findings so far:**
  - **Best confirmed result overall: 82.3% full-data — plain LCA + Tier-1 readout
    (`lca_plain_tier1`).** Readout-only swap (similarity+mean → softmax-CE +
    logsumexp) on the same plain body beat the prior 79.3% record by +3 pt and the
    proxy tracked cleanly (subset 81.9 → full 82.3). **Readout choice is the
    single biggest lever — it beat every architecture change tried.** LSA sits at
    63.3%, so the LCA≫LSA advantage (~19 pt) is a full-data phenomenon.
  - **Depth needs the FFN; attention-stacking never beats a single plain block.**
    Full-data: attn-only (no FFN) d1 47.1 ≈ d2 47.7 (depth gives ~nothing, worst
    family); +FFN lifts to d1 63.7 → d2 69.3, where depth finally pays +5.6 pt.
    But the best stack (69.3) still trails plain-tier1 by ~13 pt. **The earlier
    subset claim "depth scales without the FFN, +19 pt" was a proxy artifact the
    full-data confirm reversed** (attn-d2 subset-#1 → 35.4% full).
  - **Confirmed leaderboard:** `lca_plain_tier1` 82.3 > `lca_plain_cb` 79.3 >
    `lca` 78.1 ≫ `lca_d2_rezero_cb` 69.3 > `lca_d1_rezero_cb` 63.7 > `lsa` 63.3 >
    `lca_d2_rezero` 61.7 > `lca_attn_d2` 47.7 > `lca_attn_d1` 47.1.
  - FFN+ReZero confirms show peak→final collapse (d1 configs 44→9 / 36→0.4; d2
    −25 pt); restore_best saves the peak, but the instability is real. The Tier-1
    readout notably *fixes* retention too (top configs hold full_final ~79–80%).
  - Removing the FFN removes the NaN (attn-only 0 NaN vs 13–32 in every rezero+FFN
    sweep) — the FFN is the blow-up source — but that stability bought no accuracy.
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
