# Linchpin: does the FFN help on the SAME task in both frameworks?

**Question.** Audio KWS (torch) finds no-FFN best; local TIR (Julia) finds the FFN
load-bearing. Is that a **task** difference or an **implementation** (Julia↔torch)
difference? Run the identical FFN-on/off ablation on the same synthetic task (TIR)
in both frameworks and compare.

Setup: TIR HARD config (D=48, L=32, n_vals=16, m=3, distract=16, noise=0.35,
spread), LCA/LSA `rezero` block, `use_ffn` on/off (== Julia FFN on/off:
`PhasorTransformerBlock`-with-FFN vs bare ReZero attention residual), RMSprop
lr 1e-3 + α×5, readout codes = value codebook. torch: `scripts/linchpin_tir_ffn.py`;
Julia: `scripts/temporal_scaling_sweep.jl` (`tir_trial`, +`input_embed` option).
2 seeds. Chance = 0.0625.

## Results

| readout | body | FFN on | FFN off | Δ(on−off) |
|---|---|---|---|---|
| **pool 25%** (torch/audio default) | LCA | 0.464 | 0.448 | **+0.016** |
| pool 25% | LSA | 0.430 | 0.414 | +0.016 |
| **single-pos** (torch, readout_frac→1) | LCA | 0.451 | 0.366 | **+0.085** |
| single-pos | LSA | 0.432 | 0.288 | **+0.145** |
| **Julia native** (single-pos, no input embed) | LCA | 0.418 | 0.241 | **+0.177** |
| Julia native | LSA (ffn.csv) | 0.445 | 0.176 | +0.269 |

Sub-experiment — **input embedding is NOT the cause** (Julia, single-pos):
adding a leading input-embedding `PhasorDense` left FFN-off ~unchanged
(0.241 → 0.251); ruled out.

## Conclusion — the readout pooling is the dominant driver

1. **Frameworks initially disagreed** (torch pool-25% FFN Δ≈+0.016 vs Julia
   single-pos Δ≈+0.18) — but that was a **readout confound**, not the FFN itself.
2. **Switching torch to single-position readout restores the FFN's value**
   (LCA Δ +0.016 → +0.085; LSA +0.016 → +0.145): the pooling readout was making
   the FFN redundant.
3. **Unification:** the FFN's role is temporal integration/denoising. In the
   audio pipeline the **SSMReadout pooling** (last 25% of 500 steps) already does
   that → FFN redundant → **no-FFN wins**. In single-position TIR nothing else
   does it → **FFN load-bearing**. It is *not* "the FFN is bad"; it is "the FFN is
   redundant once another layer integrates over time."
4. **Residual framework gap:** torch attn-only single-pos (LCA 0.366) still trains
   better than Julia attn-only (0.241) — the readout doesn't fully close it. The
   difference lives entirely in the **attn-only (FFN-off)** case (FFN-on matches:
   torch 0.451 vs Julia 0.418). Prime suspect: the **untested backward pass**
   (Julia↔torch gradients are never compared) and/or optimizer/init details in the
   attention+residual path. → next: the backward-parity harness (RECONCILIATION_PLAN §3).

## Implications for the reconciliation
- The audio "no-FFN wins" is **largely explained by the pooling readout** (+ the
  ResonantSTFT frontend and LCA templates doing the rest), not by the FFN being
  intrinsically harmful. Our synthetic benchmark under-weighted the readout's role.
- **Benchmark fix:** TIR should be run with a **pooling readout** (and ideally a
  frontend) to be representative of the audio pipeline; the single-position readout
  over-states the FFN's importance.
- **Still open (implementation):** the attn-only training gap → run the backward-
  parity check before trusting any FFN/depth verdict as framework-independent.

## RESOLUTION (supersedes the "readout is dominant" reading above)

Running the **Julia** side with the pooling readout (and later the input
embedding) corrected the earlier torch-only inference. Full decomposition,
Julia LCA, TIR HARD depth-2:

| Julia config | attn-only (FFN off) | FFN on | Δ(on−off) |
|---|---|---|---|
| native (no embed, single-pos), ρ=0.9 | 0.241 | 0.418 | **+0.18** |
| **embed + pool**, ρ=0.9 | 0.375 | 0.392 | **+0.017** |
| **embed + pool**, ρ=0.99 | 0.435 | 0.452 | **+0.017** |
| torch (embed + pool), α=0.99 | 0.448 | 0.464 | +0.016 |

The FFN divergence decomposes into **two additive factors**:

1. **Pipeline = input embedding + pooling readout, *together* (the FFN-redundancy
   driver).** Adding both flips the FFN from load-bearing (Δ+0.18) to redundant
   (Δ+0.017) and lifts attn-only 0.241→0.375. **Each alone is ~null** (embed alone
   0.241→0.251; pool alone 0.241→0.255) — the effect is the *combination*, which
   is why the single-variable tests earlier looked negative. Both elements are
   present in the audio pipeline (ResonantSTFT-fed input embed + SSMReadout
   pooling) and absent in minimal single-position TIR.
2. **Optimizer ρ (0.9 → 0.99) — a minor level shift** (0.375 → 0.435), *not* the
   FFN driver. Julia `Optimisers.RMSProp` defaults ρ=0.9; torch `RMSprop`
   defaults α=0.99. Matching it brings Julia (0.435) in line with torch (0.448).

**Implementation is sound.** Forward parity was already verified; the new
**backward-parity harness** (`verify_gradparity_phasor_{dense,lca}.jl`) shows
Julia Zygote and torch autograd gradients agree to **~1e-6** (short) / ~1e-5
(long FFT) for `PhasorDense` (incl. the `_exp_kdt` rrule, causal-conv/bias-kernel
backward, and the near-origin `complex_to_angle` 1e-3 gate) and for `PhasorLCA`
(k/v projections, λ, anchors, scale, `similarity_outer_heads`, anchor-mix, bind).
With pipeline + optimizer matched, the two frameworks **agree end-to-end**
(0.435 vs 0.448). So the audio "no-FFN wins" is a genuine **task/pipeline**
result (the frontend + pooling readout do the FFN's integration job), not a
port bug.

### Implications
- **Benchmark fix (confirmed):** TIR must use an **input embedding + pooling
  readout** to represent the audio pipeline; the minimal single-position headless
  TIR over-states the FFN. Re-verify the knob findings (depth/width/modes) under
  `input_embed=true, pool_frac=0.25`.
- **Optimizer:** align RMSProp ρ (0.9 vs 0.99) across the two stacks for
  cross-framework comparability, or note it as a known regime difference.
- **Port:** faithful in both forward and backward — safe to trust for training.

## Reproduce
```
# torch (pool vs single-pos sweep)
conda run -n nubun python phasor_torch/scripts/linchpin_tir_ffn.py
# julia (input_embed × FFN, single-pos)
julia --project=. /tmp/confirm_embed.jl   # (inline runner; see git history)
```
