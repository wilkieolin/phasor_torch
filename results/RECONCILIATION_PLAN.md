# Reconciliation plan: audio task vs synthetic benchmark

Why does our best **audio** (30-class keyword-spotting, `phasor_torch` on Aurora)
network use **no FFN** (`lca_plain_cb`, 79.3% full-data; every FFN+ReZero variant
confirms ~15 pt lower and is the sole NaN source), when our **local synthetic
benchmark** (TIR, PhasorNetworks.jl) finds the FFN **load-bearing** (remove →
≈chance)? Plus: audio shows **LCA ≫ LSA (~16 pt)** and "depth helps only without
the FFN," while TIR shows LCA≈LSA and depth weak.

## Divergences
| finding | local TIR (Julia) | audio KWS (torch) |
|---|---|---|
| FFN | load-bearing (remove → ≈chance) | hurts ~15 pt + sole NaN/collapse source |
| LCA vs LSA | ≈ wash | LCA ≫ LSA (~16 pt) |
| depth | weak (peak ~3) | helps only *without* FFN (d1→d2 +19 pt subset) |

## Substrate: same vs different
- **Same & forward-parity-verified** (≤1e-5/5e-4): `PhasorDense` λ-conv SSM,
  `PhasorLSA`/`PhasorLCA` (identical Dh=D/H head slicing), VSA ops, per-channel-ω
  rule, `complex_to_angle` 1e-3 backward gate.
- **Different (pipeline):** audio = **ResonantSTFT trainable-ω frontend** + input
  embedding + **SSMReadout pooling over last 25% of 500 steps** + **fixed 30-class
  vocabulary**; TIR = phasors fed directly, single-position headless readout,
  **random-per-example target**. Scale/optimizer differ (D64/L500/Adam/50+ep vs
  D48/L32/RMSProp/few-ep).
- **Parity gap:** forward parity is comprehensive; **backward/gradient parity has
  ZERO coverage** — no script compares Julia Zygote grads vs torch autograd. The
  `complex_to_angle` gate is *asserted* parity-safe, never verified.

## Leading hypothesis (to test, not assume)
The FFN's role (multi-timescale temporal transform+integration) is **redundant on
audio** (frontend does spectral decomposition, LCA anchors memorize keyword
templates, pooling readout integrates over time) but **load-bearing by elimination
on TIR** (nothing else does it). LCA≫LSA follows from the fixed-vocab structure
TIR lacks. A possible co-cause: FFN is the audio NaN/instability source, which
could be a **backward-pass** artifact (untested).

---

## (1) Inspect similarities & differences in results
Reconciliation matrix on shared axes (task-type, frontend, readout, body, FFN
verdict, LCA/LSA, depth, scale, optimizer, NaN); separate **agreements**
("depth-without-FFN helps"?, LCA depth-robustness) from **divergences** (FFN,
LCA≫LSA); normalize metrics (full-data confirm vs 2-seed TIR mean; note the audio
subset-proxy mis-ranking).

## (2) Task disparities — the "TIR-ify / audio-ify" ladder
- **E2a audio-ify TIR (Julia):** add, one at a time — (i) fixed-class template
  vocabulary, (ii) a spectral/frontend stage, (iii) a pooling readout. Predict
  (i) → LCA≫LSA appears; (ii)+(iii) → FFN benefit shrinks/vanishes.
- **E2b TIR-ify audio (torch):** from the plain-LCA winner, remove frontend / swap
  pooling readout for single-position / add FFN — find *when the FFN starts
  mattering*.
- **E2c fixed-vocab probe:** does a fixed template vocabulary in TIR reproduce the
  audio LCA≫LSA advantage? (sharpest single test of the LCA divergence)

## (3) Implementation disparities — forward vs backward parity
- Forward: run `verify_audio_e2e.jl` on the `lca_plain_cb` checkpoint to confirm
  the deployed model still matches (argmax-exact).
- **Backward (the gap):** build a gradient-parity stage — per component
  (`PhasorDense` hippo bias-free, `_PhasorFFN`/`PhasorTransformerBlock`, `PhasorLCA`,
  `v_bind` wrap, `complex_to_angle` gate) and end-to-end: identical weights + input
  + upstream cotangent → compare Zygote vs autograd grads.
- **Stress |z|→0** (the audio FFN NaN site): do forwards agree but backwards
  diverge there? That would make the FFN helpful-in-Julia yet harmful-in-torch.
- Audit silent-drift risks: ω-buffer reconstruction (not serialized), hand-applied
  permutedims / head-transpose conventions, for the audio config.

### Linchpin experiment (disambiguates 2 vs 3)
**Same task, both frameworks, FFN on/off.** Run the identical FFN-on/off ablation
on one shared task (TIR) in both Julia and torch.
- Agree → divergence is **task** (pursue §2); implementation sound.
- Disagree on the same task → **implementation/backward** bug (pursue §3).
Run this first. (Julia TIR already in hand: FFN on 0.445 vs off 0.176 at depth 2.)

## (4) Brainstorm
- **(a) Benchmark:** fixed-vocab/template TIR; add frontend + pooling readout; a
  task suite spanning integration-bound → template-bound (map *when* FFN helps); a
  KWS-like local-pattern synthetic; match audio scale/optimizer.
- **(b) Architecture:** if FFN issue is instability, stabilize it (input-embed bias
  fix in FFN, `MultiModePhasorDense`, grad-clip/norm) & re-test; else move capacity
  to frontend/readout; test our Julia knobs on audio (width, `n_modes`, `tau_max`,
  **head-count** — a free LCA win, matching audio LCA≫LSA + more-heads); pursue
  **selectivity (knob 2)** to filter clip frames (audio is a natural selectivity
  task); double down on **LCA anchor-memory** (more/better anchors).
- **(c) phasor_torch:** add the **backward-parity suite** as a permanent gate;
  **port `MultiModePhasorDense` + λ-range + head-count + `full_d_heads`**; fix the
  FFN NaN at source; finish pending confirms (attn-only depth d1/d2).
- **(d) Other:** run the audio checkpoint in Julia (fwd now, bwd once §3 exists);
  instrument audio FFN training (`|z|` dists, grad norms, peak→final collapse) vs a
  Julia run of the same config; treat the subset-proxy mis-ranking as a clue;
  test whether the win is "no FFN" or "single-timescale dense > hippo 2-layer FFN"
  (the plain dense still carries RNN λ memory).

## Order
1. Linchpin same-task FFN A/B (task vs implementation).
2. Then either the backward-parity harness (§3) or the task-morphing ladder (§2),
   per the linchpin result.
3. Targeted §4 items the diagnosis points to.
