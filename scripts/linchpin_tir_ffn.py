"""Linchpin experiment: does the FFN help on the SAME synthetic task (TIR) in
phasor_torch, the way it does in PhasorNetworks.jl?

Julia TIR (results/temporal_scaling/ffn.csv, HARD config, depth 2):
    FFN on  (d_ff=D)  -> ~0.445      FFN off (attn-only) -> ~0.176   (chance 0.0625)
i.e. the FFN is strongly load-bearing in Julia.

If torch reproduces "FFN load-bearing" on the same task, the audio divergence
(no-FFN wins) is a TASK effect (audio pipeline makes the FFN redundant), and the
implementation is sound. If torch shows the FFN NOT helping on TIR, the divergence
is (at least partly) an implementation / backward-pass effect.

Matches the Julia TIR pipeline as closely as the torch builder allows:
  - frontend = none (phasors fed in directly)
  - body = LCA / LSA, block_type = rezero, use_ffn on/off  (== Julia FFN on/off:
    PhasorTransformerBlock-with-FFN  vs  bare ReZero attention residual)
  - readout codes := the value codebook  (mirror Julia similarity-to-Vc)
  - RMSprop lr 1e-3, alpha x5  (match the Julia TIR optimizer)

Run:  conda run -n nubun python scripts/linchpin_tir_ffn.py
"""
from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phasor_torch.config import ModelConfig
from phasor_torch.train import build_model, forward_model
from phasor_torch.losses import similarity_loss, one_hot

# ---- Julia-TIR HARD config ------------------------------------------------
D          = 48
L          = 32
N_VALS     = 16       # == n_classes; chance = 1/16 = 0.0625
M_SIGNAL   = 3
N_DISTRACT = 16
NOISE      = 0.35
SIG_FRAC   = 1.0      # spread (matches the Julia FFN study, HARD)
DEPTH      = 2
N_HEADS    = 4
N_ANCHORS  = 8
B          = 48
N_TRAIN_B  = 16
N_EVAL_B   = 8
EPOCHS     = 40
LR         = 1e-3
ALPHA_MULT = 5.0

def _wrap(x):
    return np.mod(x + 1.0, 2.0) - 1.0

def setup_codebook():
    rng = np.random.default_rng(777)                       # mirror Julia Xoshiro(777)
    Vf  = (2.0 * rng.random((D, N_VALS)) - 1.0).astype(np.float32)
    cue = (2.0 * rng.random(D) - 1.0).astype(np.float32)
    return Vf, cue

def gen_tir(Vf, cue, n_samples, seed):
    """Return x (D, L, n_samples) Phase, y (n_samples,) int64. Mirrors Julia gen_tir_batch."""
    rng = np.random.default_rng(seed)
    X = np.zeros((D, L, n_samples), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int64)
    sig_hi = max(M_SIGNAL, round(SIG_FRAC * (L - 1)))       # early-window size
    qpos = L - 1
    for b in range(n_samples):
        vstar = int(rng.integers(0, N_VALS)); y[b] = vstar
        sig_pos = rng.choice(sig_hi, size=M_SIGNAL, replace=False)
        taken = set(int(p) for p in sig_pos)
        dis = []
        while len(dis) < N_DISTRACT:
            p = int(rng.integers(0, L - 1))
            if p in taken: continue
            dis.append(p); taken.add(p)
        for p in sig_pos:
            X[:, p, b] = _wrap(Vf[:, vstar] + NOISE * (2.0 * rng.random(D) - 1.0))
        for p in dis:
            u = int(rng.integers(0, N_VALS))
            X[:, p, b] = _wrap(Vf[:, u] + NOISE * (2.0 * rng.random(D) - 1.0))
        X[:, qpos, b] = cue
    return torch.from_numpy(X), torch.from_numpy(y)

def build(body, use_ffn, seed, device, Vf, readout_frac=0.25):
    cfg = ModelConfig(
        in_dims=D, d_hidden=D, frontend="none",
        body=body, n_heads=N_HEADS, n_anchors=N_ANCHORS,
        qkv_init_mode="default", ffn_init_mode="hippo",
        block_type="rezero", gate="rezero", use_ffn=use_ffn,
        recenter=False, d_ff=D, n_blocks=DEPTH,
        readout="ssm", n_classes=N_VALS, readout_frac=readout_frac,
    )
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    model, schema = build_model(cfg, generator=g)
    # readout codes := value codebook (mirror Julia similarity-to-Vc)
    schema["readout"].codes.copy_(torch.from_numpy(Vf))
    model = model.to(device)
    return model, schema

def optimizer(model):
    alpha = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("alpha")]
    rest  = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith("alpha")]
    groups = [{"params": rest, "lr": LR}]
    if alpha:
        groups.append({"params": alpha, "lr": LR * ALPHA_MULT})
    return torch.optim.RMSprop(groups)

def run_trial(body, use_ffn, seed, device, Vf, cue, readout_frac=0.25):
    torch.manual_seed(seed)
    xtr, ytr = gen_tir(Vf, cue, B * N_TRAIN_B, seed)
    xte, yte = gen_tir(Vf, cue, B * N_EVAL_B, 9999)
    xtr, ytr, xte, yte = (t.to(device) for t in (xtr, ytr, xte, yte))
    model, schema = build(body, use_ffn, seed, device, Vf, readout_frac)
    opt = optimizer(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(xtr.shape[2], device=device)
        for i in range(N_TRAIN_B):
            idx = perm[i * B:(i + 1) * B]
            xb, yb = xtr[:, :, idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            sims = forward_model(schema, xb)
            loss = similarity_loss(sims, one_hot(yb, N_VALS))
            loss.backward(); opt.step()
    # eval
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for i in range(N_EVAL_B):
            xb, yb = xte[:, :, i * B:(i + 1) * B], yte[i * B:(i + 1) * B]
            sims = forward_model(schema, xb)
            correct += int((sims.argmax(dim=0) == yb).sum()); total += yb.numel()
    return correct / total, n_params

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Vf, cue = setup_codebook()
    print(f"device={device}  chance={1/N_VALS:.3f}  (Julia ref @ single-pos readout: FFN on ~0.42 / off ~0.24)")
    # readout_frac 0.25 = pooling (torch default); 0.03 -> single last position (matches Julia TIR)
    for frac in (0.25, 0.03):
        tag = "pool-25%" if frac > 0.1 else "single-pos"
        print(f"\n########## readout = {tag} (readout_frac={frac}) ##########")
        rows = []
        for body in ("lca", "lsa"):
            for use_ffn in (True, False):
                accs = []
                for seed in (1, 2):
                    acc, npar = run_trial(body, use_ffn, seed, device, Vf, cue, frac)
                    accs.append(acc)
                    print(f"  [{tag}] {body} ffn={int(use_ffn)} seed={seed}: acc={acc:.3f} params={npar}")
                m = sum(accs) / len(accs)
                rows.append((body, use_ffn, m))
                print(f"  == [{tag}] {body} ffn={int(use_ffn)}: mean acc={m:.3f} ==")
        print(f"=== TORCH TIR SUMMARY [{tag}] (mean of 2 seeds) ===")
        for body in ("lca", "lsa"):
            on  = next(m for b, f, m in rows if b == body and f)
            off = next(m for b, f, m in rows if b == body and not f)
            print(f"  [{tag}] {body}: FFN on={on:.3f}  off={off:.3f}  delta(on-off)={on-off:+.3f}")

if __name__ == "__main__":
    main()
