"""Localize the depth-2 LCA gradient explosion: reproduce a divergent HPO trial
on local audio and, each step, measure WHERE the backward gradient blows up and
HOW it stacks toward the input.

Instruments three things per optimizer step:
  1. boundary activation-gradient magnitude (max|dL/d(layer output)|) at every
     module boundary, in forward order -> shows the backward amplification
     profile (readout side -> input side).
  2. per-layer parameter-gradient L2 norm -> which layer's params get the
     largest update (where the blow-up lands).
  3. complex_to_angle singularity: min active |z| entering any angle op and
     max |dz| out of its backward (the ~1/|z|^2 driver), plus the ReZero alphas
     (as they open, the block branches -- and their angle ops -- turn on).

Stops at the first non-finite loss/grad and dumps the full profile for that step
and the previous (finite) step. Run with nubun python (torch+cuda):
    python scripts/grad_diverge_probe.py --trial hpo_runs/lca_d2_rezero/trial_6aaff72e52
"""
from __future__ import annotations
import argparse, json, math, os
import torch

ROOT = "/home/wilkie/code/phasor_torch"
import sys; sys.path.insert(0, ROOT)
from phasor_torch.config import ModelConfig, DataConfig, TrainConfig, RunConfig
from phasor_torch.train import build_model, forward_model, build_optimizer
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.data import make_audio_dataloaders
import phasor_torch.primitives as P
from phasor_torch.layers.phasor_residual import (
    PhasorTransformerBlock, PhasorResidual, _PhasorFFN, PhaseRecenter)
from phasor_torch.layers.phasor_lca import PhasorLCA
from phasor_torch.layers.phasor_dense import PhasorDense
from phasor_torch.layers.ssm_readout import SSMReadout

LOCAL_TRAIN = "/home/wilkie/data/sound/sound_data_raw.h5"
LOCAL_TEST = "/home/wilkie/data/sound/sound_data_raw_test.h5"
# PhaseRecenter added (config-B): its circular-mean complex_to_angle is a NEW
# singularity source vs the pre-config-B blocks -- tag + capture it.
INTERESTING = (PhasorTransformerBlock, PhasorResidual, _PhasorFFN, PhasorLCA,
               PhasorDense, SSMReadout, PhaseRecenter)

# ---- complex_to_angle singularity probe (patch the autograd Function) --------
# CUR is set by forward_pre_hooks so each complex_to_angle call is attributed to
# the layer whose forward is running (frontend glue to_phase runs right after the
# frontend module, so it is tagged 'frontend').
PROBE = {"min_z": math.inf, "max_dz": 0.0, "cur": "?",
         "byz": {}, "bydz": {}}
_of = P._ComplexToAngle.forward
_ob = P._ComplexToAngle.backward
def _pf(ctx, z, threshold, grad_threshold):
    y = _of(ctx, z, threshold, grad_threshold)
    ctx.cur = PROBE["cur"]
    with torch.no_grad():
        r2 = z.real * z.real + z.imag * z.imag
        th2 = float(threshold) ** 2
        act = r2[r2 > th2]
        if act.numel():
            mz = float(act.min().sqrt())
            PROBE["min_z"] = min(PROBE["min_z"], mz)
            k = PROBE["cur"]
            PROBE["byz"][k] = min(PROBE["byz"].get(k, math.inf), mz)
    return y
def _pb(ctx, ybar):
    dz, g1, g2 = _ob(ctx, ybar)
    with torch.no_grad():
        mag = (dz.real * dz.real + dz.imag * dz.imag).sqrt()
        if mag.numel():
            md = float(mag.max())
            PROBE["max_dz"] = max(PROBE["max_dz"], md)
            k = getattr(ctx, "cur", "?")
            PROBE["bydz"][k] = max(PROBE["bydz"].get(k, 0.0), md)
    return dz, g1, g2
P._ComplexToAngle.forward = staticmethod(_pf)
P._ComplexToAngle.backward = staticmethod(_pb)


def finite(x): return math.isfinite(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="hpo_runs/lca_d2_rezero/trial_6aaff72e52")
    ap.add_argument("--train-limit", type=int, default=4096)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--lr-mult", type=float, default=1.0, help="scale lr to force faster divergence")
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--spike", type=float, default=1e3, help="dump full profile at first max|dz| over this")
    ap.add_argument("--recenter", choices=["config", "on", "off"], default="config",
                    help="override PhaseRecenter: 'off' ablates it (config-B NaN-source test)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(ROOT, args.trial, "config.json")))
    if args.recenter != "config":
        cfg["model"]["recenter"] = (args.recenter == "on")
    model_cfg = ModelConfig(**cfg["model"])
    data_cfg = DataConfig(**{**cfg["data"], "train_path": LOCAL_TRAIN,
                             "test_path": LOCAL_TEST, "train_limit": args.train_limit,
                             "test_limit": 256})
    train_cfg = TrainConfig(**{**cfg["train"], "device": "cuda",
                               "lr": cfg["train"]["lr"] * args.lr_mult})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = model_cfg.downsample_factor if model_cfg.frontend == "resonant" else 1
    print(f"trial={args.trial}  lr={train_cfg.lr:.3e} (x{args.lr_mult})  "
          f"d{model_cfg.d_hidden} h{model_cfg.n_heads} a{model_cfg.n_anchors} "
          f"blocks={model_cfg.n_blocks}/{model_cfg.block_type}  recenter={model_cfg.recenter}  "
          f"ds={ds}  device={device}", flush=True)

    g = torch.Generator(device="cpu").manual_seed(train_cfg.seed)
    train_loader, _ = make_audio_dataloaders(
        LOCAL_TRAIN, LOCAL_TEST, model_cfg.n_classes, train_cfg.batch_size,
        train_limit=args.train_limit, test_limit=256, seed=data_cfg.seed, generator=g)
    model, schema = build_model(model_cfg, generator=g)
    model = model.to(device)
    opt = build_optimizer(model, train_cfg)

    # forward hooks: capture each interesting module's output (retain grad), in fwd order
    captures = {}
    def mk_hook(name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.retain_grad(); captures[name] = out
        return hook
    for top, mod in schema.items():
        for sub, m in mod.named_modules():
            if isinstance(m, INTERESTING):
                full = top if sub == "" else f"{top}.{sub}"
                m.register_forward_hook(mk_hook(full))
    # tag complex_to_angle calls by the deepest interesting module currently
    # executing (frontend glue to_phase runs just after the frontend module -> 'frontend')
    def mk_pre(name):
        def pre(mod, inp):
            PROBE["cur"] = name
        return pre
    for top, mod in schema.items():
        for sub, m in mod.named_modules():
            full = top if sub == "" else f"{top}.{sub}"
            m.register_forward_pre_hook(mk_pre(full))
    alpha_named = {n: p for n, p in model.named_parameters() if n.endswith("alpha")}

    def act_prof():   # max|grad| at each captured boundary, forward order
        return [(n, float(t.grad.abs().max()) if t.grad is not None else 0.0)
                for n, t in captures.items()]
    def fwd_prof():   # (name, max|act|, all_finite) at each boundary, forward order
        out = []
        for n, t in captures.items():
            with torch.no_grad():
                fin = bool(torch.isfinite(t).all())
                mx = float(t.abs().max()) if fin else float("inf")
            out.append((n, mx, fin))
        return out
    def first_nonfinite(fp):   # earliest forward boundary that went non-finite
        return next((n for n, _, fin in fp if not fin), None)
    def pgrad_top():  # per top-level schema layer param-grad L2
        out = []
        for name, mod in schema.items():
            s = sum(float(p.grad.detach().norm())**2 for p in mod.parameters() if p.grad is not None)
            out.append((name, math.sqrt(s)))
        return out
    def alphas():
        return {n.split(".block")[-1] if ".block" in n else n:
                float(p.detach().mean()) for n, p in alpha_named.items()}

    def dump(tag, step, loss, ap_, pg_, probe, alph, fp_=None):
        print(f"\n--- {tag} step {step} loss={loss} ---")
        if fp_ is not None:
            print("  boundary FWD activation (max|act|, forward order); '**' = non-finite:")
            for n, mx, fin in fp_:
                flag = "" if fin else "  ** NON-FINITE **"
                print(f"    {n:38s} {mx:.3e}{flag}")
        print("  boundary act-grad (max|dL/dout|), forward order (input->output):")
        for n, v in ap_:
            print(f"    {n:38s} {v:.3e}")
        print("  per-layer param-grad L2:")
        for n, v in pg_:
            print(f"    {n:12s} {v:.3e}")
        print(f"  complex_to_angle: min|z|={probe['min_z']:.3e}  max|dz|={probe['max_dz']:.3e}")
        print("  per-layer min|z| (singularity source):  " +
              "  ".join(f"{k}={v:.2e}" for k, v in sorted(probe.get("byz", {}).items())))
        print("  per-layer max|dz| (singularity gradient): " +
              "  ".join(f"{k}={v:.2e}" for k, v in sorted(probe.get("bydz", {}).items())))
        print(f"  alphas: " + "  ".join(f"{k}={v:+.3f}" for k, v in alph.items()), flush=True)

    step = 0
    prev = None
    dumped_spike = False
    data_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); x, y = next(data_iter)
        x = x.to(device); y = y.to(device)
        PROBE["min_z"] = math.inf; PROBE["max_dz"] = 0.0
        PROBE["byz"] = {}; PROBE["bydz"] = {}
        captures.clear()
        opt.zero_grad(set_to_none=True)
        sims = forward_model(schema, x, ds)
        loss = similarity_loss(sims, one_hot(y, model_cfg.n_classes))
        loss.backward()
        lv = float(loss.detach())
        ap_ = act_prof(); pg_ = pgrad_top(); pr = dict(PROBE); al = alphas()
        fp_ = fwd_prof()
        gmax = max((v for _, v in ap_), default=0.0)
        pmax = max((v for _, v in pg_), default=0.0)
        fwd_bad = first_nonfinite(fp_)
        bad = (not finite(lv)) or (not finite(gmax)) or (not finite(pr["max_dz"])) or (fwd_bad is not None)
        if bad:
            if prev is not None:
                dump("LAST FINITE", *prev)
            dump("BLOW-UP", step, lv, ap_, pg_, pr, al, fp_)
            # forward vs backward origin: the first non-finite FORWARD boundary is
            # the source if the blow-up is in the forward pass.
            if fwd_bad is not None:
                print(f"\n  >> FORWARD blow-up: first non-finite activation at boundary '{fwd_bad}'")
            fin_ap = [(n, v) for n, v in ap_ if finite(v)]
            if fin_ap:
                top_b = max(fin_ap, key=lambda t: t[1])
                print(f"  >> largest finite boundary grad: {top_b[0]} = {top_b[1]:.3e}")
            print(f"  >> diverged at step {step} "
                  f"(~epoch {step // max(1,len(train_loader))}) lr={train_cfg.lr:.2e}", flush=True)
            return
        if (not dumped_spike) and pr["max_dz"] > args.spike:
            dump(f"SPIKE (max|dz|>{args.spike:g})", step, lv, ap_, pg_, pr, al)
            dumped_spike = True
        if step % args.every == 0:
            top_b = max(ap_, key=lambda t: t[1]) if ap_ else ("-", 0)
            top_p = max(pg_, key=lambda t: t[1]) if pg_ else ("-", 0)
            dz_src = max(pr.get("bydz", {}).items(), key=lambda t: t[1], default=("-", 0))
            fmax_n, fmax_v, _ = max(fp_, key=lambda t: t[1]) if fp_ else ("-", 0.0, True)
            amean = sum(al.values())/max(1,len(al))
            print(f"step {step:5d} loss={lv:6.3f} | fmax={fmax_v:.2e}@{fmax_n:8s} "
                  f"gmax={gmax:.2e}@{top_b[0]:8s} pmax={top_p[1]:.2e}@{top_p[0]:8s} | "
                  f"minZ={pr['min_z']:.1e} maxDZ={pr['max_dz']:.1e}@{dz_src[0]:8s} | a={amean:+.3f}",
                  flush=True)
        prev = (step, lv, ap_, pg_, pr, al, fp_)
        opt.step()
        step += 1
    print(f"\nreached max_steps={args.max_steps} without NaN (lr={train_cfg.lr:.2e}). "
          f"Try --lr-mult 1.5 or smaller --train-limit to force divergence.", flush=True)
    if prev: dump("FINAL", *prev)


if __name__ == "__main__":
    main()
