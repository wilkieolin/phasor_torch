"""Gradient-stability study for deep stacked LCA/LSA blocks.

Sweeps network depth (n_blocks) with and without v_bind residual skips and
reports, for each, the per-block gradient-norm profile from a single backward
plus a short overfit-convergence smoke. The hypothesis: without residuals the
per-block grad norm drifts away from O(1) with depth (vanishing toward the input
and/or exploding) and deep models fail to descend; with v_bind residuals the
profile stays flat and deep models train.

Synthetic phase input (no audio data, no frontend) so it isolates the block
stacking. Run:
    PYTHONPATH=. python scripts/grad_stability.py [--body lca|lsa] [--device auto]
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict

import torch

from phasor_torch.config import ModelConfig
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.train import build_model, forward_model, select_device


def _grad_norm(module) -> float:
    s = 0.0
    for p in module.parameters():
        if p.grad is not None:
            s += float(p.grad.detach().norm()) ** 2
    return math.sqrt(s)


def _block_profile(schema) -> tuple[float, list[float]]:
    """(input grad norm, [per-block grad norm], block 0 = nearest input)."""
    input_gn = _grad_norm(schema["input"])
    acc: dict[int, float] = defaultdict(float)
    for name, layer in schema.items():
        m = re.match(r"(?:body|dense|block)(\d*)$", name)
        if not m:
            continue
        idx = int(m.group(1)) if m.group(1) else 0
        acc[idx] += _grad_norm(layer) ** 2
    prof = [math.sqrt(acc[i]) for i in sorted(acc)]
    return input_gn, prof


def _build(cfg_kw, device, gen):
    cfg = ModelConfig(frontend="none", in_dims=64, d_hidden=64, n_heads=4,
                      n_anchors=16, n_classes=10, **cfg_kw)
    model, schema = build_model(cfg, generator=gen)
    return model.to(device), schema


def run_case(body, n_blocks, residual, device, *, use_bias=False, L=32, B=16,
             steps=30, lr=1e-3):
    gen = torch.Generator().manual_seed(0)
    model, schema = _build(
        dict(body=body, n_blocks=n_blocks, residual=residual, use_bias=use_bias),
        device, gen)
    xg = torch.Generator().manual_seed(1)
    x = (torch.rand(64, L, B, generator=xg) * 2 - 1).to(device)
    y = torch.randint(0, 10, (B,), generator=xg).to(device)
    oh = one_hot(y, 10)

    # one backward for the grad-norm profile
    model.zero_grad(set_to_none=True)
    loss0 = similarity_loss(forward_model(schema, x), oh)
    loss0.backward()
    input_gn, prof = _block_profile(schema)

    # short overfit smoke: can it descend?
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = float(loss0.detach())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = similarity_loss(forward_model(schema, x), oh)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    return input_gn, prof, float(loss0.detach()), last


def main(argv=None):
    p = argparse.ArgumentParser(description="Gradient stability vs depth.")
    p.add_argument("--body", default="lca", choices=["lca", "lsa"])
    p.add_argument("--device", default="auto")
    p.add_argument("--depths", default="1,2,4,8,16")
    p.add_argument("--steps", type=int, default=30)
    args = p.parse_args(argv)
    device = select_device(args.device)
    depths = [int(d) for d in args.depths.split(",")]

    print(f"device={device} body={args.body} steps={args.steps}  "
          f"(block grad norms: idx0 = nearest input)\n")
    hdr = f"{'resid':>5} {'bias':>5} {'depth':>5} {'in_gn':>9} {'blk0_gn':>9} {'blkN_gn':>9} " \
          f"{'b0/bN':>10} {'loss0':>7} {'loss_f':>7} {'descend':>7}"
    print(hdr)
    print("-" * len(hdr))
    for residual, use_bias in ((False, False), (True, False), (False, True), (True, True)):
        for d in depths:
            in_gn, prof, l0, lf = run_case(args.body, d, residual, device,
                                           use_bias=use_bias, steps=args.steps)
            b0, bN = prof[0], prof[-1]
            ratio = b0 / bN if bN > 0 else float("inf")
            descend = "yes" if lf < l0 - 1e-4 else "NO"
            print(f"{str(residual):>5} {str(use_bias):>5} {d:>5} {in_gn:>9.2e} "
                  f"{b0:>9.2e} {bN:>9.2e} {ratio:>10.2f} {l0:>7.3f} {lf:>7.3f} {descend:>7}")
        print()


if __name__ == "__main__":
    main()
