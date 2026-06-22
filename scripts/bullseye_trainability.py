"""Bull's-eye classification: an easy learnable task to check whether v_bind
residual blocks + bias help or hurt deep phasor MLPs.

Faithful port of PhasorNetworks.jl test/data.jl `bullseye_data`: 2 classes,
radius r = N(0, 0.08) + 0.4*y, random angle -> cartesian coords already in
~[-1,1]. The coords ARE the phase input (2-channel, 2D-phase path) -- no
fabricated projection. The network is a complex-valued (phasor) MLP built from
PhasorDense layers in 2D-phase mode (angle->complex->linear(+bias)->normalize->
angle; no temporal SSM), i.e. the paper's PB blocks (arXiv:2207.08953), with an
optional v_bind residual skip per block and a Codebook similarity head.

Reading: a shallow MLP should solve this easily (chance=0.5). If deep plain
stacks degrade but bias+residual recover -> they help; if bias+residual break
this easy task -> they hurt.

Run:  PYTHONPATH=. python scripts/bullseye_trainability.py
"""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from phasor_torch.layers import Codebook, PhasorDense
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.primitives import v_bind
from phasor_torch.train import select_device


def bullseye_data(n, device, seed=0):
    """Port of test/data.jl bullseye_data. Returns coords (2, n) phase, labels (n,)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 2, (n,), generator=g)
    r = torch.randn(n, generator=g) * 0.08 + 0.4 * y
    phi = (torch.rand(n, generator=g) - 1.0) * (2 * math.pi)
    coords = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=0)   # (2, n) in ~[-1,1]
    return coords.float().to(device), y.to(device)


class PhasorMLP(nn.Module):
    """input PhasorDense(2->d) -> depth x PhasorDense(d->d) [+v_bind skip] -> Codebook(d->k)."""

    def __init__(self, d, depth, k, residual, use_bias, gen):
        super().__init__()
        self.residual = residual
        self.input = PhasorDense(2, d, use_bias=use_bias, generator=gen)
        self.blocks = nn.ModuleList(
            [PhasorDense(d, d, use_bias=use_bias, generator=gen) for _ in range(depth)])
        self.readout = Codebook(d, k, generator=gen)

    def forward(self, x):                       # x: (2, B) phase
        h = self.input(x)
        for blk in self.blocks:
            out = blk(h)
            h = v_bind(h, out) if self.residual else out
        return self.readout(h)                  # (k, B)


def run_one(depth, residual, use_bias, *, device, epochs, d=64, lr=5e-3):
    model = PhasorMLP(d, depth, 2, residual, use_bias,
                      torch.Generator().manual_seed(0)).to(device)
    xtr, ytr = bullseye_data(2000, device, seed=1)
    xte, yte = bullseye_data(1000, device, seed=2)
    oh = one_hot(ytr, 2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = 0.0
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        similarity_loss(model(xtr), oh).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            best = max(best, float((model(xte).argmax(0) == yte).float().mean()))
    return best


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--depths", default="1,2,4,8,16")
    p.add_argument("--epochs", type=int, default=120)
    args = p.parse_args(argv)
    device = select_device(args.device)

    print(f"device={device} bullseye (2 classes, chance=0.50) epochs={args.epochs}\n")
    cols = ("plain", "+bias", "+resid", "+bias+resid")
    print(f"{'depth':>5} " + " ".join(f"{c:>12}" for c in cols))
    print("-" * (6 + 13 * len(cols)))
    for d in (int(x) for x in args.depths.split(",")):
        cells = []
        for residual, use_bias in ((False, False), (False, True), (True, False), (True, True)):
            acc = run_one(d, residual, use_bias, device=device, epochs=args.epochs)
            cells.append(f"{acc:>12.3f}")
        print(f"{d:>5} " + " ".join(cells))


if __name__ == "__main__":
    main()
