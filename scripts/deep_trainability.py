"""Deep-net TRAINABILITY study: does depth train, and do bias + v_bind residual
help? (The paper-faithful metric, vs the raw grad-norm probe in grad_stability.)

Trains on the synthetic copy first-token-recall task (a LEARNABLE 10-way task,
chance = 0.10) at increasing depth with each combination of bias / residual, and
reports the best test accuracy reached. Per arXiv:2207.08953 the expectation is
that deep stacks need BOTH bias (Z -> 1+0i, identity-able) and v_bind skips to
stay trainable.

Run:  PYTHONPATH=. python scripts/deep_trainability.py [--body lca|lsa]
"""

from __future__ import annotations

import argparse

from phasor_torch.config import DataConfig, ModelConfig, RunConfig, TrainConfig
from phasor_torch.train import train


def run_one(body, depth, residual, use_bias, epochs, device):
    rc = RunConfig(
        model=ModelConfig(frontend="none", body=body, d_hidden=64, n_heads=4,
                          n_anchors=16, in_dims=64, n_classes=10,
                          n_blocks=depth, residual=residual, use_bias=use_bias),
        data=DataConfig(source="synthetic", task="copy", vocab_size=10,
                        max_length=16, num_train=512, num_test=256, seed=0),
        train=TrainConfig(epochs=epochs, batch_size=32, lr=1e-3, device=device, seed=0),
    )
    hist = train(rc)["history"]
    return max(r["test_acc"] for r in hist)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--body", default="lca", choices=["lca", "lsa"])
    p.add_argument("--device", default="auto")
    p.add_argument("--depths", default="2,4,8")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args(argv)
    from phasor_torch.train import select_device
    device = select_device(args.device)
    depths = [int(d) for d in args.depths.split(",")]

    print(f"device={device} body={args.body} epochs={args.epochs}  "
          f"best test_acc (chance=0.10)\n")
    hdr = f"{'depth':>5} " + " ".join(f"{label:>12}" for label in
          ("plain", "+bias", "+resid", "+bias+resid"))
    print(hdr); print("-" * len(hdr))
    for d in depths:
        cells = []
        for residual, use_bias in ((False, False), (False, True), (True, False), (True, True)):
            acc = run_one(args.body, d, residual, use_bias, args.epochs, device)
            cells.append(f"{acc:>12.3f}")
        print(f"{d:>5} " + " ".join(cells))


if __name__ == "__main__":
    main()
