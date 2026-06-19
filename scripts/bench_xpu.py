"""Benchmark per-step time and peak device memory for the audio LCA/LSA model.

Characterizes the real audio training path (frontend at L=16000 -> body ->
readout) across a grid of (d_hidden, batch_size) so you can (a) estimate
node-hours for a sweep and (b) see XPU memory headroom before scaling networks
or batch sizes. Uses synthetic random waveforms of the real shape (1, L, B) —
identical compute/memory to real audio, no data staging needed.

Run on an Aurora node (one tile via the affinity wrapper, or just xpu:0):
    module load frameworks && source <venv>/bin/activate
    PYTHONPATH=. python scripts/bench_xpu.py --device xpu \
        --body lca --batch 8,16,32 --d-hidden 64,128,256

Memory uses torch.{xpu,cuda} peak stats; on CPU it reports timing only.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from phasor_torch.config import ModelConfig
from phasor_torch.losses import one_hot, similarity_loss
from phasor_torch.train import build_model, forward_model, select_device


def _mem_mod(device: torch.device):
    if device.type == "xpu":
        return torch.xpu
    if device.type == "cuda":
        return torch.cuda
    return None


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def bench_cell(device, body, d_hidden, batch, args) -> dict:
    """Time one train step and measure peak memory for a single config."""
    mm = _mem_mod(device)
    if mm is not None:
        mm.empty_cache()
        mm.reset_peak_memory_stats()

    cfg = ModelConfig(
        frontend="resonant", body=body, d_hidden=d_hidden, n_heads=args.n_heads,
        n_anchors=args.n_anchors, n_freqs=args.n_freqs,
        downsample_factor=args.downsample, n_classes=args.n_classes,
        readout="ssm", readout_frac=0.25, t_period=1.0,
    )
    g = torch.Generator().manual_seed(0)
    model, schema = build_model(cfg, generator=g)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    x = torch.randn(1, args.L, batch, device=device)          # synthetic waveform batch
    y = torch.randint(0, args.n_classes, (batch,), device=device)
    oh = one_hot(y, args.n_classes)
    ds = args.downsample

    def step():
        opt.zero_grad(set_to_none=True)
        sims = forward_model(schema, x, ds)
        loss = similarity_loss(sims, oh)
        loss.backward()
        opt.step()
        return loss

    for _ in range(args.warmup):                              # warm up lazy init / compile
        step()
    if mm is not None:
        mm.synchronize()
        mm.reset_peak_memory_stats()                          # measure steady-state peak

    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    if mm is not None:
        mm.synchronize()
    sec_per_step = (time.perf_counter() - t0) / args.steps

    res = {"sec_per_step": sec_per_step}
    if mm is not None:
        res["peak_alloc_gib"] = mm.max_memory_allocated() / 2**30
        res["peak_reserved_gib"] = mm.max_memory_reserved() / 2**30
        try:
            res["total_gib"] = mm.get_device_properties(device).total_memory / 2**30
        except Exception:
            res["total_gib"] = float(args.tile_mem_gb)
    del model, schema, opt, x
    if mm is not None:
        mm.empty_cache()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark audio model time + XPU memory.")
    p.add_argument("--device", default="xpu")
    p.add_argument("--body", default="lca", choices=["lca", "lsa"])
    p.add_argument("--batch", default="8,16,32", help="comma list of batch sizes")
    p.add_argument("--d-hidden", default="64,128,256", help="comma list of d_hidden")
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-anchors", type=int, default=32)
    p.add_argument("--n-freqs", type=int, default=64)
    p.add_argument("--downsample", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=30)
    p.add_argument("--L", type=int, default=16000)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--tile-mem-gb", type=float, default=64.0, help="fallback tile HBM (GB)")
    # node-hours extrapolation
    p.add_argument("--num-train", type=int, default=51088, help="train clips per epoch")
    p.add_argument("--epochs", type=int, default=50, help="epochs per trial (for estimate)")
    p.add_argument("--eval-overhead", type=float, default=1.3,
                   help="multiplier for per-epoch eval + data overhead")
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--tiles-per-node", type=int, default=12)
    p.add_argument("--nodes", type=int, default=2)
    args = p.parse_args(argv)

    device = select_device(args.device)
    print(f"device: {device}  body: {args.body}  L: {args.L}  n_freqs: {args.n_freqs} "
          f"downsample: {args.downsample}\n")

    hdr = f"{'d_hidden':>8} {'batch':>5} {'sec/step':>9} {'sec/epoch':>10} " \
          f"{'sec/trial':>10} {'peak_alloc':>11} {'peak_resv':>10} {'%tile':>6}"
    print(hdr)
    print("-" * len(hdr))

    cells = []
    for d_hidden in _ints(args.d_hidden):
        for batch in _ints(args.batch):
            steps_per_epoch = math.ceil(args.num_train / batch)
            try:
                r = bench_cell(device, args.body, d_hidden, batch, args)
            except RuntimeError as e:
                msg = "OOM" if "out of memory" in str(e).lower() else f"ERR:{type(e).__name__}"
                print(f"{d_hidden:>8} {batch:>5} {msg:>9}")
                continue
            sec_epoch = r["sec_per_step"] * steps_per_epoch * args.eval_overhead
            sec_trial = sec_epoch * args.epochs
            alloc = r.get("peak_alloc_gib")
            resv = r.get("peak_reserved_gib")
            total = r.get("total_gib", args.tile_mem_gb)
            pct = (resv / total * 100.0) if resv else 0.0
            astr = f"{alloc:9.2f}G" if alloc is not None else "    n/a"
            rstr = f"{resv:8.2f}G" if resv is not None else "   n/a"
            print(f"{d_hidden:>8} {batch:>5} {r['sec_per_step']:>9.3f} {sec_epoch:>10.1f} "
                  f"{sec_trial:>10.0f} {astr:>11} {rstr:>10} {pct:>5.0f}%")
            cells.append((d_hidden, batch, sec_trial))

    # Node-hours estimate (uses the median measured sec/trial as a representative).
    if cells:
        trials = sorted(c[2] for c in cells)
        med = trials[len(trials) // 2]
        concurrent = args.nodes * args.tiles_per_node - 2   # manager + generator
        concurrent = max(1, concurrent)
        rounds = math.ceil(args.n_trials / concurrent)
        wall_h = rounds * med / 3600.0
        node_h = wall_h * args.nodes
        print(f"\nNode-hours estimate (epochs={args.epochs}, num_train={args.num_train}):")
        print(f"  representative sec/trial (median of grid): {med:.0f}s ({med/3600:.2f}h)")
        print(f"  {args.n_trials} trials / {concurrent} concurrent "
              f"({args.nodes} nodes x {args.tiles_per_node} tiles - 2) = {rounds} rounds")
        print(f"  wall ~= {wall_h:.1f}h   node-hours ~= {node_h:.0f}")
        print("  (scale sec/trial to your real epoch count; estimate ignores stragglers)")


if __name__ == "__main__":
    main()
