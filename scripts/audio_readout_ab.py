"""Audio A/B: current readout vs Tier-1 readout, on the plain-LCA winner config.

Tests the local-TIR prediction on REAL 30-class keyword spotting:
  A (baseline): loss=similarity, readout_pool=mean       (the current lca_plain_cb)
  B (tier1):    loss=softmax_ce, readout_pool=logsumexp  (the predicted improvement)

The TIR readout ablation predicted Tier-1 lifts accuracy substantially, largest
for the no-FFN (plain) config. Runs locally on the DGX (nubun env, CUDA) on a
subset before committing Aurora compute.

Run:
  conda run -n nubun python scripts/audio_readout_ab.py --train_limit 8000 --epochs 25 --seeds 0,1
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phasor_torch.config import ModelConfig, DataConfig, TrainConfig, RunConfig
from phasor_torch.train import train

TRAIN_H5 = "/home/wilkie/data/sound/sound_data_raw.h5"
TEST_H5 = "/home/wilkie/data/sound/sound_data_raw_test.h5"

ARMS = {
    # name: (loss, readout_pool)
    "A_baseline": ("similarity", "mean"),
    "B_tier1":    ("softmax_ce", "logsumexp"),
}


def run_arm(name, loss, pool, *, train_limit, test_limit, epochs, seed, batch, lr, ce_beta, kappa):
    model = ModelConfig(
        frontend="resonant", n_freqs=64, downsample_factor=32,
        omega_lo=0.2, omega_hi=2.5, resonant_activation="slerp",
        d_hidden=64, body="lca", n_heads=4, n_anchors=32, init_scale=3.0,
        block_type="plain",                      # the winning topology
        readout="ssm", readout_frac=0.25, n_classes=30, t_period=1.0,
        readout_pool=pool, logsumexp_kappa=kappa,
    )
    data = DataConfig(source="audio", train_path=TRAIN_H5, test_path=TEST_H5,
                      sample_rate=16000, train_limit=train_limit, test_limit=test_limit, seed=seed)
    tr = TrainConfig(batch_size=batch, epochs=epochs, lr=lr, seed=seed, device="auto",
                     loss=loss, ce_beta=ce_beta, restore_best=True,
                     early_stop_metric="test_acc")
    run = RunConfig(model=model, data=data, train=tr)
    out = train(run)
    best = out.get("best") or out.get("final") or {}
    return float(best.get("test_acc", float("nan"))), int(out.get("best_epoch", -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_limit", type=int, default=8000)
    ap.add_argument("--test_limit", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ce_beta", type=float, default=10.0)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--seeds", type=str, default="0")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    print(f"AUDIO READOUT A/B  train_limit={a.train_limit} test_limit={a.test_limit} "
          f"epochs={a.epochs} batch={a.batch} seeds={seeds}")
    results = {}
    for name, (loss, pool) in ARMS.items():
        accs = []
        for seed in seeds:
            acc, ep = run_arm(name, loss, pool, train_limit=a.train_limit, test_limit=a.test_limit,
                              epochs=a.epochs, seed=seed, batch=a.batch, lr=a.lr,
                              ce_beta=a.ce_beta, kappa=a.kappa)
            accs.append(acc)
            print(f"ARMRESULT {name} loss={loss} pool={pool} seed={seed}: best_test_acc={acc:.4f} @ep{ep}")
        results[name] = sum(accs) / len(accs)
    print("\n=== AUDIO A/B SUMMARY (best test_acc, mean over seeds) ===")
    for name in ARMS:
        print(f"  {name}: {results[name]:.4f}")
    if "A_baseline" in results and "B_tier1" in results:
        print(f"  delta (B_tier1 - A_baseline): {results['B_tier1'] - results['A_baseline']:+.4f}")
    print("AB_DONE")


if __name__ == "__main__":
    main()
