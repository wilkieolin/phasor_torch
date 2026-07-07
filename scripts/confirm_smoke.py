"""Local smoke of the config-B confirm path: top-K LCA-d2 points, tiny data.

Uses the REAL confirm machinery (hpo.HpoBase -> confirm.read_top ->
hpo.point_to_runconfig -> train) so it exercises exactly what the Aurora
confirm job will run, just on a small subset for a few epochs. Verifies config-B
is applied (recenter on, uniform QKV, hippo FFN) and that no config NaNs.

    python scripts/confirm_smoke.py [topk] [epochs] [train_limit]
"""
import os, math, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from phasor_torch import hpo, confirm
from phasor_torch.train import train

TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 2
EPOCHS = sys.argv[2] if len(sys.argv) > 2 else "3"
TRAIN_LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 512
# Depth is configurable so this smokes both the d2 confirm path and the new
# d1-rezero sweep topology (block-matched to d2). HP points are borrowed from the
# d2 results.csv either way -- they parameterize any depth.
N_BLOCKS = int(sys.argv[4]) if len(sys.argv) > 4 else int(os.environ.get("PHASOR_SMOKE_NBLOCKS", "2"))
BLOCK_TYPE = sys.argv[5] if len(sys.argv) > 5 else os.environ.get("PHASOR_SMOKE_BLOCK_TYPE", "rezero")

TRAIN = "/home/wilkie/data/sound/sound_data_raw.h5"
TEST = "/home/wilkie/data/sound/sound_data_raw_test.h5"
RESULTS = "hpo_runs/lca_d2_rezero/results.csv"
for p in (TRAIN, TEST, RESULTS):
    print(f"exists: {os.path.exists(p)}  {p}", flush=True)

base = hpo.HpoBase(
    body="lca", n_blocks=N_BLOCKS, block_type=BLOCK_TYPE, source="audio",
    train_path=TRAIN, test_path=TEST,
    train_limit=TRAIN_LIMIT, test_limit=256,
    device="cuda", patience=0, n_classes=30,
    outdir="/tmp/confirm_smoke",
)
print(f"HpoBase config-B: n_blocks={base.n_blocks} block_type={base.block_type} "
      f"recenter={base.recenter} qkv={base.qkv_init_mode} ffn={base.ffn_init_mode}",
      flush=True)

top = confirm.read_top(RESULTS, TOPK)
ok = True
for i, row in enumerate(top):
    point = {k: v for k, v in row.items() if k != "objective"}
    point["epochs"] = EPOCHS
    run = hpo.point_to_runconfig(point, base)
    m = run.model
    print(f"\n=== config {i}: lr={run.train.lr:.2e} d={m.d_hidden} h={m.n_heads} "
          f"a={m.n_anchors} wd={run.train.weight_decay:.1e} | recenter={m.recenter} "
          f"qkv={m.qkv_init_mode} ffn={m.ffn_init_mode} block={m.block_type} "
          f"nblocks={m.n_blocks} init={m.init_mode}", flush=True)
    assert m.recenter is True and m.qkv_init_mode == "default" and m.ffn_init_mode == "hippo", \
        "config-B not applied!"
    res = train(run)
    hist = res.get("history") or []
    rows = [(r["epoch"], r["train_loss"], r["test_loss"], r["test_acc"]) for r in hist]
    nan = any(math.isnan(a) or math.isnan(b) for _, a, b, _ in rows)
    for ep, tl, vl, va in rows:
        print(f"    epoch {ep}: train_loss={tl:.4f} test_loss={vl:.4f} test_acc={va:.3f}", flush=True)
    print(f"  NaN in losses? {nan}", flush=True)
    ok = ok and not nan

print(f"\nSMOKE {'PASS' if ok else 'FAIL'}: config-B confirm path "
      f"{'trained all configs without NaN' if ok else 'PRODUCED NaN'}", flush=True)
sys.exit(0 if ok else 1)
