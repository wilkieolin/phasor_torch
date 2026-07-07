"""List NaN-blow-up trials from a sweep, sorted by blow-up epoch, with the
config-B knobs actually stored in each config.json (also verifies the sweep ran
config-B)."""
import glob, json, math, os, sys

STUDY = sys.argv[1] if len(sys.argv) > 1 else "lca_d1_rezero"
ROOT = f"/home/wilkie/code/phasor_torch/hpo_runs/{STUDY}"


def isnan(x):
    try: return math.isnan(float(x))
    except: return False


rows = []
for d in glob.glob(f"{ROOT}/trial_*"):
    hp, cp = os.path.join(d, "history.json"), os.path.join(d, "config.json")
    if not os.path.exists(hp) or not os.path.exists(cp):
        continue
    H = json.load(open(hp)); hist = H["history"] if isinstance(H, dict) and "history" in H else H
    nan_ep = next((r.get("epoch") for r in hist
                   if isnan(r.get("train_loss")) or isnan(r.get("test_loss"))), None)
    if nan_ep is None:
        continue
    peak = max((float(r["test_acc"]) for r in hist if not isnan(r.get("test_acc"))), default=0.0)
    m = json.load(open(cp)).get("model", {})
    rows.append(dict(dir=os.path.basename(d), nan_ep=nan_ep, peak=peak,
                     d=m.get("d_hidden"), h=m.get("n_heads"), a=m.get("n_anchors"),
                     lr=json.load(open(cp)).get("train", {}).get("lr"),
                     iscale=m.get("init_scale"), recenter=m.get("recenter"),
                     qkv=m.get("qkv_init_mode"), ffn=m.get("ffn_init_mode"),
                     block=m.get("block_type"), nblk=m.get("n_blocks")))

rows.sort(key=lambda r: r["nan_ep"])
print(f"{len(rows)} NaN trials in {STUDY}\n")
for r in rows:
    print(f"{r['dir']}  nan@ep{r['nan_ep']:<3} peak={r['peak']*100:4.1f}%  "
          f"d{r['d']} h{r['h']} a{r['a']} lr{float(r['lr']):.2e} is{float(r['iscale']):.2f} | "
          f"recenter={r['recenter']} qkv={r['qkv']} ffn={r['ffn']} {r['block']}x{r['nblk']}")
