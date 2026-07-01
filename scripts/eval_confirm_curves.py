"""Reconstruct clean train/val accuracy curves for the top-k confirmed nets by
evaluating each SAVED checkpoint on the local data (unambiguous, unlike the
interleaved PBS stdout).

For each top-k confirm dir:
  - ckpt_epoch{N}.h5  -> (epoch N) train_acc (train subset) + val_acc (full test)
  - best.h5           -> peak weights (val_acc == summary best)
  - checkpoint.h5     -> deployed final (LCA: post-peak last epoch; LSA: == best after restore)

Writes results/hpo_analysis/confirm_curves_data.json; plot it with
scripts/plot_confirm_from_json.py (kept separate so plots regenerate instantly
without repeating the slow evals). Run with the nubun python (torch+cuda);
~40-60 min on DGX cuda for top-4 x 2 arches.

NOTE: for confirm runs produced after the confirm.py history.json fix, prefer
reading each dir's history.json directly (full per-epoch curve) instead of this
sparse checkpoint-eval reconstruction. This script exists for the older runs
(hpo_confirm/{lca,lsa}) that predate that fix.
"""
import ast, glob, json, os, re, time, functools, builtins
import h5py, torch
print = functools.partial(builtins.print, flush=True)

ROOT = "/home/wilkie/code/phasor_torch"
os.chdir(ROOT)
import sys; sys.path.insert(0, ROOT)
from phasor_torch.config import ModelConfig
from phasor_torch.train import build_model, evaluate
from phasor_torch.data import load_audio, make_dataloader
from phasor_torch.weights import load_state

OUT = os.path.join(ROOT, "results/hpo_analysis")
os.makedirs(OUT, exist_ok=True)
TEST = "/home/wilkie/data/sound/sound_data_raw_test.h5"
TRAIN = "/home/wilkie/data/sound/sound_data_raw.h5"
N_CLASSES = 30
TOPK = 4
TRAIN_SUB = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

t0 = time.time()
x_te, y_te = load_audio(TEST, N_CLASSES, limit=None, seed=1)
test_loader = make_dataloader(x_te, y_te, 128, shuffle=False, drop_last=False)
x_tr, y_tr = load_audio(TRAIN, N_CLASSES, limit=TRAIN_SUB, seed=0)
train_loader = make_dataloader(x_tr, y_tr, 128, shuffle=False, drop_last=False)
print(f"test={y_te.numel()} train_sub={y_tr.numel()}  (load {time.time()-t0:.1f}s)")

SUMMARY_RE = re.compile(
    r"\[confirm rank (\d+)\].*full_best_test_acc=([\d.]+) "
    r"full_final_test_acc=([\d.]+) epochs_run=(\d+)")
LOGS = {"lca": "phasor_confirm.o8555898", "lsa": "phasor_confirm_lsa.o8602854"}

def parse_summary(logf):
    s = {}
    with open(os.path.join(ROOT, logf)) as f:
        for line in f:
            m = SUMMARY_RE.search(line)
            if m:
                rk, best, fin, ep = m.groups()
                s[int(rk)] = {"best": float(best), "final": float(fin), "epochs": int(ep)}
    return s

def read_cfg(h5path):
    with h5py.File(h5path, "r") as f:
        return ast.literal_eval(f.attrs["cfg.model"])

def eval_both(schema, ds, path):
    load_state(path, schema)
    for m in schema.values():
        m.to(device).eval()
    _, va = evaluate(schema, test_loader, device, N_CLASSES, downsample_factor=ds)
    _, ta = evaluate(schema, train_loader, device, N_CLASSES, downsample_factor=ds)
    return float(ta), float(va)

data = {}
for arch, logf in LOGS.items():
    summ = parse_summary(logf)
    ranks = sorted(summ, key=lambda r: -summ[r]["best"])[:TOPK]
    print(f"\n### {arch.upper()} top-{TOPK}: ranks {ranks}")
    data[arch] = {}
    for rk in ranks:
        d = glob.glob(os.path.join(ROOT, f"hpo_confirm/{arch}/confirm_{rk:02d}_*"))[0]
        mdl = read_cfg(os.path.join(d, "best.h5"))
        pj = os.path.join(d, "point.json")   # lr lives in cfg.train, not cfg.model
        lr = float(json.load(open(pj))["lr"]) if os.path.exists(pj) else None
        ds = int(mdl.get("downsample_factor", 1))
        cfg = ModelConfig(**mdl)
        _, schema = build_model(cfg)
        for m in schema.values():
            m.to(device)
        ckpts = sorted(glob.glob(os.path.join(d, "ckpt_epoch*.h5")),
                       key=lambda p: int(re.search(r"epoch(\d+)", p).group(1)))
        eps, tr, va = [], [], []
        for c in ckpts:
            ep = int(re.search(r"epoch(\d+)", c).group(1))
            t1 = time.time()
            a_tr, a_va = eval_both(schema, ds, c)
            eps.append(ep); tr.append(a_tr); va.append(a_va)
            print(f"  rank{rk} ep{ep:>3}: train={a_tr*100:5.1f}% val={a_va*100:5.1f}% ({time.time()-t1:.0f}s)")
        tr_fin, va_fin = eval_both(schema, ds, os.path.join(d, "checkpoint.h5"))
        tr_best, va_best = eval_both(schema, ds, os.path.join(d, "best.h5"))
        print(f"  rank{rk} FINAL: train={tr_fin*100:.1f}% val={va_fin*100:.1f}% | "
              f"BEST: train={tr_best*100:.1f}% val={va_best*100:.1f}%")
        data[arch][rk] = {
            "epochs": eps, "train": tr, "val": va,
            "epochs_run": summ[rk]["epochs"],
            "final_train": tr_fin, "final_val": va_fin,
            "best_train": tr_best, "best_val": va_best,
            "d_hidden": mdl.get("d_hidden"), "n_heads": mdl.get("n_heads"),
            # n_anchors from cfg.model is a default (32) for LSA; only meaningful for LCA
            "n_anchors": mdl.get("n_anchors") if arch == "lca" else None,
            "lr": lr,
        }

with open(os.path.join(OUT, "confirm_curves_data.json"), "w") as f:
    json.dump(data, f, indent=2)
print("\nwrote confirm_curves_data.json")
print("Now plot with: python scripts/plot_confirm_from_json.py")
