"""Compare LCA HPO studies: new config-B d1-rezero vs prior sweeps.

Reads each study's results.csv (objective = -test_acc from the subset exploration
pass; crashed evals get a +1.0 penalty). Where per-trial history.json exists,
also splits failures into NaN blow-ups vs dead (stuck near chance).
"""
import csv, glob, json, math, os

ROOT = "/home/wilkie/code/phasor_torch/hpo_runs"
DISC = {"d_hidden": (64, 128, 256), "n_heads": (2, 4, 8),
        "n_anchors": (32, 64, 128, 256)}
STUDIES = [
    ("lca_plain_cb",             "d1 PLAIN LCA, current defaults (vanilla leader re-run)"),
    ("lca_attn_d1",              "d1 attn-only rezero (NO FFN)"),
    ("lca_attn_d2",              "d2 attn-only rezero (NO FFN) -- depth-scaling test"),
    ("lca_d2_rezero_cb",         "d2 rezero+FFN, current defaults"),
    ("lca_d1_rezero_cb",         "d1 rezero, current defaults (uniform+bias input, recenter OFF)"),
    ("lca_d1_rezero_norecenter", "d1 rezero, recenter OFF (config-B)"),
    ("lca_d1_rezero",            "d1 rezero, recenter ON  (config-B)"),
    ("lca_d2_rezero",            "d2 rezero (pre-config-B)"),
    ("lca",                      "d1 plain  (pre-config-B baseline)"),
    ("lsa",                      "d1 LSA plain (pre-config-B)"),
]


def resolve(row):
    def r(name):
        v = row.get(name + "_i")
        return DISC[name][int(float(v))] if v not in (None, "") else None
    return dict(d_hidden=r("d_hidden"), n_heads=r("n_heads"),
                n_anchors=r("n_anchors"),
                lr=float(row["lr"]), init_scale=float(row["init_scale"]),
                readout_frac=float(row["readout_frac"]),
                wd=float(row["weight_decay"]), epochs=int(float(row["epochs"])))


def isnan(x):
    try: return math.isnan(float(x))
    except: return False


for study, label in STUDIES:
    d = os.path.join(ROOT, study)
    rp = os.path.join(d, "results.csv")
    if not os.path.exists(rp):
        print(f"\n### {label} [{study}] -- NO results.csv"); continue
    rows = [r for r in csv.DictReader(open(rp)) if r.get("objective") not in (None, "")]
    objs = [float(r["objective"]) for r in rows]
    n = len(objs)
    crash = [o for o in objs if o > 0]                 # +1.0 penalty
    valid = [(-o, r) for o, r in zip(objs, rows) if o <= 0]  # (test_acc, row)
    accs = sorted((a for a, _ in valid), reverse=True)
    best_acc, best_row = max(valid, key=lambda t: t[0]) if valid else (0.0, None)
    med = accs[len(accs)//2] if accs else 0.0
    def pct(thr): return sum(1 for a in accs if a >= thr)
    print(f"\n### {label}  [{study}]")
    print(f"  evals={n}  crashes(obj>0)={len(crash)} ({100*len(crash)/max(1,n):.0f}%)  "
          f"valid={len(valid)}")
    print(f"  best test_acc(subset)={best_acc*100:.1f}%   median={med*100:.1f}%")
    print(f"  count >=20%: {pct(0.20)}   >=30%: {pct(0.30)}   >=40%: {pct(0.40)}   >=50%: {pct(0.50)}")
    if best_row:
        h = resolve(best_row)
        print(f"  incumbent HPs: d{h['d_hidden']} h{h['n_heads']} a{h['n_anchors']} "
              f"lr{h['lr']:.2e} init_scale{h['init_scale']:.2f} ro{h['readout_frac']:.2f} "
              f"wd{h['wd']:.1e} ep{h['epochs']}")
    # per-trial NaN vs dead split, if history is present
    trials = glob.glob(os.path.join(d, "trial_*/history.json"))
    if trials:
        nnan = ndead = 0
        nan_peaks = []; nan_eps = []
        for hp in trials:
            H = json.load(open(hp)); hist = H["history"] if isinstance(H, dict) and "history" in H else H
            first_nan = next((r.get("epoch") for r in hist
                              if isnan(r.get("train_loss")) or isnan(r.get("test_loss"))), None)
            peak = max((float(r["test_acc"]) for r in hist if not isnan(r.get("test_acc"))), default=0.0)
            if first_nan is not None:
                nnan += 1; nan_peaks.append(peak); nan_eps.append(first_nan)
            elif peak <= 0.12:
                ndead += 1
        print(f"  per-trial ({len(trials)}): NaN blow-ups={nnan}  dead(<=12%)={ndead}  "
              f"healthy={len(trials)-nnan-ndead}")
        if nan_peaks:
            nan_peaks.sort()
            med = nan_peaks[len(nan_peaks)//2]
            hi = sum(1 for p in nan_peaks if p >= 0.30)
            print(f"    NaN trials' pre-blowup peak test_acc: median={med*100:.1f}% "
                  f"max={max(nan_peaks)*100:.1f}%  ({hi}/{nnan} peaked >=30% before blowing up)")
            print(f"    NaN first-epoch: min={min(nan_eps)} median={sorted(nan_eps)[len(nan_eps)//2]} max={max(nan_eps)}")
    else:
        print("  (no per-trial history.json synced -> NaN/dead split unavailable)")
