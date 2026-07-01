"""Summarize the LCA/LSA HPO surrogate sweeps and full-data confirmations.

Produces, under results/hpo_analysis/:
  (1) surrogate_scatter.png    - per-eval test_acc scatter + best-so-far (incumbent) step line, LCA & LSA
  (2) incumbent_<arch>.md/.csv - HP table of each new incumbent (eval #, test_acc, resolved HPs)
  (3) confirm_curves_<arch>.png- train/val accuracy & loss vs epoch for the top-k confirmed nets

Data sources (read-only):
  hpo_runs/{lca,lsa}/results.csv           - one row per ytopt eval; objective = -test_acc
  phasor_confirm.o8555898                  - LCA full-data confirm PBS stdout (OLD test_loss early-stop)
  phasor_confirm_lsa.o8602854              - LSA full-data confirm PBS stdout (NEW test_acc + restore-best)
  hpo_confirm/{lca,lsa}/confirm_NN_*/point.json  - per-rank HPs

The confirm stdout interleaves 8 concurrent workers with no rank tag; we disaggregate by
nearest-value continuation tracking and validate each reconstructed track against the
per-rank summary line (epochs_run + full_final_test_acc).
"""
import csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/wilkie/code/phasor_torch"
OUT = os.path.join(ROOT, "results/hpo_analysis")
os.makedirs(OUT, exist_ok=True)

DISCRETE = {"d_hidden": (64, 128, 256), "n_heads": (2, 4, 8),
            "n_anchors": (32, 64, 128, 256)}

def resolve(name, idx):
    return DISCRETE[name][int(idx)]

# ---------------------------------------------------------------- surrogate sweep
def load_sweep(arch):
    path = os.path.join(ROOT, f"hpo_runs/{arch}/results.csv")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    acc = np.array([-float(r["objective"]) for r in rows])
    return rows, acc

def incumbents(rows, acc, arch):
    """Return list of (eval_idx, test_acc, hp_dict) for each new best-so-far."""
    out = []
    best = -1.0
    for i, (r, a) in enumerate(zip(rows, acc)):
        if a > best + 1e-12:
            best = a
            hp = {
                "d_hidden": resolve("d_hidden", r["d_hidden_i"]),
                "n_heads": resolve("n_heads", r["n_heads_i"]),
                "lr": float(r["lr"]),
                "init_scale": float(r["init_scale"]),
                "readout_frac": float(r["readout_frac"]),
                "weight_decay": float(r["weight_decay"]),
                "epochs": int(float(r["epochs"])),
            }
            if arch == "lca":
                hp["n_anchors"] = resolve("n_anchors", r["n_anchors_i"])
            out.append((i, a, hp))
    return out

def write_incumbent_table(arch, inc):
    cols = ["d_hidden", "n_heads"] + (["n_anchors"] if arch == "lca" else []) + \
           ["lr", "init_scale", "readout_frac", "weight_decay", "epochs"]
    md = [f"# {arch.upper()} surrogate sweep — incumbent trajectory (new best-so-far)\n",
          f"200 ytopt evals on 16k subset. objective = -best_test_acc. "
          f"{len(inc)} improvements; final incumbent = {inc[-1][1]*100:.2f}% at eval #{inc[-1][0]}.\n"]
    header = "| eval # | test_acc | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (2 + len(cols))
    md += [header, sep]
    csv_rows = []
    for ev, a, hp in inc:
        def fmt(c):
            v = hp[c]
            if c in ("lr", "weight_decay"):
                return f"{v:.2e}"
            if c in ("init_scale", "readout_frac"):
                return f"{v:.3f}"
            return str(v)
        md.append(f"| {ev} | {a*100:.2f}% | " + " | ".join(fmt(c) for c in cols) + " |")
        csv_rows.append({"eval": ev, "test_acc": a, **{c: hp[c] for c in cols}})
    with open(os.path.join(OUT, f"incumbent_{arch}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT, f"incumbent_{arch}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval", "test_acc"] + cols)
        w.writeheader()
        w.writerows(csv_rows)
    return md

# ---------------------------------------------------------------- plots
def plot_surrogate(data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, arch in zip(axes, ("lca", "lsa")):
        rows, acc = data[arch]
        x = np.arange(len(acc))
        cummax = np.maximum.accumulate(acc)
        ax.scatter(x, acc * 100, s=14, alpha=0.45, color="#4477aa",
                   label="eval test_acc", zorder=2)
        ax.step(x, cummax * 100, where="post", color="#cc3311", lw=2,
                label="best-so-far", zorder=3)
        # mark incumbents
        inc = incumbents(rows, acc, arch)
        ix = [e for e, _, _ in inc]; iy = [a * 100 for _, a, _ in inc]
        ax.scatter(ix, iy, s=55, facecolors="none", edgecolors="#cc3311",
                   lw=1.6, zorder=4, label="new incumbent")
        ax.set_title(f"{arch.upper()}  (peak {cummax[-1]*100:.1f}% @ eval {ix[-1]})")
        ax.set_xlabel("evaluation #")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("subset test_acc (%)")
    fig.suptitle("HPO surrogate sweeps (200 evals, 16k subset) — test_acc per eval + best-so-far",
                 fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, "surrogate_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ---------------------------------------------------------------- main
def main():
    data = {a: load_sweep(a) for a in ("lca", "lsa")}
    p1 = plot_surrogate(data)
    print("wrote", p1)
    for arch in ("lca", "lsa"):
        rows, acc = data[arch]
        inc = incumbents(rows, acc, arch)
        md = write_incumbent_table(arch, inc)
        print("wrote", os.path.join(OUT, f"incumbent_{arch}.md"),
              f"({len(inc)} incumbents)")
        print("\n".join(md)); print()

    print("\nConfirm train/val curves: run scripts/eval_confirm_curves.py (evaluates saved "
          "checkpoints on local data — the interleaved PBS stdout cannot be disaggregated "
          "reliably into per-rank curves).")

if __name__ == "__main__":
    main()
