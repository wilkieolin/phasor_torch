"""Plot confirm train/val curves from results/hpo_analysis/confirm_curves_data.json.

Separated from eval_confirm_curves.py so plots can be regenerated instantly without
repeating the (slow) checkpoint evaluations. lr is read from each rank's point.json
(it lives in cfg.train, not cfg.model, so it is not in the eval JSON).
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/wilkie/code/phasor_torch"
OUT = os.path.join(ROOT, "results/hpo_analysis")

with open(os.path.join(OUT, "confirm_curves_data.json")) as f:
    data = json.load(f)

def lr_of(arch, rank):
    p = glob.glob(os.path.join(ROOT, f"hpo_confirm/{arch}/confirm_{int(rank):02d}_*/point.json"))
    if not p:
        return None
    return float(json.load(open(p[0]))["lr"])

NOTE = {"lca": "LCA confirm: pre-fix (test_loss early-stop, NO restore) -> final = post-peak last epoch; peak = best.h5",
        "lsa": "LSA confirm: post-fix (test_acc early-stop + restore) -> checkpoint.h5 = best (peak); true pre-restore final was lower (see log)"}

for arch in ("lca", "lsa"):
    ranks = list(data[arch])
    n = len(ranks); ncol = 2; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.8 * nrow), squeeze=False)
    for k, rk in enumerate(ranks):
        ax = axes[k // ncol][k % ncol]
        r = data[arch][rk]
        ep = r["epochs"] + [r["epochs_run"]]
        tr = [t * 100 for t in r["train"]] + [r["final_train"] * 100]
        va = [v * 100 for v in r["val"]] + [r["final_val"] * 100]
        ax.plot(ep, tr, "o-", color="#4477aa", lw=1.6, label="train (2k subset)")
        ax.plot(ep, va, "o-", color="#cc3311", lw=1.8, label="val (full test)")
        ax.axhline(r["best_val"] * 100, color="gray", ls="--", lw=1,
                   label=f"peak val {r['best_val']*100:.1f}%")
        ax.scatter([r["epochs_run"]], [r["final_val"] * 100], color="#cc3311",
                   marker="s", s=45, zorder=5, label=f"final val {r['final_val']*100:.1f}%")
        lr = lr_of(arch, rk)
        htxt = f"d{r['d_hidden']} h{r['n_heads']}"
        if arch == "lca" and r.get("n_anchors"):
            htxt += f" a{r['n_anchors']}"
        if lr is not None:
            htxt += f" lr{lr:.1e}"
        ax.set_title(f"rank {rk}  ({htxt})", fontsize=10)
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
        ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="lower right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{arch.upper()} full-data confirmation — top-{n} nets "
                 f"(checkpoint evals, every-10-epoch grid + peak/final)\n{NOTE[arch]}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(OUT, f"confirm_curves_eval_{arch}.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print("wrote", p)
print("DONE")
