"""Hyperparameter covariance / cross-correlation analysis of the HPO sweeps.

For each study's results.csv (objective = -test_acc; discrete params are integer
index dims), computes:

  1. Spearman rank correlation of each swept HP vs test_acc  -> which knobs drive
     performance, and in which direction.
  2. HP<->HP Spearman correlation matrix among the TOP-quartile configs (sorted
     by test performance) vs the full set -> reveals trade-off ridges in the
     optimum basin (e.g. "high lr only wins with low d_hidden").

Spearman (rank-based) is used throughout: it is scale-free and catches any
monotonic relation, not just linear. lr and weight_decay are analyzed in log10
space (they are swept log-uniformly) though ranks make that moot for Spearman.

No scipy dependency: ranks + Pearson on ranks. Two-sided p via the t-approx
t = r*sqrt((n-2)/(1-r^2)).
"""
import csv, math, os

ROOT = os.path.join(os.path.dirname(__file__), "..", "hpo_runs")
DISC = {"d_hidden": (64, 128, 256), "n_heads": (2, 4, 8),
        "n_anchors": (32, 64, 128, 256)}
STUDIES = ["lca", "lsa", "lca_d1_rezero_cb", "lca_d1_rezero",
           "lca_d1_rezero_norecenter", "lca_d2_rezero"]
# HP columns to analyze. (name, source_col, transform)
HPS = [
    ("lr",          "lr",           "log10"),
    ("init_scale",  "init_scale",   "lin"),
    ("readout_frac","readout_frac", "lin"),
    ("weight_decay","weight_decay", "log10"),
    ("epochs",      "epochs",       "lin"),
    ("d_hidden",    "d_hidden_i",   "disc:d_hidden"),
    ("n_heads",     "n_heads_i",    "disc:n_heads"),
    ("n_anchors",   "n_anchors_i",  "disc:n_anchors"),
]


def _val(row, col, tf):
    v = row.get(col)
    if v in (None, ""):
        return None
    if tf.startswith("disc:"):
        return float(DISC[tf.split(":")[1]][int(float(v))])
    x = float(v)
    if tf == "log10":
        return math.log10(x) if x > 0 else None
    return x


def _rank(xs):
    # average ranks (ties shared)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va == 0 or vb == 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    return cov / math.sqrt(va * vb)


def spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def pval(r, n):
    if n < 3 or abs(r) >= 1.0:
        return 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    # two-sided normal approx to the t distribution (n large)
    z = abs(t)
    return math.erfc(z / math.sqrt(2))


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " "


def load(study):
    rp = os.path.join(ROOT, study, "results.csv")
    rows = [r for r in csv.DictReader(open(rp))
            if r.get("objective") not in (None, "")]
    data = []
    for r in rows:
        obj = float(r["objective"])
        acc = -obj                    # test_acc (crashes obj>0 -> acc<0 = very bad)
        cols = {name: _val(r, col, tf) for name, col, tf in HPS}
        if any(v is None for k, v in cols.items()
               if k != "n_anchors"):  # n_anchors absent for lsa
            continue
        cols["_acc"] = acc
        data.append(cols)
    return data


def col(data, name):
    return [d[name] for d in data if d.get(name) is not None]


def paired(data, x, y):
    xs, ys = [], []
    for d in data:
        if d.get(x) is not None and d.get(y) is not None:
            xs.append(d[x]); ys.append(d[y])
    return xs, ys


def main():
    print("=" * 78)
    print("HP -> test_acc  Spearman rank correlation (per study)")
    print("  +r: higher HP -> higher acc   -r: lower HP -> higher acc")
    print("=" * 78)
    names = [h[0] for h in HPS]
    header = "study".ljust(26) + "".join(n[:9].rjust(11) for n in names)
    print(header)
    per_study_acc = {}
    for s in STUDIES:
        data = load(s)
        per_study_acc[s] = data
        cells = []
        for n in names:
            xs, ys = paired(data, n, "_acc")
            if len(xs) < 5:
                cells.append("--".rjust(11)); continue
            r = spearman(xs, ys); p = pval(r, len(xs))
            cells.append(f"{r:+.2f}{stars(p)}".rjust(11))
        print(s.ljust(26) + "".join(cells))
    print("\n  signif: * p<.05  ** p<.01  *** p<.001  (n~200)")

    # Pooled HP->acc, ranking within each study first (removes cross-study
    # level differences), then correlating pooled within-study ranks.
    print("\n" + "=" * 78)
    print("HP -> test_acc  POOLED across studies (within-study ranks)")
    print("=" * 78)
    for n in names:
        allx, ally = [], []
        for s in STUDIES:
            xs, ys = paired(per_study_acc[s], n, "_acc")
            if len(xs) < 5:
                continue
            rx, ry = _rank(xs), _rank(ys)
            m = len(xs)
            allx += [v / m for v in rx]      # normalize ranks to [~0,1] per study
            ally += [v / m for v in ry]
        if len(allx) < 5:
            print(f"  {n:14s}   --"); continue
        r = _pearson(allx, ally); p = pval(r, len(allx))
        print(f"  {n:14s} {r:+.3f}{stars(p)}   (n={len(allx)})")

    # HP<->HP cross-correlations among TOP-quartile configs, per study.
    print("\n" + "=" * 78)
    print("HP <-> HP  Spearman among TOP-25% configs (trade-off ridges)")
    print("  only |r|>=0.30 with p<.05 shown; compared to full-set r")
    print("=" * 78)
    for s in STUDIES:
        data = sorted(per_study_acc[s], key=lambda d: d["_acc"], reverse=True)
        k = max(8, len(data) // 4)
        top = data[:k]
        hits = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                xs, ys = paired(top, a, b)
                if len(xs) < 6:
                    continue
                r = spearman(xs, ys); p = pval(r, len(xs))
                if abs(r) >= 0.30 and p < 0.05:
                    fx, fy = paired(data, a, b)
                    rf = spearman(fx, fy)
                    hits.append((abs(r), a, b, r, rf, len(xs)))
        hits.sort(reverse=True)
        print(f"\n[{s}]  top-{k} of {len(data)}")
        if not hits:
            print("   (no HP-HP pair with |r|>=0.30, p<.05 in the top quartile)")
        for _, a, b, r, rf, n in hits:
            note = "  (ridge: sign flips vs full)" if r * rf < 0 else ""
            print(f"   {a:12s} <-> {b:12s}  top r={r:+.2f}  full r={rf:+.2f}  n={n}{note}")


if __name__ == "__main__":
    main()
