"""plot_closedloop.py - figure for the closed-loop slice experiment -> outputs/figures_ext/e7_closedloop.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = [("stock", "stock Linux"), ("static", "short slice, hand-labelled"), ("ewma", "EWMA predictor"),
         ("koopman", "Koopman predictor"), ("ewma_q", "EWMA + conformal margin"),
         ("koopman_q", "Koopman + conformal margin"), ("oracle", "oracle, slice = burst"),
         ("oracle2", "oracle, slice = 2 x burst")]
COL = {"stock": "#52514e", "static": "#a8a6a0", "ewma": "#eb6834", "koopman": "#2a78d6", "ewma_q": "#f3a17f",
       "koopman_q": "#8db7ea", "oracle": "#6250d6", "oracle2": "#b0a6ee"}

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
for ax, (cfg, title) in zip(axes, [("c1_1cpu", "1 CPU: 3 CPU hogs + 4 request tasks (8 runs)"),
                                   ("c2_2cpu", "2 CPUs: 6 CPU hogs + 8 request tasks (6 runs)")]):
    df = pd.read_csv(os.path.join(ROOT, "data", "closedloop", cfg, "runs.csv"))
    g = df.groupby("policy")
    y = [k for k, _ in ORDER if k in g.groups]
    m = g.resp_short_mean.mean().reindex(y); lo = g.resp_short_mean.min().reindex(y); hi = g.resp_short_mean.max().reindex(y)
    pos = range(len(y))[::-1]
    ax.barh(list(pos), m.values, xerr=[m - lo, hi - m], color=[COL[k] for k in y], height=0.65,
            error_kw=dict(lw=0.8, capsize=2, ecolor="#0b0b0b"))
    base = m["stock"]
    for p, k in zip(pos, y):
        ax.text(hi[k] + 0.06, p, f"{m[k]:.2f} ms ({100 * (m[k] / base - 1):+.0f}%)" if k != "stock" else f"{m[k]:.2f} ms",
                va="center", fontsize=7)
    ax.set_yticks(list(pos)); ax.set_yticklabels([dict(ORDER)[k] for k in y])
    ax.set_title(title, loc="left", fontsize=8.5); ax.set_xlim(0, 3.2)
fig.supxlabel("mean response time of short requests (< 1 ms of CPU), ms; bars = mean, whiskers = range over runs", fontsize=8.5)
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "figures_ext", "e7_closedloop.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
