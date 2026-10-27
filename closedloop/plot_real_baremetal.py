"""plot_real_baremetal.py - real servers on bare metal (i7-7700): change in latency vs stock EEVDF per request type."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
r = pd.read_csv(sys.argv[1]); out = sys.argv[2]
metrics = [("redis_mean", "Redis\nall requests"), ("redis_p99", "Redis\n99th pct."), ("sqlite:point_mean", "SQLite\npoint queries"),
           ("sqlite:range_mean", "SQLite\nrange queries"), ("node:light_mean", "Node.js\nlight requests"), ("node:heavy_mean", "Node.js\nheavy requests")]
pols = [("static", "hand-set minimum slice", "#a8a6a0"), ("bpf_ewma_1.5", "textbook EWMA, c = 1.5", "#eb6834"),
        ("bpf_koopman_1.5", "Koopman (log-EDMD), c = 1.5", "#2a78d6")]
fig, ax = plt.subplots(figsize=(7.4, 2.9)); w = 0.26
for j, (p, lab, col) in enumerate(pols):
    m, lo, hi = [], [], []
    for k, _ in metrics:
        P = r.pivot(index="rep", columns="policy", values=k); base = P["stock"].mean()
        x = 100 * (P[p] / P["stock"] - 1); m.append(100 * (P[p].mean() / base - 1)); lo.append(m[-1] - min(m[-1], x.min())); hi.append(max(m[-1], x.max()) - m[-1])
    ax.bar(np.arange(len(metrics)) + (j - 1) * w, m, w * 0.9, color=col, label=lab, yerr=[lo, hi],
           error_kw=dict(lw=0.7, capsize=1.5, ecolor="#52514e"))
ax.axhline(0, color="#0b0b0b", lw=0.8)
ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m[1] for m in metrics])
ax.set_ylabel("change vs stock Linux (%)")
ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight"); print("saved", out)
