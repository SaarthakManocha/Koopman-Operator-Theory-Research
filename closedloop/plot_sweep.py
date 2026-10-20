"""plot_sweep.py - slice margin sweep (eBPF agent) and real-program latency -> outputs/figures_ext/e9_margin_real.png"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
df = pd.read_csv(os.path.join(ROOT, "data", "closedloop", "sweep_1cpu", "runs.csv"))
stock = df[df.policy == "stock"].resp_short_mean.mean()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
for pre, lab, col, mk in (("bpf_koopman_", "Koopman (eBPF agent)", "#2a78d6", "o"), ("bpf_ewma_", "EWMA (eBPF agent)", "#eb6834", "s")):
    cs, m, lo, hi = [], [], [], []
    for c in ("1", "1.5", "2", "3"):
        v = df[df.policy == pre + c].resp_short_mean
        cs.append(float(c)); m.append(v.mean()); lo.append(v.min()); hi.append(v.max())
    ax.errorbar(cs, m, yerr=[np.subtract(m, lo), np.subtract(hi, m)], color=col, marker=mk, ms=4, lw=1.4, capsize=2, label=lab)
cs, m = [], []
for p, c in (("oracle", 1), ("oracle1.5", 1.5), ("oracle2", 2), ("oracle3", 3)):
    cs.append(c); m.append(df[df.policy == p].resp_short_mean.mean())
ax.plot(cs, m, color="#6250d6", marker="^", ms=4, lw=1.2, ls="--", label="oracle (true burst)")
ax.axhline(stock, color="#52514e", lw=0.9, ls=":"); ax.text(2.2, stock + 0.04, "stock Linux", fontsize=7, color="#52514e")
ax.set_xlabel("slice = c x predicted burst;  c"); ax.set_ylabel("short-request response (ms)")
ax.set_xticks([1, 1.5, 2, 3]); ax.legend(fontsize=6.8, loc="lower right", frameon=False)
ax.set_title("Slice margin, 1 CPU, 6 paired runs each", loc="left", fontsize=8.5)

ax = axes[1]
r = pd.read_csv(os.path.join(ROOT, "data", "closedloop", "real_1cpu", "runs.csv"))
metrics = [("redis_mean", "Redis\n(all requests)"), ("sqlite:point_mean", "SQLite\npoint queries"),
           ("node:light_mean", "Node.js\nlight requests")]
pols = [("static", "hand-set short slice", "#a8a6a0"), ("bpf_ewma_1.5", "EWMA, c = 1.5", "#eb6834"),
        ("bpf_koopman_1.5", "Koopman, c = 1.5", "#2a78d6")]
w = 0.26
for j, (p, lab, col) in enumerate(pols):
    vals = []
    for k, _ in metrics:
        P = r.pivot(index="rep", columns="policy", values=k)
        vals.append(100 * (P[p].mean() / P["stock"].mean() - 1))
    ax.bar(np.arange(len(metrics)) + (j - 1) * w, vals, w * 0.92, color=col, label=lab)
ax.axhline(0, color="#0b0b0b", lw=0.8)
ax.set_xticks(range(len(metrics))); ax.set_xticklabels([m[1] for m in metrics])
ax.set_ylabel("mean latency vs stock Linux (%)"); ax.set_ylim(-60, 5)
ax.legend(fontsize=6.8, loc="lower left", frameon=False)
ax.set_title("Real programs sharing a CPU with 3 hogs, 10 paired runs", loc="left", fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "outputs", "figures_ext", "e9_margin_real.png"), dpi=200, bbox_inches="tight")
print("wrote e9_margin_real.png")
