"""plot_baremetal.py - bare-metal closed loop (lab machine, 1/8/16 CPUs): short-request response vs slice margin."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
root, out = sys.argv[1], sys.argv[2]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
for ax, nc, title in zip(axes, ["1cpu", "8cpu", "16cpu"], ["1 CPU, 3 hogs, 4 request tasks", "8 CPUs, 24 hogs, 32 request tasks", "16 CPUs, 48 hogs, 64 request tasks"]):
    df = pd.read_csv(os.path.join(root, f"closedloop_{nc}", "runs.csv"))
    piv = df.pivot(index="rep", columns="policy")["resp_short_mean"]
    cs = [1, 1.5, 2]
    for pre, lab, col, mk, ls in (("oracle", "oracle (true burst)", "#6250d6", "^", "--"),
                                  ("bpf_ewma_", "EWMA (textbook)", "#eb6834", "s", "-"),
                                  ("bpf_koopman_", "Koopman (log-EDMD)", "#2a78d6", "o", "-")):
        names = ["oracle", "oracle1.5", "oracle2"] if pre == "oracle" else [f"{pre}{c:g}" for c in cs]
        m = np.array([piv[n].mean() for n in names]); lo = np.array([piv[n].min() for n in names]); hi = np.array([piv[n].max() for n in names])
        ax.errorbar(cs, m, yerr=[m - lo, hi - m], color=col, marker=mk, ms=4, lw=1.4, ls=ls, capsize=2, label=lab)
    st = piv["stock"].mean()
    ax.axhline(st, color="#52514e", lw=0.9, ls=":"); ax.text(2.02, st + 0.06, f"stock Linux {st:.2f} ms", fontsize=7, color="#52514e", ha="right")
    ax.set_title(title, fontsize=8); ax.set_xticks(cs); ax.set_xticklabels(["1", "1.5", "2"]); ax.set_xlabel("slice margin c")
    ax.set_ylim(0, 2.7)
axes[0].set_ylabel("short-request response (ms)")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False); fig.savefig(out, dpi=220, bbox_inches="tight")
print("saved", out)
