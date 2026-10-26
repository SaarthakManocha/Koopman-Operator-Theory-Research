"""plot_lab_scaling.py - bare-metal lab machine (28 CPUs): waiting vs number of cores, from kernel counters (schedstat)."""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
df = pd.read_csv(sys.argv[1]); out = sys.argv[2]
df["fam"] = df.capture.str.extract(r"^([a-z]+)")[0]; df["c"] = df.capture.str.extract(r"_c(\d+)_")[0].astype(float)
df["share"] = df.wait_s_kernel / (df.wait_s_kernel + df.run_s_kernel)
df["wps"] = 1e3 * df.wait_s_kernel / df.slices_kernel
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6))
s = df[(df.fam == "scale") & (df.kind == "cpu")].groupby(["c", "capture"]).share.mean().reset_index()
g = s.groupby("c").share.agg(["mean", "min", "max"])
a.errorbar(g.index, 100 * g["mean"], yerr=[100 * (g["mean"] - g["min"]), 100 * (g["max"] - g["mean"])], color="#2a78d6", marker="o", ms=4, lw=1.4, capsize=2)
a.set_xscale("log", base=2); a.set_xticks([1, 2, 4, 8, 16, 28]); a.set_xticklabels(["1", "2", "4", "8", "16", "28"])
a.set_xlabel("cores available"); a.set_ylabel("CPU-bound jobs: time waiting (%)"); a.set_title("Same 8 jobs on more cores", fontsize=8)
for fam, lab, col, mk in (("scale", "8 jobs only (cores mostly idle)", "#a8a6a0", "s"), ("scalebusy", "plus one CPU-bound job per core", "#eb6834", "o")):
    x = df[(df.fam == fam) & (df.kind == "phased")].groupby(["c", "capture"]).wps.mean().reset_index()
    x = x[x.c >= 4]; g = x.groupby("c").wps.agg(["mean", "min", "max"])
    b.errorbar(g.index, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], color=col, marker=mk, ms=4, lw=1.4, capsize=2, label=lab)
b.set_xscale("log", base=2); b.set_xticks([4, 8, 16, 28]); b.set_xticklabels(["4", "8", "16", "28"])
b.set_xlabel("cores available"); b.set_ylabel("interactive job: wait per run (ms)"); b.set_title("Interactive (phased) jobs", fontsize=8)
b.legend(fontsize=7, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight"); print("saved", out)
