"""
plots_paper_extra.py - figures added for the long version of the paper -> outputs/figures_ext/
  a1_pipeline.png    system overview (measurement -> prediction -> slice control)
  b1_baselines.png   error of each model relative to log-AR(p*), per task group
  s1_spectra.png     Koopman eigenvalues of each real program, windows of 150 bursts
"""
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edmd import EDMD  # noqa: E402
from sched_events import cpu_bursts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "figures_ext")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, MUTED = "#0b0b0b", "#52514e"


def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 3.1)); ax.set_xlim(0, 100); ax.set_ylim(-3, 50); ax.axis("off")

    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=col, ec=MUTED, lw=0.8))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=7.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=6.4, color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1, lab=None):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=MUTED))
        if lab:
            ax.text((x0 + x1) / 2, y0 + 1.5, lab, ha="center", fontsize=6, color=MUTED)
    W, H = 20.5, 17
    xs = [1, 27, 53, 78.5]
    top = [("Kernel", "EEVDF scheduler,\nsched tracepoints,\nschedstat counters", "#e9f1fb"),
           ("Capture + validate", "tracefs buffer read\nafter the run; trace\nvs kernel within 1%", "#eef7f2"),
           ("Bursts", "CPU time between\nblocking events,\nper task", "#eef7f2"),
           ("Predictors", "EWMA, log-EDMD,\nAR, ARIMA, MLP,\nnonlinear EDMD", "#fbf3e6")]
    bot = [("eBPF observer", "BPF on sched_switch\nreports each burst\nwhen it ends", "#e9f1fb"),
           ("Online predictor", "log-scale operator,\nrecursive least\nsquares, ~30 ns", "#fbf3e6"),
           ("Slice request", "s = c x prediction,\nclamped; set with\nsched_setattr", "#f3effb"),
           ("Stock EEVDF", "earlier virtual\ndeadline: served\nsooner", "#e9f1fb")]
    for x, (t, bd, c) in zip(xs, top):
        box(x, 27, W, H, t, bd, c)
    for x, (t, bd, c) in zip(xs, bot):
        box(x, 3, W, H, t, bd, c)
    for i in range(3):
        arrow(xs[i] + W + 0.8, 35.5, xs[i + 1] - 0.8, 35.5)
    labs = ["burst", "prediction", "slice"]
    for i in range(3):
        arrow(xs[i] + W + 0.8, 11.5, xs[i + 1] - 0.8, 11.5, labs[i])
    ax.plot([xs[3] + W / 2, xs[3] + W / 2, xs[0] + W / 2, xs[0] + W / 2], [2.2, -1.5, -1.5, 2.2], color=MUTED, lw=0.9, ls="--")
    ax.text(50, -1.0, "scheduling decisions shape the next bursts", ha="center", fontsize=6.2, color=MUTED, style="italic")
    ax.text(1, 46.5, "Offline study (Sections IV-V)", fontsize=7, color=MUTED, style="italic")
    ax.text(1, 22.3, "Closed loop on the live kernel (Section VI)", fontsize=7, color=MUTED, style="italic")
    fig.savefig(os.path.join(FIG, "a1_pipeline.png"), dpi=220, bbox_inches="tight"); plt.close(fig)


def baselines():
    df = pd.read_csv(os.path.join(ROOT, "outputs", "baselines", "baselines.csv"))
    meths = [("EWMA", "EWMA (textbook)"), ("log-EDMD", "log-EDMD, 3 delays"), ("poly-EDMD", "poly-EDMD"),
             ("RFF-EDMD", "RFF-EDMD (kernel)"), ("ARIMA", "ARIMA (AIC)"), ("MLP", "neural network"),
             ("gradient boosting", "gradient boosting")]
    groups = [("VM synthetic", "#8db7ea"), ("real programs", "#2a78d6")]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for j, (g, col) in enumerate(groups):
        d = df[df.group == g]
        data = [np.log2(d[f"MAE_{m}"] / d["MAE_log-AR(p*)"]) for m, _ in meths]
        pos = np.arange(len(meths)) + (j - 0.5) * 0.36
        bp = ax.boxplot(data, positions=pos, widths=0.32, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1))
        for b in bp["boxes"]:
            b.set_facecolor(col); b.set_edgecolor(MUTED)
        ax.plot([], [], color=col, lw=6, label=f"{g} ({len(d)} tasks)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(meths))); ax.set_xticklabels([m[1] for m in meths], rotation=18, ha="right")
    ax.set_yticks(np.log2([0.5, 0.75, 1, 1.5, 2, 3])); ax.set_yticklabels(["0.5", "0.75", "1", "1.5", "2", "3"])
    ax.set_ylabel("error / error of log-AR(p*)")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "b1_baselines.png"), dpi=200, bbox_inches="tight"); plt.close(fig)


def spectra():
    cols = {"redis": "#e34948", "sqlite": "#1baf7a", "node": "#eda100", "ffmpeg": "#6250d6", "client": "#2a78d6"}
    labs = {"redis": "Redis", "sqlite": "SQLite", "node": "Node.js", "ffmpeg": "ffmpeg", "client": "load generators"}
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.85), sharex=True, sharey=True)
    ev = {k: [] for k in cols}
    for r in (1, 2, 3):
        tl = pickle.load(open(os.path.join(ROOT, "outputs", "baselines", f"tl_real_r{r}.pkl"), "rb"))
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            k = tl["pids"][pid]
            for i in range(0, len(s) - 150 + 1, 150):
                try:
                    ev[k].extend(EDMD(3, "linear", log=True).fit(s[i:i + 150]).eigenvalues())
                except np.linalg.LinAlgError:
                    pass
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, k in zip(axes, cols):
        e = np.array(ev[k])
        ax.plot(np.cos(th), np.sin(th), color=MUTED, lw=0.6)
        ax.scatter(e.real, e.imag, s=2, color=cols[k], alpha=0.35, lw=0)
        ax.set_title(labs[k], fontsize=8); ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
    axes[0].set_ylabel("Im $\\lambda$")
    fig.supxlabel("Re $\\lambda$", fontsize=8.5, y=0.02)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "s1_spectra.png"), dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    pipeline(); baselines(); spectra(); print("ok")
