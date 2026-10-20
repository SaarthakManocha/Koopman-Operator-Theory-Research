"""
plots_multicore.py - figures for the multi-core experiments (outputs/multicore/figures/).
    python src/plots_multicore.py
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "multicore")
FIG = os.path.join(OUT, "figures")
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#0b0b0b", "#52514e", "#e4e3df"
KIND = {"cpu": BLUE, "io": ORANGE, "phased": AQUA}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_validation():
    v = pd.read_csv(os.path.join(OUT, "validation.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for ax, (a, b, lab, unit) in zip(axes, [("run_s_kernel", "run_s_trace", "CPU time", "s"),
                                            ("wait_s_kernel", "wait_s_trace", "run-queue wait", "s"),
                                            ("slices_kernel", "slices_trace", "times scheduled", "")]):
        for k, g in v.groupby("kind"):
            ax.scatter(g[a], g[b], s=14, color=KIND[k], label={"cpu": "CPU-bound", "io": "I/O-bound", "phased": "phased"}[k],
                       edgecolor="white", linewidth=0.5)
        lim = [min(v[a].min(), v[b].min()) * 0.8, max(v[a].max(), v[b].max()) * 1.2]
        ax.plot(lim, lim, color=INK, lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"kernel counter ({unit})" if unit else "kernel counter")
        ax.set_ylabel(f"rebuilt from trace ({unit})" if unit else "rebuilt from trace")
        ax.set_title(lab, loc="left")
    axes[0].legend(fontsize=7, loc="upper left")
    save(fig, "m1_validation.png")


def fig_cores():
    sim = pd.read_csv(os.path.join(OUT, "simulation.csv"))
    s = sim[sim.config == "lab_trace3"]
    pols = [("FCFS", "oracle", "FCFS", MUTED, "-"), ("RR", "oracle", "Round robin (3 ms)", VIOLET, "-"),
            ("SJF", "ewma", "SJF + EWMA", ORANGE, "--"), ("SRTF", "ewma", "SRTF + EWMA", ORANGE, "-"),
            ("SRTF", "edmd", "SRTF + EDMD", BLUE, "-"), ("SRTF", "oracle", "SRTF + perfect knowledge", INK, ":")]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    for ax, met, lab in ((axes[0], "interactive_mean_response_ms", "mean"), (axes[1], "interactive_p99_response_ms", "99th percentile")):
        for pol, pred, name, col, ls in pols:
            g = s[(s.policy == pol) & (s.predictor == pred)].sort_values("cores")
            ax.plot(g.cores, g[met] + 1e-3, color=col, ls=ls, marker="o", ms=3, lw=1.3, label=name)
        ax.axvspan(0.8, 3.5, color="#f3e6dc", zorder=0)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 2, 3, 4, 8, 16, 28]); ax.set_xticklabels(["1", "2", "3", "4", "8", "16", "28"])
        ax.set_xlabel("cores"); ax.set_ylabel(f"I/O response, {lab} (ms)")
        ax.set_title(f"I/O job response time, {lab}", loc="left")
    fig.text(0.01, -0.02, "Shaded: fewer cores than CPU-bound jobs + 1. Replay of the lab trace's 6 jobs; zero shown as 0.001 ms.",
             fontsize=7, color=MUTED)
    axes[1].legend(fontsize=6.5, loc="upper right")
    save(fig, "m2_cores_vs_policy.png")


def fig_detection():
    ser = pickle.load(open(os.path.join(OUT, "koopman_series.pkl"), "rb"))
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.2), sharex=False)
    for i, r in enumerate(["regime_r1", "regime_r2", "regime_r3"]):
        d = ser[r]
        m = d["t"] <= d["eval_end"]
        axes[0].plot(d["t"][m], np.clip(d["koopman"][m], -2, 12), color=[BLUE, "#5aa0f0", "#1d5aa8"][i], lw=0.9,
                     label="Koopman surprise" if i == 0 else None)
        axes[0].plot(d["t"][m], np.clip(d["novelty"][m], -2, 12), color=ORANGE, lw=0.8, alpha=0.8,
                     label="distance-from-normal baseline" if i == 0 else None)
    axes[0].axvline(ser["regime_r1"]["change"], color=INK, lw=0.8)
    axes[0].text(ser["regime_r1"]["change"] + 0.3, 10.5, "load drops (4 of 6 CPU-bound jobs stop)", fontsize=7)
    axes[0].set_title("Load-drop runs (3 repetitions, 2 cores): operator fitted on the first 8 s", loc="left")
    for i, r in enumerate([k for k in ser if k.startswith(("scale", "oversub"))]):
        d = ser[r]
        m = d["t"] <= d["eval_end"]
        axes[1].plot(d["t"][m], np.clip(d["koopman"][m], -2, 12), color=BLUE, lw=0.6, alpha=0.6,
                     label="Koopman surprise, 9 runs without a change" if i == 0 else None)
    axes[1].set_title("Runs without a change: no alarms", loc="left")
    for ax in axes:
        ax.axhline(3, color="#e34948", ls=":", lw=0.9)
        ax.axvspan(0, 8, color="#eeeaf8", zorder=0)
        ax.set_ylabel("score (training σ)")
        ax.legend(fontsize=7, loc="upper left")
    axes[1].set_xlabel("time since start of capture (s)")
    save(fig, "m3_koopman_detection.png")


def fig_prediction():
    p = pd.read_csv(os.path.join(OUT, "prediction.csv"))
    p = p[p.config != "lab_trace3"]
    meths = ["Median baseline", "Last value", "EWMA tuned", "EDMD one-step", "Poly EDMD one-step", "Log EDMD one-step", "Gradient boosting"]
    short = ["median", "last value", "EWMA tuned", "EDMD", "poly EDMD", "log EDMD", "grad. boost"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
    for ax, kind, title in ((axes[0], "phased", "Phased jobs (positive control), 24 processes"),
                            (axes[1], "io", "I/O jobs under contention, 33 processes")):
        for j, m in enumerate(meths):
            g = p[(p.kind == kind) & (p.method == m)]
            sig_b = g.vs_ewma_hi < 1
            sig_w = g.vs_ewma_lo > 1
            y = np.full(len(g), j) + np.random.default_rng(j).uniform(-0.18, 0.18, len(g))
            ax.scatter(g.vs_ewma[~sig_b & ~sig_w], y[~sig_b & ~sig_w], s=9, color=MUTED, alpha=0.6)
            ax.scatter(g.vs_ewma[sig_b], y[sig_b], s=11, color=BLUE)
            ax.scatter(g.vs_ewma[sig_w], y[sig_w], s=11, color=ORANGE)
            ax.plot([g.vs_ewma.median()] * 2, [j - 0.3, j + 0.3], color=INK, lw=1.6)
        ax.axvline(1, color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.set_xlabel("MAE ÷ MAE of textbook EWMA (α = 0.5)")
    axes[0].set_yticks(range(len(meths))); axes[0].set_yticklabels(short)
    axes[0].invert_yaxis()
    fig.text(0.01, -0.03, "Each dot is one process. Blue: 95% interval wholly below 1 (reliably better than EWMA); "
             "orange: wholly above 1; grey: not distinguishable. Bar: median.", fontsize=7, color=MUTED)
    save(fig, "m4_prediction_vs_ewma.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_validation(); fig_cores(); fig_detection(); fig_prediction()
