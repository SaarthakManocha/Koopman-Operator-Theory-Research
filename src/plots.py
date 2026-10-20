"""
plots.py
--------
Makes every figure in the report from the CSVs written by run_forecast.py
and trace_stats.py.

    python src/plots.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = "outputs/figures"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 1.4,
})
KIND_COLOR = {"cpu": BLUE, "io": ORANGE}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_timeline(b, summary):
    t0 = b.start.min()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    cpu_exit = summary[summary.kind == "cpu"].last_s - t0
    for ax, kind, unit, k in ((axes[0], "cpu", "ms", 1e3), (axes[1], "io", "µs", 1e6)):
        sub = b[(b.kind == kind) & (b.end_state != "Z")]
        for i, (pid, g) in enumerate(sub.groupby("pid")):
            ax.scatter(g.start - t0, g.burst_s * k, s=3, alpha=0.45, color=KIND_COLOR[kind],
                       marker="osd"[i], linewidths=0, label=f"PID {pid}")
            med = pd.Series((g.burst_s * k).values).rolling(41, center=True, min_periods=5).median()
            ax.plot(g.start - t0, med, color=[INK, MUTED, "#8a8985"][i], lw=1.2)
        ax.set_yscale("log")
        ax.set_ylabel(f"burst length ({unit}, log)")
        ax.set_title(("CPU-bound" if kind == "cpu" else "I/O-bound") + " processes", loc="left")
        for x in cpu_exit:
            ax.axvline(x, color=MUTED, ls="--", lw=0.8)
        ax.legend(loc="upper left", fontsize=7, markerscale=3, ncol=3)
    axes[1].annotate("CPU-bound jobs exit", xy=(cpu_exit.max(), 30), xytext=(cpu_exit.max() + 3, 20),
                     fontsize=8, color=MUTED)
    axes[1].set_xlabel("time since start of trace (s)")
    save(fig, "fig2_burst_timeline.png")


def fig_distributions(b):
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    bins = np.logspace(-6, 1, 80)
    for kind, label in (("cpu", "CPU-bound"), ("io", "I/O-bound")):
        x = b[(b.kind == kind) & (b.end_state != "Z")].burst_s
        ax.hist(x, bins=bins, color=KIND_COLOR[kind], alpha=0.75, label=f"{label} (n={len(x)})",
                edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("burst length (s, log scale)")
    ax.set_ylabel("number of bursts")
    ax.legend()
    save(fig, "fig1_burst_distributions.png")


def fig_forecast(pred, pid, unit, k, name, title):
    g = pred[pred.pid == pid]
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    ax.plot(g.test_index, g.actual * k, color="#9c9b97", lw=1.0, label="actual")
    ax.plot(g.test_index, g["Mean baseline"] * k, color=INK, lw=1.0, ls=":", label="mean baseline")
    ax.plot(g.test_index, g["EWMA a=0.5 (textbook SJF)"] * k, color=ORANGE, lw=1.2, label="EWMA α=0.5 (SJF)")
    ax.plot(g.test_index, g["EDMD one-step"] * k, color=BLUE, lw=1.2, label="EDMD one-step")
    ax.set_xlabel("test burst index")
    ax.set_ylabel(f"burst length ({unit})")
    ax.set_title(title, loc="left")
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    save(fig, name)


def fig_recursive(pred, pid):
    g = pred[pred.pid == pid]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(g.test_index, g.actual * 1e3, color="#9c9b97", lw=1.0, label="actual")
    ax.plot(g.test_index, g["EDMD recursive"] * 1e3, color=BLUE, lw=1.6, label="EDMD recursive")
    ax.plot(g.test_index, g["Mean baseline"] * 1e3, color=INK, lw=1.0, ls=":", label="mean baseline")
    ax.set_xlabel("test burst index")
    ax.set_ylabel("burst length (ms)")
    ax.set_title(f"Recursive forecast fades to the mean (PID {pid}, CPU-bound)", loc="left")
    ax.legend(ncol=3, fontsize=7)
    save(fig, "fig8_recursive_forecast.png")


def fig_eigs(eig):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=MUTED, ls="--", lw=0.8)
    for kind, label in (("cpu", "CPU-bound"), ("io", "I/O-bound")):
        e = eig[(eig.kind == kind) & (eig.model == "raw")]
        ax.scatter(e.real, e.imag, s=40, color=KIND_COLOR[kind], edgecolor="white", linewidth=1.2,
                   label=label, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title("Koopman eigenvalues (raw-lag EDMD)", loc="left")
    ax.legend(loc="lower left", fontsize=7)
    save(fig, "fig7_eigenvalues.png")


def fig_main_forest(main):
    methods = ["Median baseline", "EWMA a=0.5 (textbook SJF)", "EDMD one-step", "Log EDMD one-step",
               "Gradient boosting"]
    short = ["median", "EWMA 0.5", "EDMD", "log EDMD", "grad. boost"]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ys, labels, y = [], [], 0
    for pid, g in main.groupby("pid", sort=True):
        kind = g.kind.iloc[0]
        for m, s in zip(methods, short):
            r = g[g.method == m].iloc[0]
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=KIND_COLOR[kind], lw=1.6)
            ax.scatter([r.MAE_vs_mean], [y], color=KIND_COLOR[kind], s=22, zorder=3, edgecolor="white", linewidth=1)
            ys.append(y)
            labels.append(f"{pid} ({kind})  {s}")
            y += 1
        y += 0.7
    ax.axvline(1, color=INK, lw=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("MAE relative to mean baseline (below 1 = better), 95% block-bootstrap interval")
    ax.set_title("Main split (last 20% of each process)", loc="left")
    save(fig, "fig3_main_split_intervals.png")


def fig_phase(ph):
    methods = ["Median baseline", "EWMA a=0.5 (textbook SJF)", "EDMD one-step", "Gradient boosting"]
    short = ["median", "EWMA 0.5", "EDMD", "grad. boost"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.6), sharey=True)
    for ax, phase in zip(axes, ("loaded", "idle")):
        ys, labels, y = [], [], 0
        for pid, g in ph[ph.phase == phase].groupby("pid", sort=True):
            for m, s in zip(methods, short):
                r = g[g.method == m].iloc[0]
                ax.plot([r.ci_lo, min(r.ci_hi, 2.0)], [y, y], color=ORANGE, lw=1.6)
                ax.scatter([min(r.MAE_vs_mean, 2.0)], [y], color=ORANGE, s=22, zorder=3, edgecolor="white")
                if r.MAE_vs_mean > 2.0:
                    ax.text(1.97, y - 0.35, f"{r.MAE_vs_mean:.2f} (clipped)", ha="right", fontsize=6.5, color=MUTED)
                ys.append(y)
                labels.append(f"{pid}  {s}")
                y += 1
            y += 0.7
        ax.axvline(1, color=INK, lw=0.9)
        ax.set_xlim(0.5, 2.05)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(f"{phase} phase", loc="left")
        ax.set_xlabel("MAE / mean baseline")
    axes[0].invert_yaxis()
    save(fig, "fig6_io_phase_intervals.png")


def fig_sjf(sjf):
    sub = sjf[sjf.setting.str.startswith("mixed")]
    fcfs = sub[sub.method == "FCFS (PID order)"].wait_vs_oracle.iloc[0]
    order = ["Mean baseline", "EDMD recursive", "Median baseline", "Gradient boosting", "Last value",
             "EDMD one-step", "Poly EDMD one-step", "EWMA a=0.5 (textbook SJF)", "Log EDMD one-step",
             "EWMA tuned", "Oracle SJF"]
    sub = sub.set_index("method").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = [MUTED if ("baseline" in m or "Oracle" in m or m == "Last value")
              else (ORANGE if "EWMA" in m else BLUE) for m in sub.method]
    y = np.arange(len(sub))
    ax.barh(y, sub.wait_vs_oracle, color=colors, height=0.62)
    ax.errorbar(sub.wait_vs_oracle, y, xerr=[sub.wait_vs_oracle - sub.ci_lo, sub.ci_hi - sub.wait_vs_oracle],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    for yi, v, hi in zip(y, sub.wait_vs_oracle, sub.ci_hi):
        ax.text(hi + 0.06, yi, f"{v:.2f}x", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(sub.method, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 4.4)
    ax.set_xlabel(f"total waiting time / oracle SJF (1 = perfect knowledge). FCFS in PID order: {fcfs:.1f}x (off scale)")
    ax.set_title("SJF ordered by each predictor, mixed batch of 3 CPU + 3 I/O jobs", loc="left")
    save(fig, "fig9_sjf_waiting_time.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    b = pd.read_csv("data/processed/bursts_trace3.csv").sort_values(["pid", "start"])
    summary = pd.read_csv("outputs/process_summary.csv")
    pred = pd.read_csv("outputs/predictions_main.csv")
    fig_timeline(b, summary)
    fig_distributions(b)
    fig_forecast(pred, 9566, "µs", 1e6, "fig4_forecast_io.png",
                 "One-step forecasts, I/O-bound PID 9566 (test period is after the level shift)")
    fig_forecast(pred, 9562, "ms", 1e3, "fig5_forecast_cpu.png", "One-step forecasts, CPU-bound PID 9562")
    fig_recursive(pred, 9562)
    fig_eigs(pd.read_csv("outputs/eigenvalues.csv"))
    fig_main_forest(pd.read_csv("outputs/metrics_main.csv"))
    fig_phase(pd.read_csv("outputs/metrics_phase.csv"))
    fig_sjf(pd.read_csv("outputs/sjf_simulation.csv"))


if __name__ == "__main__":
    main()
