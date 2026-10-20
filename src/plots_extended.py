"""
plots_extended.py - figures for the change experiments, 4-core scaling and the market study.
    python src/plots_extended.py   -> outputs/figures_ext/
"""
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "outputs", "figures_ext")
BLUE, ORANGE, AQUA, VIOLET, YELLOW, INK, MUTED, GRID = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100",
                                                        "#0b0b0b", "#52514e", "#e4e3df")
DET = {"koopman": (BLUE, "Koopman residual"), "novelty": (ORANGE, "distance from normal"),
       "persistence": (MUTED, "persistence residual"), "run_queue": (AQUA, "run-queue monitor"),
       "utilisation": (VIOLET, "utilisation monitor")}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200, "font.size": 8.5, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_change_roc():
    s = pd.read_csv(os.path.join(ROOT, "outputs", "change", "summary.csv"))
    s["fa"] = s.runs_with_false_alarm / s.runs
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (pl, title) in zip(axes, [("mac_arm_4core_live", "4-core ARM VM (Apple host), live monitor"),
                                      ("vm_intel_2core_live", "2-core Intel VM, live monitor")]):
        for det, (col, lab) in DET.items():
            g = s[(s.platform == pl) & (s.detector == det)].sort_values("threshold")
            ax.plot(g.fa, g.detection_rate, marker="o", ms=3, color=col, lw=1.3, label=lab)
            t3 = g[g.threshold == 3.0]
            ax.scatter(t3.fa, t3.detection_rate, s=40, facecolor="white", edgecolor=col, zorder=4, linewidth=1.4)
        ax.set_xlim(-0.03, 0.6); ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("share of runs with a false alarm")
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[0].set_ylabel("share of 18 changes detected within 5 s")
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.text(0.01, -0.03, "Each line sweeps the alarm threshold (2 to 8); open circles mark threshold 3. "
             "24 runs per platform: 18 scripted changes, 6 without a change.", fontsize=7, color=MUTED)
    save(fig, "e1_change_detection.png")


def fig_change_example():
    plat = pickle.load(open(os.path.join(ROOT, "outputs", "change", "runs.pkl"), "rb"))
    runs = {r["name"]: r for r in plat["mac_arm_4core_live"]}
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 3.6), sharex=True)
    for j, name in enumerate(["load_up_small_r1", "io_burst_small_r1", "none_r1a"]):
        r = runs[name]
        t = r["t"] - r["meta"]["t0_mono"]
        S = r["S"]
        ax = axes[0, j]
        ax.plot(t, S[:, 0], color=BLUE, lw=0.9, label="busy cores")
        ax.plot(t, S[:, 1], color=AQUA, lw=0.9, label="runnable tasks")
        ax.plot(t, S[:, 2], color=ORANGE, lw=0.9, label="switches/s (k)")
        ax.set_title({"load_up_small_r1": "one CPU-bound job added", "io_burst_small_r1": "two fast I/O jobs added",
                      "none_r1a": "no change"}[name], loc="left", fontsize=8.5)
        ax2 = axes[1, j]
        for det in ("koopman", "novelty", "run_queue"):
            ax2.plot(t, np.clip(r["scores"][det], -2, 15), color=DET[det][0], lw=0.9, label=DET[det][1])
        ax2.axhline(3, color="#e34948", ls=":", lw=0.9)
        for ax_ in (ax, ax2):
            ax_.axvspan(0, 8, color="#eeeaf8", zorder=0)
            if r["change"]:
                ax_.axvline(r["change"] - r["meta"]["t0_mono"], color=INK, lw=0.8)
        ax2.set_xlabel("seconds")
    axes[0, 0].set_ylabel("machine state"); axes[1, 0].set_ylabel("score (training σ)")
    axes[0, 2].legend(fontsize=6.5, loc="upper right"); axes[1, 2].legend(fontsize=6.5, loc="upper right")
    save(fig, "e2_change_examples.png")


def fig_scaling():
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "scale_mac", "*"))):
        m = json.load(open(os.path.join(d, "meta.json")))
        for f in glob.glob(os.path.join(d, "ground_truth", "gt_*.txt")):
            kv = dict(l.split(None, 1) for l in open(f) if l.strip())
            rows.append({"cores": m["cores"], "kind": kv["kind"].strip(), "run": int(kv["run_ns"]), "wait": int(kv["wait_ns"])})
    df = pd.DataFrame(rows)
    df["frac"] = df.wait / (df.wait + df.run)
    lat = pd.read_csv(os.path.join(ROOT, "outputs", "multicore", "latency.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for kind, col, lab in (("cpu", BLUE, "CPU-bound"), ("phased", AQUA, "phased"), ("io", ORANGE, "I/O-bound")):
        g = df[df.kind == kind].groupby("cores").frac.agg(["mean", "min", "max"])
        ax.errorbar(g.index, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], color=col, marker="o",
                    ms=4, lw=1.4, capsize=2, label=f"{lab}, ARM VM (kernel counters)")
        v = lat[(lat.kind == kind) & lat.config.isin(["scale_c01", "scale_c02"])].groupby("cores").frac_time_waiting.mean()
        ax.plot(v.index, v.values, color=col, ls="--", marker="s", ms=3.5, lw=1, label=f"{lab}, Intel VM (traces)")
    ax.axvline(3.5, color=MUTED, lw=0.8, ls=":")
    ax.text(3.55, 0.92, "cores > CPU-bound jobs", fontsize=7, color=MUTED)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"])
    ax.set_xlabel("cores available to the 8 jobs"); ax.set_ylabel("share of time spent waiting for a CPU")
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "e3_scaling_measured.png")


def fig_market():
    sc = pd.read_csv(os.path.join(ROOT, "outputs", "market", "scores.csv"), parse_dates=[0], index_col=0)
    ev = pd.read_csv(os.path.join(ROOT, "outputs", "market", "events.csv"), parse_dates=["event_date"])
    sp = pd.read_csv(os.path.join(ROOT, "data", "market", "sp500_1990_2026.csv"), parse_dates=["Date"]).set_index("Date").Close
    ew = pd.read_csv(os.path.join(ROOT, "outputs", "market", "eigenworld_long_run.csv"), parse_dates=[0], index_col=0)
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.6), sharex=True)
    dd = sp / sp.rolling(252, min_periods=1).max() - 1
    axes[0].fill_between(dd.index, dd.values * 100, 0, color="#c9d9ee")
    axes[0].set_ylabel("drawdown (%)")
    axes[0].set_title("S&P 500 drawdown from its 1-year high; lines mark the 22 rule-defined events", loc="left", fontsize=8.5)
    axes[1].plot(sc.index, np.clip(sc.koopman, -2, 12), color=BLUE, lw=0.6, label="Koopman residual (walk-forward)")
    axes[1].plot(sc.index, np.clip(sc.vix_z, -2, 12), color=ORANGE, lw=0.6, alpha=0.8, label="VIX z-score")
    axes[1].axhline(3, color="#e34948", ls=":", lw=0.9)
    axes[1].set_ylabel("score"); axes[1].legend(fontsize=7, loc="upper left", ncol=2)
    axes[2].plot(ew.index, ew.spectral_entropy, color=VIOLET, lw=1, marker=".", ms=2.5)
    m = ew.spectral_entropy.idxmin()
    axes[2].annotate(f"lowest value, {m.date()}\n(the original Lehman finding)", xy=(m, ew.spectral_entropy.min()),
                     xytext=(pd.Timestamp("2012-01-01"), ew.spectral_entropy.min() + 0.02), fontsize=7,
                     arrowprops=dict(arrowstyle="-", color=MUTED))
    axes[2].set_ylabel("EIGENWORLD\nspectral entropy")
    for ax in axes:
        for e in ev.event_date:
            ax.axvline(e, color=INK, lw=0.4, alpha=0.5)
    save(fig, "e4_market_timeline.png")

    sw = pd.read_csv(os.path.join(ROOT, "outputs", "market", "precursor_sweep.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    cols = {"koopman": BLUE, "novelty": ORANGE, "vix_z": VIOLET, "realised_vol_z": AQUA, "persistence": MUTED}
    labs = {"koopman": "Koopman residual", "novelty": "distance from normal", "vix_z": "VIX z-score",
            "realised_vol_z": "realised volatility z", "persistence": "persistence residual"}
    for det, g in sw.groupby("detector"):
        g = g.sort_values("far_alarms_per_year")
        ax.plot(g.far_alarms_per_year, g.hit_rate / g.chance_hit_rate.clip(lower=1e-3), marker="o", ms=3,
                color=cols[det], lw=1.2, label=labs[det])
    ax.axhline(1, color=INK, lw=0.8)
    ax.set_xlabel("alarms per year far (>60 days) from any event")
    ax.set_ylabel("precursor hits ÷ hits expected by chance")
    ax.set_title("22 corrections, 1992-2026 (each line sweeps the threshold)", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "e5_market_precursor.png")


def fig_eigenworld():
    """Paper figure: S&P 500 drawdown and EIGENWORLD's spectral entropy (seed 0), 1992-2026."""
    sp = pd.read_csv(os.path.join(ROOT, "data", "market", "sp500_1990_2026.csv"), parse_dates=["Date"]).set_index("Date").Close
    ew = pd.read_csv(os.path.join(ROOT, "outputs", "market", "eigenworld_long_run.csv"), parse_dates=[0], index_col=0)
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 3.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    dd = sp / sp.rolling(252, min_periods=1).max() - 1
    axes[0].fill_between(dd.index, dd.values * 100, 0, color="#c9d9ee")
    axes[0].set_ylabel("S&P 500\ndrawdown (%)")
    axes[1].plot(ew.index, ew.spectral_entropy, color=VIOLET, lw=1, marker=".", ms=2.5)
    m = ew.spectral_entropy.idxmin()
    axes[1].annotate(f"34-year minimum, {m.date()}", xy=(m, ew.spectral_entropy.min()),
                     xytext=(pd.Timestamp("2011-06-01"), ew.spectral_entropy.min() + 0.01), fontsize=7,
                     arrowprops=dict(arrowstyle="-", color=MUTED))
    axes[1].set_ylabel("EIGENWORLD\nspectral entropy")
    for ax in axes:
        ax.axvline(pd.Timestamp("2008-09-15"), color="#e34948", lw=0.9, ls=":")
    axes[0].text(pd.Timestamp("2008-11-01"), -8, "Lehman Brothers", fontsize=7, color="#e34948")
    save(fig, "e6_eigenworld.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_change_roc(); fig_change_example(); fig_scaling(); fig_market(); fig_eigenworld()
