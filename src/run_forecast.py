"""
run_forecast.py
---------------
Runs every experiment in the report and writes tables to outputs/.

    python src/run_forecast.py data/processed/bursts_trace3.csv

Experiments
  1. Main test: per process, train on first 80% of its bursts, test on last 20%.
     MAE / RMSE + 95% block-bootstrap interval for MAE relative to the mean baseline.
  2. Rolling origins: 5 chronological splits (train up to 50,60,70,80,90%,
     test on the next 10%) to check the result is not one lucky split.
  3. Phase analysis (I/O processes): the same test run separately inside the
     'loaded' phase (CPU-bound jobs still running) and the 'idle' phase
     (after they exited), to separate short-term memory from the level shift.
  4. Lag correlations, Koopman eigenvalues.
  5. SJF simulation: does a better burst predictor give lower waiting time?
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import block_bootstrap_ratio, lag_correlations, mae, rmse, sjf_waiting_time  # noqa: E402
from predictors import METHOD_ORDER, run_all_methods  # noqa: E402

OUT = "outputs"
TEST_FRAC = 0.20
SCALE = {"cpu": (1e3, "ms"), "io": (1e6, "us")}


def load(path):
    d = pd.read_csv(path)
    d = d[d.end_state != "Z"]            # drop each process's final exit burst
    return d.sort_values(["pid", "start"])


def main_split(d):
    rows, preds_store, info_store = [], {}, {}
    for pid, g in d.groupby("pid"):
        kind = g.kind.iloc[0]
        s = g.burst_s.values
        start = int(round(len(s) * (1 - TEST_FRAC)))
        preds, info = run_all_methods(s, start)
        y = s[start:]
        preds_store[pid], info_store[pid] = (y, preds, start), info
        k, unit = SCALE[kind]
        for m in METHOD_ORDER:
            r, lo, hi = block_bootstrap_ratio(y, preds[m], preds["Mean baseline"])
            re, lo_e, hi_e = block_bootstrap_ratio(y, preds[m], preds["EWMA a=0.5 (textbook SJF)"])
            rows.append({
                "pid": pid, "kind": kind, "n_train": start, "n_test": len(y), "method": m,
                f"MAE": mae(y, preds[m]) * k, "RMSE": rmse(y, preds[m]) * k, "unit": unit,
                "MAE_vs_mean": r, "ci_lo": lo, "ci_hi": hi,
                "MAE_vs_ewma": re, "ci_lo_ewma": lo_e, "ci_hi_ewma": hi_e,
            })
    return pd.DataFrame(rows), preds_store, info_store


def rolling(d, origins=(0.5, 0.6, 0.7, 0.8, 0.9), width=0.10):
    rows = []
    for pid, g in d.groupby("pid"):
        s_full = g.burst_s.values
        n = len(s_full)
        for o in origins:
            start, stop = int(n * o), min(n, int(n * (o + width)))
            s = s_full[:stop]
            preds, _ = run_all_methods(s, start)
            y = s[start:]
            base = mae(y, preds["Mean baseline"])
            ew = mae(y, preds["EWMA a=0.5 (textbook SJF)"])
            for m in METHOD_ORDER:
                e = mae(y, preds[m])
                rows.append({"pid": pid, "kind": g.kind.iloc[0], "origin": o, "method": m,
                             "MAE_vs_mean": e / base, "MAE_vs_ewma": e / ew})
    return pd.DataFrame(rows)


def phases(d):
    cpu = d[d.kind == "cpu"]
    loaded_end = cpu.groupby("pid").end.max().min()     # first CPU-bound job exits
    idle_start = cpu.groupby("pid").end.max().max()     # last CPU-bound job exits
    rows = []
    for pid, g in d[d.kind == "io"].groupby("pid"):
        for phase, gg in (("loaded", g[g.start < loaded_end]), ("idle", g[g.start > idle_start])):
            s = gg.burst_s.values
            start = int(round(len(s) * (1 - TEST_FRAC)))
            preds, _ = run_all_methods(s, start)
            y = s[start:]
            for m in METHOD_ORDER:
                r, lo, hi = block_bootstrap_ratio(y, preds[m], preds["Mean baseline"])
                rows.append({"pid": pid, "phase": phase, "n": len(s), "n_test": len(y), "method": m,
                             "MAE_us": mae(y, preds[m]) * 1e6, "median_burst_us": np.median(s) * 1e6,
                             "MAE_vs_mean": r, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows), loaded_end, idle_start


def correlations(d, loaded_end, idle_start):
    rows = []
    for pid, g in d.groupby("pid"):
        parts = [("whole trace", g)]
        if g.kind.iloc[0] == "io":
            parts += [("loaded phase", g[g.start < loaded_end]), ("idle phase", g[g.start > idle_start])]
        for name, gg in parts:
            for r in lag_correlations(gg.burst_s.values):
                rows.append({"pid": pid, "kind": g.kind.iloc[0], "segment": name, "n": len(gg), **r})
    return pd.DataFrame(rows)


def sjf_sim(preds_store, kinds):
    """
    Round k: the k-th test burst of every process forms one batch of jobs.
    Compare average waiting time when SJF orders jobs by each predictor,
    relative to the oracle (ordering by true length). FCFS = PID order.
    """
    settings = {"mixed (3 CPU + 3 I/O)": list(preds_store),
                "CPU-bound only": [p for p in preds_store if kinds[p] == "cpu"],
                "I/O-bound only": [p for p in preds_store if kinds[p] == "io"]}
    rows = []
    for name, pids in settings.items():
        n_rounds = min(len(preds_store[p][0]) for p in pids)
        res = {m: [] for m in METHOD_ORDER + ["FCFS (PID order)", "Oracle SJF"]}
        for k in range(n_rounds):
            true = np.array([preds_store[p][0][k] for p in pids])
            oracle = sjf_waiting_time(true, true)
            res["Oracle SJF"].append(oracle)
            res["FCFS (PID order)"].append(sjf_waiting_time(true, np.arange(len(pids))))
            for m in METHOD_ORDER:
                res[m].append(sjf_waiting_time(true, np.array([preds_store[p][1][m][k] for p in pids])))
        o = np.array(res["Oracle SJF"])
        rng = np.random.default_rng(0)
        boot_idx = rng.integers(0, n_rounds, (2000, n_rounds))   # resample rounds
        for m, v in res.items():
            v = np.array(v)
            ratios = v[boot_idx].sum(1) / o[boot_idx].sum(1)
            rows.append({"setting": name, "rounds": n_rounds, "method": m,
                         "avg_wait_ms": v.mean() * 1e3, "wait_vs_oracle": v.sum() / o.sum(),
                         "ci_lo": np.quantile(ratios, 0.025), "ci_hi": np.quantile(ratios, 0.975)})
    return pd.DataFrame(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/bursts_trace3.csv"
    os.makedirs(OUT, exist_ok=True)
    d = load(path)
    kinds = d.groupby("pid").kind.first().to_dict()

    main_df, preds_store, info = main_split(d)
    main_df.to_csv(f"{OUT}/metrics_main.csv", index=False)

    roll = rolling(d)
    roll.to_csv(f"{OUT}/metrics_rolling.csv", index=False)

    ph, loaded_end, idle_start = phases(d)
    ph.to_csv(f"{OUT}/metrics_phase.csv", index=False)

    corr = correlations(d, loaded_end, idle_start)
    corr.to_csv(f"{OUT}/lag_correlations.csv", index=False)

    eig_rows = []
    for pid, inf in info.items():
        for which in ("eig", "eig_log"):
            for lam in inf[which]:
                eig_rows.append({"pid": pid, "kind": kinds[pid], "model": "raw" if which == "eig" else "log",
                                 "real": lam.real, "imag": lam.imag, "abs": abs(lam)})
    eig = pd.DataFrame(eig_rows)
    eig.to_csv(f"{OUT}/eigenvalues.csv", index=False)

    sjf = sjf_sim(preds_store, kinds)
    sjf.to_csv(f"{OUT}/sjf_simulation.csv", index=False)

    # store predictions for plotting
    rows = []
    for pid, (y, preds, start) in preds_store.items():
        for i in range(len(y)):
            rows.append({"pid": pid, "kind": kinds[pid], "test_index": i, "actual": y[i],
                         **{m: preds[m][i] for m in METHOD_ORDER}})
    pd.DataFrame(rows).to_csv(f"{OUT}/predictions_main.csv", index=False)

    # ---- human-readable summary ----
    with open(f"{OUT}/metrics.txt", "w") as f:
        def w(s=""):
            print(s)
            f.write(s + "\n")
        w(f"Bursts file: {path}")
        w(f"Loaded phase ends (first CPU-bound exit): t={loaded_end:.3f} s; "
          f"idle phase starts (last exit): t={idle_start:.3f} s")
        w("\n=== 1. Main split (last 20% per process) ===")
        for pid in sorted(kinds):
            sub = main_df[main_df.pid == pid]
            u = sub.unit.iloc[0]
            w(f"\nPID {pid} ({kinds[pid]}), train {sub.n_train.iloc[0]}, test {sub.n_test.iloc[0]}, "
              f"EWMA tuned alpha={info[pid]['alpha']}")
            w(f"  {'method':28s} {'MAE ' + u:>10s} {'RMSE ' + u:>10s}  MAE/mean [95% CI]       MAE/EWMA0.5 [95% CI]")
            for _, r in sub.iterrows():
                w(f"  {r.method:28s} {r.MAE:10.3f} {r.RMSE:10.3f}  {r.MAE_vs_mean:5.3f} [{r.ci_lo:5.3f},{r.ci_hi:5.3f}]"
                  f"   {r.MAE_vs_ewma:5.3f} [{r.ci_lo_ewma:5.3f},{r.ci_hi_ewma:5.3f}]")
        w("\n=== 2. Rolling origins: median MAE ratio vs mean baseline (and vs EWMA 0.5) over 5 splits ===")
        agg = roll.groupby(["kind", "method"]).agg(
            med_vs_mean=("MAE_vs_mean", "median"), wins_vs_mean=("MAE_vs_mean", lambda x: int((x < 1).sum())),
            med_vs_ewma=("MAE_vs_ewma", "median"), wins_vs_ewma=("MAE_vs_ewma", lambda x: int((x < 1).sum())),
            n=("MAE_vs_mean", "size")).reset_index()
        for kind in ("cpu", "io"):
            w(f"  [{kind}]")
            for m in METHOD_ORDER:
                r = agg[(agg.kind == kind) & (agg.method == m)].iloc[0]
                w(f"    {m:28s} vs mean {r.med_vs_mean:5.3f} (beats it {r.wins_vs_mean}/{r.n})   "
                  f"vs EWMA {r.med_vs_ewma:5.3f} (beats it {r.wins_vs_ewma}/{r.n})")
        agg.to_csv(f"{OUT}/metrics_rolling_summary.csv", index=False)
        w("\n=== 3. I/O phase analysis: MAE ratio vs mean baseline [95% CI] ===")
        for (pid, phase), sub in ph.groupby(["pid", "phase"], sort=False):
            w(f"  PID {pid} {phase:6s} n={sub.n.iloc[0]} median burst {sub.median_burst_us.iloc[0]:.0f} us")
            for _, r in sub.iterrows():
                w(f"    {r.method:28s} MAE {r.MAE_us:8.2f} us  ratio {r.MAE_vs_mean:5.3f} [{r.ci_lo:5.3f},{r.ci_hi:5.3f}]")
        w("\n=== 4. Lag correlations ===")
        w(corr.round(3).to_string(index=False))
        w("\n=== 5. Koopman eigenvalues (raw-lag and log-lag EDMD, main split) ===")
        w(eig.round(3).to_string(index=False))
        w("\n=== 6. SJF simulation: total waiting time relative to oracle SJF (1.00 = perfect) ===")
        for name, sub in sjf.groupby("setting", sort=False):
            w(f"  {name} ({sub.rounds.iloc[0]} rounds)")
            for _, r in sub.iterrows():
                w(f"    {r.method:28s} avg wait {r.avg_wait_ms:9.3f} ms   x oracle {r.wait_vs_oracle:5.3f}"
                  f" [{r.ci_lo:5.3f},{r.ci_hi:5.3f}]")


if __name__ == "__main__":
    main()
