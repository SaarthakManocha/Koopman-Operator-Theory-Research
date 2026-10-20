"""
run_multicore.py
----------------
All multi-core experiments, on every capture in data/captures_vm/ plus the lab Trace 3.

    python src/run_multicore.py

Writes to outputs/multicore/:
    validation.csv           trace reconstruction vs kernel counters, every process of every capture
    latency.csv              how long tasks waited for a CPU, per capture and task class
    prediction.csv           burst-prediction accuracy (textbook CPU bursts), per process
    simulation.csv           multi-core scheduler replay: policies x predictors x core counts
    koopman_detection.csv    regime-change detection with the machine-level Koopman operator
    timelines/<capture>.pkl  parsed timelines (used by the figures and the visualizer)
"""
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import block_bootstrap_ratio, mae  # noqa: E402
from koopman_system import (LinearKoopman, aggregate_state, first_alarm, machine_state, novelty_score,  # noqa: E402
                            persistence_surprise, smooth)
from predictors import METHOD_ORDER, run_all_methods  # noqa: E402
from sched_events import build_timeline, cpu_bursts, load_pids  # noqa: E402
from simulate import simulate, summarize, tasks_from_bursts  # noqa: E402
from validate import validate_capture  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP = os.path.join(ROOT, "data", "captures_vm")
OUT = os.path.join(ROOT, "outputs", "multicore")
TL = os.path.join(OUT, "timelines")


def load_all():
    caps = {}
    for d in sorted(glob.glob(os.path.join(CAP, "*"))):
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "meta.json")):
            continue
        name = os.path.basename(d)
        pkl = os.path.join(TL, name + ".pkl")
        if os.path.exists(pkl):
            tl = pickle.load(open(pkl, "rb"))
        else:
            tl = build_timeline(os.path.join(d, "trace.txt.gz"), load_pids(os.path.join(d, "pids.txt")))
            pickle.dump(tl, open(pkl, "wb"))
        tl["meta"] = json.load(open(os.path.join(d, "meta.json")))
        tl["dir"] = d
        caps[name] = tl
    # lab Trace 3 (bare metal, 28 cores)
    pkl = os.path.join(TL, "lab_trace3.pkl")
    if os.path.exists(pkl):
        tl = pickle.load(open(pkl, "rb"))
    else:
        raw = os.path.join(ROOT, "data", "raw", "trace3.txt.gz")
        tl = build_timeline(raw, load_pids(os.path.join(ROOT, "evidence", "pids.txt")))
        pickle.dump(tl, open(pkl, "wb"))
    tl["meta"] = {"name": "lab_trace3", "ncpu": 28, "cpus_allowed": list(range(28)),
                  "jobs": {"cpu": 3, "io": 3, "phased": 0}}
    caps["lab_trace3"] = tl
    return caps


def config_of(name):
    return name.rsplit("_r", 1)[0] if "_r" in name else name


# ------------------------------------------------------------------ 1 validation
def run_validation(caps):
    rows = []
    for name, tl in caps.items():
        if name == "lab_trace3":
            continue
        df, meta = validate_capture(tl["dir"])
        df["lost_events"] = meta["lost_events"]
        rows.append(df)
    v = pd.concat(rows)
    v.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    return v


# ------------------------------------------------------------------ 2 latency / queues
def run_latency(caps):
    rows = []
    for name, tl in caps.items():
        w, r, m = tl["wait"], tl["run"], tl["migrations"]
        span = tl["t1"] - tl["t0"]
        ncores = len(tl["meta"]["cpus_allowed"])
        busy = tl["cpu_busy"]
        busy = busy[busy.cpu.isin(tl["meta"]["cpus_allowed"])]
        util = (busy.end - busy.start).sum() / (span * ncores)
        for kind in sorted(r.kind.dropna().unique()):
            wk = w[(w.kind == kind) & (w.cause == "wake")].dur * 1e3
            wp = w[(w.kind == kind) & (w.cause == "preempt")].dur * 1e3
            rk = r[r.kind == kind]
            rows.append({
                "capture": name, "config": config_of(name), "cores": ncores, "kind": kind,
                "n_tasks": rk.pid.nunique(), "utilisation_allowed_cores": util,
                "runnable_tasks_per_core": (w.dur.sum() + r.dur.sum()) / (span * ncores),
                "wake_to_run_median_ms": wk.median(), "wake_to_run_p95_ms": wk.quantile(.95),
                "wake_to_run_p99_ms": wk.quantile(.99), "preempt_wait_median_ms": wp.median(),
                "frac_time_waiting": w[w.kind == kind].dur.sum() / (w[w.kind == kind].dur.sum() + rk.dur.sum()),
                "migrations_per_task_per_s": len(m[m.kind == kind]) / max(rk.pid.nunique(), 1) / span,
                "cores_used_median": rk.groupby("pid").cpu.nunique().median(),
            })
    lat = pd.DataFrame(rows)
    lat.to_csv(os.path.join(OUT, "latency.csv"), index=False)
    return lat


# ------------------------------------------------------------------ 3 burst prediction
def run_prediction(caps):
    rows = []
    for name, tl in caps.items():
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            if len(s) < 60:
                continue                       # CPU-bound jobs never block: one huge burst, nothing to predict
            start = int(round(len(s) * 0.8))
            preds, info = run_all_methods(s, start)
            y = s[start:]
            for meth in METHOD_ORDER:
                r1, lo1, hi1 = block_bootstrap_ratio(y, preds[meth], preds["Mean baseline"])
                r2, lo2, hi2 = block_bootstrap_ratio(y, preds[meth], preds["EWMA a=0.5 (textbook SJF)"])
                rows.append({"capture": name, "config": config_of(name), "pid": pid, "kind": g.kind.iloc[0],
                             "n_bursts": len(s), "n_test": len(y), "method": meth,
                             "MAE_ms": mae(y, preds[meth]) * 1e3,
                             "vs_mean": r1, "vs_mean_lo": lo1, "vs_mean_hi": hi1,
                             "vs_ewma": r2, "vs_ewma_lo": lo2, "vs_ewma_hi": hi2,
                             "eig_max_log": float(np.abs(info["eig_log"]).max())})
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(OUT, "prediction.csv"), index=False)
    return p


# ------------------------------------------------------------------ 4 simulation
SIM_POLICIES = [("FCFS", "oracle"), ("RR", "oracle"),
                ("SJF", "mean"), ("SJF", "ewma"), ("SJF", "edmd"), ("SJF", "oracle"),
                ("SRTF", "mean"), ("SRTF", "ewma"), ("SRTF", "edmd"), ("SRTF", "oracle")]


def measured_linux(tl, tasks_pids):
    """Same metrics as the simulator, measured on the real trace (what EEVDF actually did)."""
    b = cpu_bursts(tl["run"], tl["wait"])
    b = b[b.pid.isin(tasks_pids)]
    return list(zip(b.pid, b.kind, b.arrival, b.first_run, b.finish, b.demand))


def run_simulation(caps):
    rows = []
    for name, tl in caps.items():
        if not (name.startswith("scale") or name.startswith("oversub") or name == "lab_trace3"):
            continue
        b = cpu_bursts(tl["run"], tl["wait"])
        tasks = tasks_from_bursts(b)
        allowed = len(tl["meta"]["cpus_allowed"])
        core_list = [1, 2, 3, 4, 6, 8, 12, 16, 28] if name == "lab_trace3" else sorted({1, 2, allowed})
        # what real Linux did (only meaningful at the real core count)
        # wait from wake-up: real wake->run waits are in tl["wait"]; burst-level response measured from first run
        s_real, _ = summarize(measured_linux(tl, set(tasks)))
        rows.append({"capture": name, "config": config_of(name), "cores": allowed, "policy": "Linux EEVDF (measured)",
                     "predictor": "-", **s_real})
        for m in core_list:
            for pol, pred in SIM_POLICIES:
                t0 = time.time()
                rec = simulate(tasks, m, pol, pred)
                s, _ = summarize(rec)
                rows.append({"capture": name, "config": config_of(name), "cores": m, "policy": pol,
                             "predictor": pred, **s, "sim_seconds": time.time() - t0})
        print(f"  simulated {name}", flush=True)
    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(OUT, "simulation.csv"), index=False)
    return sim


# ------------------------------------------------------------------ 5 Koopman regime detection
def run_koopman(caps, dt=0.010, train_s=8.0, smooth_bins=25, threshold=3.0, hold=10):
    """
    Detector: fit K on the first `train_s` seconds; score = trailing 0.25 s mean of the one-step
    residual norm in units of its training spread; alarm when the score stays above 3 for 0.1 s.
    Evaluated only while the workload is running: from the end of training until 0.5 s before
    the first full-length task exits (exits empty the machine and would trivially alarm).
    """
    rows, series = [], {}
    for name, tl in caps.items():
        # ground-truth change time = when the CPU-bound jobs that were meant to stop actually stopped
        #   regime runs: the extra_cpu jobs (listed first in pids.txt) stop at 20 s of 40 s
        #   lab Trace 3: the first CPU-bound job exits (~42.9 s; the others follow at 44.2 and 47.6 s)
        run = tl["run"]
        last_end = run.groupby("pid").end.max()
        extra = tl["meta"].get("extra_cpu")
        short_pids = []
        if extra and tl.get("dir"):
            short_pids = [int(l.split()[1]) for l in open(os.path.join(tl["dir"], "pids.txt"))][:extra[0]]
        if short_pids:
            change = last_end[short_pids].max()
        elif name == "lab_trace3":
            change = last_end[run[run.kind == "cpu"].pid.unique()].min()
        else:
            change = None
        full = [p for p in last_end.index if p not in short_pids and not (name == "lab_trace3" and p in run[run.kind == "cpu"].pid.values)]
        eval_end = last_end[full].min() - 0.5

        for variant in ("per_core", "aggregate"):
            S = machine_state(tl, dt=dt)
            if variant == "aggregate":
                S = aggregate_state(S)
            times = S.index.values
            X = S.values
            tr = times < times[0] + train_s
            km = LinearKoopman().fit(X[tr])
            scores = {"koopman": smooth(km.surprise(X), smooth_bins),
                      "persistence": smooth(persistence_surprise(X[tr], X), smooth_bins),
                      "novelty": smooth(novelty_score(X[tr], X), smooth_bins)[1:]}
            t_res = times[1:]
            inwin = t_res <= eval_end
            after = times[0] + train_s
            rec = {"capture": name, "config": config_of(name), "state": variant, "dims": int(km.keep.sum()),
                   "has_change": change is not None,
                   "change_time_s": (change - times[0]) if change is not None else np.nan,
                   "eval_end_s": eval_end - times[0], "rho_train": float(np.abs(km.eig()[0]).max())}
            for label, sc in scores.items():
                sc, tt = sc[inwin], t_res[inwin]
                if change is not None:
                    pre = tt < change
                    fa = first_alarm(sc[pre], tt[pre], threshold, after, hold)
                    al = first_alarm(sc, tt, threshold, change, hold)
                    rec[f"{label}_false_alarm"] = fa is not None
                    rec[f"{label}_delay_s"] = (al - change) if al is not None else np.nan
                else:
                    rec[f"{label}_false_alarm"] = first_alarm(sc, tt, threshold, after, hold) is not None
                    rec[f"{label}_delay_s"] = np.nan
            rows.append(rec)
            if variant == "per_core":
                series[name] = {"t": t_res - times[0], **scores,
                                "change": (change - times[0]) if change is not None else None,
                                "eval_end": eval_end - times[0], "state": S, "eig": km.eig()[0]}
    kd = pd.DataFrame(rows)
    kd.to_csv(os.path.join(OUT, "koopman_detection.csv"), index=False)
    pickle.dump(series, open(os.path.join(OUT, "koopman_series.pkl"), "wb"))
    return kd


def main():
    os.makedirs(TL, exist_ok=True)
    t = time.time()
    caps = load_all()
    print(f"loaded {len(caps)} captures in {time.time() - t:.0f}s", flush=True)
    steps = sys.argv[1:] or ["validation", "latency", "prediction", "simulation", "koopman"]
    if "validation" in steps:
        v = run_validation(caps)
        print("\nVALIDATION (relative error vs kernel counters, %)")
        print(v.groupby("kind")[["run_err_%", "wait_err_%"]].describe().round(3).to_string())
        print("orphan switch-outs per process: median", v.orphan_switch_outs.median(), "max", v.orphan_switch_outs.max())
        print("timeslice difference |trace-kernel|: median", (v.slices_trace - v.slices_kernel).abs().median(),
              "max", (v.slices_trace - v.slices_kernel).abs().max())
        print("exact count matches: slices", (v.slices_trace == v.slices_kernel).mean().round(3),
              " invol", (v.invol_trace == v.invol_kernel).mean().round(3),
              " vol", (v.vol_trace == v.vol_kernel).mean().round(3), " lost events", v.lost_events.sum())
    if "latency" in steps:
        lat = run_latency(caps)
        print("\nLATENCY")
        print(lat.groupby(["config", "kind"]).median(numeric_only=True)[
            ["cores", "utilisation_allowed_cores", "wake_to_run_median_ms", "wake_to_run_p95_ms",
             "frac_time_waiting", "migrations_per_task_per_s"]].round(3).to_string())
    if "prediction" in steps:
        run_prediction(caps)
        print("\nprediction done", flush=True)
    if "simulation" in steps:
        run_simulation(caps)
    if "koopman" in steps:
        kd = run_koopman(caps)
        print("\nKOOPMAN DETECTION")
        print(kd.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
