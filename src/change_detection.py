"""
change_detection.py
-------------------
Evaluates the scripted change experiments (kit/change_run.py) on both platforms.

  data/change_mac/   4-core ARM VM (Apple M-series host), Linux 6.8, LIVE detector:
                     kit/live_monitor.py ran as its own process; state from /proc/stat
                     (busy cores, runnable tasks, context switches) in 100 ms bins.
  data/change_vm/    2-core Intel VM, Linux 6.18, full kernel traces; machine state rebuilt
                     offline from the trace (busy cores, queued tasks, job-class CPU shares)
                     in 10 ms bins, every core kept busy (see RESULTS_MULTICORE.md, idle-exit loss).

Rules (same for every detector):
  fit / calibrate on the first 8 s; score only until 1 s before the 40 s workload ends; alarm = smoothed score above threshold for the hold time;
  detected = first alarm onset in [change, change + 5 s]; delay = onset - change;
  false alarm = an onset after training and before the change, or any onset in a null run.

Detectors: Koopman residual (one linear operator), persistence residual (K = I),
distance from normal (same state, no dynamics), and two plain monitors: utilisation
(busy cores) and run-queue length, each as a z-score against training.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from koopman_system import aggregate_state, machine_state  # noqa: E402
from sched_events import build_timeline, load_pids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "change")
WINDOW = 5.0
TRAIN = 8.0


def trailing(x, n):
    return pd.Series(x).rolling(n, min_periods=1).mean().values


def scores_from_state(S, t, n_smooth):
    """All detector scores for a state matrix S (rows = bins) with times t."""
    tr = t < t[0] + TRAIN
    mu, sd = S[tr].mean(0), S[tr].std(0)
    keep = sd > 1e-9
    Z = (S[:, keep] - mu[keep]) / sd[keep]
    Ztr = Z[tr]
    K = np.linalg.pinv(Ztr[:-1]) @ Ztr[1:]
    out = {}
    r_tr = np.sqrt(((Ztr[1:] - Ztr[:-1] @ K) ** 2).sum(1))
    r = np.r_[np.nan, np.sqrt(((Z[1:] - Z[:-1] @ K) ** 2).sum(1))]
    out["koopman"] = trailing((r - r_tr.mean()) / r_tr.std(), n_smooth)
    p_tr = np.sqrt(((Ztr[1:] - Ztr[:-1]) ** 2).sum(1))
    p = np.r_[np.nan, np.sqrt(((Z[1:] - Z[:-1]) ** 2).sum(1))]
    out["persistence"] = trailing((p - p_tr.mean()) / p_tr.std(), n_smooth)
    d_tr = np.sqrt((Ztr ** 2).sum(1))
    dist = np.sqrt((Z ** 2).sum(1))
    out["novelty"] = trailing((dist - d_tr.mean()) / d_tr.std(), n_smooth)
    out["utilisation"] = trailing(np.abs((S[:, 0] - mu[0]) / (sd[0] + 1e-9)), n_smooth)
    out["run_queue"] = trailing(np.abs((S[:, 1] - mu[1]) / (sd[1] + 1e-9)), n_smooth)
    rho = float(np.abs(np.linalg.eigvals(K)).max())
    return out, rho


def onsets(score, t, thr, hold, after, until=np.inf):
    above = (score > thr) & (t >= after) & (t <= until)
    res, run = [], 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run == hold:
            res.append(t[i - hold + 1])
    return res


def evaluate_run(sc, t, change, thr, hold, until):
    after = t[0] + TRAIN
    rec = {}
    for name, s in sc.items():
        on = onsets(np.nan_to_num(s, nan=-1e9), t, thr, hold, after, until)
        if change is None:
            rec[name] = {"false_alarm": len(on) > 0, "detected": None, "delay": None}
        else:
            pre = [o for o in on if o < change]
            post = [o for o in on if change <= o <= change + WINDOW]
            rec[name] = {"false_alarm": len(pre) > 0, "detected": len(post) > 0,
                         "delay": (post[0] - change) if post else None}
    return rec


# ---------------------------------------------------------------- platform loaders
def mac_runs(root):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mf = os.path.join(d, "meta.json")
        if not os.path.exists(mf):
            continue
        meta = json.load(open(mf))
        rows = [json.loads(l) for l in open(os.path.join(d, "monitor.jsonl"))]
        summ = [r for r in rows if r.get("summary")][0]
        bins = [r for r in rows if "busy" in r][1:]
        t = np.array([r["t"] for r in bins])
        S = np.array([[r["busy"], r["runnable"], r["switch_k"]] for r in bins])
        live = {"onsets": summ["alarms"], "overhead_pct": summ["cpu_overhead_pct_of_one_core"]}
        runs.append({"name": os.path.basename(d), "meta": meta, "t": t, "S": S, "live": live,
                     "change": meta["change_mono"], "dt": summ.get("dt", 0.1), "until": meta["t0_mono"] + 39.0})
    return runs


def vm_runs(root, dt=0.010):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        mf = os.path.join(d, "meta.json")
        if not os.path.exists(mf):
            continue
        meta = json.load(open(mf))
        tl = build_timeline(os.path.join(d, "trace.txt.gz"), load_pids(os.path.join(d, "pids.txt")))
        st = aggregate_state(machine_state(tl, dt=dt))
        cols = ["busy_total", "queue_total", "share_cpu", "share_io", "share_phased"]
        S = st[cols].values
        runs.append({"name": os.path.basename(d), "meta": meta, "t": st.index.values, "S": S,
                     "change": meta["change_mono"], "dt": dt, "orphans": sum(tl["orphans"].values()),
                     "until": meta["t0_mono"] + 39.0})
    return runs


def analyse(runs, platform, smooth_s, hold_s):
    rows, sweep = [], []
    for r in runs:
        n_smooth = max(1, int(round(smooth_s / r["dt"])))
        hold = max(1, int(round(hold_s / r["dt"])))
        sc, rho = scores_from_state(r["S"], r["t"], n_smooth)
        r["scores"], r["rho"] = sc, rho
        for thr in [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]:
            ev = evaluate_run(sc, r["t"], r["change"], thr, hold, r["until"])
            for name, e in ev.items():
                sweep.append({"platform": platform, "run": r["name"], "kind": r["meta"]["kind"],
                              "size": r["meta"]["size"], "detector": name, "threshold": thr, **e})
        if "live" in r:      # the alarms the live process actually raised, online, at threshold 3
            on = [o for o in r["live"]["onsets"] if r["t"][0] + TRAIN - 0.2 <= o <= r["until"]]
            if r["change"] is None:
                rec = {"false_alarm": len(on) > 0, "detected": None, "delay": None}
            else:
                post = [o for o in on if r["change"] <= o <= r["change"] + WINDOW]
                rec = {"false_alarm": any(o < r["change"] for o in on), "detected": bool(post),
                       "delay": (post[0] - r["change"]) if post else None}
            rows.append({"platform": platform, "run": r["name"], "kind": r["meta"]["kind"], "size": r["meta"]["size"],
                         "overhead_pct": r["live"]["overhead_pct"], **rec})
    return pd.DataFrame(sweep), pd.DataFrame(rows)


def summarise(sweep):
    g = []
    for (pl, det, thr), s in sweep.groupby(["platform", "detector", "threshold"]):
        ch = s[s.kind != "none"]
        g.append({"platform": pl, "detector": det, "threshold": thr,
                  "changes": len(ch), "detected": int(ch.detected.sum()),
                  "detection_rate": ch.detected.mean(),
                  "median_delay_s": ch.delay.dropna().median() if ch.detected.any() else np.nan,
                  "runs": len(s), "runs_with_false_alarm": int(s.false_alarm.sum())})
    return pd.DataFrame(g)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = []
    live_all = []
    plat = {}
    for label, sub in (("mac_arm_4core_live", "change_mac"), ("vm_intel_2core_live", "change_vm")):
        dd = os.path.join(ROOT, "data", sub)
        if os.path.isdir(dd):
            rr = mac_runs(dd)
            sw, live = analyse(rr, label, smooth_s=0.5, hold_s=0.2)
            out.append(sw); live_all.append(live); plat[label] = rr
    if "--trace" in sys.argv:          # cross-check: VM state rebuilt from kernel traces
        vr = vm_runs(os.path.join(ROOT, "data", "change_vm"))
        sw, _ = analyse(vr, "vm_intel_2core_trace", smooth_s=0.25, hold_s=0.1)
        out.append(sw); plat["vm_trace"] = vr
        print("VM orphan switch-outs per run:", [r["orphans"] for r in vr])
    sweep = pd.concat(out)
    sweep.to_csv(os.path.join(OUT, "sweep.csv"), index=False)
    summ = summarise(sweep)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    if live_all:
        live = pd.concat(live_all)
        live.to_csv(os.path.join(OUT, "live_alarms.csv"), index=False)
        for pl, lv in live.groupby("platform"):
            ch = lv[lv.kind != "none"]
            print(f"\n{pl}: LIVE detector (online, separate process, threshold 3): detected "
                  f"{int(ch.detected.sum())}/{len(ch)} changes, median delay {ch.delay.dropna().median():.2f} s, "
                  f"runs with a false alarm {int(lv.false_alarm.sum())}/{len(lv)}, CPU overhead median "
                  f"{lv.overhead_pct.median():.2f}% of one core")
        ch = live[live.kind != "none"]
        print(f"\nLIVE detector (online, separate process, threshold 3): detected {int(ch.detected.sum())}/{len(ch)} "
              f"changes, median delay {ch.delay.dropna().median():.2f} s, runs with a false alarm "
              f"{int(live.false_alarm.sum())}/{len(live)}, CPU overhead median {live.overhead_pct.median():.2f}% of one core")
        print(live.groupby(["platform", "kind", "size"]).agg(detected=("detected", "sum"), n=("detected", "size"),
                                                 delay=("delay", "median")).to_string())
    pd.set_option("display.width", 200)
    print("\nAt threshold 3:")
    print(summ[summ.threshold == 3.0].round(3).to_string(index=False))
    by_kind = sweep[(sweep.threshold == 3.0) & (sweep.kind != "none")].groupby(
        ["platform", "detector", "kind", "size"]).detected.mean().unstack([2, 3]).round(2)
    print("\nDetection rate by change type, threshold 3:")
    print(by_kind.to_string())
    import pickle
    pickle.dump(plat, open(os.path.join(OUT, "runs.pkl"), "wb"))


if __name__ == "__main__":
    main()
