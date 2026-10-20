#!/usr/bin/env python3
"""
change_run.py - scripted load changes with a live Koopman detector watching.

No root needed. Each run lasts 40 s:
  * a background workload runs the whole time
        max(1, N//2) CPU-bound jobs, 2 I/O-bound jobs (sleep 50 ms), 1 phased job   (N = cores)
  * at a random moment t_c in [15, 25] s (not told to the detector) ONE change happens:
        load_up    small: +1 CPU-bound job        large: +N CPU-bound jobs
        load_down  small: 1 extra job stops       large: N extra jobs stop  (they ran from the start)
        io_burst   small: +2 fast I/O jobs        large: +2N fast I/O jobs  (write 10 kB, sleep 5 ms)
        none       (null run: nothing changes)
  * kit/live_monitor.py runs as a separate process from the start, fits its operator on
    the first 8 s and logs every 20 ms bin, its score and its alarms.

    python3 change_run.py --list                 # show the plan
    python3 change_run.py --out runs --from 0 --count 3
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "workloads")
DUR = 40.0


def plan(reps=3):
    runs = []
    for rep in range(1, reps + 1):
        for kind in ("load_up", "load_down", "io_burst"):
            for size in ("small", "large"):
                runs.append({"name": f"{kind}_{size}_r{rep}", "kind": kind, "size": size, "rep": rep})
        for k in range(2):
            runs.append({"name": f"none_r{rep}{'ab'[k]}", "kind": "none", "size": "-", "rep": rep})
    return runs


def mono():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def launch(script, args, env):
    return subprocess.Popen([sys.executable, os.path.join(WL, script)] + [str(x) for x in args], env=env, cwd=WL)


def trace_start():
    import capture as C
    C.ensure_tracefs()
    T = C.T
    C.w(f"{T}/tracing_on", 0); C.w(f"{T}/trace", ""); C.w(f"{T}/trace_clock", "mono")
    C.w(f"{T}/buffer_size_kb", 131072); C.w(f"{T}/events/enable", 0)
    for e in C.EVENTS:
        if os.path.exists(f"{T}/events/sched/{e}"):
            C.w(f"{T}/events/sched/{e}/enable", 1)
    C.w(f"{T}/tracing_on", 1)


def trace_stop(d):
    import gzip
    import shutil
    import capture as C
    T = C.T
    C.w(f"{T}/tracing_on", 0)
    with open(f"{T}/trace") as fi, gzip.open(os.path.join(d, "trace.txt.gz"), "wt", compresslevel=6) as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    stats = C.ring_stats()
    C.w(f"{T}/events/enable", 0)
    return sum(int(v.get("overrun", 0)) for v in stats.values())


def run_one(spec, outroot, trace=False, baseline_cpu=None):
    n = os.cpu_count()
    d = os.path.join(outroot, spec["name"])
    os.makedirs(os.path.join(d, "ground_truth"), exist_ok=True)
    env = dict(os.environ, GT_DIR=os.path.join(d, "ground_truth"), PYTHONDONTWRITEBYTECODE="1")
    rnd = random.Random(spec["name"])
    t_c = rnd.uniform(15.0, 25.0)
    k_small, k_large = 1, n
    count = {"small": 1, "large": n}.get(spec["size"], 0)

    if trace:
        trace_start()
    t0 = mono()
    monitor = None
    if True:
        monitor = subprocess.Popen([sys.executable, os.path.join(HERE, "live_monitor.py"), "--secs", str(DUR + 0.5),
                                    "--log", os.path.join(d, "monitor.jsonl"), "--quiet"], env=env)
    time.sleep(0.05)
    procs = []
    for _ in range(baseline_cpu or max(1, n // 2)):
        procs.append(("cpu", launch("cpu_timed.py", [DUR], env)))
    for _ in range(2):
        procs.append(("io", launch("io_timed.py", [DUR], env)))
    procs.append(("phased", launch("phased.py", [DUR], env)))
    stoppers = []
    if spec["kind"] == "load_down":
        for _ in range(count):
            p = launch("cpu_timed.py", [t_c], env)
            procs.append(("cpu_extra", p))
            stoppers.append(p)
    exit_times = []
    for p in stoppers:
        threading.Thread(target=lambda p=p: (p.wait(), exit_times.append(mono())), daemon=True).start()

    change_mono = None
    time.sleep(max(0.0, t0 + t_c - mono()))
    if spec["kind"] == "load_up":
        change_mono = mono()
        for _ in range(count):
            procs.append(("cpu_extra", launch("cpu_timed.py", [DUR - t_c], env)))
    elif spec["kind"] == "io_burst":
        change_mono = mono()
        for _ in range(2 * count):
            procs.append(("io_extra", launch("io_timed.py", [DUR - t_c, 0.005], env)))
    for _, p in procs:
        p.wait()
    if monitor:
        monitor.wait()
    lost = trace_stop(d) if trace else None
    if spec["kind"] == "load_down":
        change_mono = min(exit_times) if exit_times else None

    meta = {**spec, "ncpu": n, "t0_mono": t0, "planned_change_s": t_c,
            "change_mono": change_mono, "change_s": (change_mono - t0) if change_mono else None,
            "jobs": [(k, p.pid, p.returncode) for k, p in procs], "kernel": os.uname().release,
            "traced": trace, "lost_events": lost, "live_monitor": monitor is not None,
            "baseline_cpu": baseline_cpu or max(1, n // 2),
            "machine": os.uname().machine}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(d, "pids.txt"), "w") as f:
        for k, p in procs:
            f.write(f"{k.replace('_extra', '')} {p.pid}\n")
    ok = all(p.returncode == 0 for _, p in procs) and (monitor is None or monitor.returncode == 0) and not lost
    print(f"{spec['name']:22s} change at {meta['change_s'] if meta['change_s'] else '-'}  {'OK' if ok else 'CHECK'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "change_runs"))
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--count", type=int, default=999)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--trace", action="store_true", help="also record a full tracefs trace (root)")
    ap.add_argument("--baseline-cpu", type=int, help="CPU-bound background jobs (default: half the cores)")
    a = ap.parse_args()
    runs = plan(a.reps)
    if a.list:
        for i, r in enumerate(runs):
            print(i, r["name"])
        return
    os.makedirs(a.out, exist_ok=True)
    for spec in runs[a.start:a.start + a.count]:
        run_one(spec, a.out, a.trace, a.baseline_cpu)


if __name__ == "__main__":
    main()
