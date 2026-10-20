#!/usr/bin/env python3
"""
scale_counters.py - contention vs core count from kernel counters only (no root, no tracing).

The same 8 jobs as the traced scaling runs (3 CPU-bound, 3 I/O-bound, 2 phased) are pinned
to 1, 2, 4, ... cores. Each job saves the kernel's own /proc/self/schedstat at exit:
run time, run-queue wait time and number of times scheduled. Mean wait per scheduling
= wait_ns / timeslices, the average time a job sat in a run queue before each turn on a CPU.

    python3 scale_counters.py --out scale_runs --cores 1 2 4 --reps 3 --secs 30
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "workloads")


def run(out, cores, rep, secs):
    name = f"c{cores:02d}_r{rep}"
    d = os.path.join(out, name, "ground_truth")
    os.makedirs(d, exist_ok=True)
    env = dict(os.environ, GT_DIR=d, PYTHONDONTWRITEBYTECODE="1")
    aff = set(range(cores))
    jobs = [("cpu", "cpu_timed.py")] * 3 + [("io", "io_timed.py")] * 3 + [("phased", "phased.py")] * 2
    t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
    ps = [(k, subprocess.Popen([sys.executable, os.path.join(WL, s), str(secs)], env=env, cwd=WL,
                               preexec_fn=lambda: os.sched_setaffinity(0, aff))) for k, s in jobs]
    rc = [p.wait() for _, p in ps]
    meta = {"cores": cores, "rep": rep, "secs": secs, "ncpu": os.cpu_count(), "kernel": os.uname().release,
            "machine": os.uname().machine, "exit_codes": rc, "wall_s": time.clock_gettime(time.CLOCK_MONOTONIC) - t0}
    json.dump(meta, open(os.path.join(out, name, "meta.json"), "w"), indent=1)
    print(name, "OK" if all(c == 0 for c in rc) else "CHECK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "scale_runs"))
    ap.add_argument("--cores", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rep-from", type=int, default=1)
    ap.add_argument("--secs", type=float, default=30)
    a = ap.parse_args()
    for rep in range(a.rep_from, a.rep_from + a.reps):
        for c in a.cores:
            if c <= os.cpu_count():
                run(a.out, c, rep, a.secs)


if __name__ == "__main__":
    main()
