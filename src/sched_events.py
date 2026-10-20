"""
sched_events.py
---------------
One parser for both trace formats used in this project:
  * `perf sched script` text (lab Trace 3)
  * raw tracefs / ftrace text (captures made with kit/capture.py)

It rebuilds, for every task we care about, a complete timeline of the three
things the scheduler decides:

  RUN    intervals  task is on a CPU            (pid, cpu, start, end, end_state)
  WAIT   intervals  task is runnable but queued (pid, cpu, start, end, cause)
                    cause = 'wake'    : woke up and is waiting for a CPU
                            'preempt' : was taken off a CPU while still runnable
                            'new'     : newly created task waiting for its first run
  MIGRATIONS        task moved between CPUs     (pid, ts, from, to)

and, for every CPU, the occupancy of ALL tasks (to compute per-core load).
"""

import gzip
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------- ftrace (tracefs) format ----------
F_HEAD = re.compile(r"\[(\d+)\]\s+\S+\s+(\d+\.\d+):\s+(sched_\w+):\s(.*)$")
F_SWITCH = re.compile(r"prev_comm=(.*) prev_pid=(\d+) prev_prio=\d+ prev_state=(\S+) ==> "
                      r"next_comm=(.*) next_pid=(\d+) next_prio=\d+")
F_WAKE = re.compile(r"comm=(.*) pid=(\d+) prio=\d+ target_cpu=(\d+)")
F_MIG = re.compile(r"comm=(.*) pid=(\d+) prio=\d+ orig_cpu=(\d+) dest_cpu=(\d+)")
# ---------- perf sched script format ----------
P_HEAD = re.compile(r"\[(\d+)\]\s+(\d+\.\d+):\s+sched:(sched_\w+):\s+(.*)$")
P_SWITCH = re.compile(r"(.+?):(\d+) \[\d+\] (\S+) ==> (.+?):(\d+) \[\d+\]")


def open_text(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") else open(path, errors="replace")


def iter_events(path):
    """Yield (ts, cpu, event, fields_dict) for switch / waking / wakeup / wakeup_new / migrate."""
    with open_text(path) as f:
        for line in f:
            if "sched_" not in line:
                continue
            m = F_HEAD.search(line)
            perf = False
            if not m:
                m = P_HEAD.search(line)
                perf = True
                if not m:
                    continue
            cpu, ts, ev, rest = int(m[1]), float(m[2]), m[3], m[4]
            if ev == "sched_switch":
                s = (P_SWITCH if perf else F_SWITCH).search(rest)
                if s:
                    yield ts, cpu, "switch", {"prev_comm": s[1], "prev_pid": int(s[2]), "prev_state": s[3],
                                              "next_comm": s[4], "next_pid": int(s[5])}
            elif ev in ("sched_waking", "sched_wakeup", "sched_wakeup_new"):
                s = F_WAKE.search(rest)
                if s:
                    yield ts, cpu, ev[6:], {"comm": s[1], "pid": int(s[2]), "target": int(s[3])}
            elif ev == "sched_migrate_task":
                s = F_MIG.search(rest)
                if s:
                    yield ts, cpu, "migrate", {"comm": s[1], "pid": int(s[2]),
                                               "orig": int(s[3]), "dest": int(s[4])}


def build_timeline(path, tracked):
    """
    tracked : dict pid -> kind   (the workload processes)
    Returns dict of DataFrames: run, wait, migrations, cpu_busy (all tasks, per CPU), plus
    t0/t1 and the set of CPUs seen.
    """
    run, wait, mig = [], [], []
    running = {}            # pid -> (start, cpu)
    queued = {}             # pid -> (start, cpu, cause)
    on_cpu = {}             # cpu -> (pid, start)   for ALL tasks (pid 0 = idle)
    busy = defaultdict(list)  # cpu -> list of (start, end, pid) for non-idle tasks
    comm = {}
    waking = {}             # pid -> (ts, target) from sched_waking, fallback only
    orphans = defaultdict(int)
    t0 = t1 = None
    cpus = set()

    for ts, cpu, ev, f in iter_events(path):
        t0 = ts if t0 is None else t0
        t1 = ts
        cpus.add(cpu)
        if ev == "switch":
            p, n, st = f["prev_pid"], f["next_pid"], f["prev_state"]
            # per-CPU occupancy for every task
            if cpu in on_cpu and on_cpu[cpu][0] != 0:
                busy[cpu].append((on_cpu[cpu][1], ts, on_cpu[cpu][0]))
            on_cpu[cpu] = (n, ts)
            # tracked task leaving the CPU
            if p in tracked:
                comm[p] = f["prev_comm"]
                if p in running:
                    start, c = running.pop(p)
                    run.append((p, c, start, ts, st[0]))
                else:
                    orphans[p] += 1          # switch-in was never recorded
                if st[0] == "R":                       # preempted while runnable -> queued
                    queued[p] = (ts, cpu, "preempt")
            # tracked task entering the CPU
            if n in tracked:
                comm[n] = f["next_comm"]
                if n in queued:
                    qs, qc, cause = queued.pop(n)
                    wait.append((n, cpu, qs, ts, cause))
                elif n in waking:
                    qs, _ = waking[n]
                    wait.append((n, cpu, qs, ts, "wake"))
                waking.pop(n, None)
                running[n] = (ts, cpu)
        elif ev in ("wakeup", "wakeup_new"):
            # enqueue on a run queue: the moment the task starts waiting for a CPU
            pid = f["pid"]
            if pid in tracked and pid not in running:
                queued[pid] = (ts, f["target"], "new" if ev == "wakeup_new" else "wake")
                waking.pop(pid, None)
        elif ev == "waking":
            # emitted just before the enqueue; used only if the trace has no sched_wakeup events
            pid = f["pid"]
            if pid in tracked and pid not in running and pid not in queued:
                waking[pid] = (ts, f["target"])
        elif ev == "migrate":
            pid = f["pid"]
            if pid in tracked:
                mig.append((pid, ts, f["orig"], f["dest"]))
                if pid in queued:
                    qs, _, cause = queued[pid]
                    queued[pid] = (qs, f["dest"], cause)

    runs = pd.DataFrame(run, columns=["pid", "cpu", "start", "end", "end_state"])
    runs["dur"] = runs.end - runs.start
    runs["kind"] = runs.pid.map(tracked)
    waits = pd.DataFrame(wait, columns=["pid", "cpu", "start", "end", "cause"])
    waits["dur"] = waits.end - waits.start
    waits["kind"] = waits.pid.map(tracked)
    migs = pd.DataFrame(mig, columns=["pid", "ts", "orig", "dest"])
    migs["kind"] = migs.pid.map(tracked)
    rows = [(c, s, e, p) for c, lst in busy.items() for s, e, p in lst]
    cpu_busy = pd.DataFrame(rows, columns=["cpu", "start", "end", "pid"])
    return {"run": runs, "wait": waits, "migrations": migs, "cpu_busy": cpu_busy,
            "t0": t0, "t1": t1, "cpus": sorted(cpus), "comm": comm, "orphans": dict(orphans)}


def load_pids(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                out[int(parts[1])] = parts[0]
    return out


def cpu_bursts(runs, waits=None):
    """
    Textbook CPU bursts: CPU time between two blocking events. Consecutive RUN
    intervals that ended in R (preemption) are merged, because the task never
    stopped wanting the CPU.
    arrival  = when the task became runnable (its wake-up / creation enqueue time, taken from the
               WAIT interval that ends where the burst's first piece starts); if no such wait was
               recorded (the task got a CPU instantly), arrival = start of the first piece
    first_run= start of the first piece, finish = end of the last piece, demand = CPU time used.
    """
    wake_start = {}
    if waits is not None and len(waits):
        w = waits[waits.cause.isin(["wake", "new"])]
        wake_start = {(p, round(e, 7)): s for p, e, s in zip(w.pid.values, w.end.values, w.start.values)}
    out = []
    for pid, g in runs.sort_values(["pid", "start"]).groupby("pid"):
        acc, first, pieces = 0.0, None, 0
        for r in g.itertuples():
            if first is None:
                first = r.start
            acc += r.dur
            pieces += 1
            if r.end_state != "R":
                arr = wake_start.get((pid, round(first, 7)), first)
                out.append((pid, g.kind.iloc[0], arr, first, r.end, acc, pieces, r.end_state))
                acc, first, pieces = 0.0, None, 0
        if first is not None:
            arr = wake_start.get((pid, round(first, 7)), first)
            out.append((pid, g.kind.iloc[0], arr, first, g.end.iloc[-1], acc, pieces, "end"))
    return pd.DataFrame(out, columns=["pid", "kind", "arrival", "first_run", "finish", "demand", "pieces", "end_state"])
