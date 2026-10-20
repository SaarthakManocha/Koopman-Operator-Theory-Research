"""
simulate.py
-----------
Trace-driven multi-core scheduling simulator.

Each traced task is replayed as a closed loop, exactly as it behaved in the
real trace:
        CPU burst d1 -> blocks for s1 -> CPU burst d2 -> blocks for s2 -> ...
The burst lengths d_k (CPU demand) and blocking times s_k (sleep / disk wait)
come from the trace. Blocking time does not depend on the scheduler, so a task
that finishes a burst later in the simulation also comes back later; the
arrival times shift with the policy, as they would on a real system.

Policies (m identical cores, one shared run queue, no migration cost):
    FCFS        first come first served, non-preemptive
    RR          round robin, quantum q (default 3 ms, close to the EEVDF base slice)
    SJF[p]      shortest predicted job first, non-preemptive
    SRTF[p]     shortest predicted remaining time first, preemptive on arrival
where p is a burst predictor that only sees the task's own previous bursts:
    oracle      true length (upper bound, impossible in practice)
    ewma        textbook exponential average, alpha = 0.5
    edmd        EDMD one-step forecast (delay 3), refitted online every 25 bursts
    last        previous burst length
    mean        running mean of the task's previous bursts

Metrics per burst: response = arrival -> first time on a CPU,
                   turnaround = arrival -> completion,  wait = turnaround - demand.
"""

import heapq
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from edmd import EDMD  # noqa: E402

TAU0 = 0.010  # prior guess for a task's first burst (10 ms), as in the textbook


# ------------------------------------------------------------------ predictors
class Predictor:
    def __init__(self, kind):
        self.kind = kind
        self.hist = []
        self.tau = TAU0
        self.model = None

    def predict(self, true_len):
        if self.kind == "oracle":
            return true_len
        if not self.hist:
            return TAU0
        if self.kind == "ewma":
            return self.tau
        if self.kind == "last":
            return self.hist[-1]
        if self.kind == "mean":
            return float(np.mean(self.hist))
        if self.kind == "edmd":
            if self.model is None:
                return self.tau
            h = np.array(self.hist[-3:])
            return float(max(self.model.predict_one_step(np.r_[h, 0.0], 3)[0], 1e-6))
        raise ValueError(self.kind)

    def update(self, true_len):
        self.hist.append(true_len)
        self.tau = 0.5 * true_len + 0.5 * self.tau
        n = len(self.hist)
        if self.kind == "edmd" and n >= 30 and n % 25 == 0:
            try:
                self.model = EDMD(3, "linear", log=True).fit(np.array(self.hist[-500:]))
            except np.linalg.LinAlgError:
                pass


# ------------------------------------------------------------------ simulator
def simulate(tasks, m, policy, predictor="oracle", quantum=0.003, horizon=None):
    """
    tasks : dict pid -> {"kind": str, "start": t0, "bursts": [d1, d2, ...], "blocks": [s1, s2, ...]}
    Returns list of per-burst records (pid, kind, arrival, first_run, finish, demand).
    """
    preemptive = policy in ("RR", "SRTF")
    seq = itertools.count()
    events = []                       # (time, seq, type, pid)
    state = {}
    for pid, t in tasks.items():
        state[pid] = {"k": 0, "pred": Predictor(predictor), "kind": t["kind"]}
        heapq.heappush(events, (t["start"], next(seq), "arrive", pid))
    ready = []                        # list of pids (ordering done at dispatch)
    cores = [None] * m                # pid running or None
    run_since = [0.0] * m
    slice_end = [None] * m
    job = {}                          # pid -> current burst info
    records = []
    now = 0.0

    def key(pid):
        j = job[pid]
        if policy in ("FCFS", "RR"):
            return (j["enq"], 0)
        if policy == "SJF":
            return (j["pred_len"], j["enq"])
        return (predicted_remaining(j, 0.0), j["enq"])           # SRTF

    def predicted_remaining(j, extra):
        # a job that has already run longer than predicted is assumed to need at least as long
        # again (otherwise a never-ending CPU-bound burst would look like 0 ms left forever)
        done = j["done"] + extra
        return j["pred_len"] - done if done < j["pred_len"] else done

    def dispatch(t):
        for c in range(m):
            if cores[c] is None and ready:
                ready.sort(key=key)
                pid = ready.pop(0)
                cores[c] = pid
                run_since[c] = t
                j = job[pid]
                if j["first_run"] is None:
                    j["first_run"] = t
                remaining = j["demand"] - j["done"]
                end = t + remaining
                if policy == "RR" and remaining > quantum:
                    end = t + quantum
                slice_end[c] = (end, next(seq))
                heapq.heappush(events, (end, slice_end[c][1], "slice", (c, pid)))

    def preempt_check(t, newpid):
        if policy != "SRTF":
            return
        # preempt the running job with the largest predicted remaining time if the newcomer is shorter
        cand = [(predicted_remaining(job[p], t - run_since[c]), c)
                for c, p in enumerate(cores) if p is not None]
        if not cand or None in cores:
            return
        worst, c = max(cand)
        if job[newpid]["pred_len"] < worst:
            p = cores[c]
            job[p]["done"] += t - run_since[c]
            job[p]["enq"] = t
            cores[c] = None
            slice_end[c] = None
            ready.append(p)

    while events:
        t, s, typ, data = heapq.heappop(events)
        if horizon and t > horizon:
            break
        now = t
        if typ == "arrive":
            pid = data
            st = state[pid]
            d = tasks[pid]["bursts"][st["k"]]
            job[pid] = {"arrival": t, "demand": d, "done": 0.0, "first_run": None, "enq": t,
                        "pred_len": st["pred"].predict(d)}
            ready.append(pid)
            preempt_check(t, pid)
        elif typ == "slice":
            c, pid = data
            if cores[c] != pid or slice_end[c] is None or slice_end[c][1] != s:
                continue                      # stale event (job was preempted)
            j = job[pid]
            j["done"] += t - run_since[c]
            cores[c] = None
            slice_end[c] = None
            if j["done"] >= j["demand"] - 1e-12:
                st = state[pid]
                records.append((pid, st["kind"], j["arrival"], j["first_run"], t, j["demand"]))
                st["pred"].update(j["demand"])
                st["k"] += 1
                if st["k"] < len(tasks[pid]["bursts"]):
                    block = tasks[pid]["blocks"][st["k"] - 1]
                    heapq.heappush(events, (t + block, next(seq), "arrive", pid))
            else:                              # RR quantum expired
                j["enq"] = t
                ready.append(pid)
        dispatch(now)
    return records


def tasks_from_bursts(bursts):
    """bursts: DataFrame from sched_events.cpu_bursts. Blocking time = next wake-up minus finish."""
    tasks = {}
    for pid, g in bursts.sort_values(["pid", "arrival"]).groupby("pid"):
        arr, fin, dem = g.arrival.values, g.finish.values, g.demand.values
        blocks = list(np.maximum(arr[1:] - fin[:-1], 0.0))
        tasks[pid] = {"kind": g.kind.iloc[0], "start": float(arr[0]), "bursts": list(dem), "blocks": blocks}
    return tasks


def summarize(records, kinds=("io", "phased")):
    import pandas as pd
    df = pd.DataFrame(records, columns=["pid", "kind", "arrival", "first_run", "finish", "demand"])
    df["response"] = df.first_run - df.arrival
    df["turnaround"] = df.finish - df.arrival
    df["wait"] = df.turnaround - df.demand
    inter = df[df.kind.isin(kinds)]
    return {
        "n_bursts": len(df),
        "interactive_mean_response_ms": 1e3 * inter.response.mean(),
        "interactive_p95_response_ms": 1e3 * inter.response.quantile(0.95),
        "interactive_p99_response_ms": 1e3 * inter.response.quantile(0.99),
        "interactive_max_response_ms": 1e3 * inter.response.max(),
        "interactive_mean_turnaround_ms": 1e3 * inter.turnaround.mean(),
        "all_mean_wait_ms": 1e3 * df.wait.mean(),
        "batch_finish_s": df[df.kind == "cpu"].finish.max() - df.arrival.min() if (df.kind == "cpu").any() else np.nan,
    }, df
