"""
trace_stats.py
--------------
Descriptive statistics that explain WHY bursts end.

  * summary per process: number of bursts, median / mean / max burst,
    total CPU time, number of distinct CPUs used, how bursts ended (R/S/D)
  * preemptors: for bursts that ended in state R (preempted), which task
    took the CPU next (kworker, migration thread, another python3, ...)

    python src/trace_stats.py data/raw/trace3.txt data/processed/bursts_trace3.csv evidence/pids.txt
"""

import collections
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from parse_trace import SWITCH_RE, load_pids, open_text  # noqa: E402


def task_family(comm):
    """kworker/12:1 -> kworker ; migration/3 -> migration ; python3 -> python3"""
    return re.sub(r"[/\-].*$", "", comm)


def preemptors(trace_path, kinds):
    counts = collections.Counter()
    with open_text(trace_path) as f:
        for line in f:
            if "sched_switch" not in line:
                continue
            m = SWITCH_RE.search(line)
            if m and int(m["prev_pid"]) in kinds and m["prev_state"] == "R":
                counts[(int(m["prev_pid"]), task_family(m["next_comm"]))] += 1
    rows = [{"pid": p, "kind": kinds[p], "preempted_by": fam, "count": c} for (p, fam), c in counts.items()]
    return pd.DataFrame(rows).sort_values(["pid", "count"], ascending=[True, False])


def summary(bursts):
    rows = []
    for pid, g in bursts.groupby("pid"):
        st = g.end_state.value_counts()
        rows.append({
            "pid": pid, "kind": g.kind.iloc[0], "bursts": len(g),
            "first_s": g.start.min(), "last_s": g.end.max(),
            "median_burst_ms": g.burst_s.median() * 1e3, "mean_burst_ms": g.burst_s.mean() * 1e3,
            "max_burst_ms": g.burst_s.max() * 1e3, "total_cpu_s": g.burst_s.sum(),
            "cpus_used": g.cpu.nunique(),
            "same_cpu_resume": float((g.cpu.values[1:] == g.cpu.values[:-1]).mean()),
            "ended_R": int(st.get("R", 0)), "ended_S": int(st.get("S", 0)),
            "ended_D": int(st.get("D", 0)), "ended_Z": int(st.get("Z", 0)),
        })
    return pd.DataFrame(rows)


def main():
    trace, bursts_csv, pids = sys.argv[1:4]
    kinds = load_pids(pids)
    b = pd.read_csv(bursts_csv).sort_values(["pid", "start"])
    os.makedirs("outputs", exist_ok=True)
    s = summary(b)
    s.to_csv("outputs/process_summary.csv", index=False)
    p = preemptors(trace, kinds)
    p.to_csv("outputs/preemptors.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    print()
    print(p.to_string(index=False))


if __name__ == "__main__":
    main()
