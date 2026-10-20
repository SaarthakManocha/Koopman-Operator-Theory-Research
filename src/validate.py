"""
validate.py
-----------
Checks that the trace parser reproduces the kernel's own accounting.

For every workload process, the kernel keeps counters that have nothing to do
with tracing (saved by the workload just before it exits):
    run_ns       total time on a CPU
    wait_ns      total time runnable but waiting in a run queue
    timeslices   number of times it was put on a CPU
    voluntary / involuntary context switches
We rebuild the same quantities from the trace and compare.

The ground truth is read a few hundred microseconds before the process exits,
so the trace always contains a little more (the final exit burst). Agreement is
reported as relative error; anything above 1% for run/wait time is flagged.

    python src/validate.py <capture_dir> [<capture_dir> ...]
"""
import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sched_events import build_timeline, load_pids  # noqa: E402


def read_gt(d):
    rows = []
    for f in glob.glob(os.path.join(d, "ground_truth", "gt_*.txt")):
        kv = dict(line.split(None, 1) for line in open(f) if line.strip())
        rows.append({k: (v.strip() if k == "kind" else int(v)) for k, v in kv.items()})
    return pd.DataFrame(rows).set_index("pid")


def validate_capture(d):
    pids = load_pids(os.path.join(d, "pids.txt"))
    tl = build_timeline(os.path.join(d, "trace.txt.gz"), pids)
    gt = read_gt(d)
    run, wait = tl["run"], tl["wait"]
    rows = []
    for pid, kind in pids.items():
        r = run[run.pid == pid]
        wt = wait[wait.pid == pid]
        g = gt.loc[pid]
        rows.append({
            "capture": os.path.basename(d), "pid": pid, "kind": kind,
            "run_s_trace": r.dur.sum(), "run_s_kernel": g.run_ns / 1e9,
            "wait_s_trace": wt.dur.sum(), "wait_s_kernel": g.wait_ns / 1e9,
            "slices_trace": len(r), "slices_kernel": g.timeslices,
            "invol_trace": int((r.end_state == "R").sum()), "invol_kernel": g.involuntary,
            "vol_trace": int(r.end_state.isin(["S", "D"]).sum()), "vol_kernel": g.voluntary,
            "orphan_switch_outs": tl["orphans"].get(pid, 0),
        })
    df = pd.DataFrame(rows)
    df["run_err_%"] = 100 * (df.run_s_trace - df.run_s_kernel) / df.run_s_kernel
    df["wait_err_%"] = 100 * (df.wait_s_trace - df.wait_s_kernel) / df.wait_s_kernel.where(df.wait_s_kernel > 0)
    meta = json.load(open(os.path.join(d, "meta.json")))
    return df, meta


def main():
    dirs = sys.argv[1:]
    all_rows = []
    for d in dirs:
        df, meta = validate_capture(d)
        all_rows.append(df)
        print(f"\n{os.path.basename(d)}  lost events: {meta['lost_events']}")
        cols = ["pid", "kind", "run_s_trace", "run_s_kernel", "run_err_%", "wait_s_trace", "wait_s_kernel",
                "wait_err_%", "slices_trace", "slices_kernel", "invol_trace", "invol_kernel", "vol_trace", "vol_kernel", "orphan_switch_outs"]
        print(df[cols].round(4).to_string(index=False))
    out = pd.concat(all_rows)
    return out


if __name__ == "__main__":
    main()
