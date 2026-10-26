import sys, glob, os, json
sys.path.insert(0, "/home/claude/koopman-cpu-scheduling/src")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sched_events import build_timeline, load_pids, cpu_bursts
def one(d):
    pids = load_pids(d + "/pids.txt"); tl = build_timeline(d + "/trace.txt.gz", pids)
    run = tl["run"]; b = cpu_bursts(run, tl["wait"])
    t0 = run.start.min()
    # cpu of each burst = cpu of its last piece
    out = []
    for pid, kind in pids.items():
        if kind != "io": continue
        r = run[run.pid == pid].sort_values("start")
        g = b[b.pid == pid].sort_values("arrival")
        cpus = np.interp(g.first_run.values, r.start.values, r.cpu.values) if len(r) else []
        for (_, x), c in zip(g.iterrows(), cpus):
            out.append(dict(cap=os.path.basename(d), pid=pid, t=x.arrival - t0, demand_us=x.demand * 1e6, cpu=int(round(c))))
    return out
if __name__ == "__main__":
    ds = sorted(glob.glob("/home/claude/lab1/captures/freq_*"))
    with Pool(6) as p: res = p.map(one, ds)
    df = pd.DataFrame([r for x in res for r in x]); df.to_csv("freq_io_bursts.csv", index=False)
    print(df.columns.tolist(), len(df))
