"""analyse_real_closed.py - latency of real programs under each slice policy (real_closed.py runs).

    python3 closedloop/analyse_real_closed.py data/closedloop/real_1cpu
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def load(d):
    m = json.load(open(f"{d}/meta.json"))
    rows = {}
    for prog, f in (("redis", "redis.txt"), ("node", "http.txt"), ("sqlite", "sqlite.txt")):
        x = pd.read_csv(f"{d}/{f}", sep=" ", names=["kind", "lat"])
        x["lat"] *= 1e3
        rows[f"{prog}_mean"] = x.lat.mean(); rows[f"{prog}_p99"] = x.lat.quantile(.99); rows[f"{prog}_n"] = len(x)
        for k, g in x.groupby("kind"):
            rows[f"{prog}:{k}_mean"] = g.lat.mean()
    rows["hog_cpu_s"] = sum(int(open(f).read().split()[1]) for f in glob.glob(f"{d}/hog*.txt")) / 1e3
    if os.path.exists(f"{d}/agent.txt"):
        kv = dict(l.split() for l in open(f"{d}/agent.txt")); rows["agent_cpu_s"] = float(kv["agent_cpu_s"])
    return {"policy": m["policy"], "rep": m["rep"], **rows}


def main():
    root = sys.argv[1]
    df = pd.DataFrame([load(d) for d in sorted(glob.glob(f"{root}/r*_*"))])
    df.to_csv(f"{root}/runs.csv", index=False)
    cols = [c for c in df if c.endswith("_mean") or c.endswith("_p99")] + ["hog_cpu_s"]
    g = df.groupby("policy")[cols].mean()
    print(g.round(3).T.to_string())
    out = []
    for metric in ["redis_mean", "redis_p99", "node_mean", "node_p99", "sqlite_mean", "sqlite_p99",
                   "node:light_mean", "node:heavy_mean", "sqlite:point_mean", "sqlite:scan_mean", "hog_cpu_s"]:
        P = df.pivot(index="rep", columns="policy", values=metric)
        for a, b in [(p, "stock") for p in P if p != "stock"] + [("bpf_koopman_1.5", "bpf_ewma_1.5"), ("bpf_koopman_2", "bpf_ewma_2")]:
            if a in P and b in P:
                dd = (P[a] - P[b]).dropna()
                out.append({"metric": metric, "a": a, "b": b, "change_%": 100 * (P[a].mean() / P[b].mean() - 1),
                            "a_lower_in": int((dd < 0).sum()), "of": len(dd),
                            "wilcoxon_p": stats.wilcoxon(dd).pvalue if len(dd) >= 5 and np.any(dd != 0) else np.nan})
    o = pd.DataFrame(out); o.to_csv(f"{root}/paired.csv", index=False)
    print(o.round(4).to_string())


if __name__ == "__main__":
    main()
