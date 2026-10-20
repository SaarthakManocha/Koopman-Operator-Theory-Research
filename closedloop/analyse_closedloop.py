"""
analyse_closedloop.py - summarise closed-loop runs.

    python3 closedloop/analyse_closedloop.py data/closedloop/c1 [data/closedloop/c2 ...]

Per run: request wake-up latency (wake - deadline), response time (finish -
deadline), split by burst size; hog CPU obtained; context switches; agent cost;
predictor accuracy (MAE of the burst the agent predicted vs the burst that came).
Validation: each request task's CPU time from its own log vs /proc/<pid>/schedstat.
Paired comparison: per repetition, policy vs stock and koopman vs ewma.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def load_run(d):
    m = json.load(open(f"{d}/meta.json"))
    pids = [l.split() for l in open(f"{d}/pids.txt")]
    reqs, val = [], []
    req_pids = [int(p) for k, p in pids if k == "req"]
    for i, pid in enumerate(req_pids):
        x = pd.read_csv(f"{d}/req{i}.txt", sep=" ")
        x["task"] = i
        reqs.append(x)
        g = m["gt"].get(str(pid))
        if g:
            val.append((x.work_ms.sum(), g["run_ns"] / 1e6))
    x = pd.concat(reqs)
    x["wake_ms"] = (x.wake - x.deadline) * 1e3
    x["resp_ms"] = (x.finish - x.deadline) * 1e3
    hog = sum(int(open(f).read().split()[1]) for f in glob.glob(f"{d}/hog*.txt"))
    ag = json.load(open(f"{d}/agent.json")) if os.path.exists(f"{d}/agent.json") else {}
    acc = []
    for p, hb in ag.get("bursts", {}).items():
        pr = ag["preds"][p]
        hb, pr = np.array(hb), np.array(pr)
        if len(hb) > 40:
            acc.append(np.mean(np.abs(pr[30:-1] - hb[31:])) * 1e3)
    short = x.work_ms < 1.0
    return {"policy": m["policy"], "rep": m["rep"], "n_req": len(x),
            "wake_p50": x.wake_ms.median(), "wake_p99": x.wake_ms.quantile(.99), "wake_mean": x.wake_ms.mean(),
            "resp_short_p50": x.resp_ms[short].median(), "resp_short_p99": x.resp_ms[short].quantile(.99),
            "resp_short_mean": x.resp_ms[short].mean(),
            "resp_long_mean": x.resp_ms[~short].mean(), "resp_mean": x.resp_ms.mean(),
            "hog_cpu_s": hog / 1e3, "ctxt_per_s": m["ctxt"] / m["secs"],
            "agent_cpu_pct": 100 * ag.get("agent_cpu_s", 0) / m["secs"],
            "pred_mae_ms": np.mean(acc) if acc else np.nan,
            "val_cpu_err_pct": float(np.median([abs(a - b) / b * 100 for a, b in val])) if val else np.nan}


def main():
    for root in sys.argv[1:]:
        df = pd.DataFrame([load_run(d) for d in sorted(glob.glob(f"{root}/r*_*"))])
        df.to_csv(f"{root}/runs.csv", index=False)
        cols = ["wake_p50", "wake_p99", "resp_short_p50", "resp_short_p99", "resp_short_mean", "resp_long_mean",
                "resp_mean", "hog_cpu_s", "ctxt_per_s", "agent_cpu_pct", "pred_mae_ms", "val_cpu_err_pct"]
        summ = df.groupby("policy")[cols].mean().round(3)
        print(f"\n=== {root}  ({df.rep.nunique()} repetitions)\n{summ.to_string()}")
        piv = df.pivot(index="rep", columns="policy")
        lines = []
        for metric in ("wake_p50", "wake_p99", "resp_short_mean", "resp_short_p99", "resp_mean", "hog_cpu_s"):
            for a, b in [("koopman", "stock"), ("ewma", "stock"), ("koopman", "ewma"), ("static", "stock"),
                         ("koopman", "static"), ("oracle", "koopman"), ("fixed", "stock"),
                         ("koopman_q", "stock"), ("ewma_q", "stock"), ("koopman_q", "ewma_q"),
                         ("koopman_q", "koopman"), ("oracle2", "stock"), ("oracle2", "koopman_q")]:
                if a in piv[metric] and b in piv[metric]:
                    d = (piv[metric][a] - piv[metric][b]).dropna()
                    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
                    lines.append({"metric": metric, "a": a, "b": b, "mean_a": piv[metric][a].mean(),
                                  "mean_b": piv[metric][b].mean(), "a_lower_in": int((d < 0).sum()), "of": len(d),
                                  "ratio": piv[metric][a].mean() / piv[metric][b].mean(), "wilcoxon_p": p})
        pw = pd.DataFrame(lines)
        pw.to_csv(f"{root}/paired.csv", index=False)
        print(pw.round(4).to_string())


if __name__ == "__main__":
    main()
