"""
conformal_cost.py - (a) prediction intervals for CPU bursts, (b) cost of one prediction.

    python src/conformal_cost.py     -> outputs/extras/conformal.csv, cost.csv, conformal_summary.txt

(a) Split conformal prediction. For every process with >= 60 bursts in the
validated captures (and lab Trace 3): fit on the first 60%, calibrate on the
next 20%, test on the last 20%. The score is the absolute log-ratio
|log(actual / predicted)|; the interval is predicted * exp(+-q), with q the
ceil((n+1)(1-a))/n quantile of calibration scores. Coverage on the test part is
guaranteed >= 1-a on average if bursts are exchangeable; they are a time
series, so we measure it. Done for log-EDMD and the textbook EWMA, a = 0.1.

(b) Wall-clock cost of a single next-burst prediction and of refitting, for
the numpy implementation used here and a plain C implementation (3 lags,
online recursive least squares), averaged over 10^5-10^7 calls.
"""
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edmd import EDMD  # noqa: E402
from predictors import ewma_path  # noqa: E402
from sched_events import cpu_bursts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "extras")
TL = os.path.join(ROOT, "outputs", "multicore", "timelines")
ALPHA = 0.1


def series_all():
    out = []
    for f in sorted(os.listdir(TL)):
        tl = pickle.load(open(os.path.join(TL, f), "rb"))
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            if len(s) >= 60:
                out.append((f[:-4], pid, g.kind.iloc[0], s))
    return out


def conformal():
    rows = []
    for cap, pid, kind, s in series_all():
        n = len(s); a, b = int(n * 0.6), int(n * 0.8)
        m = EDMD(delay=3, kind="linear", log=True).fit(s[:a])
        p_edmd = m.predict_one_step(s, a)            # predictions for indices a..n-1
        p_ewma = ewma_path(s, 0.5)[a:]
        y = s[a:]
        for name, p in (("log-EDMD", p_edmd), ("EWMA", p_ewma)):
            p = np.maximum(p, 1e-7)
            sc = np.abs(np.log(np.maximum(y, 1e-7) / p))
            cal, te = sc[:b - a], sc[b - a:]
            k = int(np.ceil((len(cal) + 1) * (1 - ALPHA)))
            q = np.sort(cal)[min(k, len(cal)) - 1]
            cover = float(np.mean(te <= q))
            rows.append({"capture": cap, "pid": pid, "kind": kind, "method": name, "n_cal": len(cal),
                         "n_test": len(te), "q_log": q, "width_factor": float(np.exp(2 * q)),
                         "coverage": cover,
                         "median_upper_ms": float(np.median(p[b - a:] * np.exp(q)) * 1e3),
                         "median_actual_ms": float(np.median(y[b - a:]) * 1e3)})
    return pd.DataFrame(rows)


C_SRC = r"""
#include <stdio.h>
#include <time.h>
#include <math.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(void){
  /* 3-lag linear operator on log bursts, online recursive least squares (forgetting 0.995) */
  double A[3]={0.5,0.2,0.1}, P[3][3]={{100,0,0},{0,100,0},{0,0,100}}, x[3]={0.1,0.2,0.3}, lam=0.995;
  volatile double sink=0; long N=10000000; double t0=now();
  for(long i=0;i<N;i++){ double y=A[0]*x[0]+A[1]*x[1]+A[2]*x[2]; sink+=y; x[2]=x[1]; x[1]=x[0]; x[0]=y*0.9+1e-3*(i&7);}
  double tp=(now()-t0)/N; N=2000000; t0=now();
  for(long i=0;i<N;i++){
    double z=0.3*(i&3)-0.4, Px[3], den=lam, g[3], e;
    for(int r=0;r<3;r++){Px[r]=P[r][0]*x[0]+P[r][1]*x[1]+P[r][2]*x[2]; den+=x[r]*Px[r];}
    for(int r=0;r<3;r++) g[r]=Px[r]/den;
    e=z-(A[0]*x[0]+A[1]*x[1]+A[2]*x[2]);
    for(int r=0;r<3;r++){A[r]+=g[r]*e; for(int c=0;c<3;c++) P[r][c]=(P[r][c]-g[r]*Px[c])/lam;}
    x[2]=x[1]; x[1]=x[0]; x[0]=z; sink+=A[0];
  }
  double tu=(now()-t0)/N;
  printf("%.3f %.3f\n", tp*1e9, tu*1e9); return (int)(sink*0);
}
"""


def cost():
    rows = []
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(-7, 0.5, 300))
    m = EDMD(delay=3, kind="linear", log=True).fit(s)
    x = s[-4:]
    n = 2000
    t = time.perf_counter()
    for _ in range(n):
        m.predict_one_step(x, 3)
    rows.append({"impl": "Python/numpy (this study)", "op": "predict next burst", "ns": (time.perf_counter() - t) / n * 1e9})
    t = time.perf_counter()
    for _ in range(200):
        EDMD(delay=3, kind="linear", log=True).fit(s)
    rows.append({"impl": "Python/numpy (this study)", "op": "refit on 300 bursts", "ns": (time.perf_counter() - t) / 200 * 1e9})
    a = A = m.A; z = np.array([0.1, 0.2, 0.3])
    t = time.perf_counter()
    for _ in range(20000):
        z = A @ z
    rows.append({"impl": "Python/numpy (this study)", "op": "3x3 operator step only", "ns": (time.perf_counter() - t) / 20000 * 1e9})
    src = os.path.join(OUT, "koopman_cost.c"); exe = os.path.join(OUT, "koopman_cost")
    open(src, "w").write(C_SRC)
    subprocess.run(["gcc", "-O2", "-o", exe, src, "-lm"], check=True)
    tp, tu = map(float, subprocess.run([exe], capture_output=True, text=True).stdout.split())
    rows.append({"impl": "C, -O2", "op": "predict next burst", "ns": tp})
    rows.append({"impl": "C, -O2", "op": "online RLS update", "ns": tu})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    c = conformal(); c.to_csv(os.path.join(OUT, "conformal.csv"), index=False)
    k = cost(); k.to_csv(os.path.join(OUT, "cost.csv"), index=False)
    lines = [f"Split conformal, target coverage {1 - ALPHA:.0%}, {c.pid.nunique()} processes "
             f"({c[c.method == 'EWMA'].shape[0]} series)"]
    for (kind, meth), g in c.groupby(["kind", "method"]):
        lines.append(f"  {kind:7s} {meth:9s} n={len(g):3d} coverage median {g.coverage.median():.3f} "
                     f"(min {g.coverage.min():.2f}, share >= 0.85: {np.mean(g.coverage >= 0.85):.2f}); "
                     f"interval width factor median {g.width_factor.median():.2f}")
    piv = c.pivot_table(index=["capture", "pid"], columns="method", values="width_factor")
    lines.append(f"  log-EDMD interval narrower than EWMA in {(piv['log-EDMD'] < piv['EWMA']).sum()}/{len(piv)} series; "
                 f"median width ratio {np.median(piv['log-EDMD'] / piv['EWMA']):.3f}")
    lines.append("\nCost per call:")
    for _, r in k.iterrows():
        lines.append(f"  {r.impl:28s} {r.op:24s} {r.ns:12.1f} ns")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "conformal_summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
