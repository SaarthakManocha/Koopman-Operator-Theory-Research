"""
baselines.py - is the Koopman predictor more than an autoregressive model?

    python src/baselines.py   -> outputs/baselines/

For every process/thread with >= 60 bursts (contended VM captures, lab Trace 3,
real programs): fit on the first 80%, predict the last 20% one step ahead,
using only earlier bursts. Methods:

  EWMA          textbook SJF estimator, alpha = 0.5 (the reference)
  log-EDMD      Koopman/Hankel-DMD, 3 lags, linear dictionary, log scale (as in the paper)
  log-AR(3)     ordinary least squares with intercept on 3 log lags
  log-AR(p*)    order 1..10 chosen by AIC on training data
  ARIMA         (p,d,q) in {0..2}x{0,1}x{0,1} chosen by AIC on the last 1,500 training log bursts (statsmodels)
  MLP           sklearn MLP, 2x32 units, on 6 log lags
  poly-EDMD     Koopman with squares and products of the 3 log lags
  RFF-EDMD      Koopman with 3 log lags + 64 random Fourier features (approximate kernel EDMD)
  gradient boosting on 3 raw lags (as in the paper)

Reports MAE ratios to EWMA and to log-AR(p*), sign tests with Holm correction,
and the numerical equivalence of log-EDMD and log-AR(3) (Hankel DMD = AR).
"""
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edmd import EDMD, delay_embed  # noqa: E402
from predictors import ewma_path  # noqa: E402
from sched_events import build_timeline, cpu_bursts, load_pids  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "baselines")
EPS = 1e-7


def datasets():
    out = []
    tl_dir = os.path.join(ROOT, "outputs", "multicore", "timelines")
    for f in sorted(os.listdir(tl_dir)):
        tl = pickle.load(open(os.path.join(tl_dir, f), "rb"))
        b = cpu_bursts(tl["run"], tl["wait"])
        grp = "lab Trace 3" if f.startswith("lab") else "VM synthetic"
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            if len(s) >= 60:
                out.append((grp, g.kind.iloc[0], f[:-4], pid, s))
    for d in sorted(glob.glob(os.path.join(ROOT, "data", "real_programs", "real_r*"))):
        pkl = os.path.join(OUT, "tl_" + os.path.basename(d) + ".pkl")
        if os.path.exists(pkl):
            tl = pickle.load(open(pkl, "rb"))
        else:
            pids = load_pids(os.path.join(d, "pids.txt"))
            tl = build_timeline(os.path.join(d, "trace.txt.gz"), pids)
            tl["pids"] = pids
            pickle.dump(tl, open(pkl, "wb"))
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            if len(s) >= 60:
                out.append(("real programs", tl["pids"][pid], os.path.basename(d), pid, s))
    return out


def lag_matrix(l, p, start):
    """rows for targets t = start..n-1 : [l_{t-1}, ..., l_{t-p}]"""
    return np.column_stack([l[start - k - 1: len(l) - k - 1] for k in range(p)])


def ar_fit_predict(l, start, p):
    Xtr = lag_matrix(l[:start], p, p); ytr = l[p:start]
    A = np.column_stack([np.ones(len(Xtr)), Xtr])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    resid = ytr - A @ coef
    aic = len(ytr) * np.log(np.mean(resid ** 2) + 1e-300) + 2 * (p + 1)
    Xte = lag_matrix(l, p, start)
    return np.column_stack([np.ones(len(Xte)), Xte]) @ coef, aic


def arima_predict(l, start):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    fit_on = l[max(0, start - 1500):start]          # last 1,500 training bursts keep ARIMA fitting tractable
    for p in range(3):
        for d in (0, 1):
            for q in range(2):
                if p == q == 0 and d == 0:
                    continue
                try:
                    m = ARIMA(fit_on, order=(p, d, q), trend="c" if d == 0 else "n").fit()
                    if best is None or m.aic < best[0]:
                        best = (m.aic, (p, d, q), m)
                except Exception:
                    pass
    full = best[2].apply(l)                       # same parameters, filter over the full series
    pred = full.get_prediction(start=start, end=len(l) - 1).predicted_mean
    return np.asarray(pred), best[1]


def rff_edmd_predict(l, start, D=64, seed=0):
    mu, sd = l[:start].mean(), l[:start].std() + 1e-12
    z = (l - mu) / sd
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (3, D)); bph = rng.uniform(0, 2 * np.pi, D)

    def psi(X):
        return np.hstack([X, np.sqrt(2 / D) * np.cos(X @ W + bph)])
    X = delay_embed(z[:start], 3)                 # rows [z_t, z_{t-1}, z_{t-2}]
    P, Pn = psi(X[:-1]), psi(X[1:])
    G = P.T @ P + 1e-3 * np.eye(P.shape[1])
    K = np.linalg.solve(G, P.T @ Pn)
    preds = []
    for t in range(start, len(z)):
        x = z[t - 3:t][::-1][None, :]
        preds.append((psi(x) @ K)[0, 0])
    return np.array(preds) * sd + mu


def run_one(s):
    s = np.asarray(s, float)
    n = len(s); start = int(round(n * 0.8)); y = s[start:]
    l = np.log(np.maximum(s, EPS))
    P = {}
    P["EWMA"] = ewma_path(s, 0.5)[start:]
    P["log-EDMD"] = EDMD(3, "linear", log=True).fit(s[:start]).predict_one_step(s, start)
    p3, _ = ar_fit_predict(l, start, 3); P["log-AR(3)"] = np.exp(p3)
    best = min((ar_fit_predict(l, start, p) + (p,) for p in range(1, 11) if start > 4 * p), key=lambda r: r[1])
    P["log-AR(p*)"] = np.exp(best[0]); order = best[2]
    try:
        pa, aorder = arima_predict(l, start); P["ARIMA"] = np.exp(pa)
    except Exception:
        P["ARIMA"] = np.exp(best[0]); aorder = None
    X6 = lag_matrix(l[:start], 6, 6)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=2000, random_state=0, early_stopping=True).fit(X6, l[6:start])
    P["MLP"] = np.exp(mlp.predict(lag_matrix(l, 6, start)))
    P["poly-EDMD"] = EDMD(3, "poly2", log=True).fit(s[:start]).predict_one_step(s, start)
    P["RFF-EDMD"] = np.exp(rff_edmd_predict(l, start))
    Xr = delay_embed(s[:start], 3)
    gb = GradientBoostingRegressor(random_state=0).fit(Xr[:-1], s[3:start])
    P["gradient boosting"] = gb.predict(lag_matrix(s, 3, start)[:, :])
    mae = {k: float(np.mean(np.abs(v - y))) for k, v in P.items()}
    eq = float(np.max(np.abs(np.log(P["log-EDMD"]) - p3)))
    return mae, order, aorder, eq


METHODS = ["EWMA", "log-EDMD", "log-AR(3)", "log-AR(p*)", "ARIMA", "MLP", "poly-EDMD", "RFF-EDMD", "gradient boosting"]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for grp, kind, cap, pid, s in datasets():
        mae, order, aorder, eq = run_one(s)
        rows.append({"group": grp, "kind": kind, "capture": cap, "pid": pid, "n": len(s), "ar_order": order,
                     "arima_order": str(aorder), "edmd_vs_ar3_max_logdiff": eq, **{f"MAE_{k}": v for k, v in mae.items()}})
        print(f"{grp:14s} {kind:7s} {cap:12s} {pid:7d} n={len(s):5d} p*={order}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "baselines.csv"), index=False)
    lines = [f"{len(df)} series. Median ratio of MAE (method / reference) and wins; sign test, Holm over methods within each block."]
    for ref in ("EWMA", "log-AR(p*)"):
        for grp, g in [("ALL", df)] + list(df.groupby("group")):
            lines.append(f"\n[{grp}] reference = {ref}  (n={len(g)})")
            ps, recs = [], []
            for m in METHODS:
                if m == ref:
                    continue
                r = g[f"MAE_{m}"] / g[f"MAE_{ref}"]
                w = int((r < 1).sum()); l_ = int((r > 1).sum())
                p = stats.binomtest(w, w + l_, 0.5).pvalue if w + l_ else 1.0
                ps.append(p); recs.append((m, r.median(), w, len(g)))
            order = np.argsort(ps); adj = np.empty(len(ps)); run = 0
            for rank, i in enumerate(order):
                run = max(run, min(1, ps[i] * (len(ps) - rank))); adj[i] = run
            for (m, med, w, n), pa in zip(recs, adj):
                lines.append(f"  {m:18s} median ratio {med:.3f}   better in {w:3d}/{n}   p_Holm={pa:.2g}")
    lines.append(f"\nlog-EDMD vs log-AR(3): max |log prediction difference| median {df.edmd_vs_ar3_max_logdiff.median():.2e}, "
                 f"max {df.edmd_vs_ar3_max_logdiff.max():.2e}")
    lines.append(f"AIC-chosen AR order: {df.ar_order.value_counts().sort_index().to_dict()}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
