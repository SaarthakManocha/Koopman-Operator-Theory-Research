"""
real_programs_analysis.py - validation, burst prediction and spectral fingerprints
for the real-program captures (kit/real_programs.py).

    python src/real_programs_analysis.py   -> outputs/real_programs/

1. Validation: every tracked thread, trace vs /proc/<tid>/schedstat.
2. Burst prediction: same methods and split (80/20) as the main study, for every
   thread with >= 60 bursts; ratio to the textbook EWMA with block-bootstrap
   intervals; sign test per program.
3. Spectral fingerprint: cut each thread's burst series into windows of 150
   bursts; fit log-EDMD (3 lags) per window; features = the operator's
   eigenvalues (moduli sorted, real parts, largest |imaginary part|). Classify
   which program a window came from, with leave-one-capture-out cross-validation
   (train on two captures, test on the third), against features made of simple
   burst statistics (log mean, log spread, lag-1 autocorrelation) and both.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edmd import EDMD  # noqa: E402
from evaluate import block_bootstrap_ratio, mae  # noqa: E402
from predictors import METHOD_ORDER, run_all_methods  # noqa: E402
from sched_events import build_timeline, cpu_bursts, load_pids  # noqa: E402
from validate import validate_capture  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "real_programs")
OUT = os.path.join(ROOT, "outputs", "real_programs")
WIN = 150


def spectral_features(s):
    m = EDMD(delay=3, kind="linear", log=True).fit(s)
    ev = m.eigenvalues()
    mod = np.sort(np.abs(ev))[::-1]
    re = np.sort(ev.real)[::-1]
    return list(mod) + list(re) + [np.abs(ev.imag).max()]


def stat_features(s):
    l = np.log(np.maximum(s, 1e-7))
    return [l.mean(), l.std(), np.corrcoef(l[:-1], l[1:])[0, 1]]


def main():
    os.makedirs(OUT, exist_ok=True)
    caps = sorted(d for d in glob.glob(os.path.join(DATA, "real_r*")) if os.path.isdir(d))
    # 1 validation
    vals = []
    for d in caps:
        df, meta = validate_capture(d)
        df["lost_events"] = meta["lost_events"]
        vals.append(df)
    val = pd.concat(vals)
    val.to_csv(os.path.join(OUT, "validation.csv"), index=False)
    # 2 prediction + 3 windows
    rows, wins = [], []
    for d in caps:
        pids = load_pids(os.path.join(d, "pids.txt"))
        tl = build_timeline(os.path.join(d, "trace.txt.gz"), pids)
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            kind = pids[pid]
            for k in range(0, len(s) - WIN + 1, WIN):
                w = s[k:k + WIN]
                try:
                    wins.append({"capture": os.path.basename(d), "pid": pid, "kind": kind,
                                 **{f"spec{i}": v for i, v in enumerate(spectral_features(w))},
                                 **{f"stat{i}": v for i, v in enumerate(stat_features(w))}})
                except np.linalg.LinAlgError:
                    pass
            if len(s) < 60:
                continue
            start = int(round(len(s) * 0.8))
            preds, info = run_all_methods(s, start)
            y = s[start:]
            for meth in METHOD_ORDER:
                r, lo, hi = block_bootstrap_ratio(y, preds[meth], preds["EWMA a=0.5 (textbook SJF)"])
                rows.append({"capture": os.path.basename(d), "pid": pid, "kind": kind, "n_bursts": len(s),
                             "median_burst_ms": float(np.median(s) * 1e3), "method": meth,
                             "MAE_ms": mae(y, preds[meth]) * 1e3, "vs_ewma": r, "vs_ewma_lo": lo, "vs_ewma_hi": hi})
    pr = pd.DataFrame(rows)
    pr.to_csv(os.path.join(OUT, "prediction.csv"), index=False)
    W = pd.DataFrame(wins).replace([np.inf, -np.inf], np.nan).dropna()
    W.to_csv(os.path.join(OUT, "fingerprint_windows.csv"), index=False)

    lines = []
    lines.append(f"VALIDATION ({len(val)} threads, {len(caps)} captures, lost events {val.lost_events.unique().tolist()})")
    lines.append(f"  CPU time  |err| median {val['run_err_%'].abs().median():.2f}%  max {val['run_err_%'].abs().max():.2f}%")
    lines.append(f"  wait time |err| median {val['wait_err_%'].abs().median():.2f}%  max {val['wait_err_%'].abs().max():.2f}%")
    lines.append(f"  orphan switch-outs total {int(val.orphan_switch_outs.sum())}")
    lines.append("\nBURST PREDICTION, ratio of MAE to textbook EWMA (<1 = better), threads with >= 60 bursts")
    for meth in ["Log EDMD one-step", "EDMD one-step", "Poly EDMD one-step", "EWMA tuned", "Gradient boosting", "Last value"]:
        g = pr[pr.method == meth]
        for kind, gk in g.groupby("kind"):
            n_better = int((gk.vs_ewma < 1).sum()); n = len(gk)
            p = stats.binomtest(n_better, n, 0.5).pvalue if n else np.nan
            lines.append(f"  {meth:20s} {kind:7s} n={n:2d} median ratio {gk.vs_ewma.median():.3f}  "
                         f"better in {n_better}/{n}  (sign test p={p:.3g})  CI-excludes-1-and-better: {int((gk.vs_ewma_hi < 1).sum())}")
        n_better = int((g.vs_ewma < 1).sum())
        lines.append(f"  {meth:20s} ALL     n={len(g):2d} median ratio {g.vs_ewma.median():.3f}  better in {n_better}/{len(g)} "
                     f"(p={stats.binomtest(n_better, len(g), 0.5).pvalue:.3g})")
    # 3 fingerprint
    lines.append(f"\nSPECTRAL FINGERPRINT: {len(W)} windows of {WIN} bursts, classes {W.kind.value_counts().to_dict()}")
    spec = [c for c in W if c.startswith("spec")]; st = [c for c in W if c.startswith("stat")]
    res = {}
    for name, cols in (("eigenvalues only", spec), ("burst statistics only", st), ("both", spec + st)):
        yt, yp = [], []
        for cap in W.capture.unique():
            tr, te = W[W.capture != cap], W[W.capture == cap]
            clf = RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced").fit(tr[cols], tr.kind)
            yt += list(te.kind); yp += list(clf.predict(te[cols]))
        res[name] = (balanced_accuracy_score(yt, yp), yt, yp)
        lines.append(f"  {name:22s} balanced accuracy {res[name][0]:.3f}  (chance {1 / W.kind.nunique():.3f})")
    labels = sorted(W.kind.unique())
    cm = confusion_matrix(res["eigenvalues only"][1], res["eigenvalues only"][2], labels=labels)
    lines.append("  confusion (eigenvalues only), rows = true: " + str(labels))
    for l, row in zip(labels, cm):
        lines.append(f"    {l:7s} {row.tolist()}")
    med = W.groupby("kind")[["spec0", "spec6"]].median()
    lines.append("  median spectral radius |lambda1| and largest |Im lambda| per program:")
    for k, r in med.iterrows():
        lines.append(f"    {k:7s} {r.spec0:.3f}  {r.spec6:.3f}")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "summary.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
