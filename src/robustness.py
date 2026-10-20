"""
robustness.py
-------------
1. Sensitivity of the change-detection comparison to the detector settings
   (training length, smoothing window, threshold) on both live platforms.
2. Holm-Bonferroni correction of the burst-prediction comparisons against the textbook EWMA:
   per process, a one-sided p-value from the block bootstrap of the MAE ratio, corrected over
   all method x process tests within each job type.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import change_detection as cd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "robustness")


def sensitivity():
    plat = pickle.load(open(os.path.join(ROOT, "outputs", "change", "runs.pkl"), "rb"))
    rows = []
    for train in (5.0, 8.0, 12.0):
        cd.TRAIN = train
        for smooth in (0.2, 0.5, 1.0):
            for thr in (2.5, 3.0, 4.0):
                for pl, runs in plat.items():
                    if pl == "vm_trace":
                        continue
                    for r in runs:
                        n_s = max(1, int(round(smooth / r["dt"])))
                        hold = max(1, int(round(0.2 / r["dt"])))
                        sc, _ = cd.scores_from_state(r["S"], r["t"], n_s)
                        ev = cd.evaluate_run(sc, r["t"], r["change"], thr, hold, r["until"])
                        for det, e in ev.items():
                            rows.append({"platform": pl, "train_s": train, "smooth_s": smooth, "threshold": thr,
                                         "detector": det, "change": r["meta"]["kind"] != "none", **e})
    cd.TRAIN = 8.0
    df = pd.DataFrame(rows)
    g = df.groupby(["platform", "train_s", "smooth_s", "threshold", "detector"]).apply(
        lambda s: pd.Series({"detection_rate": s[s.change].detected.mean(), "fa_runs": s.false_alarm.mean()}),
        include_groups=False).reset_index()
    g.to_csv(os.path.join(OUT, "detection_sensitivity.csv"), index=False)
    # how often does Koopman beat the distance-from-normal detector (higher detection AND no more false alarms)?
    piv = g.pivot_table(index=["platform", "train_s", "smooth_s", "threshold"], columns="detector",
                        values=["detection_rate", "fa_runs"])
    k_better = ((piv["detection_rate"]["koopman"] >= piv["detection_rate"]["novelty"]) &
                (piv["fa_runs"]["koopman"] <= piv["fa_runs"]["novelty"]) &
                ((piv["detection_rate"]["koopman"] > piv["detection_rate"]["novelty"]) |
                 (piv["fa_runs"]["koopman"] < piv["fa_runs"]["novelty"])))
    n_better = ((piv["detection_rate"]["novelty"] >= piv["detection_rate"]["koopman"]) &
                (piv["fa_runs"]["novelty"] <= piv["fa_runs"]["koopman"]) &
                ((piv["detection_rate"]["novelty"] > piv["detection_rate"]["koopman"]) |
                 (piv["fa_runs"]["novelty"] < piv["fa_runs"]["koopman"])))
    print(f"Settings tried: {len(piv)} (3 training lengths x 3 smoothing windows x 3 thresholds x 2 platforms)")
    print(f"Koopman strictly better than distance-from-normal: {int(k_better.sum())}; "
          f"distance-from-normal strictly better: {int(n_better.sum())}; neither: {int(len(piv) - k_better.sum() - n_better.sum())}")
    return g


def holm():
    sys.path.insert(0, os.path.dirname(__file__))
    from evaluate import mae  # noqa: F401,E402
    p = pd.read_csv(os.path.join(ROOT, "outputs", "multicore", "prediction.csv"))
    p = p[p.config != "lab_trace3"]
    # recompute one-sided bootstrap p-values from the stored predictions is not possible; approximate
    # p from the 95% interval: an interval wholly below 1 means p < 0.025 one-sided. For Holm we need
    # exact p-values, so we redo the bootstrap here.
    import pickle as pk
    from predictors import run_all_methods  # noqa: E402
    from sched_events import cpu_bursts  # noqa: E402
    rows = []
    tl_dir = os.path.join(ROOT, "outputs", "multicore", "timelines")
    rng = np.random.default_rng(0)
    for f in sorted(os.listdir(tl_dir)):
        if f.startswith("lab"):
            continue
        tl = pk.load(open(os.path.join(tl_dir, f), "rb"))
        b = cpu_bursts(tl["run"], tl["wait"])
        for pid, g in b.groupby("pid"):
            s = g.sort_values("arrival").demand.values
            if len(s) < 60:
                continue
            start = int(round(len(s) * 0.8))
            preds, _ = run_all_methods(s, start)
            y = s[start:]
            eb = np.abs(y - preds["EWMA a=0.5 (textbook SJF)"])
            n, block = len(y), 10
            idx = [(rng.integers(0, n - block + 1, int(np.ceil(n / block)))[:, None] + np.arange(block)).ravel()[:n]
                   for _ in range(2000)]
            for m in ["EDMD one-step", "Poly EDMD one-step", "Log EDMD one-step", "Gradient boosting", "EWMA tuned"]:
                em = np.abs(y - preds[m])
                ratios = np.array([em[i].sum() / eb[i].sum() for i in idx])
                rows.append({"capture": f[:-4], "pid": pid, "kind": g.kind.iloc[0], "method": m,
                             "ratio": em.sum() / eb.sum(), "p_better": (np.sum(ratios >= 1) + 1) / (len(ratios) + 1)})
    df = pd.DataFrame(rows)
    out = []
    for kind, s in df.groupby("kind"):
        s = s.sort_values("p_better").reset_index(drop=True)
        m = len(s)
        adj = np.maximum.accumulate([(m - i) * pv for i, pv in enumerate(s.p_better)])
        s["p_holm"] = np.minimum(adj, 1.0)
        out.append(s)
    df = pd.concat(out)
    df.to_csv(os.path.join(OUT, "prediction_holm.csv"), index=False)
    t = df.groupby(["kind", "method"]).agg(processes=("pid", "size"), median_ratio=("ratio", "median"),
                                           better_unadjusted=("p_better", lambda x: int((x < 0.025).sum())),
                                           better_after_holm=("p_holm", lambda x: int((x < 0.05).sum())))
    print(t.to_string())
    return df


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    sensitivity()
    holm()
