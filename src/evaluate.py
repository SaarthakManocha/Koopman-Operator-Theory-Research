"""
evaluate.py
-----------
Error metrics, block-bootstrap confidence intervals, autocorrelation and a
small SJF scheduling simulation.
"""

import numpy as np
from scipy.stats import spearmanr


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def block_bootstrap_ratio(y, p_method, p_base, n_boot=2000, block=10, seed=0):
    """
    95% interval for  MAE(method) / MAE(base)  on the same test bursts.
    Moving-block bootstrap: resample contiguous blocks of `block` bursts, so
    that short-range dependence between neighbouring bursts is preserved.
    A ratio below 1 means the method has lower typical error than the base.
    """
    y, pm, pb = map(np.asarray, (y, p_method, p_base))
    em, eb = np.abs(y - pm), np.abs(y - pb)
    n = len(y)
    block = min(block, n)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        ratios[i] = em[idx].sum() / max(eb[idx].sum(), 1e-15)
    return float(em.sum() / eb.sum()), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def lag_correlations(b, lags=(1, 2, 3)):
    b = np.asarray(b, dtype=float)
    rows = []
    for L in lags:
        rows.append({
            "lag": L,
            "pearson_raw": float(np.corrcoef(b[:-L], b[L:])[0, 1]),
            "spearman_rank": float(spearmanr(b[:-L], b[L:])[0]),
        })
    return rows


def sjf_waiting_time(true_bursts, predicted):
    """
    Non-preemptive SJF on one batch of jobs that all arrive at time 0,
    ordered by PREDICTED length; waiting time is computed with the TRUE
    lengths. Returns the average waiting time.
    """
    order = np.argsort(predicted, kind="stable")
    t, total = 0.0, 0.0
    for j in order:
        total += t
        t += true_bursts[j]
    return total / len(true_bursts)
