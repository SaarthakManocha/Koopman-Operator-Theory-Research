"""
predictors.py
-------------
Every method answers the same question: given the bursts before index t,
what is burst t?  Each function returns predictions for indices
start ... len(series)-1, using only data before each index (no peeking).

Baselines
  mean / median   : one constant learned from the training part
  last value      : predict that the next burst equals the previous one
  EWMA            : the textbook SJF estimator (Silberschatz, ch. 5)
                        tau_{n+1} = alpha * t_n + (1 - alpha) * tau_n
Learned models
  EDMD variants   : see edmd.py
  gradient boost  : nonlinear trees on the same last-3-bursts input
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from edmd import EDMD, delay_embed

DELAY = 3


def pred_constant(series, start, value):
    return np.full(len(series) - start, value, dtype=float)


def pred_last(series, start):
    s = np.asarray(series, dtype=float)
    return s[start - 1:-1].copy()


def ewma_path(series, alpha):
    """tau[i] = prediction for series[i] made before seeing it."""
    s = np.asarray(series, dtype=float)
    tau = np.empty(len(s))
    tau[0] = s[0]
    for i in range(1, len(s)):
        tau[i] = alpha * s[i - 1] + (1 - alpha) * tau[i - 1]
    return tau


def pred_ewma(series, start, alpha):
    return ewma_path(series, alpha)[start:]


def tune_alpha(train):
    """Pick alpha on TRAINING data only (smallest one-step absolute error)."""
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    errs = [np.mean(np.abs(ewma_path(train, a)[1:] - np.asarray(train)[1:])) for a in grid]
    return float(grid[int(np.argmin(errs))])


def pred_gbm(series, start):
    s = np.asarray(series, dtype=float)
    X = delay_embed(s[:start], DELAY)
    Xtr, ytr = X[:-1], s[DELAY:start]
    model = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
    Xte = np.array([s[t - DELAY:t][::-1] for t in range(start, len(s))])
    return model.predict(Xte)


def run_all_methods(series, start):
    """
    series : full burst sequence of one process (seconds)
    start  : index of the first test burst (everything before is training)
    returns dict  method_name -> predictions for series[start:]
    and a dict of fitted info (alpha, eigenvalues).
    """
    s = np.asarray(series, dtype=float)
    train = s[:start]
    h = len(s) - start
    alpha = tune_alpha(train)

    lin = EDMD(DELAY, "linear").fit(train)
    preds = {
        "Mean baseline": pred_constant(s, start, train.mean()),
        "Median baseline": pred_constant(s, start, np.median(train)),
        "Last value": pred_last(s, start),
        "EWMA a=0.5 (textbook SJF)": pred_ewma(s, start, 0.5),
        f"EWMA tuned": pred_ewma(s, start, alpha),
        "EDMD one-step": lin.predict_one_step(s, start),
        "EDMD recursive": lin.predict_recursive(train, h),
        "Poly EDMD one-step": EDMD(DELAY, "poly2").fit(train).predict_one_step(s, start),
        "Log EDMD one-step": EDMD(DELAY, "linear", log=True).fit(train).predict_one_step(s, start),
        "Gradient boosting": pred_gbm(s, start),
    }
    info = {"alpha": alpha, "eig": lin.eigenvalues(),
            "eig_log": EDMD(DELAY, "linear", log=True).fit(train).eigenvalues()}
    return preds, info


METHOD_ORDER = [
    "Mean baseline", "Median baseline", "Last value", "EWMA a=0.5 (textbook SJF)", "EWMA tuned",
    "EDMD one-step", "EDMD recursive", "Poly EDMD one-step", "Log EDMD one-step", "Gradient boosting",
]
