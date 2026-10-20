"""
koopman_system.py
-----------------
Koopman analysis of the WHOLE multi-core machine, not one process.

The state of the machine in each time bin (default 10 ms) is a vector:
    busy_c      fraction of the bin that core c was running any task         (one per core)
    queue_c     average number of workload tasks waiting in core c's queue   (one per core)
    share_k     fraction of the machine's capacity used by workload class k  (cpu / io / phased)
This is the same construction as the network-traffic state in the DRISHTI paper:
a vector of measured quantities, standardised on training data, and a single
linear operator K fitted by least squares,  s_{t+1} ~= s_t K  (row convention).

From K we read:
    eigenvalues          how fast patterns in the machine's load decay / oscillate
    modes                which cores and queues move together
    one-step residual    r_t = s_{t+1} - s_t K, the operator's "surprise"; a run of
                         large residuals means the machine left the regime K was fitted on
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ state construction
def occupancy(intervals, keys, key_col, t0, t1, dt):
    """Fraction of each bin covered by intervals, per key. intervals: DataFrame with start, end, key_col."""
    nb = int(np.ceil((t1 - t0) / dt))
    out = np.zeros((nb, len(keys)))
    idx = {k: i for i, k in enumerate(keys)}
    for k, s, e in zip(intervals[key_col].values, intervals.start.values, intervals.end.values):
        if k not in idx:
            continue
        j = idx[k]
        s, e = max(s, t0), min(e, t1)
        if e <= s:
            continue
        b0, b1 = int((s - t0) // dt), int((e - t0) // dt)
        b1 = min(b1, nb - 1)
        if b0 == b1:
            out[b0, j] += e - s
        else:
            out[b0, j] += (t0 + (b0 + 1) * dt) - s
            out[b0 + 1:b1, j] += dt
            out[b1, j] += e - (t0 + b1 * dt)
    return out / dt


def aggregate_state(S):
    """Ablation (added after the per-core results on the 28-core lab trace): five machine-wide
    numbers instead of one per core: total busy cores, total queued tasks, class shares."""
    b = S[[c for c in S.columns if c.startswith("busy_")]].sum(1)
    q = S[[c for c in S.columns if c.startswith("queue_")]].sum(1)
    out = S[[c for c in S.columns if c.startswith("share_")]].copy()
    out.insert(0, "queue_total", q)
    out.insert(0, "busy_total", b)
    return out


def machine_state(tl, dt=0.010, t0=None, t1=None, classes=("cpu", "io", "phased")):
    t0 = tl["t0"] if t0 is None else t0
    t1 = tl["t1"] if t1 is None else t1
    cpus = tl["cpus"]
    busy = occupancy(tl["cpu_busy"], cpus, "cpu", t0, t1, dt)
    queue = occupancy(tl["wait"], cpus, "cpu", t0, t1, dt)
    run = tl["run"]
    share = occupancy(run, list(classes), "kind", t0, t1, dt) / len(cpus)
    cols = [f"busy_{c}" for c in cpus] + [f"queue_{c}" for c in cpus] + [f"share_{k}" for k in classes]
    times = t0 + dt * (np.arange(busy.shape[0]) + 0.5)
    return pd.DataFrame(np.hstack([busy, queue, share]), columns=cols, index=times)


# ------------------------------------------------------------------ operator
class LinearKoopman:
    """K = X^+ Y on standardised state (row convention), exactly as in the DRISHTI paper."""

    def fit(self, S):
        S = np.asarray(S, float)
        self.mu = S.mean(0)
        self.sd = S.std(0)
        self.keep = self.sd > 1e-9                     # drop constant columns (e.g. an unused core)
        Z = (S[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        X, Y = Z[:-1], Z[1:]
        self.K = np.linalg.pinv(X) @ Y
        self.train_resid = Y - X @ self.K
        self.res_scale = np.sqrt((self.train_resid ** 2).sum(1)).std() + 1e-12
        self.res_mean = np.sqrt((self.train_resid ** 2).sum(1)).mean()
        return self

    def z(self, S):
        return (np.asarray(S, float)[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]

    def eig(self):
        lam, V = np.linalg.eig(self.K.T)   # column convention for modes: K^T v = lambda v
        order = np.argsort(-np.abs(lam))
        return lam[order], V[:, order]

    def residual_norm(self, S):
        Z = self.z(S)
        R = Z[1:] - Z[:-1] @ self.K
        return np.sqrt((R ** 2).sum(1))

    def surprise(self, S):
        """Residual norm in units of the training residual's spread (0 = typical training error)."""
        return (self.residual_norm(S) - self.res_mean) / self.res_scale


def persistence_surprise(S_train, S):
    """Baseline: same residual idea with K = I (predict 'nothing changes')."""
    mu, sd = S_train.mean(0), S_train.std(0)
    keep = sd > 1e-9
    Zt = (S_train[:, keep] - mu[keep]) / sd[keep]
    rt = np.sqrt(((Zt[1:] - Zt[:-1]) ** 2).sum(1))
    Z = (S[:, keep] - mu[keep]) / sd[keep]
    r = np.sqrt(((Z[1:] - Z[:-1]) ** 2).sum(1))
    return (r - rt.mean()) / (rt.std() + 1e-12)


def novelty_score(S_train, S):
    """Baseline with the same inputs as the Koopman detector but no dynamics: distance of the
    current state from the training mean (z-scored norm), in units of its training spread."""
    mu, sd = S_train.mean(0), S_train.std(0)
    keep = sd > 1e-9
    dt = np.sqrt((((S_train[:, keep] - mu[keep]) / sd[keep]) ** 2).sum(1))
    d = np.sqrt((((S[:, keep] - mu[keep]) / sd[keep]) ** 2).sum(1))
    return (d - dt.mean()) / (dt.std() + 1e-12)


def sliding_spectrum(S, win, step):
    """Spectral radius and top eigenvalues of K fitted on sliding windows of the state."""
    rows = []
    for a in range(0, len(S) - win + 1, step):
        seg = S[a:a + win]
        try:
            m = LinearKoopman().fit(seg)
        except np.linalg.LinAlgError:
            continue
        lam, _ = m.eig()
        rows.append({"i0": a, "i1": a + win, "rho": float(np.abs(lam).max()),
                     "lam": lam[:6]})
    return rows


def smooth(x, n):
    """Trailing (causal) moving average: the value at t uses only t-n+1 ... t, never the future."""
    if n <= 1:
        return np.asarray(x, float)
    x = np.asarray(x, float)
    c = np.cumsum(np.r_[0.0, x])
    out = np.empty_like(x)
    for i in range(len(x)):
        a = max(0, i - n + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def first_alarm(score, times, threshold, after, hold):
    """First time after `after` that the smoothed score stays above threshold for `hold` bins."""
    above = score > threshold
    run = 0
    for i, (a, t) in enumerate(zip(above, times)):
        if t < after:
            run = 0
            continue
        run = run + 1 if a else 0
        if run >= hold:
            return times[i - hold + 1]
    return None
