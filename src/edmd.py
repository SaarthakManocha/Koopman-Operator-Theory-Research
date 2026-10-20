"""
edmd.py
-------
Extended Dynamic Mode Decomposition (EDMD), a finite approximation of the
Koopman operator.

Idea in one paragraph:
    The system moves x(t) -> x(t+1) through some unknown, possibly nonlinear
    map F. The Koopman operator K does not act on x itself but on functions
    ("observables") g of x:  K g(x(t)) = g(x(t+1)).  K is LINEAR even when F
    is not. EDMD picks a finite list of observables psi(x) (the
    "dictionary"), collects pairs (psi(x_t), psi(x_{t+1})) from data, and
    solves a least-squares problem for the matrix A with
        psi(x_{t+1}) ~= A psi(x_t).
    The eigenvalues of A describe how quickly patterns decay (|lambda| < 1)
    or oscillate (complex lambda).

Here the "state" of a process is its last `delay` burst lengths
(time-delay embedding):  x_t = [b_t, b_{t-1}, ..., b_{t-delay+1}].
"""

import numpy as np


def delay_embed(series, delay):
    """Rows are [b_t, b_{t-1}, ..., b_{t-delay+1}] for t = delay-1 ... n-1."""
    series = np.asarray(series, dtype=float)
    n = len(series) - delay + 1
    return np.column_stack([series[delay - 1 - k: delay - 1 - k + n] for k in range(delay)])


def dictionary(X, kind="linear"):
    """
    Observables psi(x).
      linear : the lags themselves                  -> EDMD == linear AR(delay)
      poly2  : lags + squares + pairwise products   -> genuinely nonlinear observables
    """
    if kind == "linear":
        return X
    if kind == "poly2":
        d = X.shape[1]
        cols = [X]
        cols.append(X ** 2)
        for i in range(d):
            for j in range(i + 1, d):
                cols.append((X[:, i] * X[:, j])[:, None])
        return np.hstack(cols)
    raise ValueError(kind)


class EDMD:
    """
    Fit A so that psi(x_{t+1}) ~= A psi(x_t).  Data are z-scored with
    TRAINING statistics only, so there is no information leak from test data.

    log=True fits the model on log(burst) instead of burst, which shrinks the
    few very long bursts that otherwise dominate least squares.
    """

    def __init__(self, delay=3, kind="linear", log=False, ridge=1e-6):
        self.delay, self.kind, self.log, self.ridge = delay, kind, log, ridge

    # ---- transforms -------------------------------------------------------
    def _fwd(self, b):
        b = np.asarray(b, dtype=float)
        if self.log:
            b = np.log(np.maximum(b, 1e-9))
        return (b - self.mu) / self.sd

    def _inv(self, z):
        v = z * self.sd + self.mu
        return np.exp(v) if self.log else np.maximum(v, 0.0)

    # ---- fit --------------------------------------------------------------
    def fit(self, train):
        train = np.asarray(train, dtype=float)
        raw = np.log(np.maximum(train, 1e-9)) if self.log else train
        self.mu, self.sd = raw.mean(), raw.std() + 1e-12
        z = self._fwd(train)
        X = delay_embed(z, self.delay)
        P, Pn = dictionary(X[:-1], self.kind), dictionary(X[1:], self.kind)
        # least squares with a tiny ridge for numerical safety:  A = Pn^T P (P^T P + rI)^-1
        G = P.T @ P + self.ridge * np.eye(P.shape[1])
        self.A = np.linalg.solve(G, P.T @ Pn).T
        return self

    def eigenvalues(self):
        return np.linalg.eigvals(self.A)

    # ---- forecasts --------------------------------------------------------
    def predict_one_step(self, history_and_test, start):
        """
        For every index t >= start, predict b_t from the REAL previous
        `delay` bursts b_{t-1}, ..., b_{t-delay}.
        """
        z = self._fwd(history_and_test)
        preds = []
        for t in range(start, len(z)):
            x = z[t - self.delay:t][::-1][None, :]          # [b_{t-1}, b_{t-2}, ...]
            psi_next = dictionary(x, self.kind) @ self.A.T
            preds.append(psi_next[0, 0])                     # first observable = newest lag
        return self._inv(np.array(preds))

    def predict_recursive(self, train, horizon):
        """Start from the last training window and feed predictions back in."""
        z = list(self._fwd(train)[-self.delay:][::-1])       # newest first
        out = []
        for _ in range(horizon):
            x = np.array(z[:self.delay])[None, :]
            nxt = (dictionary(x, self.kind) @ self.A.T)[0, 0]
            out.append(nxt)
            z.insert(0, nxt)
        return self._inv(np.array(out))
