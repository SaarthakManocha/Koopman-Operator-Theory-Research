"""
market_koopman.py
-----------------
The same Koopman change detector, applied to the S&P 500 (daily, 1990-2026).

Pre-registered protocol (fixed before any result was looked at):

State (one vector per trading day, the EIGENWORLD feature set):
    log return, 21-day and 63-day realised volatility, their ratio,
    21-day and 63-day cumulative return, volume z-score (63-day)

Detector (identical in form to the scheduler and network detectors):
    K fitted by least squares on the previous 504 trading days (2 years) of
    standardised state, refitted every 21 days (walk-forward, never sees the future);
    score = one-step residual norm in units of the training residual spread,
    trailing 5-day mean; alarm onset when the score first exceeds the threshold.

Events (found by rule, not by hand):
    event day = first close >= 10% below the trailing 252-day high, after the index
    had been within 5% of its high (one event per drawdown episode).

Questions:
    Precursor: is there an alarm in the 20 trading days BEFORE the event?
    Compared with chance: the same number of alarm days placed at random.
    False alarms: alarm onsets more than 60 trading days from any event, per year.
Baselines with the same alarm logic: VIX z-score, 21-day realised-volatility
z-score, distance-from-normal (same state, no dynamics), persistence residual.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "market")
OUT = os.path.join(ROOT, "outputs", "market")

TRAIN, REFIT, SMOOTH = 504, 21, 5
LEAD, FAR = 20, 60
NAMED = {  # secondary check, well-known stress dates (event = first trading day on/after)
    "LTCM / Russia": "1998-08-04", "Dot-com peak to slide": "2000-04-14", "9/11 reopening": "2001-09-17",
    "Bear Stearns funds": "2007-07-26", "Lehman Brothers": "2008-09-15", "Flash crash": "2010-05-06",
    "US downgrade": "2011-08-08", "China devaluation": "2015-08-24", "Volmageddon": "2018-02-05",
    "Q4 2018 selloff": "2018-12-24", "COVID crash": "2020-03-09", "2022 rate shock": "2022-06-13",
}


def load():
    sp = pd.read_csv(os.path.join(DATA, "sp500_1990_2026.csv"), parse_dates=["Date"]).set_index("Date")
    vix = pd.read_csv(os.path.join(DATA, "vix_1990_2026.csv"), parse_dates=["Date"]).set_index("Date")
    return sp, vix.Close.reindex(sp.index).ffill()


def features(sp):
    close, vol = sp.Close.astype(float), sp.Volume.astype(float)
    r = np.log(close / close.shift(1))
    f = pd.DataFrame({
        "log_return": r,
        "vol_21": r.rolling(21).std(), "vol_63": r.rolling(63).std(),
        "ret_21": r.rolling(21).sum(), "ret_63": r.rolling(63).sum(),
        "volume_z": (vol - vol.rolling(63).mean()) / vol.rolling(63).std(),
    })
    f["vol_ratio"] = f.vol_21 / f.vol_63
    return f.dropna()


def events(close, dd_trigger=-0.10, reset=-0.05):
    peak = close.rolling(252, min_periods=1).max()
    dd = close / peak - 1
    out, armed = [], True
    for d, v in dd.items():
        if armed and v <= dd_trigger:
            out.append(d)
            armed = False
        elif not armed and v >= reset:
            armed = True
    return out, dd


def walk_forward_scores(F):
    """Koopman residual, persistence residual and distance-from-normal, walk-forward."""
    X = F.values
    n = len(X)
    k_res = np.full(n, np.nan)
    p_res = np.full(n, np.nan)
    nov = np.full(n, np.nan)
    rho = np.full(n, np.nan)
    for start in range(TRAIN, n, REFIT):
        tr = X[start - TRAIN:start]
        mu, sd = tr.mean(0), tr.std(0) + 1e-12
        Z = (tr - mu) / sd
        Xa, Ya = Z[:-1], Z[1:]
        K = np.linalg.pinv(Xa) @ Ya
        r_tr = np.sqrt(((Ya - Xa @ K) ** 2).sum(1))
        p_tr = np.sqrt(((Ya - Xa) ** 2).sum(1))
        d_tr = np.sqrt((Z ** 2).sum(1))
        lam = np.abs(np.linalg.eigvals(K)).max()
        end = min(n, start + REFIT)
        Zn = (X[start - 1:end] - mu) / sd
        r = np.sqrt(((Zn[1:] - Zn[:-1] @ K) ** 2).sum(1))
        p = np.sqrt(((Zn[1:] - Zn[:-1]) ** 2).sum(1))
        dn = np.sqrt((Zn[1:] ** 2).sum(1))
        k_res[start:end] = (r - r_tr.mean()) / r_tr.std()
        p_res[start:end] = (p - p_tr.mean()) / p_tr.std()
        nov[start:end] = (dn - d_tr.mean()) / d_tr.std()
        rho[start:end] = lam
    idx = F.index
    sm = lambda a: pd.Series(a, idx).rolling(SMOOTH, min_periods=1).mean()  # noqa: E731
    return pd.DataFrame({"koopman": sm(k_res), "persistence": sm(p_res), "novelty": sm(nov), "rho": rho}, index=idx)


def zscore_trailing(s, win=504):
    return (s - s.rolling(win).mean().shift(1)) / s.rolling(win).std().shift(1)


def onsets(score, thr):
    above = (score > thr).fillna(False).values
    return [score.index[i] for i in range(len(above)) if above[i] and (i == 0 or not above[i - 1])]


def evaluate(score, ev, thr, days):
    pos = {d: i for i, d in enumerate(days)}
    on = [d for d in onsets(score, thr) if not np.isnan(score.loc[d])]
    on_i = np.array([pos[d] for d in on])
    ev_i = np.array([pos[d] for d in ev if d in pos])
    hits = sum(any((on_i >= e - LEAD) & (on_i < e)) for e in ev_i)
    far = [o for o in on_i if len(ev_i) == 0 or np.min(np.abs(ev_i - o)) > FAR]
    valid = score.dropna()
    years = len(valid) / 252
    # alarm-day rate and chance level of a hit with the same number of alarm onsets placed at random
    p_day = len(on_i) / len(valid)
    chance = 1 - (1 - p_day) ** LEAD
    return {"threshold": thr, "alarm_onsets": len(on_i), "events": len(ev_i), "precursor_hits": int(hits),
            "hit_rate": hits / max(len(ev_i), 1), "chance_hit_rate": chance,
            "far_alarms_per_year": len(far) / years}


def run():
    os.makedirs(OUT, exist_ok=True)
    sp, vix = load()
    F = features(sp)
    sc = walk_forward_scores(F)
    sc["vix_z"] = zscore_trailing(vix.reindex(sc.index)).rolling(SMOOTH, min_periods=1).mean()
    sc["realised_vol_z"] = zscore_trailing(F.vol_21).rolling(SMOOTH, min_periods=1).mean()
    valid_from = sc.koopman.first_valid_index()
    sc = sc.loc[valid_from:]
    ev, dd = events(sp.Close)
    ev = [e for e in ev if e >= valid_from + pd.Timedelta(days=40)]
    days = list(sc.index)
    rows = []
    for name in ["koopman", "persistence", "novelty", "vix_z", "realised_vol_z"]:
        for thr in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
            rows.append({"detector": name, **evaluate(sc[name], ev, thr, days)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "precursor_sweep.csv"), index=False)
    sc.to_csv(os.path.join(OUT, "scores.csv"))
    pd.DataFrame({"event_date": ev, "drawdown_at_event": [dd.loc[e] for e in ev]}).to_csv(os.path.join(OUT, "events.csv"), index=False)

    # named stress dates: first alarm in the 20 trading days before, per detector, at threshold 3
    named = []
    for label, ds in NAMED.items():
        d = sc.index[sc.index.searchsorted(pd.Timestamp(ds))]
        i = days.index(d)
        rec = {"event": label, "date": d.date()}
        for name in ["koopman", "novelty", "vix_z", "realised_vol_z"]:
            win = sc[name].iloc[max(0, i - LEAD):i]
            first = win[win > 3.0]
            rec[f"{name}_lead_days"] = (i - days.index(first.index[0])) if len(first) else None
            rec[f"{name}_max_before"] = round(float(win.max()), 2)
        named.append(rec)
    named = pd.DataFrame(named)
    named.to_csv(os.path.join(OUT, "named_events.csv"), index=False)

    # spectral radius before events vs all other days (EIGENWORLD's 'stability margin' claim)
    pre = np.zeros(len(sc), bool)
    for e in ev:
        i = days.index(e)
        pre[max(0, i - 60):i] = True
    rho = sc.rho.values
    rng = np.random.default_rng(0)
    obs = np.nanmean(rho[pre]) - np.nanmean(rho[~pre])
    # block permutation: shift the pre-event mask by random offsets (keeps its block structure)
    null = [np.nanmean(rho[np.roll(pre, k)]) - np.nanmean(rho[~np.roll(pre, k)])
            for k in rng.integers(252, len(rho) - 252, 2000)]
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
    with open(os.path.join(OUT, "summary.txt"), "w") as f:
        def w(s=""):
            print(s)
            f.write(s + "\n")
        w(f"Trading days scored: {len(sc)} ({sc.index[0].date()} to {sc.index[-1].date()}); events: {len(ev)}")
        w("Events: " + ", ".join(str(e.date()) for e in ev))
        w("\nPrecursor sweep (hit = alarm onset within 20 trading days before the event):")
        w(res.round(3).to_string(index=False))
        w("\nNamed stress dates (lead days of first score > 3 in the 20 days before; blank = none):")
        w(named.to_string(index=False))
        w(f"\nSpectral radius of K: mean in the 60 days before events {np.nanmean(rho[pre]):.4f}, "
          f"other days {np.nanmean(rho[~pre]):.4f}, difference {obs:+.4f}, block-permutation p = {p:.3f}")
    return res, named


if __name__ == "__main__":
    run()
