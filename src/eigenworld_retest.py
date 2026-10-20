"""
eigenworld_retest.py
--------------------
Runs EIGENWORLD's own, unmodified pipelines (third_party/eigenworld) on the S&P 500
from 1990 to 2026 and tests its claim on all 22 rule-defined events, not one.

  1. Reproduction check on the original 2004-2010 file (entropy minimum before Lehman).
  2. Long run: diffusion-map spectral entropy and Koopman stability margin every 21 days.
  3. Test: are entropy and stability margin lower in windows ending in the 60 trading days
     before an event than in other windows? Block-permutation test (the pre-event mask is
     circularly shifted, keeping its block structure), two-sided, 5,000 shifts.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))
sys.path.insert(0, os.path.dirname(__file__))
from eigenworld.koopman.pipeline import KoopmanPipeline  # noqa: E402
from eigenworld.manifold.pipeline import ManifoldPipeline  # noqa: E402
from market_koopman import events  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "market")


def load_long():
    df = pd.read_csv(os.path.join(ROOT, "data", "market", "sp500_1990_2026.csv"), parse_dates=["Date"])
    return df[["Date", "Close", "High", "Low", "Open", "Volume"]]


def perm_test(values, mask, n=5000, seed=0):
    v = np.asarray(values, float)
    obs = v[mask].mean() - v[~mask].mean()
    rng = np.random.default_rng(seed)
    null = []
    for k in rng.integers(3, len(v) - 3, n):
        m = np.roll(mask, k)
        null.append(v[m].mean() - v[~m].mean())
    null = np.array(null)
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1)


class seeded:
    """EIGENWORLD estimates its kernel bandwidth from 500 random point pairs drawn with an
    UNSEEDED numpy generator, so its output varies slightly from run to run. Inside this
    context every unseeded default_rng() call is fed from one master generator with a fixed
    seed: the method is unchanged, the run becomes reproducible, and different seeds show how
    much the conclusions depend on the random draw."""

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.orig = np.random.default_rng
        master = self.orig(self.seed)
        orig = self.orig
        np.random.default_rng = lambda s=None: orig(int(master.integers(2 ** 32))) if s is None else orig(s)

    def __exit__(self, *a):
        np.random.default_rng = self.orig


def run_once(df, seed):
    with seeded(seed):
        man = ManifoldPipeline(10, 500, 21).run(df)
        kop = KoopmanPipeline().run(df)
    return man.join(kop, how="inner")


def main():
    os.makedirs(OUT, exist_ok=True)
    df = load_long()
    seeds = [int(x) for x in sys.argv[1:]] or [0]
    sweep = []
    for seed in seeds[1:]:
        r = run_once(df, seed)
        sp_ = df.set_index("Date").Close
        ev_, _ = events(sp_)
        ev_ = [e for e in ev_ if e > r.index[0]]
        d_ = sp_.index
        pre_ = np.array([any(0 < d_.searchsorted(e) - d_.searchsorted(x) <= 60 for e in ev_) for x in r.index])
        o1, p1 = perm_test(r.spectral_entropy.values, pre_)
        o2, p2 = perm_test(r.stability_margin.values, pre_)
        sweep.append({"seed": seed, "entropy_diff": o1, "entropy_p": p1, "margin_diff": o2, "margin_p": p2,
                      "min_date": r.spectral_entropy.idxmin().date(), "min_value": r.spectral_entropy.min()})
        print(sweep[-1], flush=True)
    if sweep:
        pd.DataFrame(sweep).to_csv(os.path.join(OUT, "eigenworld_seed_sweep.csv"), index=False)
    res = run_once(df, seeds[0])
    res.to_csv(os.path.join(OUT, "eigenworld_long_run.csv"))
    sp = df.set_index("Date").Close
    ev, _ = events(sp)
    days = sp.index
    ev = [e for e in ev if e > res.index[0]]
    pre = np.zeros(len(res), bool)
    for i, d in enumerate(res.index):
        for e in ev:
            gap = days.searchsorted(e) - days.searchsorted(d)
            if 0 < gap <= 60:
                pre[i] = True
    lines = [f"(main run: bandwidth sample seed {seeds[0]})", f"EIGENWORLD pipelines on {res.index[0].date()} .. {res.index[-1].date()}: {len(res)} windows, "
             f"{pre.sum()} end in the 60 trading days before one of {len(ev)} events"]
    for col, claim in (("spectral_entropy", "lower before crashes"), ("stability_margin", "smaller before crashes")):
        obs, p = perm_test(res[col].values, pre)
        lines.append(f"{col:17s}: mean before events {res[col][pre].mean():.4f}, other windows "
                     f"{res[col][~pre].mean():.4f}, difference {obs:+.4f}, permutation p = {p:.3f}  (claim: {claim})")
    # where does the long-run entropy minimum fall, and how many events are preceded by a window in the lowest 10%?
    q10 = res.spectral_entropy.quantile(0.10)
    low = res.spectral_entropy <= q10
    lines.append(f"Global entropy minimum: {res.spectral_entropy.min():.4f} on {res.spectral_entropy.idxmin().date()}")
    lines.append(f"Windows in the lowest 10% of entropy: {low.sum()}; of these, {int((low & pre).sum())} are pre-event "
                 f"(expected by chance {low.sum() * pre.mean():.1f})")
    lehman = res.loc["2008-06-01":"2008-10-31"]
    lines.append("Around Lehman (EIGENWORLD's original case): " + ", ".join(
        f"{d.date()} {v:.3f}" for d, v in lehman.spectral_entropy.items()))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "eigenworld_retest.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
