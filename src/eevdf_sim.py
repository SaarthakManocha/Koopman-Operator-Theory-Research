"""
eevdf_sim.py - event-driven simulator of Linux EEVDF with per-task slice requests.

STATUS: NOT VALIDATED, NOT USED IN THE PAPER. Compared with the measured closed-loop
runs (1 CPU, 3 hogs, 4 request tasks) it gets stock-Linux short-request response
about 45% too high (3.2-3.4 ms against 2.14-2.27 ms measured) and does not reproduce
the measured effect of the oracle slices (exact slice no better than stock, twice the
burst 46% better). Some mechanism of the real kernel (accounting points, delayed
dequeue, lag scaling, overheads) is missing. Kept for reference only.

Purpose: reproduce the closed-loop experiment (closedloop/) at core counts the
VM does not have. It is validated against the measured runs first
(src/eevdf_validate.py) and only then used to scale.

Model (kernel 6.12+ fair class, all tasks nice 0):
  * per-CPU run queue; vruntime advances at wall-clock rate while running;
    V = mean vruntime of runnable entities (avg_vruntime);
  * an entity is eligible when v <= V; pick = eligible entity with the earliest
    virtual deadline d = v + slice (pick_eevdf);
  * RUN_TO_PARITY: a picked entity is protected until its deadline is renewed;
  * slice expiry is noticed at the scheduler tick (HZ, no hrtick): if v >= d the
    deadline is renewed (d = v + slice), protection ends and the CPU re-picks;
  * wake-up: placed with its saved lag (v = V - vlag, |vlag| <= max(2*slice, tick)),
    d = v + slice; if the woken task has a shorter slice than the running one and
    an earlier deadline, the running task's protection is cancelled
    (PREEMPT_SHORT); the CPU then re-picks and switches if the pick changed;
  * sleeping: vlag = V - v is saved;
  * placement across CPUs: previous CPU if idle, else any idle CPU, else the
    previous CPU. Hogs are pinned round-robin. No periodic load balancing,
    no delayed dequeue, no context-switch cost.

Workload: identical generator to closedloop/task.c (two-regime log-AR(1) bursts,
periods 15/20/25 ms, overrun -> next request starts at once).

Slice policies: 'stock' (default slice), ('ewma', c), ('koopman', c) with exact
per-burst online predictors (log-scale 3-lag operator updated by recursive least
squares, as in the eBPF agent), ('oracle', c).
"""
import heapq
import math
import random

import numpy as np

DEF_SLICE = 1.4e-3
MIN_SLICE = 1e-4


class Pred:
    def __init__(self, kind):
        self.kind, self.n, self.ewma = kind, 0, None
        self.l = [0.0, 0.0, 0.0]
        self.th = np.zeros(4); self.P = np.eye(4) * 100.0

    def update(self, b):
        lb = math.log(max(b, 1e-7))
        self.ewma = b if self.ewma is None else 0.5 * b + 0.5 * self.ewma
        if self.kind == "koopman" and self.n >= 3:
            x = np.array([1.0, *self.l]); Px = self.P @ x; den = 0.995 + x @ Px
            e = lb - self.th @ x
            self.th = self.th + Px / den * e
            self.P = (self.P - np.outer(Px, Px) / den) / 0.995
        self.l = [lb, self.l[0], self.l[1]]; self.n += 1
        if self.kind == "koopman" and self.n >= 10:
            return math.exp(min(0.0, self.th @ np.array([1.0, *self.l])))
        return self.ewma


class Task:
    __slots__ = ("id", "kind", "slice", "v", "d", "vlag", "protect", "cpu", "rem", "on_rq", "cpu_time",
                 "rng", "regime", "lw", "w", "period", "dl", "wake_t", "pred", "policy", "pinned", "log",
                 "mus", "mul", "psw", "pred_val", "vprot")

    def __init__(self, id_, kind):
        self.id, self.kind = id_, kind
        self.slice, self.v, self.d, self.vlag = DEF_SLICE, 0.0, DEF_SLICE, 0.0
        self.protect, self.cpu, self.rem, self.on_rq, self.cpu_time = False, 0, math.inf, False, 0.0
        self.log = []
        self.vprot = 0.0
        self.pred_val = None
        self.wake_t = None


class CPU:
    def __init__(self, i):
        self.i, self.rq, self.curr, self.t_start, self.ver, self.vbase = i, [], None, 0.0, 0, 0.0


class Sim:
    def __init__(self, n_cpus, n_hogs, n_reqs, secs, policy, seed, hz=250, mus=0.3, mul=4.0, psw=0.05, overhead=0.0):
        self.m, self.T, self.policy, self.tick, self.ovh = n_cpus, secs, policy, 1.0 / hz, overhead
        self.cpus = [CPU(i) for i in range(n_cpus)]
        self.ev, self.seq, self.now = [], 0, 0.0
        rng = random.Random(1000 * seed + 7)
        self.tasks = []
        for i in range(n_hogs):
            t = Task(len(self.tasks), "hog"); t.cpu = i % n_cpus; t.pinned = True; t.rem = math.inf
            self.tasks.append(t); self.enqueue(t, initial=True)
        for i in range(n_reqs):
            t = Task(len(self.tasks), "req"); t.pinned = False
            t.rng = random.Random(rng.randrange(1, 10 ** 6))
            t.period = rng.choice([15, 20, 25]) * 1e-3
            t.regime, t.lw, t.mus, t.mul, t.psw = 0, math.log(mus), math.log(mus), math.log(mul), psw
            t.w = math.exp(t.lw)
            t.pred = Pred(policy[0]) if policy[0] in ("ewma", "koopman") else None
            t.cpu = i % n_cpus
            t.dl = t.period
            self.tasks.append(t); self.push(t.dl, "wake", t)
        for c in self.cpus:
            self.push(self.tick * (1 + c.i / max(1, n_cpus)), "tick", c)
            self.resched(c)

    # ------------------------------------------------------------- helpers
    def push(self, t, kind, obj, ver=None):
        self.seq += 1; heapq.heappush(self.ev, (t, self.seq, kind, obj, ver))

    def vcur(self, c):
        """bring the running entity's vruntime up to now"""
        t = c.curr
        if t is not None:
            dt = self.now - c.t_start
            t.v += dt; t.cpu_time += dt
            if t.kind == "req":
                t.rem -= dt
            c.t_start = self.now
            if t.v >= t.d:                      # update_deadline(): slice used up
                t.d = t.v + t.slice
                t.protect = False

    def V(self, c):
        if not c.rq:
            return c.vbase
        return sum(t.v for t in c.rq) / len(c.rq)

    def slice_for(self, t):
        if t.kind == "hog" or self.policy[0] == "stock":
            return DEF_SLICE
        kind, cfac = self.policy
        if kind == "oracle":
            s = cfac * t.w * 1e-3
        else:
            s = cfac * (t.pred_val if t.pred_val is not None else DEF_SLICE)
        return min(max(s, MIN_SLICE), DEF_SLICE)

    def enqueue(self, t, initial=False):
        c = self.cpus[t.cpu]
        V = self.V(c)
        if initial:
            t.v = V; t.d = t.v + t.slice / 2
        else:
            lim = max(2 * t.slice, self.tick)
            t.v = V - max(-lim, min(lim, t.vlag))
            t.d = t.v + t.slice
        t.on_rq = True; t.protect = False
        c.rq.append(t)

    def pick(self, c):
        cur = c.curr
        V = self.V(c)
        # RUN_TO_PARITY (6.17+): the running entity keeps the CPU while it is still eligible and
        # has not used up its protected budget, min(deadline, vruntime at pick + smallest queued slice)
        if cur is not None and cur.on_rq and cur.protect and cur.v <= V + 1e-12 and cur.v < cur.vprot:
            return cur
        el = [t for t in c.rq if t.v <= V + 1e-12]
        if not el:
            el = c.rq
        return min(el, key=lambda t: (t.d, t.id)) if el else None

    def resched(self, c):
        self.vcur(c)
        nxt = self.pick(c)
        if nxt is c.curr:
            if nxt is not None and not nxt.protect:
                nxt.protect = True
                nxt.vprot = min(nxt.d, nxt.v + min(t.slice for t in c.rq))
            return
        c.curr = nxt; c.t_start = self.now; c.ver += 1
        if nxt is not None:
            nxt.protect = True
            nxt.vprot = min(nxt.d, nxt.v + min(t.slice for t in c.rq))
            if nxt.kind == "req" and nxt.wake_t is None:
                nxt.wake_t = self.now           # first time on a CPU after waking
            if nxt.kind == "req":
                self.push(self.now + nxt.rem, "done", c, c.ver)

    # ------------------------------------------------------------- events
    def on_wake(self, t):
        # choose CPU
        prev = self.cpus[t.cpu]
        if prev.curr is not None:
            idle = [c for c in self.cpus if c.curr is None]
            if idle:
                t.cpu = idle[0].i
        c = self.cpus[t.cpu]
        self.vcur(c)
        t.rem = t.w * 1e-3 + self.ovh          # wake-up and system-call work on top of the burst
        t.slice = self.slice_for(t)
        t.wake_t = None
        self.enqueue(t)
        cur = c.curr
        if cur is not None:
            if t.slice < cur.slice and t.v <= self.V(c) and t.d < cur.d:
                cur.protect = False                         # PREEMPT_SHORT
            else:
                cur.vprot = min(cur.vprot, cur.v + t.slice)  # update_protect_slice
        self.resched(c)

    def on_done(self, c):
        t = c.curr
        self.vcur(c)
        fin = self.now
        t.log.append((t.dl, t.wake_t, fin, t.w))
        # sleep
        V = self.V(c)
        t.vlag = V - t.v
        c.rq.remove(t); t.on_rq = False
        c.vbase = self.V(c) if c.rq else V
        c.curr = None; c.ver += 1
        # predictor sees the finished burst
        if t.pred is not None:
            t.pred_val = t.pred.update(t.w * 1e-3)
        # next request
        if t.rng.random() < t.psw:
            t.regime = 1 - t.regime
        mu = t.mul if t.regime else t.mus
        t.lw = mu + 0.6 * (t.lw - mu) + 0.35 * t.rng.gauss(0, 1)
        t.w = min(30.0, max(0.05, math.exp(t.lw)))
        t.dl += t.period
        if fin > t.dl:
            t.dl = fin
        if t.dl < self.T:
            self.push(t.dl, "wake", t)
        self.resched(c)

    def on_tick(self, c):
        self.vcur(c)
        self.resched(c)
        self.push(self.now + self.tick, "tick", c)

    def run(self):
        while self.ev:
            t, _, kind, obj, ver = heapq.heappop(self.ev)
            if t > self.T + 1.0:
                break
            self.now = t
            if kind == "wake":
                self.on_wake(obj)
            elif kind == "done":
                if ver == obj.ver:
                    self.on_done(obj)
            elif kind == "tick":
                if t <= self.T:
                    self.on_tick(obj)
        return self

    def results(self):
        rows = [r for t in self.tasks if t.kind == "req" for r in t.log]
        a = np.array(rows)
        resp = (a[:, 2] - a[:, 0]) * 1e3; wake = (a[:, 1] - a[:, 0]) * 1e3; w = a[:, 3]
        short = w < 1.0
        hog = sum(t.cpu_time for t in self.tasks if t.kind == "hog")
        return {"n_req": len(a), "resp_short_mean": resp[short].mean(), "resp_long_mean": resp[~short].mean(),
                "resp_mean": resp.mean(), "wake_p50": float(np.median(wake)), "hog_cpu_s": hog}


def simulate(n_cpus, n_hogs, n_reqs, secs, policy, seed, hz=250, overhead=0.0):
    return Sim(n_cpus, n_hogs, n_reqs, secs, policy, seed, hz, overhead=overhead).run().results()


if __name__ == "__main__":
    import time
    for pol in [("stock",), ("ewma", 1.0), ("koopman", 1.0), ("oracle", 1.0), ("oracle", 2.0)]:
        t0 = time.time()
        r = simulate(1, 3, 4, 20, pol, 0, overhead=float(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 0.0)
        print(pol, {k: round(v, 3) for k, v in r.items()}, f"{time.time() - t0:.1f}s")
