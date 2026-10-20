#!/usr/bin/env python3
"""
live_monitor.py - a Koopman change detector for a running Linux machine.

No root, no tracing, no knowledge of which processes run. It reads /proc/stat,
which every Linux kernel updates on each timer tick:
    cpu  user nice system idle iowait irq softirq steal ...   CPU time (jiffies)
    procs_running N                                         runnable tasks right now
    ctxt  C                                                 context switches since boot
It samples every 10 ms and forms one state vector per 100 ms bin:
    busy     cores busy (from the jiffy counters, 0 .. n_cores)
    runnable average number of runnable tasks over the bin's samples
    switch   context switches per second, in thousands
(/proc/schedstat was tried first, but its per-core run time is only credited
when a task leaves a core, so a CPU-bound task produces spikes, not a signal.)

After `train` seconds it fits ONE linear operator K (s_{t+1} ~= s_t K,
standardised, least squares) and from then on scores every bin by its one-step
residual, in units of the training residual spread, averaged over the last
0.5 s. It raises an alarm when the score stays above `threshold` for 0.2 s.

    python3 live_monitor.py                         # print status every second
    python3 live_monitor.py --log run.jsonl --secs 40

The monitor measures its own CPU use and reports it at exit.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

CLK = os.sysconf("SC_CLK_TCK")


def read_proc_stat():
    """Return (busy_jiffies, total_jiffies, procs_running, ctxt) from /proc/stat."""
    busy = total = running = ctxt = 0
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                v = [int(x) for x in line.split()[1:9]]
                total = sum(v)
                busy = total - v[3] - v[4]          # minus idle and iowait
            elif line.startswith("procs_running"):
                running = int(line.split()[1])
            elif line.startswith("ctxt"):
                ctxt = int(line.split()[1])
    return busy, total, running, ctxt


class OnlineKoopman:
    def __init__(self, threshold=3.0, smooth_s=0.5, hold_s=0.2, dt=0.1):
        self.threshold, self.dt = threshold, dt
        self.n_smooth = max(1, int(round(smooth_s / dt)))
        self.n_hold = max(1, int(round(hold_s / dt)))
        self.K = None
        self.recent = []
        self.above = 0
        self.prev_z = None

    def fit(self, S):
        S = np.asarray(S, float)
        self.mu, self.sd = S.mean(0), S.std(0)
        self.keep = self.sd > 1e-9
        Z = (S[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        X, Y = Z[:-1], Z[1:]
        self.K = np.linalg.pinv(X) @ Y
        r = np.sqrt(((Y - X @ self.K) ** 2).sum(1))
        self.r_mean, self.r_sd = r.mean(), r.std() + 1e-12
        self.prev_z = Z[-1]
        lam = np.linalg.eigvals(self.K)
        return lam

    def step(self, s):
        """Score one new state vector; returns (score, alarm_now)."""
        z = (np.asarray(s, float)[self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        r = np.sqrt(((z - self.prev_z @ self.K) ** 2).sum())
        self.prev_z = z
        self.recent.append((r - self.r_mean) / self.r_sd)
        if len(self.recent) > self.n_smooth:
            self.recent.pop(0)
        score = float(np.mean(self.recent))
        self.above = self.above + 1 if score > self.threshold else 0
        return score, self.above == self.n_hold          # True exactly once per alarm onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1, help="state bin (s)")
    ap.add_argument("--sample", type=float, default=0.01, help="sampling interval inside a bin (s)")
    ap.add_argument("--train", type=float, default=8.0)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--secs", type=float, default=0, help="stop after this many seconds (0 = run until Ctrl-C)")
    ap.add_argument("--log", help="write one JSON line per bin to this file")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    ncpu = os.cpu_count()
    log = open(a.log, "w") if a.log else None
    det = OnlineKoopman(a.threshold, dt=a.dt)
    per_bin = max(1, int(round(a.dt / a.sample)))
    t_start = time.clock_gettime(time.CLOCK_MONOTONIC)
    cpu0 = time.process_time()
    b0, tot0, _, c0 = read_proc_stat()
    t_bin = t_start
    train, alarms, n_bins = [], [], 0
    next_t = t_start + a.sample
    last_print = t_start
    run_samples = []
    try:
        while True:
            time.sleep(max(0.0, next_t - time.clock_gettime(time.CLOCK_MONOTONIC)))
            next_t += a.sample
            b, tot, running, c = read_proc_stat()
            run_samples.append(running - 1)          # do not count the monitor itself
            if len(run_samples) < per_bin:
                continue
            now = time.clock_gettime(time.CLOCK_MONOTONIC)
            dtot = max(tot - tot0, 1)
            s = [ncpu * (b - b0) / dtot, float(np.mean(run_samples)), (c - c0) / (now - t_bin) / 1000.0]
            b0, tot0, c0, t_bin = b, tot, c, now
            run_samples = []
            n_bins += 1
            el = now - t_start
            rec = {"t": round(now, 6), "busy": round(s[0], 4), "runnable": round(s[1], 3), "switch_k": round(s[2], 4)}
            if det.K is None:
                train.append(s)
                if el >= a.train:
                    lam = det.fit(train[1:])         # first bin straddles start-up
                    rec["fitted"] = True
                    rec["rho"] = round(float(np.abs(lam).max()), 4)
                    if not a.quiet:
                        print(f"[{el:7.2f}s] operator fitted on {len(train) - 1} bins, spectral radius "
                              f"{np.abs(lam).max():.3f}", flush=True)
            else:
                score, onset = det.step(s)
                rec["score"] = round(score, 3)
                if onset:
                    t_alarm = now - (det.n_hold - 1) * a.dt
                    alarms.append(t_alarm)
                    rec["alarm"] = True
                    if not a.quiet:
                        print(f"[{el:7.2f}s] ALARM  surprise {score:.1f} sigma  (busy {s[0]:.2f} cores, "
                              f"runnable {s[1]:.1f}, {s[2]:.1f}k switches/s)", flush=True)
                if not a.quiet and now - last_print >= 1.0:
                    print(f"[{el:7.2f}s] surprise {score:5.1f}  busy {s[0]:.2f}  runnable {s[1]:.1f}", flush=True)
                    last_print = now
            if log:
                log.write(json.dumps(rec) + "\n")
            if a.secs and el >= a.secs:
                break
    except KeyboardInterrupt:
        pass
    wall = time.clock_gettime(time.CLOCK_MONOTONIC) - t_start
    cpu = time.process_time() - cpu0
    summary = {"summary": True, "bins": n_bins, "wall_s": round(wall, 3), "cpu_s": round(cpu, 4),
               "cpu_overhead_pct_of_one_core": round(100 * cpu / wall, 3), "alarms": [round(x, 6) for x in alarms],
               "dt": a.dt, "sample": a.sample, "train_s": a.train, "threshold": a.threshold}
    if log:
        log.write(json.dumps(summary) + "\n")
        log.close()
    if not a.quiet:
        print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
