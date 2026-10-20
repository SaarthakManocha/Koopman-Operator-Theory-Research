"""
agent.py - userspace burst predictor that steers EEVDF through per-task slice requests.

    python3 agent.py <policy> <pidfile> <seconds> <out.json>

Every POLL seconds it reads, for each task in pidfile,
    /proc/<pid>/schedstat   (CPU time used, ns)
    /proc/<pid>/status      (voluntary context switches)
A CPU burst is the CPU time between two voluntary switches (the task blocked or
slept); if k voluntary switches happened since the last poll the accumulated CPU
time is split evenly, so the agent sees the same burst definition as the
offline study. The agent never uses information the kernel does not expose.

policies
    ewma     next burst = 0.5*last + 0.5*previous estimate (textbook SJF)
    koopman  log-EDMD, 3 lags, linear dictionary, refit on the last 300 bursts
    fixed    every task gets the minimum slice once
    static   tasks labelled 'req' get the minimum, 'hog' keep the default (needs labels)
    ewma_q, koopman_q
             as above, but the slice is the conformal upper bound of the prediction:
             pred * exp(q), q = 90th percentile of log(actual/predicted) over the
             task's last 200 bursts (so ~90% of bursts fit inside their slice)
The predicted burst b is mapped to slice = clamp(b, SMIN, default slice): the agent
only shortens slices, never lengthens them. A task with no
voluntary switch while it has used more than SMAX of CPU is treated as long.
"""
import ctypes
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from edmd import EDMD  # noqa: E402

POLL = 0.010
SMIN = 0.1e-3                       # kernel minimum
SMAX = None                         # the kernel default slice, read at start: the agent only ever shortens
libc = ctypes.CDLL(None, use_errno=True)
NR_SETATTR = 314 if os.uname().machine == "x86_64" else 274


def get_slice(pid=0):
    buf = ctypes.create_string_buffer(56)
    libc.syscall(NR_SETATTR + 1, pid, buf, 56, 0)
    return struct.unpack("IIQiIQQQ", buf.raw[:48])[5] / 1e9


def set_slice(pid, sec):
    ns = int(min(max(sec, SMIN), SMAX) * 1e9)
    buf = ctypes.create_string_buffer(struct.pack("IIQiIQQQ", 48, 0, 0, 0, 0, ns, 0, 0))
    return libc.syscall(NR_SETATTR, pid, buf, 0)


def read(pid):
    with open(f"/proc/{pid}/schedstat") as f:
        run = int(f.read().split()[0])
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("voluntary_ctxt_switches"):
                return run, int(line.split()[1])


class Koopman:
    def __init__(self):
        self.m, self.n_at_fit = None, 0

    def predict(self, hist, ewma):
        if len(hist) < 30:
            return ewma
        if self.m is None or len(hist) - self.n_at_fit >= 20:
            self.m = EDMD(delay=3, kind="linear", log=True).fit(np.array(hist[-300:]))
            self.n_at_fit = len(hist)
        x = np.array(hist[-4:])
        return float(self.m.predict_one_step(x, 3)[0])


def main():
    global SMAX
    SMAX = get_slice()
    policy, pidfile, dur, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    tasks = {}
    for line in open(pidfile):
        kind, pid = line.split()
        tasks[int(pid)] = {"kind": kind, "hist": [], "ewma": None, "acc": 0, "last": None, "k": Koopman(),
                           "pred": [], "slices": 0}
    if policy == "fixed":
        for pid in tasks:
            set_slice(pid, SMIN)
    if policy == "static":
        for pid, t in tasks.items():
            set_slice(pid, SMIN if t["kind"] == "req" else SMAX)
    t_end = time.monotonic() + dur
    t_cpu0 = time.process_time()
    calls = 0
    while time.monotonic() < t_end and policy in ("ewma", "koopman", "ewma_q", "koopman_q"):
        for pid, t in tasks.items():
            try:
                run, vol = read(pid)
            except (OSError, TypeError):
                continue
            if t["last"] is None:
                t["last"] = (run, vol)
                continue
            dr, dv = run - t["last"][0], vol - t["last"][1]
            t["last"] = (run, vol)
            t["acc"] += dr
            if dv > 0:
                b = t["acc"] / dv / 1e9
                t["acc"] = 0
                prev = t["ewma"]
                t["hist"].append(b)
                t["ewma"] = b if prev is None else 0.5 * b + 0.5 * prev
                if t["pred"] and t["pred"][-1] > 0:
                    t.setdefault("lr", []).append(np.log(max(b, 1e-7) / t["pred"][-1]))
                p = t["k"].predict(t["hist"], t["ewma"]) if policy.startswith("koopman") else t["ewma"]
                t["pred"].append(p)
                if policy.endswith("_q") and len(t.get("lr", [])) >= 20:
                    p = p * float(np.exp(np.quantile(t["lr"][-200:], 0.9)))
                set_slice(pid, p); calls += 1
            elif t["acc"] > SMAX * 1e9 and t.get("long") is not True:
                set_slice(pid, SMAX); t["long"] = True; calls += 1
            if dv > 0:
                t["long"] = False
        time.sleep(POLL)
    res = {"policy": policy, "agent_cpu_s": time.process_time() - t_cpu0, "default_slice_s": SMAX, "setattr_calls": calls,
           "bursts": {str(p): t["hist"] for p, t in tasks.items() if t["kind"] == "req"},
           "preds": {str(p): t["pred"] for p, t in tasks.items() if t["kind"] == "req"}}
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
