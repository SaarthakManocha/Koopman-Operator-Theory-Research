"""
run_closedloop.py - closed-loop test: does a burst predictor that sets EEVDF slices
lower request latency on a contended Linux machine?

    sudo python3 closedloop/run_closedloop.py --cpus 0 --hogs 3 --reqs 4 --reps 5 --secs 20 --out data/closedloop/c1
    sudo python3 closedloop/run_closedloop.py --cpus 0-1 --hogs 6 --reqs 8 --reps 5 --secs 20 --out data/closedloop/c2

Every repetition runs all policies with the same random seeds (paired design), in a
shuffled order. Ground truth per task comes from /proc/<pid>/schedstat, sampled
every 20 ms until the task exits.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = ["stock", "fixed", "static", "ewma", "koopman", "oracle", "ewma_q", "koopman_q", "oracle2"]


def ctxt():
    for line in open("/proc/stat"):
        if line.startswith("ctxt"):
            return int(line.split()[1])


def one_run(a, policy, rep, outdir):
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(1000 * rep + 7)
    procs, pids = [], []
    ts = ["taskset", "-c", a.cpus]
    scx = None
    if policy.startswith("scx_"):
        # scx_<none|ewma|koopman>_<margin %>: whole-machine sched_ext scheduler (kit/sched_ext)
        _, pred, margin = policy.split("_")
        exe = os.path.join(HERE, "..", "sched_ext", "build", "scx_koopman")
        if not os.path.exists(exe):
            exe = os.path.join(HERE, "..", "kit", "sched_ext", "build", "scx_koopman")
        scx = subprocess.Popen([exe, "-p", pred, "-m", margin, "-t", str(a.secs + 3), "-o", f"{outdir}/scx.txt"])
        time.sleep(1.5)
    for i in range(a.hogs):
        p = subprocess.Popen(ts + [os.path.join(HERE, "task"), "hog", str(a.secs), f"{outdir}/hog{i}.txt"])
        procs.append(p); pids.append(("hog", p.pid))
    for i in range(a.reqs):
        seed = rng.randrange(1, 10 ** 6)
        period = rng.choice([15, 20, 25])
        args = [os.path.join(HERE, "task"), "req", str(a.secs), f"{outdir}/req{i}.txt", str(seed), str(period),
                "0.3", "4", "0.05"] + ([policy] if policy.startswith("oracle") else [])
        p = subprocess.Popen(ts + args)
        procs.append(p); pids.append(("req", p.pid))
    with open(f"{outdir}/pids.txt", "w") as f:
        for k, pid in pids:
            f.write(f"{k} {pid}\n")
    gt, stop = {}, threading.Event()

    def sampler():
        while not stop.is_set():
            for k, pid in pids:
                try:
                    run, wait, sl = map(int, open(f"/proc/{pid}/schedstat").read().split())
                    gt[pid] = {"kind": k, "run_ns": run, "wait_ns": wait, "slices": sl}
                except OSError:
                    pass
            time.sleep(0.02)
    th = threading.Thread(target=sampler); th.start()
    c0 = ctxt()
    ag = None
    if policy.startswith("bpf_"):
        # bpf_<ewma|koopman>_<margin>: event-driven C agent with the eBPF burst observer
        _, pred, c = policy.split("_")
        ag = subprocess.Popen(ts + [os.path.join(HERE, "bpf", "cagent"), pred, c, f"{outdir}/pids.txt",
                                    str(a.secs + 0.5), f"{outdir}/agent.txt"])
        time.sleep(0.3)
    elif policy != "stock":
        ap = "static" if policy.startswith("oracle") else policy
        ag = subprocess.Popen(ts + [sys.executable, os.path.join(HERE, "agent.py"), ap, f"{outdir}/pids.txt",
                                    str(a.secs), f"{outdir}/agent.json"])
    for p in procs:
        p.wait()
    c1 = ctxt()
    if ag:
        ag.wait()
    stop.set(); th.join()
    if scx:
        scx.wait()
    json.dump({"policy": policy, "rep": rep, "cpus": a.cpus, "hogs": a.hogs, "reqs": a.reqs, "secs": a.secs,
               "ctxt": c1 - c0, "gt": gt}, open(f"{outdir}/meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpus", default="0"); ap.add_argument("--hogs", type=int, default=3)
    ap.add_argument("--reqs", type=int, default=4); ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--secs", type=float, default=20); ap.add_argument("--out", required=True)
    ap.add_argument("--policies", default=",".join(POLICIES))
    a = ap.parse_args()
    pol = a.policies.split(",")
    for rep in range(a.reps):
        order = pol[:]; random.Random(rep).shuffle(order)
        for policy in order:
            t = time.time()
            one_run(a, policy, rep, os.path.join(a.out, f"r{rep}_{policy}"))
            print(f"rep {rep} {policy:8s} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
