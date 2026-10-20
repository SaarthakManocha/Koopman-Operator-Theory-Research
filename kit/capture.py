#!/usr/bin/env python3
"""
capture.py - record real Linux scheduler events while running known workloads.

Uses the kernel's tracefs interface directly (no `perf` needed), so it works on
any Linux with tracing enabled, including WSL2.

Every capture records, besides the trace itself, three kinds of ground truth so
the analysis can be checked against the kernel's own accounting:
  * per-process /proc/<pid>/schedstat and context-switch counts (saved by each workload at exit)
  * /proc/stat per-CPU time counters sampled every 100 ms (CPU busy time, steal time)
  * per-CPU ring-buffer statistics (proves no events were lost)

Usage (as root):
    sudo python3 capture.py --check                       # is this machine able to capture?
    sudo python3 capture.py --plan default                # run the standard experiment plan
    sudo python3 capture.py --name test --cpus 0 --cpu-jobs 1 --io-jobs 1 --phased-jobs 1 --secs 5
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WL = os.path.join(HERE, "workloads")
T = "/sys/kernel/tracing"
EVENTS = ["sched_switch", "sched_waking", "sched_wakeup", "sched_wakeup_new",
          "sched_migrate_task", "sched_process_fork", "sched_process_exec", "sched_process_exit"]


def w(path, value):
    with open(path, "w") as f:
        f.write(str(value))


def r(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


# ---------------------------------------------------------------- checks
def ensure_tracefs():
    if not os.path.exists(os.path.join(T, "events")):
        os.makedirs(T, exist_ok=True)
        subprocess.run(["mount", "-t", "tracefs", "nodev", T], check=False)
    return os.path.exists(os.path.join(T, "events", "sched", "sched_switch"))


def check(verbose=True):
    ok = True
    msgs = []
    if os.geteuid() != 0:
        msgs.append("FAIL  not running as root (use sudo)")
        ok = False
    elif not ensure_tracefs():
        msgs.append("FAIL  tracefs / sched tracepoints not available on this kernel")
        ok = False
    else:
        missing = [e for e in EVENTS if not os.path.exists(f"{T}/events/sched/{e}")]
        msgs.append("OK    tracefs mounted, sched tracepoints present" +
                    (f" (missing optional: {missing})" if missing else ""))
    msgs.append(f"INFO  kernel {platform.release()}, {os.cpu_count()} CPUs, "
                f"{'WSL' if 'microsoft' in platform.release().lower() else 'native/VM'}")
    hv = "hypervisor" in r("/proc/cpuinfo")
    msgs.append(f"INFO  running under a hypervisor: {hv}")
    msgs.append(f"INFO  cpufreq available: {os.path.exists('/sys/devices/system/cpu/cpu0/cpufreq')}")
    if verbose:
        print("\n".join(msgs))
        print("READY TO CAPTURE" if ok else "CANNOT CAPTURE - send this output back")
    return ok


def selftest():
    """Does this kernel emit a sched_switch for EVERY switch, including when a core leaves idle?
    (Found during validation: some VM guest kernels drop part of the idle-exit switches.)"""
    ensure_tracefs()
    w(f"{T}/tracing_on", 0)
    w(f"{T}/trace", "")
    w(f"{T}/events/enable", 0)
    w(f"{T}/events/sched/sched_switch/enable", 1)
    w(f"{T}/tracing_on", 1)
    p = subprocess.Popen([sys.executable, "-c", "import time\nfor _ in range(200): time.sleep(0.01)"])
    p.wait()
    w(f"{T}/tracing_on", 0)
    txt = r(f"{T}/trace")
    w(f"{T}/events/enable", 0)
    outs = txt.count(f"prev_pid={p.pid} ")
    ins = txt.count(f"next_pid={p.pid} ")
    ok = ins == outs
    print(f"SELFTEST  a sleeping task was switched out {outs} times and in {ins} times -> "
          + ("COMPLETE: idle exits are traced" if ok else
             f"INCOMPLETE: {outs - ins} idle-exit switches missing; use only runs that keep every core busy"))
    return ok


# ---------------------------------------------------------------- samplers
def proc_stat_sampler(path, stop):
    """Every 100 ms: monotonic timestamp + per-CPU jiffies (user nice system idle iowait irq softirq steal)."""
    with open(path, "w") as out:
        while not stop.is_set():
            ts = time.clock_gettime(time.CLOCK_MONOTONIC)
            lines = [l for l in r("/proc/stat").splitlines() if l.startswith("cpu") and not l.startswith("cpu ")]
            out.write(f"{ts:.6f} " + " | ".join(lines) + "\n")
            freqs = [r(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq").strip() for i in range(os.cpu_count())]
            if any(freqs):
                out.write(f"{ts:.6f} FREQ " + " ".join(freqs) + "\n")
            stop.wait(0.1)


def ring_stats():
    out = {}
    for cpu in sorted(os.listdir(f"{T}/per_cpu")):
        st = {}
        for line in r(f"{T}/per_cpu/{cpu}/stats").splitlines():
            k, _, v = line.partition(":")
            st[k.strip()] = v.strip()
        out[cpu] = st
    return out


# ---------------------------------------------------------------- one capture
def set_governor(gov):
    """Set every CPU's cpufreq governor; returns the previous one (or None if cpufreq is absent)."""
    paths = [f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor" for i in range(os.cpu_count())]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return None
    old = r(paths[0]).strip()
    for p in paths:
        try:
            w(p, gov)
        except OSError:
            pass
    return old


def capture(name, cpus, n_cpu, n_io, n_phased, secs, cpu_secs=None, outroot=None, extra_cpu=None, governor=None):
    outroot = outroot or os.path.join(HERE, "captures")
    d = os.path.join(outroot, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    gt_dir = os.path.join(d, "ground_truth")
    os.makedirs(gt_dir)
    cpu_secs = cpu_secs or secs
    print(f"\n=== {name}: cpus={cpus or 'all'}  cpu={n_cpu} io={n_io} phased={n_phased}  {secs}s ===", flush=True)

    old_gov = set_governor(governor) if governor else None
    # --- configure tracing
    w(f"{T}/tracing_on", 0)
    w(f"{T}/trace", "")
    w(f"{T}/trace_clock", "mono")          # same clock as time.CLOCK_MONOTONIC -> aligned with samplers
    # Big ring buffer, read only AFTER the run. Streaming the trace while recording makes the
    # reader process wake on every event and perturb the very scheduling we measure
    # (found during validation: a ~50k switches/s feedback loop on idle cores).
    avail_kb = int([l for l in r("/proc/meminfo").splitlines() if l.startswith("MemAvailable")][0].split()[1])
    per_cpu_kb = int(min(262144, max(16384, avail_kb * 0.30 / os.cpu_count())))
    w(f"{T}/buffer_size_kb", per_cpu_kb)
    w(f"{T}/events/enable", 0)
    for e in EVENTS:
        if os.path.exists(f"{T}/events/sched/{e}"):
            w(f"{T}/events/sched/{e}/enable", 1)

    stop = threading.Event()
    sampler = threading.Thread(target=proc_stat_sampler, args=(os.path.join(d, "procstat.txt"), stop))
    sampler.start()

    env = dict(os.environ, GT_DIR=gt_dir, PYTHONDONTWRITEBYTECODE="1")
    affinity = set(cpus) if cpus else set(range(os.cpu_count()))
    jobs = ([("cpu", "cpu_timed.py", cpu_secs)] * n_cpu + [("io", "io_timed.py", secs)] * n_io
            + [("phased", "phased.py", secs)] * n_phased)
    if extra_cpu:                                   # (count, seconds): CPU jobs with a different length
        jobs = [("cpu", "cpu_timed.py", extra_cpu[1])] * extra_cpu[0] + jobs

    t_start = time.clock_gettime(time.CLOCK_MONOTONIC)
    w(f"{T}/tracing_on", 1)
    time.sleep(0.2)
    procs = []
    for kind, script, s in jobs:
        p = subprocess.Popen([sys.executable, os.path.join(WL, script), str(s)], env=env, cwd=WL,
                             preexec_fn=lambda: os.sched_setaffinity(0, affinity))
        procs.append((kind, p))
    with open(os.path.join(d, "pids.txt"), "w") as f:
        for kind, p in procs:
            f.write(f"{kind} {p.pid}\n")
    rc = [p.wait() for _, p in procs]
    time.sleep(0.3)
    w(f"{T}/tracing_on", 0)
    t_end = time.clock_gettime(time.CLOCK_MONOTONIC)

    # read the whole buffer once, after tracing has stopped
    with open(f"{T}/trace") as fi, open(os.path.join(d, "trace.txt"), "w") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    stop.set()
    sampler.join()
    stats = ring_stats()
    w(f"{T}/events/enable", 0)
    if governor and old_gov:
        set_governor(old_gov)

    lost = sum(int(s.get("overrun", 0)) + int(s.get("dropped events", 0) or 0) for s in stats.values())
    meta = {
        "buffer_kb_per_cpu": per_cpu_kb, "reader": "post-run read of tracefs 'trace' (no streaming)",
        "name": name, "kernel": platform.release(), "ncpu": os.cpu_count(),
        "cpus_allowed": sorted(affinity), "jobs": {"cpu": n_cpu, "io": n_io, "phased": n_phased},
        "secs": secs, "cpu_secs": cpu_secs, "extra_cpu": extra_cpu, "governor": governor or "unchanged", "t_start_mono": t_start, "t_end_mono": t_end,
        "exit_codes": rc, "lost_events": lost, "ring_stats": stats,
        "hypervisor": "hypervisor" in r("/proc/cpuinfo"),
        "wsl": "microsoft" in platform.release().lower(),
        "trace_clock": "mono", "captured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    # compress + hash
    src = os.path.join(d, "trace.txt")
    h = hashlib.sha256()
    with open(src, "rb") as fi, gzip.open(src + ".gz", "wb", compresslevel=6) as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            h.update(chunk)
            fo.write(chunk)
    os.remove(src)
    meta["sha256_trace_txt"] = h.hexdigest()
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    n_gt = len([x for x in os.listdir(gt_dir) if x.startswith("gt_")])
    status = "OK" if (lost == 0 and n_gt == len(procs) and all(c == 0 for c in rc)) else "CHECK"
    print(f"    {status}: lost events={lost}, ground-truth files={n_gt}/{len(procs)}, "
          f"exit codes={set(rc)}, trace.gz={os.path.getsize(src + '.gz') / 1e6:.1f} MB", flush=True)
    return status


# ---------------------------------------------------------------- experiment plans
def default_plan(reps=3, idle_ok=True):
    """
    scale_cNN : the same 8 jobs (3 CPU-bound, 3 I/O-bound, 2 phased) on 1, 2, 4, ... cores
    oversub   : 2 CPU-bound jobs per core plus interactive jobs, all cores
    regime    : load drops half-way (half the CPU-bound jobs stop at 20 s of 40 s) while every
                core stays busy, so the change can be detected without relying on idle-exit events
    If the self-test showed missing idle-exit events (idle_ok False), scaling runs above 3 cores
    use one CPU-bound job per core (scalebusy_cNN) so that no core is ever idle.
    """
    n = os.cpu_count()
    plan = []
    core_counts = sorted({c for c in (1, 2, 4, 8, 16, 32) if c <= n} | {n})
    for rep in range(1, reps + 1):
        for c in core_counts:
            if idle_ok or c <= 3:
                # same 8 jobs on more and more cores (cores beyond 3 may sit idle)
                plan.append(dict(name=f"scale_c{c:02d}_r{rep}", cpus=list(range(c)), n_cpu=3, n_io=3, n_phased=2, secs=30))
            else:
                # kernel drops idle-exit events here: one CPU-bound job per core keeps every core busy
                plan.append(dict(name=f"scalebusy_c{c:02d}_r{rep}", cpus=list(range(c)), n_cpu=c, n_io=3, n_phased=2, secs=30))
        plan.append(dict(name=f"oversub_r{rep}", cpus=None, n_cpu=2 * n, n_io=max(2, n // 2), n_phased=2, secs=30))
        plan.append(dict(name=f"regime_r{rep}", cpus=None, n_cpu=n, n_io=3, n_phased=2, secs=40, cpu_secs=40,
                         extra_cpu=(2 * n, 20)))
    return plan


def lab_plan(reps=3):
    """Everything still pending that needs the bare-metal lab machine:
    the default plan (contended scaling up to all cores, overload, load drop) plus the
    clock-speed test: the Trace 3 workload (3 CPU-bound jobs for 40 s, I/O and phased jobs
    for 70 s, all cores) with the default governor and with the 'performance' governor.
    If the threefold I/O slow-down after the CPU jobs exit disappears with 'performance',
    CPU frequency scaling caused it. CPU frequency is sampled every 100 ms (procstat.txt)."""
    plan = default_plan(reps, True)
    n = os.cpu_count()
    for rep in range(1, reps + 1):
        for c in sorted({c for c in (4, 8, 16) if c <= n} | {n}):
            # contended scaling: one CPU-bound job per core keeps every core busy, plus the interactive jobs
            plan.append(dict(name=f"scalebusy_c{c:02d}_r{rep}", cpus=list(range(c)), n_cpu=c, n_io=3, n_phased=2, secs=30))
        plan.append(dict(name=f"freq_default_r{rep}", cpus=None, n_cpu=3, n_io=3, n_phased=2, secs=70, cpu_secs=40))
        plan.append(dict(name=f"freq_performance_r{rep}", cpus=None, n_cpu=3, n_io=3, n_phased=2, secs=70,
                         cpu_secs=40, governor="performance"))
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--plan", choices=["default", "quick", "lab"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--name")
    ap.add_argument("--cpus", default="")
    ap.add_argument("--cpu-jobs", type=int, default=3)
    ap.add_argument("--io-jobs", type=int, default=3)
    ap.add_argument("--phased-jobs", type=int, default=2)
    ap.add_argument("--secs", type=float, default=30)
    ap.add_argument("--cpu-secs", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()

    if not check(verbose=True):
        sys.exit(1)
    idle_ok = selftest()
    if a.check or a.selftest:
        return
    out = a.out or os.path.join(HERE, "captures")
    os.makedirs(out, exist_ok=True)
    # machine description
    with open(os.path.join(out, "machine.txt"), "w") as f:
        f.write(subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout)
        f.write(subprocess.run(["lscpu"], capture_output=True, text=True).stdout)
        f.write(subprocess.run(["lscpu", "-e"], capture_output=True, text=True).stdout)

    if a.plan:
        if a.plan == "lab":
            plan = lab_plan(a.reps) if idle_ok else default_plan(a.reps, idle_ok)
        elif a.plan == "default":
            plan = default_plan(a.reps, idle_ok)
        else:
            plan = [dict(name="quick", cpus=[0], n_cpu=1, n_io=1, n_phased=1, secs=5)]
        print(f"Plan: {len(plan)} captures, about {sum(p['secs'] for p in plan) / 60 + len(plan) * 0.3:.0f} minutes")
        with open(os.path.join(out, "selftest.txt"), "w") as f:
            f.write(f"idle_exit_events_complete {idle_ok}\n")
        results = [capture(outroot=out, **p) for p in plan]
        print(f"\nFinished: {results.count('OK')}/{len(results)} OK")
    else:
        cpus = [int(c) for c in a.cpus.split(",")] if a.cpus else None
        capture(a.name or "manual", cpus, a.cpu_jobs, a.io_jobs, a.phased_jobs, a.secs, a.cpu_secs, out)


if __name__ == "__main__":
    main()
