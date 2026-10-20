"""
export_viz.py
-------------
Packs parsed traces + Koopman analysis into one compact JSON per capture for the
scheduler visualizer (viz/). Every number shown in the visualizer comes from here.

    python src/export_viz.py

Times are integer microseconds from the start of the capture.
"""
import base64
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from koopman_system import LinearKoopman, aggregate_state, machine_state, smooth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TL = os.path.join(ROOT, "outputs", "multicore", "timelines")
OUT = os.path.join(ROOT, "viz", "data")

STATE = {"R": 0, "S": 1, "D": 2}
CAUSE = {"wake": 0, "preempt": 1, "new": 2}

CHOICES = [
    ("lab_trace3", "Lab machine, 28 cores",
     "Bare-metal HP Pro Tower, kernel 7.0, 3 CPU-bound and 3 I/O-bound jobs. Cores are mostly idle; "
     "watch the CPU-bound jobs hop between cores and the I/O jobs slow down after the CPU jobs exit at ~47 s."),
    ("scale_c01_r1", "1 core, 8 jobs",
     "Eight jobs share a single core. Every wake-up has to queue, so this is where scheduling order matters."),
    ("scale_c02_r1", "2 cores, 8 jobs",
     "The same eight jobs on two cores. Queues are shorter and jobs migrate between the two cores."),
    ("oversub_r1", "2 cores, overloaded",
     "Four CPU-bound jobs plus interactive jobs on two cores: the queues never empty."),
    ("regime_r1", "2 cores, load drops at 20 s",
     "Six CPU-bound jobs for 20 s, then two. The Koopman operator was fitted on the first 8 s; watch its "
     "surprise score when the load changes."),
]


def ints(a):
    return [int(x) for x in a]


def b64_u8(a):
    return base64.b64encode(np.asarray(a, dtype=np.uint8).tobytes()).decode()


def merged_other(cpu_busy, tracked, cpus, t0, min_gap=20e-6):
    """Occupancy of all OTHER tasks per core, merged across tiny gaps to keep the file small."""
    out = {}
    ob = cpu_busy[~cpu_busy.pid.isin(tracked)]
    for c in cpus:
        g = ob[ob.cpu == c].sort_values("start")
        s, e = g.start.values, g.end.values
        if len(s) == 0:
            out[str(c)] = []
            continue
        ms, me = [s[0]], [e[0]]
        for a, b in zip(s[1:], e[1:]):
            if a - me[-1] < min_gap:
                me[-1] = max(me[-1], b)
            else:
                ms.append(a)
                me.append(b)
        ms, me = np.array(ms), np.array(me)
        st = np.round((ms - t0) * 1e6).astype(np.int64)
        du = np.maximum(np.round((me - ms) * 1e6).astype(np.int64), 1)
        flat = np.empty(2 * len(st), dtype=np.int64)
        flat[0::2] = np.diff(np.r_[0, st])          # delta-encoded starts
        flat[1::2] = du
        out[str(c)] = ints(flat)
    return out


def koopman_track(tl, dt=0.010, train_s=8.0, win_s=2.0, step_s=0.1):
    # machine-wide state (busy cores, queued tasks, CPU share per job type): stable on short
    # windows even for 28 cores, where a 52-variable operator overfits a 2 s window
    S = aggregate_state(machine_state(tl, dt=dt))
    X = S.values
    times = S.index.values - tl["t0"]
    tr = times < times[0] + train_s
    km = LinearKoopman().fit(X[tr])
    surprise = smooth(km.surprise(X), 25)
    # sliding-window eigenvalues
    win, step = int(win_s / dt), int(step_s / dt)
    eig_t, eig = [], []
    for a in range(0, len(X) - win + 1, step):
        try:
            m = LinearKoopman().fit(X[a:a + win])
            lam, _ = m.eig()
        except (np.linalg.LinAlgError, ValueError):
            continue
        lam = lam[:8]
        eig_t.append(round(float(times[a + win - 1]), 3))
        eig.append([[round(float(z.real), 4), round(float(z.imag), 4)] for z in lam])
    lam_tr, V = km.eig()
    return {
        "dt": dt, "train_s": train_s,
        "surprise_t0": round(float(times[1]), 4),
        "surprise": [round(float(v), 2) for v in surprise],
        "eig_t": eig_t, "eig": eig,
        "rho_train": round(float(np.abs(lam_tr).max()), 4),
        "state_dims": int(km.keep.sum()),
        "state_cols": [c for c, k in zip(S.columns, km.keep) if k],
    }


def marks(name, tl, t0):
    """Annotated moments for the timeline: when CPU-bound jobs stopped."""
    run = tl["run"]
    last = run.groupby("pid").end.max()
    out = []
    if name == "lab_trace3":
        for i, (p, e) in enumerate(sorted(last[run[run.kind == "cpu"].pid.unique()].items(), key=lambda x: x[1])):
            out.append({"t": round(e - t0, 3), "label": f"CPU-bound job {i + 1} exits"})
    extra = tl["meta"].get("extra_cpu")
    if extra:
        pids = [int(l.split()[1]) for l in open(os.path.join(ROOT, "data", "captures_vm", name, "pids.txt"))][:extra[0]]
        out.append({"t": round(last[pids].max() - t0, 3), "label": f"{extra[0]} of {extra[0] + 2} CPU-bound jobs stop"})
    return out


def export(name, label, desc):
    tl = pickle.load(open(os.path.join(TL, name + ".pkl"), "rb"))
    mp = os.path.join(ROOT, "data", "captures_vm", name, "meta.json")
    tl["meta"] = json.load(open(mp)) if os.path.exists(mp) else {"cpus_allowed": list(range(28))}
    t0, t1 = tl["t0"], tl["t1"]
    run, wait, mig = tl["run"], tl["wait"], tl["migrations"]
    pids = sorted(run.pid.unique(), key=lambda p: ({"cpu": 0, "phased": 1, "io": 2}.get(run[run.pid == p].kind.iloc[0], 3), p))
    idx = {p: i for i, p in enumerate(pids)}
    kinds = {p: run[run.pid == p].kind.iloc[0] for p in pids}
    tasks = []
    counts = {}
    for p in pids:
        k = kinds[p]
        counts[k] = counts.get(k, 0) + 1
        tasks.append({"pid": int(p), "kind": k, "label": f"{ {'cpu': 'CPU', 'io': 'I/O', 'phased': 'Phased'}[k]} {counts[k]}"})

    r = run.sort_values("start")
    runs = np.column_stack([r.pid.map(idx).values, r.cpu.values, np.round((r.start - t0) * 1e6),
                            np.maximum(np.round(r.dur * 1e6), 1), r.end_state.map(lambda s: STATE.get(s, 1)).values])
    w = wait.sort_values("start")
    waits = np.column_stack([w.pid.map(idx).values, w.cpu.values, np.round((w.start - t0) * 1e6),
                             np.maximum(np.round(w.dur * 1e6), 1), w.cause.map(CAUSE).values])
    m = mig.sort_values("ts")
    migs = np.column_stack([m.pid.map(idx).values, np.round((m.ts - t0) * 1e6), m.orig.values, m.dest.values])

    cpus = sorted(set(int(c) for c in tl["meta"].get("cpus_allowed", tl["cpus"])) | set(int(c) for c in r.cpu.unique()))
    if name == "lab_trace3":
        cpus = list(range(28))
    data = {
        "name": name, "label": label, "desc": desc,
        "duration_us": int(round((t1 - t0) * 1e6)), "cpus": cpus, "tasks": tasks,
        "run": ints(runs.ravel()), "wait": ints(waits.ravel()), "mig": ints(migs.ravel()),
        "other": merged_other(tl["cpu_busy"], set(pids), cpus, t0),
        "koopman": koopman_track(tl),
        "marks": marks(name, tl, t0),
        "jobs_cpus": sorted(int(c) for c in tl["meta"].get("cpus_allowed", cpus)),
        "stats": {"switches": int(len(r)), "migrations": int(len(m)),
                  "median_wait_us": float(np.median(w.dur) * 1e6) if len(w) else 0.0},
    }
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{name}: {os.path.getsize(path) / 1e6:.2f} MB, {len(r)} runs, {len(w)} waits, {len(m)} migrations")
    return path


if __name__ == "__main__":
    for c in CHOICES:
        export(*c)
