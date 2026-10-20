"""
parse_trace.py
--------------
Reconstruct per-process CPU bursts from `perf sched script` text output.

A burst is the time between a process being switched IN on a CPU
(`... ==> comm:PID [prio]`) and the next time that same process is
switched OUT (`comm:PID [prio] <state> ==> ...`).

The state letter at switch-out tells us WHY the burst ended:
    R  = still runnable, so it was preempted (time slice used up / higher
         priority task woke up)
    S  = went to sleep voluntarily (e.g. time.sleep, waiting on an event)
    D  = uninterruptible sleep, usually waiting on disk / I/O
Anything else is kept as-is.

Usage:
    python src/parse_trace.py data/raw/trace3.txt evidence/pids.txt data/processed/bursts_trace3.csv

The trace file may also be gzip-compressed (.gz).
"""

import csv
import gzip
import re
import sys

# Example sched_switch line:
#  swapper  0 [019]  924.127283:  sched:sched_switch: swapper/19:0 [120] R ==> python3:9565 [120]
SWITCH_RE = re.compile(
    r"\[(?P<cpu>\d+)\]\s+(?P<ts>\d+\.\d+):\s+sched:sched_switch:\s+"
    r"(?P<prev_comm>.+?):(?P<prev_pid>\d+)\s+\[(?P<prev_prio>\d+)\]\s+(?P<prev_state>\S+)\s+==>\s+"
    r"(?P<next_comm>.+?):(?P<next_pid>\d+)\s+\[(?P<next_prio>\d+)\]"
)


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


def load_pids(path):
    """pids.txt has lines like 'cpu 9562' or 'io 9565'."""
    kinds = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                kinds[int(parts[1])] = parts[0]
    return kinds


def parse(trace_path, pid_kinds):
    """Return a list of bursts for the PIDs we care about, in time order."""
    running = {}   # pid -> (start_ts, cpu)
    bursts = []
    n_lines = n_switch = 0
    with open_text(trace_path) as f:
        for line in f:
            n_lines += 1
            if "sched_switch" not in line:
                continue
            m = SWITCH_RE.search(line)
            if not m:
                continue
            n_switch += 1
            ts = float(m["ts"])
            cpu = int(m["cpu"])
            prev_pid = int(m["prev_pid"])
            next_pid = int(m["next_pid"])

            # 1) the previous task leaves the CPU -> close its burst
            if prev_pid in pid_kinds and prev_pid in running:
                start, start_cpu = running.pop(prev_pid)
                bursts.append({
                    "pid": prev_pid,
                    "kind": pid_kinds[prev_pid],
                    "start": start,
                    "end": ts,
                    "burst_s": ts - start,
                    "cpu": start_cpu,
                    "end_state": m["prev_state"],
                })

            # 2) the next task gets the CPU -> open a new burst
            if next_pid in pid_kinds:
                running[next_pid] = (ts, cpu)

    bursts.sort(key=lambda b: (b["pid"], b["start"]))
    return bursts, n_lines, n_switch


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    trace_path, pids_path, out_csv = sys.argv[1:]
    kinds = load_pids(pids_path)
    bursts, n_lines, n_switch = parse(trace_path, kinds)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "kind", "start", "end", "burst_s", "cpu", "end_state"])
        w.writeheader()
        for b in bursts:
            w.writerow({**b, "start": f"{b['start']:.6f}", "end": f"{b['end']:.6f}",
                        "burst_s": f"{b['burst_s']:.6f}"})
    print(f"lines read          : {n_lines:,}")
    print(f"sched_switch events : {n_switch:,}")
    print(f"bursts written      : {len(bursts):,} -> {out_csv}")
    for pid, kind in sorted(kinds.items()):
        n = sum(1 for b in bursts if b["pid"] == pid)
        print(f"  {kind:>3} pid {pid}: {n} bursts")


if __name__ == "__main__":
    main()
