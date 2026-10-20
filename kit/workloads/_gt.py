"""Ground truth: before exiting, each workload saves what the KERNEL says about it.
/proc/self/schedstat = total time on CPU (ns), total time waiting in run queue (ns), number of timeslices.
/proc/self/status    = voluntary / involuntary context switches.
These are counted by the kernel itself, independently of the trace, so they let us
check that the trace parser reconstructs bursts and waits correctly."""
import os


def dump_ground_truth(kind):
    out = os.environ.get("GT_DIR")
    if not out:
        return
    with open("/proc/self/schedstat") as f:
        run_ns, wait_ns, slices = f.read().split()[:3]
    vol = invol = "?"
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("voluntary_ctxt_switches"):
                vol = line.split()[1]
            elif line.startswith("nonvoluntary_ctxt_switches"):
                invol = line.split()[1]
    with open(os.path.join(out, f"gt_{os.getpid()}.txt"), "w") as f:
        f.write(f"pid {os.getpid()}\nkind {kind}\nrun_ns {run_ns}\nwait_ns {wait_ns}\n"
                f"timeslices {slices}\nvoluntary {vol}\ninvoluntary {invol}\n")
