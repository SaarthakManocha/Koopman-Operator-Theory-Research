"""Positive control: a job whose burst lengths have real, known structure.
Every 2 s it switches phase:  short phase = compute 1 ms then sleep 5 ms,
                              long phase  = compute 20 ms then sleep 5 ms.
If a predictor cannot find structure here, it cannot find it anywhere.
Runs for argv[1] seconds."""
import sys, time
from _gt import dump_ground_truth

end = time.time() + float(sys.argv[1])
start = time.time()


def spin(cpu_seconds):
    # counts CPU time, not wall time, so the burst is 1 ms / 20 ms of real work
    # even when the job has to share its core with others
    t = time.process_time() + cpu_seconds
    while time.process_time() < t:
        pass


while time.time() < end:
    long_phase = int((time.time() - start) / 2.0) % 2 == 1
    spin(0.020 if long_phase else 0.001)
    time.sleep(0.005)
dump_ground_truth("phased")
