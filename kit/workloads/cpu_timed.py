"""CPU-bound job: pure computation for argv[1] seconds of wall-clock time."""
import sys, time
from _gt import dump_ground_truth

end = time.time() + float(sys.argv[1])
x = 0
while time.time() < end:
    for i in range(100_000):
        x += i
dump_ground_truth("cpu")
