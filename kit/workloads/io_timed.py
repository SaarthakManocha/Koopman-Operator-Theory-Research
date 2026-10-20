"""I/O-bound job: write 10,000 bytes to a file, sleep 50 ms, repeat for argv[1] seconds.
Same loop body as io_long.py used in the lab Trace 3. Optional argv[2] = sleep in seconds
(the change experiments use 0.005 for an I/O burst)."""
import os, sys, tempfile, time
from _gt import dump_ground_truth

end = time.time() + float(sys.argv[1])
fname = os.path.join(tempfile.gettempdir(), f"tmp_io_{os.getpid()}.txt")
while time.time() < end:
    with open(fname, "w") as f:
        f.write("x" * 10000)
    time.sleep(float(sys.argv[2]) if len(sys.argv) > 2 else 0.05)
os.remove(fname)
dump_ground_truth("io")
