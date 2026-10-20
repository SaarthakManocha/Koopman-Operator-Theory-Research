"""
run_all.py  -  one command to reproduce every number and figure in the report.

    python run_all.py

Steps
  1. verify the raw trace against the SHA-256 recorded in the lab
  2. parse the trace into per-process CPU bursts
  3. descriptive statistics (how bursts end, who preempts whom)
  4. forecasting experiments  -> outputs/metrics.txt and CSVs
  5. figures                  -> outputs/figures/
"""

import gzip
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
PY = sys.executable
RAW_GZ = "data/raw/trace3.txt.gz"
RAW_TXT = "data/raw/trace3.txt"
EXPECTED_SHA = "7fd1c314b0545c2575f6804b14ec113b470201e0a63e264a094e885ee77cfe48"  # from evidence/evidence3.txt


def sha256_of_trace():
    h = hashlib.sha256()
    opener = (lambda: open(RAW_TXT, "rb")) if os.path.exists(RAW_TXT) else (lambda: gzip.open(RAW_GZ, "rb"))
    with opener() as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def step(title, args):
    print(f"\n=== {title} ===", flush=True)
    subprocess.run([PY] + args, check=True)


def main():
    trace = RAW_TXT if os.path.exists(RAW_TXT) else RAW_GZ
    print("=== 1. Integrity check ===")
    got = sha256_of_trace()
    print(f"SHA-256 of trace3.txt : {got}")
    print(f"recorded in the lab   : {EXPECTED_SHA}")
    print("MATCH" if got == EXPECTED_SHA else "MISMATCH - trace differs from the one captured in the lab!")

    os.makedirs("data/processed", exist_ok=True)
    step("2. Parse trace into bursts",
         ["src/parse_trace.py", trace, "evidence/pids.txt", "data/processed/bursts_trace3.csv"])
    step("3. Trace statistics",
         ["src/trace_stats.py", trace, "data/processed/bursts_trace3.csv", "evidence/pids.txt"])
    step("4. Forecasting experiments", ["src/run_forecast.py", "data/processed/bursts_trace3.csv"])
    step("5. Figures", ["src/plots.py"])
    print("\nDone. Open outputs/metrics.txt and outputs/figures/.")


if __name__ == "__main__":
    main()
