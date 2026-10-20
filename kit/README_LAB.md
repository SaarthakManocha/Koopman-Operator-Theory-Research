# Lab kit: the experiments that need the bare-metal lab machine

Everything else in the paper is done. These runs need a real (not virtual) Linux
machine with many cores, which is why they wait for the lab. Total: about 50 minutes.

Requirements: Linux 6.6 or newer, sudo, python3 with numpy and pandas
(`sudo apt install -y python3-numpy python3-pandas zip`). Close everything else.

## 1. Check (10 seconds)

    sudo python3 capture.py --check

Must print `READY TO CAPTURE` and `SELFTEST ... COMPLETE`. On the lab machine Trace 3
showed complete events, so this should pass. If it says INCOMPLETE, stop and report.

## 2. Traced captures (about 35 minutes)

    sudo python3 capture.py --plan lab

This runs, three times each:
- the same 8 jobs on 1, 2, 4, 8, 16 and all cores (`scale_*`)
- one CPU-bound job per core plus interactive jobs on 4, 8, 16 and all cores (`scalebusy_*`):
  contended scaling, the multi-core question at full size
- the whole machine overloaded (`oversub_*`) and a load drop half-way (`regime_*`)
- the clock-speed test (`freq_default_*`, `freq_performance_*`): the Trace 3 workload with the
  normal CPU governor and with the governor locked to `performance`. CPU frequency is logged
  every 100 ms. If the threefold I/O slow-down after the CPU jobs finish disappears with
  `performance`, frequency scaling caused it.

## 3. Live detector on bare metal (about 17 minutes, no sudo needed)

    python3 change_run.py --out captures/change_lab

24 runs with scripted load changes; `live_monitor.py` watches from /proc/stat.

## 4. Closed loop: predictor steering the real scheduler (about 50 minutes, sudo)

Needs kernel 6.12 or newer (per-task EEVDF slice requests). Check with `uname -r`.

    cd closedloop && gcc -O2 -o task task.c -lm
    sudo python3 run_closedloop.py --cpus 0 --hogs 3 --reqs 4 --reps 8 --secs 20 --out ../captures/closedloop_1cpu
    sudo python3 run_closedloop.py --cpus 0-7 --hogs 24 --reqs 32 --reps 6 --secs 20 --out ../captures/closedloop_8cpu
    python3 analyse_closedloop.py ../captures/closedloop_1cpu ../captures/closedloop_8cpu

Same test as on the VM, on bare metal: a user-space agent predicts each task's next CPU
burst (EWMA or Koopman) and sets its EEVDF slice; request latency, CPU-hog throughput and
the agent's own cost are recorded.

## 5. Real programs (about 4 minutes, sudo; needs redis-server, node, ffmpeg)

    sudo python3 real_programs.py --reps 3 --secs 60 --out captures/real_programs

## 6. Optional: scaling from counters only (5 minutes, no sudo)

    python3 scale_counters.py --out captures/scale_lab --cores 1 2 4 8 16

## 7. Bring it back

    zip -r lab_results.zip captures

`tools/validate.py captures/<run>` compares any traced run against the kernel's own counters.
