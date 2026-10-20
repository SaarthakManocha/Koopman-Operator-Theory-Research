# Lab kit: the experiments that need the bare-metal lab machine

Everything else in the paper is done. These runs need a real (not virtual) Linux machine
with many cores. Do the steps in order. **Total: about 3 hours**, mostly waiting; steps
2-6 can each be started and left running.

Rules while it runs: close every other program (browser, IDE, music); do not use the
machine; keep it plugged in.

## 0. Install and check (10 minutes)

    sudo apt install -y python3-numpy python3-pandas python3-scipy zip gcc clang \
        libbpf-dev libelf-dev zlib1g-dev linux-tools-$(uname -r) \
        redis-server nodejs ffmpeg
    sudo systemctl stop redis-server     # the kit starts its own Redis on another port
    uname -r                             # note the kernel version
    nproc                                # note the number of CPUs
    cat /sys/kernel/sched_ext/state      # if this file exists, step 6 is possible

Write down `uname -r`, `nproc`, and whether the sched_ext file exists. Put the three
lines in a file `captures/machine_notes.txt`, which you need to create:

    mkdir -p captures && (uname -a; nproc; ls /sys/kernel/sched_ext 2>&1) > captures/machine_notes.txt

- If the kernel is older than **6.12**, skip steps 4, 5 and 6. They need per-task slice
  requests or sched_ext.
- If `apt` cannot find `linux-tools-$(uname -r)`, install `linux-tools-generic` instead.

## 1. Tracing self-test (10 seconds)

    sudo python3 capture.py --check

It must print `READY TO CAPTURE` and `SELFTEST ... COMPLETE`. If it says INCOMPLETE, stop
and send me the output.

## 2. Traced captures: scaling, overload, frequency (about 35 minutes)

    sudo python3 capture.py --plan lab

This runs, three times each:

- `scale_*`: the same 8 jobs on 1, 2, 4, 8, 16 and all cores;
- `scalebusy_*`: one CPU-bound job per core plus interactive jobs, on 4, 8, 16 and all
  cores. This is contended scaling, the multi-core question at full size;
- `oversub_*` (the whole machine overloaded) and `regime_*` (a load drop half-way);
- `freq_default_*` and `freq_performance_*`: the Trace 3 workload with the normal CPU
  governor, and with the governor locked to `performance`. The CPU frequency is logged.
  This tests whether power management caused the threefold I/O slowdown in Trace 3.

## 3. Live change detector (about 17 minutes, no sudo)

    python3 change_run.py --out captures/change_lab

## 4. Steering EEVDF with predicted slices (about 70 minutes, sudo, kernel >= 6.12)

This is the paper's main closed-loop result (Tables V-VI, Fig. 9 left), now on bare metal.

    cd closedloop
    gcc -O2 -o task task.c -lm
    cd bpf && make && cd ..              # builds the eBPF observer + agent (bpf/cagent)
    P=stock,oracle,oracle1.5,oracle2,bpf_ewma_1,bpf_ewma_1.5,bpf_ewma_2,bpf_koopman_1,bpf_koopman_1.5,bpf_koopman_2
    sudo python3 run_closedloop.py --cpus 0   --hogs 3  --reqs 4  --reps 6 --secs 20 --policies $P --out ../captures/closedloop_1cpu
    sudo python3 run_closedloop.py --cpus 0-7 --hogs 24 --reqs 32 --reps 6 --secs 20 --policies $P --out ../captures/closedloop_8cpu
    python3 analyse_closedloop.py ../captures/closedloop_1cpu ../captures/closedloop_8cpu
    cd ..

- If `make` fails, send me the error. You can still run the polling agent with
  `--policies stock,ewma,koopman,oracle,oracle2`, which needs no eBPF.
- If the machine has fewer than 8 CPUs, use `--cpus 0-3 --hogs 12 --reqs 16` for the second run.

## 5. Real programs (about 30 minutes, sudo)

5a. Capture the bursts of Redis, Node.js, SQLite and ffmpeg (prediction study, Fig. 5):

    sudo python3 real_programs.py --reps 3 --secs 60 --out captures/real_programs

5b. Steer the real servers (Table VII, Fig. 9 right). This needs the eBPF agent from step 4:

    cd closedloop
    sudo python3 real_closed.py --reps 10 --secs 20 --out ../captures/real_closed
    python3 analyse_real_closed.py ../captures/real_closed
    cd ..

## 6. Koopman scheduler in the kernel (sched_ext), only if the file in step 0 exists (about 45 minutes)

Follow `sched_ext/README.md`: build with `make` and do the 10-second smoke test first. If
the smoke test works, run the experiment in that README. It writes to
`captures/scx_8cpu`. If anything goes wrong, Ctrl-C returns the machine to EEVDF. If the
smoke test fails, send me its output and the last lines of `dmesg`.

## 6b. Optional: scaling from counters only (5 minutes, no sudo)

    python3 scale_counters.py --out captures/scale_lab --cores 1 2 4 8 16

## 7. Bring it back

    zip -r lab_results.zip captures

Send `lab_results.zip`, plus any error output. `tools/validate.py captures/<run>` checks any
traced run against the kernel's own counters, if you want to look yourself.

## Checklist

- [ ] 0 install, `machine_notes.txt`
- [ ] 1 self-test passes
- [ ] 2 `capture.py --plan lab`
- [ ] 3 `change_run.py`
- [ ] 4 closed loop, 1 CPU and 8 CPUs (kernel >= 6.12)
- [ ] 5a real-program captures
- [ ] 5b real-server steering (kernel >= 6.12)
- [ ] 6 sched_ext (if supported)
- [ ] 7 `lab_results.zip`
