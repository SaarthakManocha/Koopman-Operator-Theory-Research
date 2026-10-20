# scx_koopman: a Koopman-predicting scheduler for sched_ext

Status: the BPF program compiles (checked against a kernel 7.3 `vmlinux.h`); it has
**not been run**, because the machine it was written on has no sched_ext. Run it first
on the lab machine, as below. If anything goes wrong, the kernel unloads it and returns
to EEVDF on its own (sched_ext watchdog), and Ctrl-C always restores EEVDF.

What it does: a global virtual-time scheduler (like the kernel's `scx_simple`) that
orders runnable tasks by `virtual time + predicted next burst` and gives each a slice
of `margin x predicted burst`. Bursts are measured in BPF; the predictor (EWMA or the
log-scale Koopman operator, updated by recursive least squares) runs in user space.

## 1. Check the kernel supports it

    cat /sys/kernel/sched_ext/state        # file must exist ("disabled" is fine)
    grep SCHED_CLASS_EXT /boot/config-$(uname -r)   # =y

## 2. Build

    sudo apt install clang libbpf-dev linux-tools-$(uname -r) libelf-dev zlib1g-dev
    make

## 3. Smoke test (10 seconds)

    sudo ./build/scx_koopman -p koopman -t 10
    cat /sys/kernel/sched_ext/root/ops 2>/dev/null; dmesg | tail

## 4. Experiment (about 40 minutes)

From `kit/closedloop` (after `gcc -O2 -o task task.c -lm`):

    sudo python3 run_closedloop.py --cpus 0-7 --hogs 24 --reqs 32 --reps 6 --secs 20 \
        --policies stock,scx_none_150,scx_ewma_150,scx_koopman_150,scx_koopman_100,scx_koopman_200 \
        --out ../captures/scx_8cpu
    python3 analyse_closedloop.py ../captures/scx_8cpu

`stock` is EEVDF; `scx_none` is the same BPF scheduler without predictions (the fair
baseline for the scheduler itself); `scx_<predictor>_<margin%>` use predictions.
