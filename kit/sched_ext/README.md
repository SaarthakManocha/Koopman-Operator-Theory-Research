# scx_koopman: a Koopman-predicting scheduler for sched_ext

Status: run on bare metal (Intel Core i7-7700, Ubuntu 24.04, kernel 7.0) on 27 Oct 2026: loaded and
unloaded cleanly in 30 of 30 runs; results in the paper (Section VII-G). If anything goes wrong, the kernel
unloads it and returns to EEVDF on its own (sched_ext watchdog), and Ctrl-C always restores EEVDF.

Needs libbpf >= 1.5. Ubuntu 24.04 ships 1.3, and `make` then fails with
"'struct bpf_map_skeleton' has no member named 'link'". Build libbpf 1.5 in your home folder and link it:

    git clone --depth 1 -b v1.5.0 https://github.com/libbpf/libbpf ~/libbpf15
    make -C ~/libbpf15/src -j8 BUILD_STATIC_ONLY=y OBJDIR=$HOME/libbpf15/obj DESTDIR=$HOME/libbpf15/inst install
    make build/scx_koopman.bpf.skel.h
    gcc -O2 -I$HOME/libbpf15/inst/usr/include -Ibuild -Iinclude -o build/scx_koopman scx_koopman.c \
        $(find ~/libbpf15/inst -name libbpf.a) -lelf -lz -lm

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
