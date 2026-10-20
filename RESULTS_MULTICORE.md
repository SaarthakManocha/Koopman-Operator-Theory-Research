# Multi-core extension, visualisation, live detection and markets: methods, results, validation

This document records the work done after the case-study presentation, in
answer to two requests:

1. How does the study apply to a **multi-core** processor and the Linux scheduler?
2. Can the scheduler's core decisions (which process runs where, how processes
   enter and leave run queues, migrations), and the Koopman view of them, be
   **visualised**?

Every number below is produced by code in `src/` from data in `data/`, and can
be regenerated with the commands at the end.

---

## 1. Data

| Source | Machine | Kernel / scheduler | Captured with | Runs |
|---|---|---|---|---|
| Lab Trace 3 | HP Pro Tower 280 G9, 28 logical CPUs, bare metal | 7.0.0-30, EEVDF | `perf sched record` | 1 (75.5 s) |
| VM captures | Intel Xeon, 2 vCPUs, KVM guest | 6.18.44, EEVDF | `kit/capture.py` (tracefs) | 12 (4 set-ups x 3 repetitions, 30-40 s each) |

VM set-ups (8 workload processes unless stated):

| Set-up | Cores allowed | Jobs | Purpose |
|---|---|---|---|
| `scale_c01` | 1 | 3 CPU-bound, 3 I/O-bound, 2 phased | maximum contention |
| `scale_c02` | 2 | same | same jobs, more cores |
| `oversub` | 2 | 4 CPU-bound, 2 I/O-bound, 2 phased | queues never empty |
| `regime` | 2 | 6 CPU-bound (4 stop at 20 s), 3 I/O, 2 phased | known load change for Koopman detection |

Workloads (`kit/workloads/`): `cpu_timed.py` (pure computation), `io_timed.py`
(write 10 kB, sleep 50 ms; same loop as the lab's `io_long.py`), and
`phased.py`, a **positive control** whose bursts switch every 2 s between
1 ms and 20 ms of CPU time, so its burst lengths have known structure.

## 2. Validation (is the data right?)

Each workload saves the kernel's own accounting at exit
(`/proc/self/schedstat`: CPU time, run-queue wait time, times scheduled;
`/proc/self/status`: voluntary/involuntary switches). The trace parser
(`src/sched_events.py`) rebuilds the same quantities independently from the
trace. Over 105 processes in 12 captures (`outputs/multicore/validation.csv`, figure `m1_validation.png`):

| Quantity | Agreement with kernel |
|---|---|
| CPU time | median 0.93% absolute difference, max 3.45% |
| Run-queue wait time | median 0.22%, max 2.07% |
| Times scheduled | median 4 switches out of thousands (max 0.57% relative) |
| Lost trace events | 0 in every capture |

The small excess of CPU time in the trace comes from the process's exit path
(after the counters are saved) and from interrupt time, which the kernel's
CPU-time accounting excludes but trace intervals include.

### Two problems found by validation and fixed

1. **Observer effect.** The first version of the capture tool streamed events
   out of the kernel while recording. Each event woke the reader process, which
   preempted the workload and produced more events: a feedback loop of about
   50,000 context switches per second on any core with spare capacity. Traces
   were 15-23 MB instead of 0.5-0.8 MB, and counts disagreed with the kernel by
   up to 9%. Fix: record into a large kernel buffer and read it only after the
   run. One contaminated capture is kept as evidence in
   `data/observer_effect_example/`.
2. **Missing idle-exit events on the VM.** On this virtual machine's kernel, part
   of the `sched_switch` events for a core leaving idle are never emitted (both
   tracefs and perf show it: a task that slept 201 times was seen switching back
   in only 1 to 176 times across seven self-tests). It does not occur on
   the lab machine: Trace 3 has zero unmatched switch-outs. Fix: the parser
   counts unmatched switch-outs, the capture tool runs a self-test, and VM
   experiments keep every core busy. With busy cores the counts match the
   kernel (on one core, exactly in 54% of processes and within 4 in all).

## 3. Results

### 3.1 How the scheduler behaves with few and many cores (`outputs/multicore/latency.csv`)

| Set-up | Cores | Load on allowed cores | I/O wake-up to run, median / p95 | Phased wake-up to run, median / p95 |
|---|---|---|---|---|
| Lab Trace 3 | 28 | 8% | 0.016 / 0.066 ms | (none) |
| `scale_c01` | 1 | 98% | 0.069 / 2.6 ms | 2.6 / 7.9 ms |
| `scale_c02` | 2 | 98% | 0.010 / 1.4 ms | 0.010 / 3.9 ms |
| `oversub` | 2 | 98% | 0.009 / 1.9 ms | 1.7 / 4.0 ms |

On the 28-core lab machine, only 2.3 of 28 cores were busy on average and
processes waited about 16 microseconds for a CPU. No real queue ever formed,
so the order in which a scheduler picks jobs could not matter there. Queues
form only when runnable work exceeds the cores.

All 769 preemptions of the lab's CPU-bound processes were by kernel threads
(55% `kworker`, 45% `migration/N`), never by the other workload processes.

### 3.2 Does better burst prediction help a multi-core scheduler? (`simulation.csv`, figure `m2`)

Trace-driven simulator (`src/simulate.py`): each process is replayed as a closed
loop of its real CPU bursts and real blocking times, on m identical cores with
one shared queue. Policies: FCFS, round robin (3 ms), non-preemptive SJF and
preemptive SRTF, each with a predictor (running mean, textbook EWMA, EDMD, or
perfect knowledge). Real Linux (EEVDF) is measured on the same trace.

Findings:

- **Core count against runnable work decides whether scheduling order matters at all.**
  Replaying the lab's 6 jobs, I/O response time is zero for every policy from
  4 cores up (3 CPU-bound jobs + 1). Below that, FCFS and SJF make I/O jobs
  wait tens of milliseconds on average.
- **Preemption matters more than prediction.** Non-preemptive SJF performs like
  FCFS whatever the predictor, because an I/O job that arrives while a CPU-bound
  job holds the core must wait for it. On 2 cores (`scale_c02`): mean I/O
  response FCFS 10.4 ms, SJF+EWMA 10.2 ms, SJF+perfect knowledge 10.2 ms.
- **With preemption, the predictor barely matters.** SRTF with the mean, EWMA or
  EDMD gives 0.011 ms on 2 cores; the gain comes from telling short jobs from
  long ones, which every predictor does.
- **Real Linux already behaves like predicted SRTF for interactive jobs.** On one
  core, measured EEVDF mean I/O and phased response 1.87 ms; simulated SRTF with
  predictors 1.5-1.9 ms; round robin 6.8 ms; FCFS 16.4 ms.

Caveats: the simulator has no context-switch or migration cost and one shared
queue (Linux has one queue per core plus load balancing); averages are driven by
a few long waits, so p95 and p99 are reported alongside.

### 3.3 Burst prediction under contention (`prediction.csv`, figure `m4`)

Textbook CPU bursts (CPU time between two blocking events), per process, last
20% of each process as test, 95% block-bootstrap intervals.

| Process type | Processes | Best methods vs mean guess | EDMD family vs textbook EWMA |
|---|---|---|---|
| Phased (positive control) | 24 | every recent-burst method: 90-96% lower error, all 24 significant | EDMD variants 5-23% **worse** than EWMA (median ratios 1.05-1.23; sign test, Holm-corrected); tuned EWMA better than EWMA in 24 of 24 |
| I/O under contention | 33 | gradient boosting 17% lower (29 of 33 significant) | **log EDMD better than EWMA in 33 of 33 processes, median 14% lower error** (Holm p = 2e-9); poly EDMD 32 of 33; gradient boosting 33 of 33 |

(Per-process intervals: log EDMD's 95% interval lies wholly below EWMA's in 30 of 33 processes.
Population-level test: one sign test per method, Holm-corrected over ten comparisons,
`outputs/robustness/prediction_sign_tests.csv`.)

The positive control shows the methods find structure when it exists. Under
contention, I/O bursts are noisy and the textbook EWMA overreacts; a log-scale
EDMD forecast is reliably better. The lab's uncontended I/O bursts (Trace 3) did
not show this; there the change in burst level dominated.

### 3.4 Koopman operator on the whole machine (`koopman_detection.csv`, figure `m3`)

State every 10 ms: busy fraction and queued tasks per core plus the CPU share of
each job type (7 variables on 2 cores, 52 on the lab machine), standardised; one
operator K fitted by least squares on the first 8 s, exactly as in the DRISHTI
network-traffic paper. Detector: trailing 0.25 s mean of the one-step residual
in training standard deviations; alarm when above 3 for 0.1 s; evaluated only
while the workload is running.

| | Load-drop runs (3) | Runs without a change (9) |
|---|---|---|
| Koopman residual | detected all 3, **77-94 ms** after the change, no false alarm before it | 0 false alarms |
| Persistence residual (K = I) | detected 0 | 0 false alarms |
| Distance from normal state (same inputs, no dynamics) | detected 0 (score rose to about 1.7, below threshold) | 0 false alarms |

Core utilisation stays at 100% through the change (the cores never idle), so a
utilisation monitor cannot see it; the change shows up in how queues evolve,
which the operator models. After the change, the Koopman score also shows a
clean 2-second square wave: the phased jobs' 2 s cycle, which becomes visible
once the heavy load is gone.

Lab Trace 3 (28 cores, one change: first CPU-bound job exits at 42.9 s): with the
52-variable per-core state, every detector raises false alarms before the
change, because 8 s of a mostly idle 28-core machine does not describe "normal"
well. With a 4-variable machine-wide state (busy cores, queued tasks, job-type
shares; an ablation added after seeing the per-core result), the Koopman
detector finds the change 32 ms after it with no false alarm; the distance
baseline also finds it (52 ms); persistence misses it.

The evidence base is small (3 changes in the VM, 1 on the lab machine).

## 4. Visualisation

`viz/index.html` (published as the Scheduler Observatory artifact) replays five
traces in slow motion: a tile per core glowing in the colour of the running
process, run-queue tokens with growing wait bars, a sleep/disk pool, migration
arcs, a decision log narrating each scheduling decision with its cause and
wait, a per-core timeline, and a Koopman lens with the eigenvalues of a
sliding-window operator and the live surprise score. All data comes from
`src/export_viz.py`. The lens uses the machine-wide state for every trace
because a 52-variable operator overfits 2-second windows.

## 5. Added after the second request (scaling, live detection, markets)

### 5.1 Measured scaling to 4 cores (`data/scale_mac/`, figure `e3`)

ARM VM on an Apple M-series host (4 vCPUs, Linux 6.8), same 8 jobs pinned to 1, 2 and 4 cores,
3 repetitions, kernel per-process counters only (no root available there). Share of time spent
waiting for a CPU: CPU-bound 76% / 50% / 1%; phased 80% / 58% / 39%; I/O 87% / 71% / 74%.
The Intel VM traces give the same CPU-bound values on 1 and 2 cores (77% / 50%). 8 and 16 cores
need a Linux machine with that many cores (lab).

### 5.2 Live change detection, 48 scripted runs on two architectures (`outputs/change/`, figures `e1`, `e2`)

`kit/live_monitor.py` reads only /proc/stat (no root, no tracing): busy cores, runnable tasks and
context switches in 100 ms bins, operator fitted on the first 8 s. `kit/change_run.py` applies one
change at a random time in 15-25 s (load up / load down / I/O burst, small or large) or none; the
monitor is not told. 24 runs per platform (18 changes, 6 null).

| Platform | Detected online | Median delay | Runs with a false alarm | Monitor CPU |
|---|---|---|---|---|
| ARM VM, 4 cores | 17/18 | 0.12 s | 10/24 | 0.5% of one core |
| Intel VM, 2 cores | 12/18 | 0.26 s | 10/24 | 1.1% of one core |

Baselines on the same state, same rules (threshold 3): distance-from-normal 18/18 and 16/18
(false alarms 7/24, 10/24); run-queue monitor 12/18 and 15/18 (0/24, 4/24); utilisation 12/18 and 6/18.
Simple monitors miss I/O bursts (visible only in the switch rate). Across 54 settings (training
length x smoothing x threshold x platform) the Koopman residual is never strictly better than the
distance-from-normal score and is strictly worse in 51 (`outputs/robustness/`).

`/proc/schedstat` was tried first and rejected: its per-CPU run time is credited only when a task
leaves a core, so a CPU-bound task produces spikes, not a signal.

### 5.3 Markets: pre-registered test on 1990-2026 (`outputs/market/`, figures `e4`, `e5`)

Data: S&P 500 and VIX daily from yfinance, 1990-01-02 to 2026-10-19 (hashes in `data/market/sha256.txt`).
22 rule-defined events (first close 10% below the 1-year high after recovering to within 5%).
Walk-forward operator (2-year training, refit every 21 days) on EIGENWORLD's 7 features.

- Precursor (alarm in the 20 trading days before an event): Koopman best 7/22 vs 3.4 expected by
  chance (p = 0.043, not significant after correcting for 35 settings); at threshold 3, 0/22.
  VIX z-score 11/22 (p = 0.0003, survives correction).
- Named stress dates: Koopman warned only before COVID.
- Spectral radius before events vs other days: p = 0.62.
- EIGENWORLD re-test with its own unmodified code (`third_party/eigenworld`, `src/eigenworld_retest.py`).
  Its kernel bandwidth uses an UNSEEDED random sample, so it was run under 11 fixed seeds:
  on the original 2004-2010 file, 10 of 11 seeds reproduce the published minimum on 2008-08-27
  (values 1.960-1.982; published 1.972, matched exactly by seed 4). On 1992-2026 the 34-year minimum
  falls in July-August 2008 for all 11 seeds, but over 22 events entropy is HIGHER before events for
  every seed (+0.021 to +0.026, p = 0.055-0.087) and the stability margin does not differ
  (p = 0.83-0.997). The Lehman signal is real but unique.

## 6. Not yet done

- Contended captures on 8 and 16 cores of bare metal: needs the lab machine (no other Linux machine
  with that many cores was available; the user's laptop is a Mac).
- The clock-speed explanation for the lab's 3x I/O slow-down: needs the lab
  machine (VMs do not expose CPU frequency).

## 7. Reproduce

    python src/run_multicore.py            # validation, latency, prediction, simulation, Koopman
    python src/plots_multicore.py          # figures m1-m4
    python src/change_detection.py         # live detection, 48 runs
    python src/robustness.py               # sensitivity + Holm-corrected tests
    python src/market_koopman.py           # market study
    python src/eigenworld_retest.py        # EIGENWORLD on 36 years
    python src/plots_extended.py           # figures e1-e5
    python src/export_viz.py               # visualiser data
    cd paper && latexmk -pdf main.tex      # the paper
