# Predictive CPU Scheduling via Koopman Operator Theory

Can we predict how long a process will run on the CPU next, from how long it ran its last few times? 
We test EDMD (a Koopman-operator method) against simple guesses and against the textbook SJF
exponential average. The data is a real Linux scheduler trace with 3 CPU-bound and 3 I/O-bound processes (Trace 3).

## Quick start (VS Code)

1. Open this folder in VS Code (`File > Open Folder`).
2. Open a terminal (`` Ctrl+` ``) and install the libraries once:
   ```
   pip install -r requirements.txt
   ```
3. Run everything:
   ```
   python run_all.py
   ```
   Or press **F5** and pick **"Run full pipeline"**.

It takes about 15 seconds. It checks the trace's SHA-256 against the value
recorded in the lab, then rebuilds every table and figure in the report.

## Round 2: baselines, eBPF agent, margin sweep, real-program loop, sched_ext (20 Oct, evening)

    python3 src/baselines.py                                    # Table II (AR / ARIMA / MLP / nonlinear EDMD)
    cd closedloop/bpf && make && cd ../..                       # eBPF observer + C agent (needs clang, libbpf-dev, bpftool)
    sudo python3 closedloop/run_closedloop.py --cpus 0 --hogs 3 --reqs 4 --reps 6 --secs 20 \
        --policies stock,oracle,oracle1.5,oracle2,oracle3,bpf_ewma_1,bpf_ewma_1.5,bpf_ewma_2,bpf_ewma_3,bpf_koopman_1,bpf_koopman_1.5,bpf_koopman_2,bpf_koopman_3 \
        --out data/closedloop/sweep_1cpu
    sudo python3 closedloop/real_closed.py --reps 10 --secs 20 --out data/closedloop/real_1cpu
    python3 closedloop/analyse_real_closed.py data/closedloop/real_1cpu
    python3 closedloop/plot_sweep.py                            # Fig. 9 (margin sweep + real programs)
    python3 src/plots_paper_extra.py                            # Figs. 1, 6, 7 (overview, baselines, spectra)

`kit/sched_ext/` holds the sched_ext policy for the lab (compiles, not yet run). Details: RESULTS_EXTRA.md, sections 5-9.

## Closed loop, real programs, intervals and cost (added 20 Oct)

See **RESULTS_EXTRA.md**. Regenerate with:

    sudo python3 closedloop/run_closedloop.py --cpus 0 --hogs 3 --reqs 4 --reps 8 --secs 20 --out data/closedloop/c1_1cpu
    python3 closedloop/analyse_closedloop.py data/closedloop/c1_1cpu data/closedloop/c2_2cpu   # Fig. 7 data
    python3 closedloop/plot_closedloop.py                                                     # Fig. 7
    python3 src/real_programs_analysis.py && python3 src/plot_real_programs.py                # Fig. 4, real programs
    python3 src/conformal_cost.py                                                             # intervals + cost
    python3 src/plots_extended.py                                                             # Fig. 11 (EIGENWORLD)

```
closedloop/                task.c (workload), agent.py (slice controller), run/analyse/plot scripts
data/closedloop/           126 closed-loop runs (1 CPU and 2 CPUs), runs.csv and paired.csv per set-up
data/real_programs/        3 traced captures of Redis, Node.js, SQLite, ffmpeg (+ kernel ground truth)
kit/real_programs.py       capture script for the real programs
```

## Multi-core extension and visualizer (added after the presentation)

See **RESULTS_MULTICORE.md** for methods, results and validation, and **paper/main.pdf** for the paper.
Still to do on the lab machine: see **kit/README_LAB.md** (one command, about 50 minutes).

```
kit/                       capture kit (tracefs + kernel ground truth), live_monitor.py (unprivileged
                           Koopman change detector), change_run.py, scale_counters.py; README_LAB.md
data/change_vm, change_mac 48 scripted change runs (Intel 2-core VM with traces, ARM 4-core VM)
data/scale_mac             1/2/4-core scaling from kernel counters (ARM VM)
data/market                S&P 500 and VIX 1990-2026 (with SHA-256)
third_party/eigenworld     EIGENWORLD's code, unmodified, for the re-test
src/change_detection.py    live and offline change detection vs baselines
src/robustness.py          sensitivity sweep + Holm-corrected tests
src/market_koopman.py      pre-registered market study; src/eigenworld_retest.py
paper/                     LaTeX source and PDF
data/captures_vm/          12 validated captures (2-core Linux 6.18 VM, 4 set-ups x 3 repetitions)
data/observer_effect_example/  a capture contaminated by the streaming reader, kept as evidence
src/sched_events.py        one parser for perf and tracefs traces: run / queue / migration timelines
src/validate.py            checks every reconstructed process against /proc/<pid>/schedstat
src/simulate.py            trace-driven multi-core scheduler simulator (FCFS, RR, SJF, SRTF + predictors)
src/koopman_system.py      Koopman operator on the whole machine's state; regime-change detection
src/run_multicore.py       runs all multi-core experiments -> outputs/multicore/
src/plots_multicore.py     figures m1-m4
src/export_viz.py          data for the visualizer
viz/index.html             Scheduler Observatory (open with: python -m http.server, then visit /viz/)
```

    python src/run_multicore.py      # about 5 minutes
    python src/plots_multicore.py
    python src/export_viz.py

## Folder layout (original case study)

```
run_all.py                 one command: verify -> parse -> stats -> experiments -> figures
cpu_bound.py, io_long.py   the two workload scripts that were run in the lab
src/
  parse_trace.py           perf sched text -> per-process CPU bursts (CSV)
  trace_stats.py           how bursts end (R/S/D), who preempts the CPU-bound jobs
  edmd.py                  EDMD / Koopman: delay embedding, dictionaries, fit, eigenvalues, forecasts
  predictors.py            all methods: mean, median, last value, EWMA (SJF), EDMD variants, gradient boosting
  evaluate.py              MAE/RMSE, block-bootstrap intervals, lag correlations, SJF waiting-time simulation
  run_forecast.py          runs every experiment, writes outputs/metrics.txt + CSVs
  plots.py                 all figures
data/raw/trace3.txt.gz     raw `perf sched script` output (1,899,303 lines, gzip; parser reads it directly)
data/processed/            reconstructed bursts (bursts_trace3.csv)
outputs/                   metrics.txt (human-readable summary), CSV tables, figures/
evidence/                  pids.txt, evidence3.txt (uname, hostname, date, nproc, paranoid, SHA-256),
                           session3.log (raw terminal recording) + a readable copy
report/                    the case study report (.docx)
WALKTHROUGH.md             plain-English explanation of the code and likely viva questions
```

## Where each report number comes from

| Report section | File |
|---|---|
| Process summary, preemptors | `outputs/process_summary.csv`, `outputs/preemptors.csv` |
| Main forecast table + intervals | `outputs/metrics_main.csv` (section 1 of `metrics.txt`) |
| Rolling splits | `outputs/metrics_rolling_summary.csv` (section 2) |
| I/O phase analysis | `outputs/metrics_phase.csv` (section 3) |
| Lag correlations | `outputs/lag_correlations.csv` (section 4) |
| Eigenvalues | `outputs/eigenvalues.csv` (section 5) |
| SJF simulation | `outputs/sjf_simulation.csv` (section 6) |

## How the trace was captured (lab, Ubuntu, kernel 7.0.0-30-generic, 28 CPUs)

```
echo -1 > /proc/sys/kernel/perf_event_paranoid
rm -f pids.txt
perf sched record -o cap3.data -- bash -c '
  for i in 1 2 3; do python3 cpu_bound.py & echo "cpu $!" >> pids.txt; done;
  for i in 1 2 3; do python3 io_long.py  & echo "io $!"  >> pids.txt; done; wait'
perf sched script -i cap3.data > trace3.txt
sha256sum trace3.txt
```
(The terminal recording `session3.log` is cut off partway through the record
command, so the end of that command is reconstructed from `pids.txt` and
the trace contents.)
