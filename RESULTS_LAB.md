# Bare-metal results, lab day 1 (26 Oct 2026)

Machine: Intel Core i7-14700 (8 P-cores x 2 threads = CPUs 0-15, 12 E-cores = CPUs 16-27), 28 CPUs, Ubuntu 26.04, kernel 7.0.0-38, default EEVDF slice 2.8 ms, sched_ext available.
Raw captures (518 MB) are kept outside the repository; `data/lab_day1/` holds the derived tables, logs and the live change-detection runs.

| Experiment | Result | Script |
| --- | --- | --- |
| Tracing self-test | 204 switch-outs, 204 switch-ins: no missing idle exits | kit/capture.py --check |
| Validation, 42 captures, 903 processes | CPU time within 0.44% for every process with >= 0.5 s CPU (median 0.024%); 0 lost events; 0 unmatched switch-outs | src/validate_lab.py |
| Closed loop, eBPF, 1/8/16 CPUs, 180 runs | short-request response -57..-59% (log-EDMD c=1.5-2), EWMA -50..-54%, oracle c=1 -8..-9%, c=1.5 -72%; Koopman beats EWMA 48/54 (p=9e-8); hogs within +-0.8% | closedloop/analyse_closedloop.py, closedloop/plot_baremetal.py |
| Frequency test | I/O bursts +65% after CPU jobs exit with default governor (9/9), none with performance governor | src/freq_test_lab.py |
| Scaling | CPU-bound waiting 76.5/50.2/0.05/0% on 1/2/4/8+ CPUs; with one CPU hog per core, interactive wait 1.75 ms per run at 4-16 CPUs | src/plot_lab_scaling.py |
| Live change detection, 24 runs | Koopman residual 8/18 (large changes 7/9, 0.04 s), run-queue/utilization monitors 14/18 | src/change_detection.py (mac_runs loader) |

Not yet run on bare metal: real-program captures (ffmpeg -nostdin fix applied), real-server steering, scx_koopman.

# Bare-metal results, lab day 2 (27 Oct 2026)

Machine: Intel Core i7-7700 (4 cores x 2 threads, 8 CPUs), Ubuntu 24.04, kernel 7.0.0-38, sched_ext available. Self-test 201/201.

| Experiment | Result | Script |
| --- | --- | --- |
| Real servers, 1 CPU + 3 hogs, 60 runs | log-EDMD c=1.5: Redis -43%, SQLite point -77%, Node.js light -34% (10/10 pairs each); EWMA -48/-23/-34%; Redis p99 +21% and SQLite range +32% with log-EDMD | closedloop/analyse_real_closed.py, closedloop/plot_real_baremetal.py |
| Real-program captures, 3 x 60 s | threads >= 0.5 s CPU within 3.1% of kernel (median 0.33%), 0 lost events; log-EDMD beats EWMA in 24/30 threads (median ratio 0.81); Redis 0.55, SQLite 0.56 | src/real_programs_analysis.py |
| sched_ext scx_koopman, 8 CPUs, 36 runs | log-EDMD margin 1.5: short -44%, long -41%, all -41% vs stock EEVDF; no-prediction baseline 14x worse; 30/30 clean load/unload | closedloop/analyse_closedloop.py |
