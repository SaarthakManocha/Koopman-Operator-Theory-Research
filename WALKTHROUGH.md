# Walkthrough: how the project works, in plain words

Read this before presenting. It follows the code in the order `run_all.py` runs it.

## The question

SJF (Shortest Job First) gives the lowest average waiting time, but it needs
the length of each process's **next CPU burst**, which nobody knows in
advance. Can we predict it from the last few bursts?

## Step 1: capture (done in the lab)

`perf sched record` records every scheduler event in the kernel. We launched
3 copies of `cpu_bound.py` (a pure compute loop) and 3 copies of
`io_long.py` (write a file, sleep 50 ms, 1,500 times) **inside** the perf
command, and saved their PIDs to `pids.txt`.

Launching them inside the perf command is what fixed the pilot. Before, the
I/O processes were missing from the trace.

## Step 2: `src/parse_trace.py`, from events to bursts

Every `sched_switch` line says: task A leaves the CPU in state X, task B
comes in. For our 6 PIDs we note the time each one comes in and the time it
leaves. The difference is one **burst**. The state letter when it leaves
tells us why:

- `R`: preempted (still wanted the CPU)
- `S`: went to sleep (our `time.sleep`)
- `D`: waiting on disk (our file write)

Output: `data/processed/bursts_trace3.csv`, with 6,128 bursts.

Sanity check you can quote: each I/O process has **exactly 1,500 `S`
endings**, one per loop iteration.

## Step 3: `src/trace_stats.py`, why bursts end

- CPU-bound processes: every preemption was by `kworker` (kernel worker) or
  `migration/N` (the load balancer moving the process to another CPU). None
  was by our other processes, because there are 28 CPUs for 6 processes.
- I/O bursts are about 1,000 times shorter than CPU bursts (µs against ms).

## Step 4: `src/edmd.py`, the Koopman part

- **State** = last 3 bursts `[b_t, b_t-1, b_t-2]` (delay embedding).
- **Dictionary** = what we compute from the state. "linear" means the 3
  values as they are. "poly" also adds squares and products.
- **Fit**: find the matrix `A` so that `dictionary(next state) ≈ A × dictionary(state)`.
  This is ordinary least squares (`np.linalg.solve` on the normal equations).
- **One-step forecast**: give it the 3 real previous bursts and read off the next.
- **Recursive forecast**: feed its own guesses back in. Because every
  eigenvalue of `A` is below 1, this always fades to the mean.
- **Eigenvalues of A**: |λ| close to 1 means long memory, |λ| close to 0 means
  short memory.

If asked "isn't linear EDMD just an AR(3) model?": **yes**, and the report
says so. The Koopman view adds the eigenvalue interpretation, and the poly
dictionary is the nonlinear extension.

## Step 5: `src/predictors.py`, everything we compare

| Method | One-line idea |
|---|---|
| Mean / Median | always guess one number from training |
| Last value | next = previous |
| EWMA α=0.5 | textbook SJF: `τ(n+1) = α·t(n) + (1−α)·τ(n)` |
| EWMA tuned | same, α picked on training data |
| EDMD one-step / recursive / poly / log | Koopman variants |
| Gradient boosting | nonlinear ML on the same 3 previous bursts |

Rule followed everywhere: **no peeking**. Every prediction uses only bursts
that came before it, and scaling uses training data only.

## Step 6: `src/evaluate.py` + `src/run_forecast.py`, the tests

1. **Main split**: per process, train on the first 80%, test on the last 20%.
2. **Bootstrap interval**: resample the test set in blocks of 10 consecutive
   bursts, 2,000 times. If the whole 95% interval of "MAE ÷ mean-baseline
   MAE" is below 1, the method is reliably better.
3. **Rolling splits**: repeat with 5 different cut points, to check the
   result isn't one lucky split.
4. **Phase analysis**: I/O bursts tripled when the CPU jobs finished (about
   100 µs to about 330 µs). We re-test inside each phase separately to see
   what is left once that jump is removed.
5. **SJF simulation**: order a batch of jobs by each predictor and measure
   the waiting time against an oracle that knows the true lengths.

## The findings, in four lines

1. The biggest win is knowing CPU-bound from I/O-bound. That alone takes SJF
   from 10.9× to about 2.5× the oracle waiting time.
2. I/O bursts tripled when system load dropped. The textbook EWMA tracked
   this as well as or better than EDMD, and nothing reliably beat it.
3. Inside a stable phase, I/O bursts were barely more predictable than a constant.
4. EDMD's eigenvalues were a good **diagnosis**: a near-1 eigenvalue flagged
   the level shift. As a forecaster it did not beat the textbook method.

## Likely questions

- **Why compare against such simple baselines?** A fancy model is only worth
  using if it beats the simple one. Here, for I/O, it didn't.
- **Is this the scheduler Linux uses?** No. Linux (EEVDF since kernel 6.6;
  the lab runs 7.0) shares CPU time fairly and doesn't predict bursts. This
  study tests whether prediction would be possible.
- **Why the log scale?** CPU bursts are heavy-tailed (median about 60 ms,
  max 4 s). A few huge bursts dominate least squares; the log scale shrinks them.
- **Why block bootstrap and not ordinary bootstrap?** Neighbouring bursts are
  related. Resampling single bursts would make the intervals too narrow.
- **Why did the I/O bursts get longer?** Probably CPU frequency and power
  saving when the machine went idle. We did not measure it, and the report
  says so (Future Work 1).
- **How do I know the data is real?** `run_all.py` recomputes the SHA-256 of
  the trace and compares it with the value recorded in the lab
  (`evidence/evidence3.txt`).

## Running it live

```
pip install -r requirements.txt
python run_all.py
```
Then open `outputs/metrics.txt` and `outputs/figures/`. Or press F5 in VS Code.
