/*
 * cagent.c - event-driven slice controller: eBPF burst observer + C predictor.
 *
 *   ./cagent <ewma|koopman> <margin c> <pidfile> <seconds> <out.txt>
 *
 * For every burst a tracked task finishes (reported by burst.bpf.c the moment
 * it ends), the agent updates that task's predictor and sets its EEVDF slice
 * to clamp(c * prediction, 0.1 ms, default slice) with sched_setattr.
 *
 * Predictors (per task):
 *   ewma     tau = 0.5 * burst + 0.5 * tau                        (textbook SJF)
 *   koopman  log-scale linear operator on 3 delay coordinates (Hankel DMD /
 *            log-EDMD with an intercept), updated after every burst by
 *            recursive least squares with forgetting factor 0.995; EWMA until
 *            the task has 10 bursts.
 * Only tasks labelled "req" in the pidfile are controlled; others are ignored.
 * At exit it writes: agent CPU time, number of bursts, mean predictor cost,
 * and per-task bursts/predictions for accuracy.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/sched/types.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "burst.skel.h"

#define MAXT 256
#define MAXB 200000
#define LAM 0.995

struct ev { unsigned int pid, pad; unsigned long long burst_ns, ts; };

struct task {
    int pid, n;
    double ewma, l[3];          /* last three log bursts, newest first */
    double th[4], P[4][4];      /* RLS state for [1, l0, l1, l2] */
    double pred;                /* last prediction (seconds) */
};

static struct task T[MAXT]; static int NT;
static int koop; static double C, SDEF;
static FILE *trace;             /* pid burst pred */
static long nev; static double tpred;

static double now(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void set_slice(int pid, double sec) {
    if (sec < 1e-4) sec = 1e-4;
    if (sec > SDEF) sec = SDEF;
    struct sched_attr a; memset(&a, 0, sizeof a);
    a.size = sizeof a; a.sched_runtime = (unsigned long long)(sec * 1e9);
    syscall(SYS_sched_setattr, pid, &a, 0);
}

static double predict_update(struct task *t, double b) {
    double lb = log(b > 1e-7 ? b : 1e-7);
    t->ewma = t->n ? 0.5 * b + 0.5 * t->ewma : b;
    if (koop && t->n >= 3) {                 /* RLS update with x = [1, l0, l1, l2] -> target lb */
        double x[4] = {1, t->l[0], t->l[1], t->l[2]}, Px[4], den = LAM, e = lb;
        for (int r = 0; r < 4; r++) { Px[r] = 0; for (int c = 0; c < 4; c++) Px[r] += t->P[r][c] * x[c]; den += x[r] * Px[r]; }
        for (int r = 0; r < 4; r++) e -= t->th[r] * x[r];
        for (int r = 0; r < 4; r++) t->th[r] += Px[r] / den * e;
        for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) t->P[r][c] = (t->P[r][c] - Px[r] * Px[c] / den) / LAM;
    }
    t->l[2] = t->l[1]; t->l[1] = t->l[0]; t->l[0] = lb; t->n++;
    if (koop && t->n >= 10) {
        double y = t->th[0] + t->th[1] * t->l[0] + t->th[2] * t->l[1] + t->th[3] * t->l[2];
        if (y > log(1.0)) y = log(1.0);
        return exp(y);
    }
    return t->ewma;
}

static int on_event(void *ctx, void *data, size_t sz) {
    struct ev *e = data;
    for (int i = 0; i < NT; i++) if (T[i].pid == (int)e->pid) {
        struct task *t = &T[i];
        double b = e->burst_ns * 1e-9;
        if (t->n) fprintf(trace, "%d %.9f %.9f\n", t->pid, b, t->pred);
        double t0 = now(CLOCK_MONOTONIC);
        t->pred = predict_update(t, b);
        tpred += now(CLOCK_MONOTONIC) - t0;
        set_slice(t->pid, C * t->pred);
        nev++;
        break;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    koop = !strcmp(argv[1], "koopman"); C = atof(argv[2]);
    double dur = atof(argv[4]);
    { struct sched_attr g; memset(&g, 0, sizeof g);
      SDEF = syscall(SYS_sched_getattr, 0, &g, sizeof g, 0) == 0 ? g.sched_runtime * 1e-9 : 1.4e-3; }
    struct burst_bpf *sk = burst_bpf__open_and_load();
    if (!sk) { fprintf(stderr, "bpf load failed\n"); return 1; }
    FILE *pf = fopen(argv[3], "r"); char kind[16]; int pid;
    while (fscanf(pf, "%15s %d", kind, &pid) == 2) {
        if (strcmp(kind, "req")) continue;
        struct task *t = &T[NT++]; memset(t, 0, sizeof *t); t->pid = pid;
        for (int r = 0; r < 4; r++) t->P[r][r] = 100.0;
        struct { unsigned long long on_since, acc; } z = {0, 0};
        bpf_map__update_elem(sk->maps.tracked, &pid, sizeof pid, &z, sizeof z, BPF_ANY);
    }
    fclose(pf);
    char tp[512]; snprintf(tp, sizeof tp, "%s.bursts", argv[5]); trace = fopen(tp, "w");
    if (burst_bpf__attach(sk)) { fprintf(stderr, "attach failed\n"); return 1; }
    struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(sk->maps.events), on_event, NULL, NULL);
    double end = now(CLOCK_MONOTONIC) + dur;
    while (now(CLOCK_MONOTONIC) < end) ring_buffer__poll(rb, 100);
    struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    double cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6 + ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    FILE *o = fopen(argv[5], "w");
    fprintf(o, "agent_cpu_s %.6f\nbursts %ld\npredict_ns %.1f\ndefault_slice_s %.6f\n", cpu, nev, nev ? tpred / nev * 1e9 : 0, SDEF);
    fclose(o); fclose(trace);
    ring_buffer__free(rb); burst_bpf__destroy(sk);
    return 0;
}
