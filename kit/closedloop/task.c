/*
 * task.c - workload for the closed-loop slice experiment.
 *
 *   ./task hog  <seconds> <out>                       CPU hog, counts work loops
 *   ./task req  <seconds> <out> <seed> <period_ms> <short_ms> <long_ms> <p_switch> [oracle]
 *
 * "req" is a request-serving task: it sleeps until an absolute deadline (period
 * apart), then does a CPU burst of w ms (thread CPU time), then records
 *   deadline, wake time, finish time, w
 * Burst lengths follow a two-regime log-AR(1) process (short / long regime,
 * switching with probability p_switch per request), so recent history predicts
 * the next burst but not perfectly.
 * With "oracle", the task sets its own EEVDF slice to its next burst before
 * sleeping (perfect knowledge, an upper bound for any predictor).
 */
#define _GNU_SOURCE
#include <linux/sched/types.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static double now(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static void spin_cpu(double sec) {
    double end = now(CLOCK_THREAD_CPUTIME_ID) + sec;
    volatile uint64_t x = 0;
    while (now(CLOCK_THREAD_CPUTIME_ID) < end) for (int i = 0; i < 200; i++) x += i;
}

static uint64_t SDEF = 0;   /* this task's default slice, read at start */
static int set_slice(uint64_t ns) {
    if (ns < 100000) ns = 100000;
    if (SDEF && ns > SDEF) ns = SDEF;
    struct sched_attr a; memset(&a, 0, sizeof a);
    a.size = sizeof a; a.sched_policy = 0; a.sched_runtime = ns;
    return syscall(SYS_sched_setattr, 0, &a, 0);
}

/* xorshift + Box-Muller */
static uint64_t s;
static double urand(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return (s >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { double u = urand() + 1e-12, v = urand(); return sqrt(-2 * log(u)) * cos(2 * M_PI * v); }

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    double dur = atof(argv[2]);
    FILE *f = fopen(argv[3], "w");
    if (!strcmp(argv[1], "hog")) {
        double end = now(CLOCK_MONOTONIC) + dur; uint64_t loops = 0;
        while (now(CLOCK_MONOTONIC) < end) { spin_cpu(0.001); loops++; }
        fprintf(f, "hog_ms_cpu %llu\n", (unsigned long long)loops);
        fclose(f); return 0;
    }
    s = strtoull(argv[4], 0, 10) * 2654435761ULL + 88172645463325252ULL;
    double period = atof(argv[5]) * 1e-3, mus = log(atof(argv[6])), mul = log(atof(argv[7])), psw = atof(argv[8]);
    int oracle = argc > 9 && !strncmp(argv[9], "oracle", 6);
    double ofac = (argc > 9 && !strcmp(argv[9], "oracle2")) ? 2.0 : 1.0;   /* oracle2: slice = 2 x burst */
    { struct sched_attr g; memset(&g, 0, sizeof g);
      if (syscall(SYS_sched_getattr, 0, &g, sizeof g, 0) == 0) SDEF = g.sched_runtime; }
    const double phi = 0.6, sig = 0.35;
    int regime = 0; double lw = mus;
    double t0 = now(CLOCK_MONOTONIC), dl = t0 + period, end = t0 + dur;
    /* first burst length */
    double w = exp(lw);
    fprintf(f, "deadline wake finish work_ms\n");
    while (dl < end) {
        if (oracle) set_slice((uint64_t)(ofac * w * 1e6));
        struct timespec ts = { (time_t)dl, (long)((dl - (time_t)dl) * 1e9) };
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, 0);
        double wake = now(CLOCK_MONOTONIC);
        spin_cpu(w * 1e-3);
        double fin = now(CLOCK_MONOTONIC);
        fprintf(f, "%.9f %.9f %.9f %.4f\n", dl - t0, wake - t0, fin - t0, w);
        /* next burst */
        if (urand() < psw) regime = !regime;
        double mu = regime ? mul : mus;
        lw = mu + phi * (lw - mu) + sig * nrand();
        w = exp(lw); if (w > 30) w = 30; if (w < 0.05) w = 0.05;
        dl += period;
        if (fin > dl) dl = fin;               /* overrun: start next request now */
    }
    fclose(f);
    return 0;
}
