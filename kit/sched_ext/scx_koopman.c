/* SPDX-License-Identifier: GPL-2.0 */
/*
 * scx_koopman: user-space half. Loads the BPF scheduler, reads every finished
 * CPU burst from the ring buffer, updates that task's predictor and writes the
 * predicted next burst (ns) into the `pred` map.
 *
 *   sudo ./scx_koopman -p koopman|ewma|none [-m margin_pct] [-t seconds] [-o stats.txt]
 *
 * -p none loads the same scheduler without predictions (the vtime baseline).
 * Ctrl-C or -t ends it; the kernel falls back to EEVDF automatically.
 */
#include <assert.h>
#include <libgen.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <bpf/bpf.h>
#include <scx/common.h>
#include "scx_koopman.bpf.skel.h"

#define MAXT 65536
#define LAM 0.995

struct pstate { int used; double ewma, l[3], th[4], P[4][4]; long n; };
static struct pstate *S;
static int use_koop, pred_fd;
static long nev;
static volatile int exit_req;

static void on_sig(int s) { exit_req = 1; }

static double update(struct pstate *t, double b)
{
	double lb = log(b > 1e-7 ? b : 1e-7);

	t->ewma = t->n ? 0.5 * b + 0.5 * t->ewma : b;
	if (use_koop && t->n >= 3) {
		double x[4] = {1, t->l[0], t->l[1], t->l[2]}, Px[4], den = LAM, e = lb;
		for (int r = 0; r < 4; r++) { Px[r] = 0; for (int c = 0; c < 4; c++) Px[r] += t->P[r][c] * x[c]; den += x[r] * Px[r]; }
		for (int r = 0; r < 4; r++) e -= t->th[r] * x[r];
		for (int r = 0; r < 4; r++) t->th[r] += Px[r] / den * e;
		for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) t->P[r][c] = (t->P[r][c] - Px[r] * Px[c] / den) / LAM;
	}
	t->l[2] = t->l[1]; t->l[1] = t->l[0]; t->l[0] = lb; t->n++;
	if (use_koop && t->n >= 10) {
		double y = t->th[0] + t->th[1] * t->l[0] + t->th[2] * t->l[1] + t->th[3] * t->l[2];
		return exp(y < 0 ? y : 0);
	}
	return t->ewma;
}

struct burst_ev { __u32 pid, pad; __u64 burst_ns; };

static int on_burst(void *ctx, void *data, size_t sz)
{
	struct burst_ev *e = data;
	struct pstate *t = &S[e->pid % MAXT];
	if (!t->used) { memset(t, 0, sizeof *t); t->used = 1; for (int r = 0; r < 4; r++) t->P[r][r] = 100; }
	__u64 p = (__u64)(update(t, e->burst_ns * 1e-9) * 1e9);
	bpf_map_update_elem(pred_fd, &e->pid, &p, BPF_ANY);
	nev++;
	return 0;
}

int main(int argc, char **argv)
{
	const char *pol = "koopman", *out = NULL;
	int margin = 150, opt;
	double secs = 0;

	while ((opt = getopt(argc, argv, "p:m:t:o:h")) != -1) {
		switch (opt) {
		case 'p': pol = optarg; break;
		case 'm': margin = atoi(optarg); break;
		case 't': secs = atof(optarg); break;
		case 'o': out = optarg; break;
		default:
			fprintf(stderr, "usage: %s -p koopman|ewma|none [-m margin_pct] [-t seconds] [-o stats]\n", basename(argv[0]));
			return 1;
		}
	}
	use_koop = !strcmp(pol, "koopman");
	S = calloc(MAXT, sizeof *S);
	signal(SIGINT, on_sig); signal(SIGTERM, on_sig);

	struct scx_koopman *skel = SCX_OPS_OPEN(koopman_ops, scx_koopman);
	skel->rodata->mode = strcmp(pol, "none") != 0;
	skel->rodata->margin_pct = margin;
	SCX_OPS_LOAD(skel, koopman_ops, scx_koopman, uei);
	struct bpf_link *link = SCX_OPS_ATTACH(skel, koopman_ops, scx_koopman);
	pred_fd = bpf_map__fd(skel->maps.pred);
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.bursts), on_burst, NULL, NULL);
	fprintf(stderr, "scx_koopman running: predictor=%s margin=%d%%\n", pol, margin);

	struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
	while (!exit_req && !UEI_EXITED(skel, uei)) {
		ring_buffer__poll(rb, 100);
		if (secs > 0) {
			struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
			if (t.tv_sec - t0.tv_sec + (t.tv_nsec - t0.tv_nsec) * 1e-9 > secs) break;
		}
	}
	if (out) {
		FILE *f = fopen(out, "w");
		fprintf(f, "predictor %s\nmargin_pct %d\nbursts %ld\n", pol, margin, nev);
		fclose(f);
	}
	ring_buffer__free(rb);
	bpf_link__destroy(link);
	UEI_REPORT(skel, uei);
	scx_koopman__destroy(skel);
	return 0;
}
