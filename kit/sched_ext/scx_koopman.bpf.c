/* SPDX-License-Identifier: GPL-2.0 */
/*
 * scx_koopman: a sched_ext scheduler that orders tasks by predicted next CPU burst.
 *
 * Global weighted-vtime scheduler (as in the kernel's scx_simple) with two changes:
 *
 *  1. Every CPU burst (CPU time between two moments the task stops being runnable)
 *     is measured here and sent to user space through a ring buffer. User space
 *     (scx_koopman.c) runs the predictor (textbook EWMA or the log-scale Koopman
 *     operator, updated by recursive least squares) and writes the predicted next
 *     burst, in ns, into the `pred` hash map keyed by pid.
 *
 *  2. When a task is queued, its key in the shared queue is
 *         vtime + predicted burst            (earliest predicted finish first)
 *     and its time slice is  margin * predicted burst,  clamped to
 *     [min_slice_ns, SCX_SLICE_DFL]. Tasks without a prediction yet use
 *     SCX_SLICE_DFL for both, i.e. they behave as in scx_simple.
 *
 * mode 0 = no prediction (plain vtime scheduler, the baseline);
 * mode 1 = use predictions from user space.
 */
#include <scx/common.bpf.h>

char _license[] SEC("license") = "GPL";

const volatile u32 mode = 1;
const volatile u64 margin_pct = 150;           /* slice = margin_pct/100 * prediction */
const volatile u64 min_slice_ns = 100000;

static u64 vtime_now;
UEI_DEFINE(uei);

#define SHARED_DSQ 0

struct task_ctx {
	u64 on_since;   /* ns timestamp when it last started running, 0 = not running */
	u64 acc;        /* CPU time of the current burst */
};

struct {
	__uint(type, BPF_MAP_TYPE_TASK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct task_ctx);
} tctx SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 65536);
	__type(key, u32);
	__type(value, u64);
} pred SEC(".maps");

struct burst_ev {
	u32 pid;
	u32 pad;
	u64 burst_ns;
};

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 22);
} bursts SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
	__uint(key_size, sizeof(u32));
	__uint(value_size, sizeof(u64));
	__uint(max_entries, 3);                 /* [direct, queued, predicted] */
} stats SEC(".maps");

static void stat_inc(u32 idx)
{
	u64 *c = bpf_map_lookup_elem(&stats, &idx);
	if (c)
		(*c)++;
}

static u64 predicted(struct task_struct *p)
{
	u32 pid = p->pid;
	u64 *v;

	if (!mode)
		return 0;
	v = bpf_map_lookup_elem(&pred, &pid);
	return v ? *v : 0;
}

static u64 slice_for(u64 pr)
{
	u64 s;

	if (!pr)
		return SCX_SLICE_DFL;
	s = pr * margin_pct / 100;
	if (s < min_slice_ns)
		s = min_slice_ns;
	if (s > SCX_SLICE_DFL)
		s = SCX_SLICE_DFL;
	return s;
}

s32 BPF_STRUCT_OPS(koopman_select_cpu, struct task_struct *p, s32 prev_cpu, u64 wake_flags)
{
	bool is_idle = false;
	s32 cpu;

	cpu = scx_bpf_select_cpu_dfl(p, prev_cpu, wake_flags, &is_idle);
	if (is_idle) {
		stat_inc(0);
		scx_bpf_dsq_insert(p, SCX_DSQ_LOCAL, slice_for(predicted(p)), 0);
	}
	return cpu;
}

void BPF_STRUCT_OPS(koopman_enqueue, struct task_struct *p, u64 enq_flags)
{
	u64 vtime = p->scx.dsq_vtime;
	u64 pr = predicted(p);

	stat_inc(1);
	if (pr)
		stat_inc(2);
	if (time_before(vtime, vtime_now - SCX_SLICE_DFL))
		vtime = vtime_now - SCX_SLICE_DFL;
	p->scx.dsq_vtime = vtime;
	scx_bpf_dsq_insert_vtime(p, SHARED_DSQ, slice_for(pr), vtime + (pr ? pr : SCX_SLICE_DFL), enq_flags);
}

void BPF_STRUCT_OPS(koopman_dispatch, s32 cpu, struct task_struct *prev)
{
	scx_bpf_dsq_move_to_local(SHARED_DSQ);
}

void BPF_STRUCT_OPS(koopman_running, struct task_struct *p)
{
	struct task_ctx *c;

	if (time_before(vtime_now, p->scx.dsq_vtime))
		vtime_now = p->scx.dsq_vtime;
	c = bpf_task_storage_get(&tctx, p, 0, BPF_LOCAL_STORAGE_GET_F_CREATE);
	if (c)
		c->on_since = bpf_ktime_get_ns();
}

void BPF_STRUCT_OPS(koopman_stopping, struct task_struct *p, bool runnable)
{
	struct task_ctx *c;
	u64 now = bpf_ktime_get_ns();

	c = bpf_task_storage_get(&tctx, p, 0, 0);
	if (c && c->on_since) {
		u64 ran = now - c->on_since;

		/* charge virtual time by the time actually used, scaled by weight */
		p->scx.dsq_vtime += ran * 100 / p->scx.weight;
		c->acc += ran;
		c->on_since = 0;
		if (!runnable) {                /* the task blocked: the burst is over */
			struct burst_ev *e = bpf_ringbuf_reserve(&bursts, sizeof(*e), 0);

			if (e) {
				e->pid = p->pid;
				e->burst_ns = c->acc;
				bpf_ringbuf_submit(e, 0);
			}
			c->acc = 0;
		}
	}
}

void BPF_STRUCT_OPS(koopman_enable, struct task_struct *p)
{
	p->scx.dsq_vtime = vtime_now;
}

s32 BPF_STRUCT_OPS_SLEEPABLE(koopman_init)
{
	return scx_bpf_create_dsq(SHARED_DSQ, -1);
}

void BPF_STRUCT_OPS(koopman_exit, struct scx_exit_info *ei)
{
	UEI_RECORD(uei, ei);
}

SCX_OPS_DEFINE(koopman_ops,
	       .select_cpu		= (void *)koopman_select_cpu,
	       .enqueue			= (void *)koopman_enqueue,
	       .dispatch		= (void *)koopman_dispatch,
	       .running			= (void *)koopman_running,
	       .stopping		= (void *)koopman_stopping,
	       .enable			= (void *)koopman_enable,
	       .init			= (void *)koopman_init,
	       .exit			= (void *)koopman_exit,
	       .name			= "koopman");
