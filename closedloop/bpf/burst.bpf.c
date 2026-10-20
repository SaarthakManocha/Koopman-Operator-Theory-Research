// burst.bpf.c - report every CPU burst of tracked tasks the moment it ends.
//
// A burst is the CPU time a task uses between two voluntary switch-outs
// (it blocked or slept). Preemptions do not end a burst; the time is summed.
// User space adds the pids to track to `tracked` and reads finished bursts
// from the ring buffer `events`.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

struct acct {
    __u64 on_since;   // when the task was last switched in (0 = not on a CPU)
    __u64 acc;        // CPU time of the current burst so far
};

struct event {
    __u32 pid;
    __u32 _pad;
    __u64 burst_ns;
    __u64 ts;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u32);
    __type(value, struct acct);
} tracked SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 22);
} events SEC(".maps");

SEC("tp_btf/sched_switch")
int BPF_PROG(on_switch, bool preempt, struct task_struct *prev, struct task_struct *next, unsigned int prev_state)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 ppid = BPF_CORE_READ(prev, pid);
    __u32 npid = BPF_CORE_READ(next, pid);

    struct acct *a = bpf_map_lookup_elem(&tracked, &ppid);
    if (a && a->on_since) {
        a->acc += now - a->on_since;
        a->on_since = 0;
        if (prev_state != 0) {                       // TASK_RUNNING == 0: preempted, burst continues
            struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
            if (e) {
                e->pid = ppid; e->burst_ns = a->acc; e->ts = now;
                bpf_ringbuf_submit(e, 0);
            }
            a->acc = 0;
        }
    }
    struct acct *b = bpf_map_lookup_elem(&tracked, &npid);
    if (b)
        b->on_since = now;
    return 0;
}
