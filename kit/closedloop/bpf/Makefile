# Build the eBPF burst observer and the C slice controller.
# Needs clang, libbpf-dev and bpftool; vmlinux.h is generated from the running kernel.
BPFTOOL ?= $(shell ls /usr/lib/linux-tools*/bpftool 2>/dev/null | head -1 || echo bpftool)
ARCH := $(shell uname -m | sed 's/x86_64/x86/;s/aarch64/arm64/')

all: cagent

vmlinux.h:
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

burst.bpf.o: burst.bpf.c vmlinux.h
	clang -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -c $< -o $@

burst.skel.h: burst.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

cagent: cagent.c burst.skel.h
	gcc -O2 -o $@ cagent.c -lbpf -lelf -lz -lm

clean:
	rm -f burst.bpf.o burst.skel.h cagent
