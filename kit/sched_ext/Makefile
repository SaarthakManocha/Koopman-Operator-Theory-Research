# Build scx_koopman on a machine whose kernel has sched_ext (CONFIG_SCHED_CLASS_EXT=y, kernel >= 6.12).
# Needs: clang, libbpf-dev (>= 1.4), bpftool matching the kernel (linux-tools-$(uname -r)), libelf, zlib.
BPFTOOL ?= $(shell ls /usr/lib/linux-tools/$(shell uname -r)/bpftool 2>/dev/null || which bpftool)
ARCH := $(shell uname -m | sed 's/x86_64/x86/;s/aarch64/arm64/')

all: build/scx_koopman

build/vmlinux.h:
	mkdir -p build
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

build/scx_koopman.bpf.o: scx_koopman.bpf.c build/vmlinux.h
	clang -O2 -g -target bpf -mcpu=v3 -mlittle-endian -D__TARGET_ARCH_$(ARCH) -Ibuild -Iinclude -Iinclude/scx -c $< -o $@

build/scx_koopman.bpf.skel.h: build/scx_koopman.bpf.o
	$(BPFTOOL) gen skeleton $< name scx_koopman > $@

build/scx_koopman: scx_koopman.c build/scx_koopman.bpf.skel.h
	gcc -O2 -Ibuild -Iinclude -o $@ scx_koopman.c -lbpf -lelf -lz -lm

clean:
	rm -rf build
