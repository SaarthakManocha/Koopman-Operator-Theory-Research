"""
real_programs.py - kernel traces of real programs instead of synthetic loops.

    sudo python3 kit/real_programs.py --reps 3 --secs 45 --out data/real_programs

One capture runs, at the same time, on all CPUs:
    redis      redis-server, driven by a paced client (GET/SET plus occasional LRANGE/SORT)
    node       a Node.js HTTP server, driven by a paced client (light and heavy JSON endpoints)
    sqlite     a Python process running SQL queries of mixed size against SQLite, with think time
    ffmpeg     real-time (-re) H.264 encoding of a synthetic 640x360 video, all encoder threads
    clients    the two load generators themselves
and, untracked, a parallel gcc build and xz compression as background load, plus one
SCHED_IDLE spinner per CPU so cores never go idle (avoids the idle-exit trace loss
seen on this VM).
Every tracked thread's CPU time, wait time and timeslices are read from
/proc/<tid>/schedstat just before it stops, for validation against the trace.
"""
import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capture as C  # noqa: E402

PY = sys.executable

REDIS_CLIENT = r"""
import random, socket, sys, time
rng = random.Random(int(sys.argv[2])); end = time.time() + float(sys.argv[1])
s = socket.create_connection(("127.0.0.1", 6390)); f = s.makefile("rb")
def cmd(*a):
    s.sendall(("*%d\r\n" % len(a) + "".join("$%d\r\n%s\r\n" % (len(x), x) for x in map(str, a))).encode())
    line = f.readline()
    if line[:1] == b"$" and int(line[1:]) >= 0: f.read(int(line[1:]) + 2)
    elif line[:1] == b"*":
        for _ in range(int(line[1:])):
            l2 = f.readline(); f.read(int(l2[1:]) + 2) if l2[:1] == b"$" else None
cmd("DEL", "big")
cmd("RPUSH", "big", *[rng.randrange(10**6) for _ in range(30000)])
while time.time() < end:
    r = rng.random()
    if r < 0.85: cmd("SET", "k%d" % rng.randrange(1000), "v" * rng.randrange(10, 2000))
    elif r < 0.97: cmd("GET", "k%d" % rng.randrange(1000))
    elif r < 0.995: cmd("LRANGE", "big", 0, rng.randrange(100, 5000))
    else: cmd("SORT", "big", "LIMIT", 0, 10)
    time.sleep(rng.expovariate(1 / 0.004))
"""

NODE_SERVER = r"""
const http = require('http');
http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x'); const n = parseInt(u.searchParams.get('n') || '10');
  const a = []; for (let i = 0; i < n; i++) a.push({id: i, v: Math.sqrt(i) * Math.random(), s: 'x'.repeat(i % 50)});
  a.sort((p, q) => p.v - q.v);
  res.end(JSON.stringify(a.slice(0, 20)));
}).listen(8099, '127.0.0.1');
"""

HTTP_CLIENT = r"""
import http.client, random, sys, time
rng = random.Random(int(sys.argv[2])); end = time.time() + float(sys.argv[1])
c = http.client.HTTPConnection("127.0.0.1", 8099)
heavy = False
while time.time() < end:
    if rng.random() < 0.03: heavy = not heavy            # phases of heavy and light requests
    n = rng.randrange(20000, 60000) if heavy else rng.randrange(50, 2000)
    c.request("GET", "/q?n=%d" % n); c.getresponse().read()
    time.sleep(rng.expovariate(1 / 0.008))
"""

SQLITE_WORKER = r"""
import random, sqlite3, sys, time, os
rng = random.Random(int(sys.argv[2])); end = time.time() + float(sys.argv[1])
db = sqlite3.connect(sys.argv[3])
while time.time() < end:
    r = rng.random()
    if r < 0.6: db.execute("SELECT v FROM t WHERE id = ?", (rng.randrange(200000),)).fetchall()
    elif r < 0.85: db.execute("SELECT g, avg(v), count(*) FROM t WHERE id BETWEEN ? AND ? GROUP BY g", (a := rng.randrange(190000), a + rng.randrange(1000, 10000))).fetchall()
    elif r < 0.97:
        db.executemany("UPDATE t SET v = ? WHERE id = ?", [(rng.random(), rng.randrange(200000)) for _ in range(rng.randrange(10, 200))]); db.commit()
    else: db.execute("SELECT g, sum(v) FROM t GROUP BY g ORDER BY 2 DESC").fetchall()
    time.sleep(rng.expovariate(1 / 0.006))
"""


def prepare(work):
    os.makedirs(work, exist_ok=True)
    for name, src in (("redis_client.py", REDIS_CLIENT), ("node_server.js", NODE_SERVER),
                      ("http_client.py", HTTP_CLIENT), ("sqlite_worker.py", SQLITE_WORKER)):
        open(os.path.join(work, name), "w").write(textwrap.dedent(src))
    db = os.path.join(work, "bench.db")
    if not os.path.exists(db):
        import sqlite3
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER, v REAL)")
        rng = random.Random(0)
        c.executemany("INSERT INTO t VALUES (?,?,?)", ((i, i % 97, rng.random()) for i in range(200000)))
        c.commit(); c.close()
    src = os.path.join(work, "build"); os.makedirs(src, exist_ok=True)
    for i in range(24):                                         # a small C code base for the background build
        with open(os.path.join(src, f"m{i}.c"), "w") as f:
            for j in range(60):
                f.write(f"double f{i}_{j}(double x){{double s=0;for(int k=0;k<{j + 10};k++)s+=x*k/(k+1.0);"
                        f"return s>1e9?f{i}_{(j + 1) % 60}(x/2):s;}}\n")
    open(os.path.join(src, "Makefile"), "w").write(
        "SRC=$(wildcard *.c)\nOBJ=$(SRC:.c=.o)\nall: $(OBJ)\n%.o: %.c\n\tgcc -O2 -c $< -o $@\nclean:\n\trm -f *.o\n")
    blob = os.path.join(work, "blob.bin")
    if not os.path.exists(blob):
        with open(blob, "wb") as f:
            rng = random.Random(1)
            for _ in range(64):
                f.write(bytes(rng.randrange(64) for _ in range(1 << 16)))   # compressible
    return db, src, blob


def threads_of(pid):
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []


def schedstat(tid):
    try:
        run, wait, sl = map(int, open(f"/proc/{tid}/schedstat").read().split())
        st = {l.split(":")[0]: l.split()[1] for l in open(f"/proc/{tid}/status") if "ctxt_switches" in l}
        return {"run_ns": run, "wait_ns": wait, "slices": sl, "vol": int(st["voluntary_ctxt_switches"]),
                "invol": int(st["nonvoluntary_ctxt_switches"])}
    except OSError:
        return None


def one_capture(name, secs, outroot, work, seed):
    db, src, blob = prepare(work)
    d = os.path.join(outroot, name)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    T = C.T
    C.ensure_tracefs()
    C.w(f"{T}/tracing_on", 0); C.w(f"{T}/trace", ""); C.w(f"{T}/trace_clock", "mono")
    C.w(f"{T}/buffer_size_kb", 131072); C.w(f"{T}/events/enable", 0)
    for e in C.EVENTS:
        if os.path.exists(f"{T}/events/sched/{e}"):
            C.w(f"{T}/events/sched/{e}/enable", 1)
    bg_stop = threading.Event()

    def background():
        while not bg_stop.is_set():
            subprocess.run(["make", "-s", "-C", src, "clean"], capture_output=True)
            subprocess.run(["make", "-s", "-j2", "-C", src], capture_output=True)
            if bg_stop.is_set():
                break
            subprocess.run(["xz", "-6", "-T1", "-c", blob], stdout=subprocess.DEVNULL)

    # One SCHED_IDLE spinner per CPU keeps every core out of the idle loop (this VM's kernel drops
    # part of the switches out of idle; see the main study). SCHED_IDLE tasks only get CPU time
    # nothing else wants, so the programs under test are barely affected.
    spinners = [subprocess.Popen(["chrt", "--idle", "0", "taskset", "-c", str(c), PY, "-c", "while True: pass"])
                for c in range(os.cpu_count())]
    redis = subprocess.Popen(["redis-server", "--port", "6390", "--save", "", "--appendonly", "no",
                              "--io-threads", "1"], stdout=subprocess.DEVNULL)
    node = subprocess.Popen(["node", os.path.join(work, "node_server.js")])
    time.sleep(1.5)
    C.w(f"{T}/tracing_on", 1)
    t0 = time.clock_gettime(time.CLOCK_MONOTONIC)
    bg = threading.Thread(target=background); bg.start()
    rc = subprocess.Popen([PY, os.path.join(work, "redis_client.py"), str(secs), str(seed)])
    hc = subprocess.Popen([PY, os.path.join(work, "http_client.py"), str(secs), str(seed + 1)])
    sq = subprocess.Popen([PY, os.path.join(work, "sqlite_worker.py"), str(secs), str(seed + 2), db])
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-re", "-f", "lavfi", "-i",
                           f"testsrc2=size=640x360:rate=25:duration={secs}", "-c:v", "libx264", "-preset", "veryfast",
                           "-threads", "2", "-f", "null", "-"])
    time.sleep(2.0)
    tracked = []
    for kind, p in (("redis", redis), ("node", node), ("sqlite", sq), ("client", rc), ("client", hc)):
        tracked.append((kind, p.pid))                     # main thread of each
    for tid in threads_of(ff.pid):
        tracked.append(("ffmpeg", tid))
    with open(os.path.join(d, "pids.txt"), "w") as f:
        for k, t in tracked:
            f.write(f"{k} {t}\n")
    gt = {}

    def sample():
        for k, t in tracked:
            s = schedstat(t)
            if s:
                gt[t] = {"kind": k, **s}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            sample(); time.sleep(0.02)
    sm = threading.Thread(target=sampler); sm.start()
    for p in (rc, hc, sq, ff):
        p.wait()
    sample()
    stop.set(); sm.join()
    C.w(f"{T}/tracing_on", 0)
    t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
    bg_stop.set()
    redis.send_signal(signal.SIGTERM); node.send_signal(signal.SIGTERM)
    redis.wait(); node.wait(); bg.join()
    for sp in spinners:
        sp.kill()
    with open(f"{T}/trace") as fi, open(os.path.join(d, "trace.txt"), "w") as fo:
        shutil.copyfileobj(fi, fo, 1 << 20)
    stats = C.ring_stats()
    C.w(f"{T}/events/enable", 0)
    lost = sum(int(s.get("overrun", 0)) + int(s.get("dropped events", 0) or 0) for s in stats.values())
    subprocess.run(["gzip", "-f", os.path.join(d, "trace.txt")])
    # ground truth in the same format the validator reads
    gtd = os.path.join(d, "ground_truth"); os.makedirs(gtd)
    for t, g in gt.items():
        with open(os.path.join(gtd, f"gt_{t}.txt"), "w") as f:
            f.write(f"pid {t}\nkind {g['kind']}\nrun_ns {g['run_ns']}\nwait_ns {g['wait_ns']}\ntimeslices {g['slices']}\n"
                    f"voluntary {g['vol']}\ninvoluntary {g['invol']}\n")
    meta = {"name": name, "secs": secs, "ncpu": os.cpu_count(), "cpus_allowed": list(range(os.cpu_count())),
            "t_start_mono": t0, "t_end_mono": t1, "lost_events": lost, "tracked": tracked,
            "ground_truth_note": "schedstat read while each thread is alive, last sample before it stopped",
            "programs": {"redis": subprocess.run(["redis-server", "--version"], capture_output=True, text=True).stdout.strip(),
                         "node": subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip(),
                         "ffmpeg": "libx264 veryfast 640x360 25 fps -re", "sqlite": __import__("sqlite3").sqlite_version},
            "jobs": {}}
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"{name}: tracked {len(tracked)} threads, lost events {lost}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--secs", type=float, default=45)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = os.path.join(tempfile.gettempdir(), "koopman_real_programs")
    for r in range(a.reps):
        one_capture(f"real_r{r + 1}", a.secs, a.out, work, seed=100 * (r + 1))


if __name__ == "__main__":
    main()
