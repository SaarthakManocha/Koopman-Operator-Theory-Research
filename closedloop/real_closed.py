"""
real_closed.py - the slice controller on real programs.

    sudo python3 closedloop/real_closed.py --reps 5 --secs 20 --out data/closedloop/real_1cpu

CPU 0 (contended): three CPU hogs, a Redis server, a Node.js HTTP server and a
SQLite query worker. CPU 1: the two load generators, so client-side timing is not
itself delayed by the hogs. Latency is measured end to end:
  redis   client send -> reply received
  node    client send -> response received (light and heavy request phases)
  sqlite  intended query start (paced schedule) -> query finished
Policies: stock EEVDF; 'static' (the three servers get the minimum slice by hand);
and the eBPF agent with EWMA or Koopman predictions and margin c, controlling
the three server threads only (bpf_<ewma|koopman>_<c>). Each repetition runs every
policy with the same seeds, in shuffled order.
"""
import argparse
import ctypes
import json
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import textwrap
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

REDIS_CLIENT = r"""
import random, socket, sys, time
rng = random.Random(int(sys.argv[2])); end = time.time() + float(sys.argv[1]); out = open(sys.argv[3], "w")
s = socket.create_connection(("127.0.0.1", 6391)); f = s.makefile("rb")
def cmd(*a):
    s.sendall(("*%d\r\n" % len(a) + "".join("$%d\r\n%s\r\n" % (len(x), x) for x in map(str, a))).encode())
    line = f.readline()
    if line[:1] == b"$" and int(line[1:]) >= 0: f.read(int(line[1:]) + 2)
    elif line[:1] == b"*":
        for _ in range(int(line[1:])):
            l2 = f.readline()
            if l2[:1] == b"$": f.read(int(l2[1:]) + 2)
cmd("DEL", "big"); cmd("RPUSH", "big", *[rng.randrange(10**6) for _ in range(30000)])
while time.time() < end:
    r = rng.random(); t0 = time.perf_counter()
    if r < 0.85: cmd("SET", "k%d" % rng.randrange(1000), "v" * rng.randrange(10, 2000)); k = "set"
    elif r < 0.97: cmd("GET", "k%d" % rng.randrange(1000)); k = "get"
    else: cmd("LRANGE", "big", 0, rng.randrange(1000, 20000)); k = "lrange"
    out.write("%s %.9f\n" % (k, time.perf_counter() - t0))
    time.sleep(rng.expovariate(1 / 0.004))
"""

HTTP_CLIENT = r"""
import http.client, random, sys, time
rng = random.Random(int(sys.argv[2])); end = time.time() + float(sys.argv[1]); out = open(sys.argv[3], "w")
c = http.client.HTTPConnection("127.0.0.1", 8098); heavy = False
while time.time() < end:
    if rng.random() < 0.03: heavy = not heavy
    n = rng.randrange(20000, 60000) if heavy else rng.randrange(50, 2000)
    t0 = time.perf_counter(); c.request("GET", "/q?n=%d" % n); c.getresponse().read()
    out.write("%s %.9f\n" % ("heavy" if heavy else "light", time.perf_counter() - t0))
    time.sleep(rng.expovariate(1 / 0.008))
"""

NODE_SERVER = r"""
const http = require('http');
http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x'); const n = parseInt(u.searchParams.get('n') || '10');
  const a = []; for (let i = 0; i < n; i++) a.push({id: i, v: Math.sqrt(i) * Math.random(), s: 'x'.repeat(i % 50)});
  a.sort((p, q) => p.v - q.v); res.end(JSON.stringify(a.slice(0, 20)));
}).listen(8098, '127.0.0.1');
"""

SQLITE_WORKER = r"""
import random, sqlite3, sys, time
rng = random.Random(int(sys.argv[2])); T = float(sys.argv[1]); out = open(sys.argv[4], "w")
db = sqlite3.connect(sys.argv[3]); t0 = time.perf_counter(); due = t0
while due - t0 < T:
    now = time.perf_counter()
    if now < due: time.sleep(due - now)
    r = rng.random()
    if r < 0.6: db.execute("SELECT v FROM t WHERE id = ?", (rng.randrange(200000),)).fetchall(); k = "point"
    elif r < 0.9:
        a = rng.randrange(190000); db.execute("SELECT g, avg(v), count(*) FROM t WHERE id BETWEEN ? AND ? GROUP BY g", (a, a + rng.randrange(1000, 10000))).fetchall(); k = "range"
    else: db.execute("SELECT g, sum(v) FROM t GROUP BY g ORDER BY 2 DESC").fetchall(); k = "scan"
    out.write("%s %.9f\n" % (k, time.perf_counter() - due))
    due += rng.expovariate(1 / 0.010)
    due = max(due, time.perf_counter() - 0.0)
"""

POLICIES = ["stock", "static", "bpf_ewma_1.5", "bpf_koopman_1.5", "bpf_ewma_2", "bpf_koopman_2"]


def setslice(pid, ns):
    libc = ctypes.CDLL(None, use_errno=True)
    buf = ctypes.create_string_buffer(struct.pack("IIQiIQQQ", 48, 0, 0, 0, 0, ns, 0, 0))
    libc.syscall(314, pid, buf, 0)


def prepare(work):
    os.makedirs(work, exist_ok=True)
    for n, src in (("redis_client.py", REDIS_CLIENT), ("http_client.py", HTTP_CLIENT),
                   ("node_server.js", NODE_SERVER), ("sqlite_worker.py", SQLITE_WORKER)):
        open(os.path.join(work, n), "w").write(textwrap.dedent(src))
    db = os.path.join(work, "bench.db")
    if not os.path.exists(db):
        import sqlite3
        c = sqlite3.connect(db); c.execute("CREATE TABLE t(id INTEGER PRIMARY KEY, g INTEGER, v REAL)")
        rng = random.Random(0)
        c.executemany("INSERT INTO t VALUES (?,?,?)", ((i, i % 97, rng.random()) for i in range(200000)))
        c.commit(); c.close()
    return db


def one(policy, rep, secs, outdir, work, db):
    shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
    seed = 100 + rep
    c0 = ["taskset", "-c", "0"]; c1 = ["taskset", "-c", "1"]
    redis = subprocess.Popen(c0 + ["redis-server", "--port", "6391", "--save", "", "--appendonly", "no"],
                             stdout=subprocess.DEVNULL)
    node = subprocess.Popen(c0 + ["node", os.path.join(work, "node_server.js")])
    time.sleep(1.5)
    hogs = [subprocess.Popen(c0 + [os.path.join(HERE, "task"), "hog", str(secs + 1), f"{outdir}/hog{i}.txt"]) for i in range(3)]
    sq = subprocess.Popen(c0 + [PY, os.path.join(work, "sqlite_worker.py"), str(secs), str(seed), db, f"{outdir}/sqlite.txt"])
    servers = [("req", redis.pid), ("req", node.pid), ("req", sq.pid)]
    with open(f"{outdir}/pids.txt", "w") as f:
        for k, p in servers:
            f.write(f"{k} {p}\n")
        for h in hogs:
            f.write(f"hog {h.pid}\n")
    ag = None
    if policy == "static":
        for _, p in servers:
            setslice(p, 100000)
    elif policy.startswith("bpf_"):
        _, pred, c = policy.split("_")
        ag = subprocess.Popen(c0 + [os.path.join(HERE, "bpf", "cagent"), pred, c, f"{outdir}/pids.txt",
                                    str(secs + 1), f"{outdir}/agent.txt"])
    time.sleep(0.3)
    rc = subprocess.Popen(c1 + [PY, os.path.join(work, "redis_client.py"), str(secs), str(seed), f"{outdir}/redis.txt"])
    hc = subprocess.Popen(c1 + [PY, os.path.join(work, "http_client.py"), str(secs), str(seed + 1), f"{outdir}/http.txt"])
    for p in (rc, hc, sq):
        p.wait()
    for h in hogs:
        h.wait()
    if ag:
        ag.wait()
    redis.send_signal(signal.SIGTERM); node.send_signal(signal.SIGTERM); redis.wait(); node.wait()
    json.dump({"policy": policy, "rep": rep, "secs": secs}, open(f"{outdir}/meta.json", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--secs", type=float, default=20)
    ap.add_argument("--out", required=True); ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--rep0", type=int, default=0, help="first repetition index (to add repetitions later)")
    a = ap.parse_args()
    work = os.path.join(tempfile.gettempdir(), "koopman_real_closed")
    db = prepare(work)
    for rep in range(a.rep0, a.rep0 + a.reps):
        order = a.policies.split(","); random.Random(rep).shuffle(order)
        for pol in order:
            t = time.time()
            one(pol, rep, a.secs, os.path.join(a.out, f"r{rep}_{pol}"), work, db)
            print(f"rep {rep} {pol:16s} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
