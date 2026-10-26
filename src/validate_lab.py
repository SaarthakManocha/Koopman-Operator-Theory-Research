import sys, glob, os
sys.path.insert(0, "/home/claude/koopman-cpu-scheduling/src")
from multiprocessing import Pool
import pandas as pd
from validate import validate_capture
def one(d):
    try:
        df, meta = validate_capture(d); df["lost"] = meta.get("lost_events"); return df
    except Exception as e:
        print("FAIL", d, e); return None
if __name__ == "__main__":
    ds = sorted(d for d in glob.glob("/home/claude/lab1/captures/*") if os.path.exists(d + "/trace.txt.gz"))
    with Pool(8) as p: res = p.map(one, ds)
    df = pd.concat([r for r in res if r is not None]); df.to_csv("validation.csv", index=False)
    print(len(ds), "captures,", len(df), "processes")
    a = df.assign(r=df["run_err_%"].abs(), w=df["wait_err_%"].abs())
    print("run |err| median %.3f%%  p95 %.3f  max %.3f" % (a.r.median(), a.r.quantile(.95), a.r.max()))
    print("wait |err| median %.3f%%  p95 %.3f  max %.3f" % (a.w.median(), a.w.quantile(.95), a.w.max()))
    print("slices diff median", (a.slices_trace - a.slices_kernel).median(), "orphans total", a.orphan_switch_outs.sum(), "lost total", a.groupby("capture").lost.first().sum())
    a["fam"] = a.capture.str.replace(r"_c?\d+.*|_r\d", "", regex=True)
    print(a.groupby("fam").agg(n=("pid","size"), run_med=("r","median"), run_max=("r","max"), wait_med=("w","median"), wait_max=("w","max"), orph=("orphan_switch_outs","sum")).round(3))
    print(a.sort_values("r", ascending=False).head(8)[["capture","pid","kind","run_s_kernel","run_err_%","wait_err_%"]])
