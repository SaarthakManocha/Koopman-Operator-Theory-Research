"""plot_real_programs.py - burst-prediction error per real program -> outputs/figures_ext/e8_real_programs.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
p = pd.read_csv(os.path.join(ROOT, "outputs", "real_programs", "prediction.csv"))
meths = [("Log EDMD one-step", "log-EDMD (Koopman)", "#2a78d6"), ("Poly EDMD one-step", "poly-EDMD (Koopman)", "#8db7ea"),
         ("Gradient boosting", "gradient boosting", "#1baf7a"), ("EWMA tuned", "EWMA, tuned alpha", "#eb6834")]
kinds = ["redis", "sqlite", "node", "ffmpeg", "client"]
lab = {"redis": "Redis", "sqlite": "SQLite", "node": "Node.js HTTP", "ffmpeg": "ffmpeg H.264", "client": "load generators"}
fig, ax = plt.subplots(figsize=(7.2, 2.9)); w = 0.2
for j, (m, l, c) in enumerate(meths):
    for i, k in enumerate(kinds):
        g = p[(p.method == m) & (p.kind == k)].vs_ewma; x = i + (j - 1.5) * w
        ax.bar(x, g.median(), w * 0.9, color=c, label=l if i == 0 else None)
        ax.scatter([x] * len(g), g, s=5, color="#0b0b0b", zorder=3)
ax.axhline(1, color="#0b0b0b", lw=0.8); ax.text(-0.45, 1.02, "textbook EWMA = 1", fontsize=7)
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels([f"{lab[k]}\n({p[(p.method == meths[0][0]) & (p.kind == k)].shape[0]} threads)" for k in kinds])
ax.set_ylabel("prediction error\nrelative to EWMA"); ax.set_ylim(0, 1.45)
ax.legend(fontsize=7, ncol=4, loc="upper left", frameon=False, bbox_to_anchor=(0, 1.12))
fig.tight_layout(); fig.savefig(os.path.join(ROOT, "outputs", "figures_ext", "e8_real_programs.png"), dpi=200, bbox_inches="tight")
print("wrote e8_real_programs.png")
