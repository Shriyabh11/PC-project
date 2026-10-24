#!/usr/bin/env python3
"""Summarise results.csv: medians over measured reps, speedup, efficiency, improvement, plots."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
df = pd.read_csv(src)
if not df["correct"].all():
    print("WARNING: some runs were INCORRECT - fix before trusting any numbers")
df = df[df["warmup"] == 0]

g = df.groupby(["z", "threads", "method"]).agg(
    total_s=("total_s", "median"), probe_s=("probe_s", "median"), seq_s=("seq_s", "first"),
    lam=("lambda", "median"), max_avg=("max_avg_part", "first"), tasks=("tasks", "median"),
    steals=("steals", "median"), steal_att=("steal_attempts", "median")).reset_index()
g["speedup_vs_seq"] = g["seq_s"] / g["total_s"]
g["efficiency"] = g["speedup_vs_seq"] / g["threads"]

def improvement(base):
    b = g[g["method"] == base].set_index(["z", "threads"])["total_s"]
    key = list(zip(g["z"], g["threads"]))
    return [100.0 * (b[k] - t) / b[k] if k in b.index else float("nan") for k, t in zip(key, g["total_s"])]

g["impr_over_static_%"] = improvement("static")
g["impr_over_paper_%"] = improvement("paper")
g["probe_impr_over_static_%"] = [
    100.0 * (s - p) / s for s, p in zip(
        [g[(g.z == z) & (g.threads == t) & (g.method == "static")]["probe_s"].iloc[0] for z, t in zip(g.z, g.threads)],
        g["probe_s"])]
g.to_csv("summary.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(g.round(3).to_string(index=False))

pmax = g["threads"].max()
sub = g[g["threads"] == pmax]
for col, title, fn in [("speedup_vs_seq", "Speedup vs sequential reference", "speedup.png"),
                       ("lam", "Thread imbalance lambda = max/avg finish time", "lambda.png"),
                       ("probe_s", "Probe-phase time (s)", "probe_time.png"),
                       ("total_s", "End-to-end time (s)", "total_time.png")]:
    plt.figure(figsize=(6, 4))
    for m, d in sub.groupby("method"):
        plt.plot(d["z"].to_numpy(), d[col].to_numpy(), marker="o", label=m)
    plt.xlabel("Zipf skew z"); plt.ylabel(title); plt.title(f"{title} ({pmax} threads)")
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(fn, dpi=150); plt.close()
# phase breakdown: where does the time go? (lowest and highest skew, max threads)
cols = [c for c in ["partition_s", "build_s", "plan_s", "probe_s"] if c in df.columns]
ph = df.groupby(["z", "threads", "method"])[cols].median().reset_index()
zs = sorted(ph["z"].unique()); picks = [zs[0], zs[-1]] if len(zs) > 1 else zs
fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4), squeeze=False)
for ax, z in zip(axes[0], picks):
    d = ph[(ph.z == z) & (ph.threads == pmax)].set_index("method")
    bottom = None
    for col, lab in [("partition_s", "partition"), ("build_s", "build"), ("plan_s", "planning"), ("probe_s", "probe")]:
        if col not in d.columns: continue
        vals = d[col].to_numpy()
        ax.bar(list(d.index), vals, bottom=bottom, label=lab)
        bottom = vals if bottom is None else bottom + vals
    ax.set_title(f"Phase breakdown, z={z}, {pmax} threads"); ax.set_ylabel("seconds"); ax.legend()
plt.tight_layout(); plt.savefig("phases.png", dpi=150); plt.close()
print("\nWrote summary.csv, speedup.png, lambda.png, probe_time.png, total_time.png, phases.png")