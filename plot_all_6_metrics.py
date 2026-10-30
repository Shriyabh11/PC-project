#!/usr/bin/env python3
"""Generate comprehensive plots for all 6 evaluation metrics across all skew levels (z),
including per-z thread scaling panels and skew sweeps, highlighting z=0.75 performance.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("analysis/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

METHOD_COLORS = {
    "static": "#1f77b4",    # Blue
    "paper": "#ff7f0e",     # Orange
    "adaptive1": "#2ca02c", # Green
    "adaptive2": "#d62728", # Red
}
METHOD_MARKERS = {
    "static": "o",
    "paper": "s",
    "adaptive1": "^",
    "adaptive2": "D",
}


def load_data(csv_path="results.csv"):
    df = pd.read_csv(csv_path)
    measured = df[df["warmup"] == 0].copy()
    
    # Aggregation
    agg = measured.groupby(["z", "threads", "method"]).agg(
        probe_ms=("probe_s", lambda x: x.median() * 1000),
        total_ms=("total_s", lambda x: x.median() * 1000),
        plan_ms=("plan_s", lambda x: x.median() * 1000),
        lam=("lambda", "median"),
        max_avg_part=("max_avg_part", "median"),
        tasks=("tasks", "median"),
        steals=("steals", "median"),
        steal_attempts=("steal_attempts", "median"),
        correct=("correct", "mean"),
    ).reset_index()

    # Calculate Speedup and Efficiency vs 1 thread of the same method
    t1_map = {}
    for (z, method), group in agg[agg["threads"] == 1].groupby(["z", "method"]):
        t1_map[(z, method)] = group["probe_ms"].values[0]

    agg["speedup"] = [
        t1_map[(z, m)] / t if (z, m) in t1_map and t > 0 else 1.0
        for z, m, t in zip(agg["z"], agg["method"], agg["probe_ms"])
    ]
    agg["efficiency"] = (agg["speedup"] / agg["threads"]) * 100.0
    agg["cpu_util"] = (1.0 / agg["lam"]) * 100.0

    return agg


def plot_6_metrics_for_z(df_agg, z_val):
    """Plot the 6 evaluation metrics for a specific z level across thread counts."""
    sub = df_agg[df_agg["z"] == z_val].copy()
    if sub.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    threads = sorted(sub["threads"].unique())

    # Metric 1: Execution Time (Probe Time)
    ax = axes[0, 0]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["probe_ms"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=1.8)
    ax.set_title("1. Execution Time (Probe Phase)")
    ax.set_ylabel("Probe Time (ms)")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Metric 2: Speedup (Probe Phase vs 1 Thread)
    ax = axes[0, 1]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["speedup"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=1.8)
    ax.plot(threads, threads, "k--", alpha=0.5, label="Ideal linear")
    ax.set_title("2. Parallel Speedup (vs. 1 Thread)")
    ax.set_ylabel("Speedup factor")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Metric 3: Parallel Efficiency (%)
    ax = axes[0, 2]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["efficiency"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=1.8)
    ax.axhline(100, color="k", linestyle="--", alpha=0.5)
    ax.set_title("3. Parallel Efficiency (%)")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Metric 4: Load Imbalance (Lambda = Max / Avg Finish Time)
    ax = axes[1, 0]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["lam"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=1.8)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Perfect balance (1.0)")
    ax.set_title("4. Load Imbalance (λ = Max / Avg Time)")
    ax.set_ylabel("Imbalance factor λ")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Metric 5: CPU Utilization (%)
    ax = axes[1, 1]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["cpu_util"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=1.8)
    ax.axhline(100, color="k", linestyle="--", alpha=0.5)
    ax.set_title("5. CPU Utilization (1/λ %)")
    ax.set_ylabel("Active Worker Time (%)")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # Metric 6: Scheduling & Redistribution Overhead (Tasks & Steals)
    ax = axes[1, 2]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["tasks"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=f"{method} tasks", lw=1.8)
    ax.set_title("6. Scheduling Overhead (Total Tasks)")
    ax.set_ylabel("Tasks Created")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.suptitle(f"Evaluation Metrics Comparison at Zipf Skew z = {z_val}", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / f"metrics_6_z_{str(z_val).replace('.', '_')}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_6_metrics_across_z(df_agg, p=4):
    """Plot the 6 evaluation metrics across skew levels z for a fixed thread count (e.g. 4 threads)."""
    sub = df_agg[df_agg["threads"] == p].copy()
    if sub.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    zs = sorted(sub["z"].unique())

    # 1. Execution Time (Probe Time)
    ax = axes[0, 0]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["probe_ms"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.set_title(f"1. Execution Time (Probe Phase, p={p})")
    ax.set_ylabel("Probe Time (ms)")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Parallel Speedup vs 1 Thread
    ax = axes[0, 1]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["speedup"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.axhline(p, color="k", linestyle="--", alpha=0.5, label=f"Ideal {p}x")
    ax.set_title(f"2. Parallel Speedup (vs. 1 Thread, p={p})")
    ax.set_ylabel("Speedup factor")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 3. Parallel Efficiency
    ax = axes[0, 2]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["efficiency"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.set_title(f"3. Parallel Efficiency (%, p={p})")
    ax.set_ylabel("Efficiency (%)")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 4. Load Imbalance Lambda
    ax = axes[1, 0]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["lam"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Perfect balance")
    ax.set_title(f"4. Load Imbalance (λ = Max/Avg Time, p={p})")
    ax.set_ylabel("Imbalance λ")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 5. CPU Utilization
    ax = axes[1, 1]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["cpu_util"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.set_title(f"5. CPU Utilization (1/λ %, p={p})")
    ax.set_ylabel("Active Worker Time (%)")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 6. Tasks and Stealing Overhead
    ax = axes[1, 2]
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("z")
        ax.plot(grp["z"], grp["tasks"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=f"{method} tasks", lw=2.0)
    ax.set_title(f"6. Task Generation Count (Budget Control, p={p})")
    ax.set_ylabel("Tasks Created")
    ax.set_xlabel("Zipf Skew z")
    ax.set_xticks(zs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.suptitle(f"Evaluation Metrics Across All Skew Levels z (4 Threads)", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / f"metrics_6_across_z_p{p}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_z075_focus_comparison():
    """Detailed comparison highlighting z=0.75 performance in count-only and expand workloads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Count-only at z=0.75 across threads
    df = pd.read_csv("results.csv")
    sub = df[(df["warmup"] == 0) & (df["z"] == 0.75)].groupby(["threads", "method"]).agg(
        probe_ms=("probe_s", lambda x: x.median() * 1000),
        lam=("lambda", "median"),
    ).reset_index()

    ax = axes[0]
    threads = sorted(sub["threads"].unique())
    for method, grp in sub.groupby("method"):
        grp = grp.sort_values("threads")
        ax.plot(grp["threads"], grp["probe_ms"], marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method), label=method, lw=2.0)
    ax.set_title("Count-Only Workload (z = 0.75)")
    ax.set_ylabel("Probe Time (ms)")
    ax.set_xlabel("Threads")
    ax.set_xticks(threads)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # 2. Expand mode with r_z=0.75, z=0.75 (Many-to-many join)
    try:
        df_exp = pd.read_csv("results_expand_rz075.csv")
        sub_exp = df_exp[(df_exp["warmup"] == 0) & (df_exp["z"] == 0.75)].groupby(["threads", "method"]).agg(
            probe_ms=("probe_s", lambda x: x.median() * 1000),
            lam=("lambda", "median"),
        ).reset_index()

        ax = axes[1]
        t_exp = sorted(sub_exp["threads"].unique())
        for method, grp in sub_exp.groupby("method"):
            grp = grp.sort_values("threads")
            ax.plot(grp["threads"], grp["probe_ms"], marker=METHOD_MARKERS.get(method, "o"),
                    color=METHOD_COLORS.get(method), label=f"{method} (λ={grp['lam'].iloc[-1]:.2f})", lw=2.0)
        ax.set_title("Many-to-Many Workload (R skew = 0.75, S skew = 0.75)")
        ax.set_ylabel("Probe Time (ms)")
        ax.set_xlabel("Threads")
        ax.set_xticks(t_exp)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    except Exception as e:
        axes[1].text(0.5, 0.5, f"Expand data not found: {e}", ha="center", va="center")

    fig.suptitle("Performance Focus: How Adaptive Outperforms at z = 0.75", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "z075_focus_comparison.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main():
    print("Loading benchmark results from results.csv...")
    df_agg = load_data("results.csv")
    
    unique_zs = sorted(df_agg["z"].unique())
    print(f"Generating 6-metric plots for each z in: {unique_zs}")
    
    generated = []
    for z in unique_zs:
        path = plot_6_metrics_for_z(df_agg, z)
        if path:
            print(f"  Saved: {path}")
            generated.append(path)
            
    print("Generating 6-metric sweep across all z (4 threads)...")
    path_sweep = plot_6_metrics_across_z(df_agg, p=4)
    if path_sweep:
        print(f"  Saved: {path_sweep}")
        generated.append(path_sweep)

    print("Generating z=0.75 focus comparison plot...")
    path_focus = plot_z075_focus_comparison()
    if path_focus:
        print(f"  Saved: {path_focus}")
        generated.append(path_focus)

    print("\nAll plots generated successfully under analysis/plots/.")


if __name__ == "__main__":
    main()
