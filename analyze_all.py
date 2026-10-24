#!/usr/bin/env python3
"""Aggregate every results*.csv file and generate a cross-experiment report."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "analysis"
GROUP_COLUMNS = ["source", "z", "threads", "method", "n", "ns", "expand", "r_z"]
MEDIAN_COLUMNS = [
    "partition_s",
    "build_s",
    "probe_s",
    "total_s",
    "seq_s",
    "lambda",
    "max_avg_part",
    "tasks",
    "steal_attempts",
    "steals",
    "plan_s",
    "tau_ns",
]


def load_results():
    paths = sorted(ROOT.glob("results*.csv"))
    if not paths:
        raise FileNotFoundError(f"No results*.csv files found in {ROOT}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = set(GROUP_COLUMNS[1:] + ["warmup", "rep", "correct"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        if not frame["correct"].eq(1).all():
            raise ValueError(f"{path.name} contains benchmark rows marked incorrect")
        frame = frame.loc[frame["warmup"] == 0].copy()
        if frame.empty:
            raise ValueError(f"{path.name} has no measured (non-warmup) rows")
        frame["source"] = path.stem
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), paths


def summarize(data):
    metrics = [column for column in MEDIAN_COLUMNS if column in data.columns]
    aggregations = {
        f"{column}_median": (column, "median") for column in metrics
    }
    aggregations["total_s_sd"] = ("total_s", "std")
    aggregations["measured_reps"] = ("rep", "nunique")
    aggregations["correct"] = ("correct", "min")
    result = (
        data.groupby(GROUP_COLUMNS, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    result["speedup_vs_seq"] = (
        result["seq_s_median"] / result["total_s_median"]
    )
    result["efficiency"] = result["speedup_vs_seq"] / result["threads"]
    paper_times = result[result["method"] == "paper"].set_index(
        ["source", "z", "threads"]
    )["total_s_median"]
    result["vs_paper_%"] = [
        100 * (paper_times[key] - runtime) / paper_times[key]
        if key in paper_times.index
        else float("nan")
        for key, runtime in zip(
            zip(result["source"], result["z"], result["threads"]),
            result["total_s_median"],
        )
    ]
    return result


def line_plot(ax, data, x_column, y_column, title, ylabel):
    for method, group in data.groupby("method"):
        group = group.sort_values(x_column)
        ax.plot(
            group[x_column].to_numpy(),
            group[y_column].to_numpy(),
            marker="o",
            label=method,
        )
    ax.set_title(title)
    ax.set_xlabel("Zipf skew z")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()


def save_baseline_plots(summary):
    baseline = summary[summary["source"] == "results"]
    if baseline.empty:
        return

    thread_counts = sorted(baseline["threads"].unique())
    fig, axes = plt.subplots(
        1, len(thread_counts), figsize=(5 * len(thread_counts), 4), squeeze=False
    )
    for ax, thread_count in zip(axes[0], thread_counts):
        subset = baseline[baseline["threads"] == thread_count]
        line_plot(
            ax,
            subset,
            "z",
            "total_s_median",
            f"End-to-end time ({thread_count} threads)",
            "Median seconds",
        )
    fig.suptitle("Baseline workload: scheduler runtime vs skew")
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_runtime_by_threads.png", dpi=160)
    plt.close(fig)

    max_threads = max(thread_counts)
    subset = baseline[baseline["threads"] == max_threads]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line_plot(
        axes[0],
        subset,
        "z",
        "speedup_vs_seq",
        f"Speedup vs sequential reference ({max_threads} threads)",
        "Sequential time / total time",
    )
    line_plot(
        axes[1],
        subset,
        "z",
        "lambda_median",
        f"Partition imbalance ({max_threads} threads)",
        "Median lambda (max/average partition size)",
    )
    line_plot(
        axes[2],
        subset,
        "z",
        "efficiency",
        f"Efficiency ({max_threads} threads)",
        "Speedup / thread count",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_speedup_and_imbalance.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    line_plot(
        axes[0],
        subset,
        "z",
        "tasks_median",
        f"Scheduled task count ({max_threads} threads)",
        "Median tasks",
    )
    line_plot(
        axes[1],
        subset,
        "z",
        "steals_median",
        f"Work-stealing activity ({max_threads} threads)",
        "Median successful steals",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_scheduler_diagnostics.png", dpi=160)
    plt.close(fig)

    phases = [
        ("partition_s_median", "partition"),
        ("build_s_median", "build"),
        ("plan_s_median", "planning"),
        ("probe_s_median", "probe"),
    ]
    skew_values = sorted(subset["z"].unique())
    selected_skews = [skew_values[0]]
    if skew_values[-1] != skew_values[0]:
        selected_skews.append(skew_values[-1])
    fig, axes = plt.subplots(
        1, len(selected_skews), figsize=(6 * len(selected_skews), 4), squeeze=False
    )
    for ax, skew in zip(axes[0], selected_skews):
        rows = subset[subset["z"] == skew].sort_values("method")
        bottom = None
        for column, label in phases:
            if column not in rows.columns:
                continue
            values = rows[column].fillna(0).to_numpy()
            ax.bar(rows["method"], values, bottom=bottom, label=label)
            bottom = values if bottom is None else bottom + values
        ax.set_title(f"Phase times at z={skew}, {max_threads} threads")
        ax.set_ylabel("Median seconds")
        ax.tick_params(axis="x", rotation=25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "baseline_phase_breakdown.png", dpi=160)
    plt.close(fig)


def save_scenario_plot(summary, source, filename, title):
    data = summary[
        (summary["source"] == source)
        & (summary["threads"] == summary.loc[summary["source"] == source, "threads"].max())
    ]
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    line_plot(axes[0], data, "z", "total_s_median", "End-to-end time", "Median seconds")
    line_plot(axes[1], data, "z", "probe_s_median", "Probe phase", "Median seconds")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=160)
    plt.close(fig)


def save_expansion_plot(summary):
    data = summary[
        summary["source"].isin(["results_expand_rz05", "results_expand_rz075"])
    ]
    if data.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, source in zip(axes, ["results_expand_rz05", "results_expand_rz075"]):
        scenario = data[
            (data["source"] == source)
            & (data["threads"] == data.loc[data["source"] == source, "threads"].max())
        ]
        line_plot(
            ax,
            scenario,
            "z",
            "total_s_median",
            f"R-side skew z={scenario['r_z'].iloc[0]}",
            "Median end-to-end seconds",
        )
    fig.suptitle("Many-to-many workload: R and S key skew")
    fig.tight_layout()
    fig.savefig(OUTPUT / "many_to_many_runtime.png", dpi=160)
    plt.close(fig)


def save_ablation_plot(summary):
    baseline = summary[
        (summary["source"] == "results")
        & (summary["threads"] == summary.loc[summary["source"] == "results", "threads"].max())
        & summary["method"].isin(["adaptive1", "adaptive2"])
        & summary["z"].isin([1.0, 1.5])
    ].copy()
    round_robin = summary[
        (summary["source"] == "results_ablation_rr")
        & (summary["threads"] == summary.loc[summary["source"] == "results_ablation_rr", "threads"].max())
    ].copy()
    if baseline.empty or round_robin.empty:
        return
    baseline["initial_assignment"] = "LPT (standard)"
    round_robin["initial_assignment"] = "round robin"
    data = pd.concat([baseline, round_robin], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, ["adaptive1", "adaptive2"]):
        scenario = data[data["method"] == method]
        for assignment, group in scenario.groupby("initial_assignment"):
            group = group.sort_values("z")
            ax.plot(
                group["z"].to_numpy(),
                group["total_s_median"].to_numpy(),
                marker="o",
                label=assignment,
            )
        ax.set_title(method)
        ax.set_xlabel("Zipf skew z")
        ax.set_ylabel("Median end-to-end seconds")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Adaptive scheduler initial-assignment ablation (4 threads)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "round_robin_ablation.png", dpi=160)
    plt.close(fig)


def save_threshold_plot(summary):
    data = summary[summary["source"].str.startswith("results_tau_")].copy()
    if data.empty:
        return
    data = data.sort_values("tau_ns_median")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(data["tau_ns_median"], data["total_s_median"], marker="o", label="total")
    axes[0].plot(data["tau_ns_median"], data["probe_s_median"], marker="o", label="probe")
    axes[0].set_title("Runtime by configured steal threshold")
    axes[0].set_xlabel("Steal threshold (ns, log scale)")
    axes[0].set_ylabel("Median seconds")
    axes[0].set_xscale("log")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(data["tau_ns_median"], data["steals_median"], marker="o", label="steals")
    axes[1].plot(
        data["tau_ns_median"],
        data["steal_attempts_median"],
        marker="o",
        label="steal attempts",
    )
    axes[1].set_title("Work-stealing activity")
    axes[1].set_xlabel("Steal threshold (ns, log scale)")
    axes[1].set_ylabel("Median count per run")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Adaptive2 steal-threshold sweep")
    fig.tight_layout()
    fig.savefig(OUTPUT / "steal_threshold_sweep.png", dpi=160)
    plt.close(fig)


def save_paper_comparison_plot(summary):
    comparable = summary[
        summary["source"].isin(
            ["results", "results_probeheavy", "results_expand_rz05", "results_expand_rz075"]
        )
        & (summary["threads"] == summary.groupby("source")["threads"].transform("max"))
        & summary["method"].isin(["static", "adaptive1", "adaptive2"])
    ].copy()
    if comparable.empty:
        return

    sources = [
        "results",
        "results_probeheavy",
        "results_expand_rz05",
        "results_expand_rz075",
    ]
    titles = {
        "results": "Baseline",
        "results_probeheavy": "Probe-heavy",
        "results_expand_rz05": "Many-to-many R z=0.5",
        "results_expand_rz075": "Many-to-many R z=0.75",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    baseline = comparable[comparable["source"] == "results"]
    for method, group in baseline.groupby("method"):
        group = group.sort_values("z")
        axes[0].plot(
            group["z"].to_numpy(),
            group["vs_paper_%"].to_numpy(),
            marker="o",
            label=method,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Baseline by S-side skew (4 threads)")
    axes[0].set_xlabel("Zipf skew z")
    axes[0].set_ylabel("Faster than paper-style baseline (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    scenario = comparable[comparable["source"].isin(sources)]
    averages = (
        scenario.groupby(["source", "method"])["vs_paper_%"]
        .mean()
        .unstack("method")
        .reindex(sources)
    )
    averages.plot(kind="bar", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Mean change vs paper-style baseline")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Average across tested skews (%)")
    axes[1].set_xticklabels([titles[source] for source in sources], rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(title="Method")
    fig.suptitle("Direct comparison with the implemented paper-style scheduler")
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison_to_paper_baseline.png", dpi=160)
    plt.close(fig)


def build_report(data, summary, paths):
    lines = [
        "# Benchmark results analysis",
        "",
        f"Analyzed {len(paths)} CSV files and {len(data)} measured rows "
        f"({data['rep'].nunique()} distinct repetition indices).",
        "Warm-up rows were excluded; measurements are summarized by median. "
        "All included rows passed the benchmark's correctness flag.",
        "",
        "## Baseline workload",
        "",
        "At four threads, the fastest measured method at each skew was:",
        "",
        "| Skew z | Fastest method | Median total time (s) | Static time (s) | Change vs static |",
        "|---:|---|---:|---:|---:|",
    ]
    baseline = summary[
        (summary["source"] == "results")
        & (summary["threads"] == summary.loc[summary["source"] == "results", "threads"].max())
    ]
    for skew in sorted(baseline["z"].unique()):
        rows = baseline[baseline["z"] == skew]
        best = rows.loc[rows["total_s_median"].idxmin()]
        static = rows[rows["method"] == "static"]
        if static.empty:
            static_time = float("nan")
            change = float("nan")
        else:
            static_time = static["total_s_median"].iloc[0]
            change = 100 * (static_time - best["total_s_median"]) / static_time
        lines.append(
            f"| {skew:g} | {best['method']} | {best['total_s_median']:.6f} "
            f"| {static_time:.6f} | {change:+.2f}% |"
        )

    if not baseline.empty and "static" in set(baseline["method"]):
        pivot = baseline.pivot(index="z", columns="method", values="total_s_median")
        comparisons = []
        for method in ["paper", "adaptive1", "adaptive2"]:
            if method in pivot.columns:
                delta = 100 * (pivot["static"] - pivot[method]) / pivot["static"]
                comparisons.append(f"{method}: {delta.mean():+.2f}%")
        lines.extend(
            [
                "",
                "Across the measured baseline skew values, average total-time change "
                "versus static at four threads (positive is faster): "
                + "; ".join(comparisons)
                + ".",
            ]
        )

    lines.extend(
        [
            "",
            "## Comparison with the paper-style scheduler",
            "",
            "Here, **paper** means the paper-style scheduler implementation in this "
            "benchmark (`paper` rows), not independently measured results from "
            "the published research paper. Positive percentages mean the method "
            "was faster; negative percentages mean it was slower. Values below "
            "average over tested skew values at the maximum thread count for "
            "each workload.",
            "",
            "| Workload | Threads | Static vs paper | Adaptive1 vs paper | Adaptive2 vs paper |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for source, label in [
        ("results", "Baseline"),
        ("results_probeheavy", "Probe-heavy"),
        ("results_expand_rz05", "Many-to-many, R z=0.5"),
        ("results_expand_rz075", "Many-to-many, R z=0.75"),
    ]:
        rows = summary[summary["source"] == source]
        if rows.empty or "paper" not in set(rows["method"]):
            continue
        thread_count = rows["threads"].max()
        rows = rows[rows["threads"] == thread_count]
        deltas = rows.groupby("method")["vs_paper_%"].mean()
        cells = []
        for method in ["static", "adaptive1", "adaptive2"]:
            value = deltas.get(method, float("nan"))
            cells.append(f"{value:+.2f}%" if pd.notna(value) else "n/a")
        lines.append(
            f"| {label} | {thread_count:g} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )

    lines.extend(
        [
            "",
            "The adaptive methods do not win uniformly: use the scenario-specific "
            "results rather than treating one average as a universal ranking.",
        ]
    )

    lines.extend(["", "## Scenario comparisons", ""])
    scenario_specs = [
        ("results_probeheavy", "Probe-heavy (S is 4x R)"),
        ("results_expand_rz05", "Many-to-many (R-side z=0.5)"),
        ("results_expand_rz075", "Many-to-many (R-side z=0.75)"),
    ]
    for source, label in scenario_specs:
        scenario = summary[summary["source"] == source]
        if scenario.empty:
            continue
        scenario = scenario[
            scenario["threads"] == scenario["threads"].max()
        ]
        skew = scenario["z"].max()
        rows = scenario[scenario["z"] == skew]
        best = rows.loc[rows["total_s_median"].idxmin()]
        static = rows[rows["method"] == "static"]
        if not static.empty:
            change = 100 * (
                static["total_s_median"].iloc[0] - best["total_s_median"]
            ) / static["total_s_median"].iloc[0]
            lines.append(
                f"- **{label}, highest tested S-side skew z={skew:g}:** "
                f"{best['method']} was fastest at {best['total_s_median']:.6f}s, "
                f"{change:.2f}% lower than static."
            )

    lines.extend(
        [
            "",
            "## Plots",
            "",
            "- [Baseline runtime by thread count](baseline_runtime_by_threads.png)",
            "- [Baseline speedup and imbalance](baseline_speedup_and_imbalance.png)",
            "- [Baseline scheduler diagnostics](baseline_scheduler_diagnostics.png)",
            "- [Direct comparison to paper-style baseline](comparison_to_paper_baseline.png)",
            "- [Baseline phase breakdown](baseline_phase_breakdown.png)",
            "- [Probe-heavy workload](probe_heavy_comparison.png)",
            "- [Many-to-many workload](many_to_many_runtime.png)",
            "- [Round-robin initial-assignment ablation](round_robin_ablation.png)",
            "- [Steal-threshold sweep](steal_threshold_sweep.png)",
            "",
            "## Interpretation notes",
            "",
            "- Results are medians of measured repetitions; warm-ups are excluded. "
            "The companion CSV also records standard deviation for total time.",
            "- Comparisons are descriptive for this machine and these workloads; "
            "they are not statistical-significance tests.",
            "- The sequential reference and workload differ between experiment "
            "families, so compare methods within a CSV/scenario rather than "
            "comparing absolute times across different files.",
            "- Speedup uses the CSV's sequential-reference time divided by total "
            "algorithm time. Efficiency is that ratio divided by thread count.",
        ]
    )
    (OUTPUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUTPUT.mkdir(exist_ok=True)
    data, paths = load_results()
    summary = summarize(data)
    summary.to_csv(OUTPUT / "all_experiments_summary.csv", index=False)

    save_baseline_plots(summary)
    save_scenario_plot(
        summary,
        "results_probeheavy",
        "probe_heavy_comparison.png",
        "Probe-heavy workload (S is four times R)",
    )
    save_expansion_plot(summary)
    save_ablation_plot(summary)
    save_threshold_plot(summary)
    save_paper_comparison_plot(summary)
    build_report(data, summary, paths)

    print(f"Analyzed {len(paths)} CSV files and {len(data)} measured rows.")
    print(f"Correctness flags: {sorted(data['correct'].unique().tolist())}")
    print(f"Report and plots written to {OUTPUT}")


if __name__ == "__main__":
    main()
