"""Shared aggregation for benchmark result CSV files."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_METRICS = {
    "total_s": ("total_s", "median"),
    "total_s_min": ("total_s", "min"),
    "total_s_max": ("total_s", "max"),
    "probe_s": ("probe_s", "median"),
    "plan_s": ("plan_s", "median"),
    "seq_s": ("seq_s", "median"),
    "lambda": ("lambda", "median"),
    "max_avg_part": ("max_avg_part", "median"),
    "tasks": ("tasks", "median"),
    "steals": ("steals", "median"),
    "steal_attempts": ("steal_attempts", "median"),
    "correct_share": ("correct", "mean"),
    "partition_s": ("partition_s", "median"),
    "build_s": ("build_s", "median"),
    "alpha": ("alpha", "median"),
    "beta": ("beta", "median"),
    "tau_ns": ("tau_ns", "median"),
    "overhead_ns": ("overhead_ns", "median"),
    "measured_reps": ("rep", "nunique"),
    "split_partitions": ("split_partitions", "median"),
    "max_chunks_per_partition": ("max_chunks_per_partition", "median"),
}
GROUP_METADATA = {
    "n": ("n", "first"),
    "ns": ("ns", "first"),
    "r_z": ("r_z", "first"),
    "init": ("init", "first"),
    "paper_factor": ("paper_factor", "first"),
    "paper_chunk": ("paper_chunk", "first"),
    "granularity": ("granularity", "first"),
    "min_chunk": ("min_chunk", "first"),
    "budget_per_thread": ("budget_per_thread", "first"),
    "include_build": ("include_build", "first"),
    "key_range": ("key_range", "first"),
    "seed_r": ("seed_r", "first"),
    "seed_s": ("seed_s", "first"),
}
COMPARISON_COLUMNS = {
    "static": ("improvement_total_vs_static_pct", "improvement_probe_vs_static_pct"),
    "paper": ("improvement_total_vs_paper_pct", "improvement_probe_vs_paper_pct"),
}


def validate_correctness(data, source):
    if "correct" not in data.columns:
        raise ValueError(f"{source}: required column 'correct' is missing")
    invalid = ~data["correct"].eq(1)
    if invalid.any():
        count = int(invalid.sum())
        raise ValueError(
            f"{source}: found {count} row(s) with correct != 1; "
            "analysis stopped."
        )


def summarize_results(data, group_columns):
    required = set(group_columns + ["warmup", "rep", "expand", "correct"])
    metric_aggregations = {
        name: aggregation
        for name, aggregation in GROUP_METRICS.items()
        if aggregation[0] in data.columns
    }
    optional_metrics = {"split_partitions", "max_chunks_per_partition"}
    required.update(
        source
        for name, (source, _) in GROUP_METRICS.items()
        if name not in optional_metrics
    )
    required.update(
        source
        for source, _ in GROUP_METADATA.values()
        if source in data.columns
    )
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    validate_correctness(data, "input CSV")
    measured = data.loc[data["warmup"] == 0].copy()
    if measured.empty:
        raise ValueError("input CSV has no non-warmup rows")

    result = (
        measured.groupby(group_columns, dropna=False)
        .agg(
            **metric_aggregations,
            **{
                name: aggregation
                for name, aggregation in GROUP_METADATA.items()
                if aggregation[0] in data.columns
            },
            expand=("expand", "first"),
        )
        .reset_index()
    )
    context_columns = [column for column in group_columns if column != "method"]
    result["steal_success_ratio"] = [
        steals / attempts if attempts else float("nan")
        for steals, attempts in zip(result["steals"], result["steal_attempts"])
    ]

    expand_mode = result["expand"].eq(1)
    result["speedup_vs_seq"] = float("nan")
    result["parallel_efficiency"] = float("nan")
    result.loc[~expand_mode, "speedup_vs_seq"] = (
        result.loc[~expand_mode, "seq_s"] / result.loc[~expand_mode, "total_s"]
    )
    result.loc[~expand_mode, "parallel_efficiency"] = (
        result.loc[~expand_mode, "speedup_vs_seq"]
        / result.loc[~expand_mode, "threads"]
    )

    for method, (total_column, probe_column) in COMPARISON_COLUMNS.items():
        baseline = result[result["method"] == method].set_index(context_columns)
        for output_column, metric in (
            (total_column, "total_s"),
            (probe_column, "probe_s"),
        ):
            baseline_values = baseline[metric].to_dict()
            improvements = []
            for _, row in result.iterrows():
                key = tuple(row[column] for column in context_columns)
                base = baseline_values.get(key)
                value = row[metric]
                if base is None or pd.isna(base) or base == 0:
                    improvements.append(float("nan"))
                else:
                    improvements.append(100.0 * (base - value) / base)
            result[output_column] = improvements

    return result


def print_summary(summary):
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 1000)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6g}",
            na_rep="N/A",
        )
    )


PLOT_METRICS = [
    ("total_s", "End-to-end time (s)", True),
    ("probe_s", "Probe time (s)", False),
    ("plan_s", "Planning time (s)", False),
    ("speedup_vs_seq", "Speedup vs sequential", False),
    ("parallel_efficiency", "Parallel efficiency", False),
    ("lambda", "Thread imbalance lambda", False),
    ("max_avg_part", "Partition imbalance", False),
    ("tasks", "Tasks", False),
    ("steals", "Successful steals", False),
    ("steal_success_ratio", "Steal success ratio", False),
    ("improvement_total_vs_static_pct", "Total improvement vs static (%)", False),
    ("improvement_total_vs_paper_pct", "Total improvement vs paper (%)", False),
    ("improvement_probe_vs_static_pct", "Probe improvement vs static (%)", False),
    ("improvement_probe_vs_paper_pct", "Probe improvement vs paper (%)", False),
    ("correct_share", "Correct runs (share)", False),
]


def save_metric_dashboards(summary, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouping = ["threads"]
    if "source" in summary.columns:
        grouping.insert(0, "source")

    outputs = []
    for key, subset in summary.groupby(grouping, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        context = dict(zip(grouping, key))
        plotted = [
            (column, title, spread)
            for column, title, spread in PLOT_METRICS
            if column in subset.columns
            and not (column in {"speedup_vs_seq", "parallel_efficiency"}
                     and subset["expand"].eq(1).all())
        ]
        columns = 4
        rows = (len(plotted) + columns - 1) // columns
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 3.4 * rows),
            squeeze=False,
        )
        for axis, (metric, title, show_spread) in zip(axes.flat, plotted):
            for method, data in subset.groupby("method"):
                data = data.sort_values("z")
                x = data["z"].to_numpy()
                y = data[metric].to_numpy(dtype=float)
                if show_spread:
                    lower = np.maximum(0, y - data["total_s_min"].to_numpy(dtype=float))
                    upper = np.maximum(0, data["total_s_max"].to_numpy(dtype=float) - y)
                    axis.errorbar(
                        x,
                        y,
                        yerr=np.vstack((lower, upper)),
                        marker="o",
                        capsize=3,
                        label=method,
                    )
                else:
                    axis.plot(x, y, marker="o", label=method)
            axis.set_title(title)
            axis.set_xlabel("Zipf skew z")
            axis.grid(alpha=0.25)
            if metric in {"improvement_total_vs_static_pct",
                          "improvement_total_vs_paper_pct",
                          "improvement_probe_vs_static_pct",
                          "improvement_probe_vs_paper_pct"}:
                axis.axhline(0, color="black", linewidth=0.7)
            if subset[metric].notna().any():
                axis.legend(fontsize="small")
            else:
                axis.text(
                    0.5,
                    0.5,
                    "Baseline not present",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

        for axis in axes.flat[len(plotted):]:
            axis.set_visible(False)

        title_parts = [f"{name}={value}" for name, value in context.items()]
        fig.suptitle("Metrics: " + ", ".join(title_parts))
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        safe_key = "_".join(str(value) for value in key)
        output = output_dir / f"{prefix}_{safe_key}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        outputs.append(output)
    return outputs
