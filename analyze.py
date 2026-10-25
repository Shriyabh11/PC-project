#!/usr/bin/env python3
"""Summarise results.csv: medians over measured reps, speedup, efficiency, improvement, plots."""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from result_metrics import (
    print_summary,
    save_metric_dashboards,
    summarize_results,
)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results.csv")
    try:
        df = pd.read_csv(src)
        summary = summarize_results(df, ["z", "threads", "method"])
    except (OSError, pd.errors.ParserError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    summary.to_csv("summary.csv", index=False)
    print_summary(summary)
    dashboards = save_metric_dashboards(summary, Path("."), "metrics")
    print("Metric dashboards:")
    for dashboard in dashboards:
        print(f"  {dashboard}")

    pmax = summary["threads"].max()
    subset = summary[summary["threads"] == pmax]
    plots = [
        ("lambda", "Thread imbalance lambda", "lambda.png"),
        ("probe_s", "Probe-phase time (s)", "probe_time.png"),
        ("total_s", "End-to-end time (s)", "total_time.png"),
    ]
    if not summary["speedup_vs_seq"].isna().all():
        plots.append(
            ("speedup_vs_seq", "Speedup vs sequential reference", "speedup.png")
        )
    for column, title, filename in plots:
        plt.figure(figsize=(6, 4))
        for method, group in subset.groupby("method"):
            plt.plot(
                group["z"].to_numpy(),
                group[column].to_numpy(),
                marker="o",
                label=method,
            )
        plt.xlabel("Zipf skew z")
        plt.ylabel(title)
        plt.title(f"{title} ({pmax} threads)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    phases = df[df["warmup"] == 0].groupby(["z", "threads", "method"])[
        ["partition_s", "build_s", "plan_s", "probe_s"]
    ].median().reset_index()
    skew_values = sorted(phases["z"].unique())
    selected_skews = [skew_values[0], skew_values[-1]] if len(skew_values) > 1 else skew_values
    fig, axes = plt.subplots(
        1, len(selected_skews), figsize=(5 * len(selected_skews), 4), squeeze=False
    )
    for axis, skew in zip(axes[0], selected_skews):
        group = phases[(phases["z"] == skew) & (phases["threads"] == pmax)].set_index("method")
        bottom = None
        for column, label in [
            ("partition_s", "partition"),
            ("build_s", "build"),
            ("plan_s", "planning"),
            ("probe_s", "probe"),
        ]:
            values = group[column].to_numpy()
            axis.bar(list(group.index), values, bottom=bottom, label=label)
            bottom = values if bottom is None else bottom + values
        axis.set_title(f"Phase breakdown, z={skew}, {pmax} threads")
        axis.set_ylabel("Seconds")
        axis.legend()
    plt.tight_layout()
    plt.savefig("phases.png", dpi=150)
    plt.close()
    print(
        "\nWrote summary.csv, lambda.png, probe_time.png, total_time.png, "
        "phases.png" + (", speedup.png" if "speedup.png" in [p[2] for p in plots] else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())