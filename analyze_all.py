#!/usr/bin/env python3
"""Analyze the focused z=0.75 tuning results without derived speedup claims."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path("analysis")


def main():
    OUT.mkdir(exist_ok=True)
    paths = sorted(Path(".").glob("tuning_z075*.csv"))
    if not paths:
        print("ERROR: no tuning_z075*.csv files found; run bash run_tuning_z075.sh first")
        return 1

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"z", "threads", "method", "warmup", "correct", "total_s", "probe_s", "lambda"}
        missing = required - set(frame.columns)
        if missing:
            print(f"ERROR: {path}: missing columns: {sorted(missing)}")
            return 1
        if not frame["correct"].eq(1).all():
            print(f"ERROR: {path}: incorrect rows found")
            return 1
        frame = frame[(frame["warmup"] == 0) & (frame["z"] == 0.75)].copy()
        frame["candidate"] = path.stem.removeprefix("tuning_z075_")
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    summary = data.groupby(["candidate", "threads", "method"], dropna=False).agg(
        total_s_median=("total_s", "median"),
        total_s_min=("total_s", "min"),
        total_s_max=("total_s", "max"),
        probe_s_median=("probe_s", "median"),
        lambda_median=("lambda", "median"),
        measured_reps=("rep", "nunique"),
    ).reset_index()
    summary.to_csv(OUT / "z075_tuning_summary.csv", index=False)

    metrics = [
        ("total_s_median", "Median end-to-end time (s)"),
        ("probe_s_median", "Median probe time (s)"),
        ("lambda_median", "Median thread imbalance lambda"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4))
    for axis, (metric, title) in zip(axes, metrics):
        for method, group in summary.groupby("method"):
            group = group.sort_values("threads")
            axis.plot(group["threads"], group[metric], marker="o", label=method)
        axis.set_title(title)
        axis.set_xlabel("Threads")
        axis.set_xticks(sorted(summary["threads"].unique()))
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(OUT / "z075_tuning.png", dpi=160)
    plt.close(fig)

    report = [
        "# z=0.75 scheduler tuning", "",
        "Results are measured median timings for the focused S-side skew `z=0.75` experiment.",
        "No speedup, parallel efficiency, or percentage-improvement claims are calculated.", "",
        "## Summary", "",
        summary.to_string(index=False, float_format=lambda value: f"{value:.9g}"), "",
        "## Interpretation", "",
        "Compare absolute median total/probe times, thread imbalance, and correctness at the same thread count and candidate setting.",
    ]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Analyzed {len(paths)} z=0.75 result files and wrote analysis/z075_tuning_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
