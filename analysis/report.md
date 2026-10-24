# Benchmark results analysis

Analyzed 9 CSV files and 2240 measured rows (10 distinct repetition indices).
Warm-up rows were excluded; measurements are summarized by median. All included rows passed the benchmark's correctness flag.

## Baseline workload

At four threads, the fastest measured method at each skew was:

| Skew z | Fastest method | Median total time (s) | Static time (s) | Change vs static |
|---:|---|---:|---:|---:|
| 0 | static | 0.055524 | 0.055524 | +0.00% |
| 0.5 | adaptive2 | 0.054707 | 0.054789 | +0.15% |
| 0.75 | adaptive1 | 0.054172 | 0.054615 | +0.81% |
| 1 | paper | 0.053730 | 0.055150 | +2.57% |
| 1.25 | adaptive1 | 0.053793 | 0.054612 | +1.50% |
| 1.5 | paper | 0.054304 | 0.056829 | +4.44% |

Across the measured baseline skew values, average total-time change versus static at four threads (positive is faster): paper: +1.10%; adaptive1: +0.95%; adaptive2: +1.05%.

## Comparison with the paper-style scheduler

Here, **paper** means the paper-style scheduler implementation in this benchmark (`paper` rows), not independently measured results from the published research paper. Positive percentages mean the method was faster; negative percentages mean it was slower. Values below average over tested skew values at the maximum thread count for each workload.

| Workload | Threads | Static vs paper | Adaptive1 vs paper | Adaptive2 vs paper |
|---|---:|---:|---:|---:|
| Baseline | 4 | -1.15% | -0.17% | -0.06% |
| Probe-heavy | 4 | -1.82% | +0.35% | +0.08% |
| Many-to-many, R z=0.5 | 4 | -26.64% | +0.09% | -3.32% |
| Many-to-many, R z=0.75 | 4 | -23.06% | +0.92% | -0.89% |

The adaptive methods do not win uniformly: use the scenario-specific results rather than treating one average as a universal ranking.

## Scenario comparisons

- **Probe-heavy (S is 4x R), highest tested S-side skew z=1.5:** adaptive1 was fastest at 0.159944s, 4.89% lower than static.
- **Many-to-many (R-side z=0.5), highest tested S-side skew z=1.5:** adaptive1 was fastest at 0.030105s, 42.55% lower than static.
- **Many-to-many (R-side z=0.75), highest tested S-side skew z=1:** adaptive1 was fastest at 0.069467s, 46.84% lower than static.

## Plots

- [Baseline runtime by thread count](baseline_runtime_by_threads.png)
- [Baseline speedup and imbalance](baseline_speedup_and_imbalance.png)
- [Baseline scheduler diagnostics](baseline_scheduler_diagnostics.png)
- [Direct comparison to paper-style baseline](comparison_to_paper_baseline.png)
- [Baseline phase breakdown](baseline_phase_breakdown.png)
- [Probe-heavy workload](probe_heavy_comparison.png)
- [Many-to-many workload](many_to_many_runtime.png)
- [Round-robin initial-assignment ablation](round_robin_ablation.png)
- [Steal-threshold sweep](steal_threshold_sweep.png)

## Interpretation notes

- Results are medians of measured repetitions; warm-ups are excluded. The companion CSV also records standard deviation for total time.
- Comparisons are descriptive for this machine and these workloads; they are not statistical-significance tests.
- The sequential reference and workload differ between experiment families, so compare methods within a CSV/scenario rather than comparing absolute times across different files.
- Speedup uses the CSV's sequential-reference time divided by total algorithm time. Efficiency is that ratio divided by thread count.
