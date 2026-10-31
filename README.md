# Skew-Resilient Parallel Hash Joins via Adaptive Fine-Grained Scheduling

A C++20 and OpenMP implementation and empirical evaluation of parallel hash joins under skewed workloads.

This project investigates how partition-level workload imbalance affects parallel hash join performance and evaluates adaptive task decomposition and work-stealing strategies for reducing probe-phase stragglers.

The implementation is inspired by the methodology in:

> Balkesen et al., *Main-Memory Hash Joins on Multi-Core CPUs: Tuning to the Underlying Hardware*, ICDE 2013.

The project does not reproduce every algorithmic detail of the original paper. In particular, the `paper` scheduler is a **paper-inspired approximation** used as a comparison baseline.

---

## 1. Project Overview

A parallel hash join typically consists of:

1. **Partitioning:** Divide the build and probe relations into smaller partitions.
2. **Build:** Construct a hash table for each build-side partition.
3. **Probe:** Probe the corresponding hash table using tuples from the probe-side partition.

When the probe relation follows a skewed key distribution, some partitions become significantly larger than others. Under static scheduling, threads assigned smaller partitions may finish early while other threads continue processing oversized partitions.

This creates a load imbalance problem.

The project evaluates four scheduling strategies:

| Method | Description |
|---|---|
| `static` | Assigns partitions to threads using OpenMP static scheduling. |
| `paper` | Uses fixed-size chunks for oversized partitions and a shared task queue. This is a simplified, paper-inspired baseline, not an exact reproduction of Balkesen et al.'s implementation. |
| `adaptive1` | Uses a cost-based partition splitting strategy and a shared dynamic task queue. |
| `adaptive2` | Uses adaptive task decomposition, per-worker Chase-Lev deques, and cost-gated work stealing. |

The main objective is to investigate whether adaptive, overhead-aware task scheduling can reduce load imbalance without introducing excessive scheduling overhead.

---

## 2. Implementation Methodology

### 2.1 Radix Partitioning

The implementation uses two-pass radix partitioning:

- First pass: 5 radix bits.
- Second pass: 5 radix bits.
- Total: \(2^{10} = 1024\) partitions.

Both relations, build relation \(R\) and probe relation \(S\), are partitioned using the same partitioning scheme.

### 2.2 Build Phase

The build relation is partitioned and a hash table is constructed for each partition.

The build phase uses parallel static scheduling. The primary focus of the scheduler comparison is the probe phase, where skew in the probe relation can create significant differences in partition processing time.

### 2.3 Probe Phase

The probe relation is distributed across the corresponding build-side hash tables.

Under uniform key distributions, partition sizes are relatively balanced. As skew increases, a small number of partitions can contain a much larger fraction of the probe tuples.

The four scheduling methods are compared on this phase.

### 2.4 Scheduling Strategies

**`static`**

Uses OpenMP static scheduling to assign partitions to threads. Each thread processes its assigned partitions, which can lead to stragglers when partition sizes are highly uneven.

**`paper`**

Uses a simplified fixed-size chunking strategy. Oversized partitions are divided into chunks of 16,384 tuples, which workers claim from a shared task list.

This is a paper-inspired comparison baseline, not an exact implementation of the original paper's scheduling algorithm.

**`adaptive1`**

Uses a cost-based splitting rule to determine whether an oversized partition should be divided into smaller tasks.

The estimated partition cost is:

\[
C_i =
\begin{cases}
\alpha |R_i| & \text{if build cost is enabled}\\
0 & \text{otherwise}
\end{cases}
+
\beta |S_i|
+
\gamma \widehat{M}_i
\]

where:

- \(|R_i|\): number of build tuples in partition \(i\).
- \(|S_i|\): number of probe tuples in partition \(i\).
- \(\widehat{M}_i\): estimated number of matching tuples, when output expansion is enabled.
- \(\alpha,\beta,\gamma\): calibrated cost coefficients.

The splitting decision uses the condition:

\[
C_i\left(1-\frac{1}{k}\right) > H(k)
\]

where \(k\) is the number of chunks and \(H(k)\) represents estimated task creation and scheduling overhead.

The number of generated tasks is bounded by a task budget proportional to the worker count.

Tasks are dynamically claimed from a shared queue.

**Important:** In the default count-only configuration, `include_build` is disabled. Therefore, the default cost estimate is primarily based on probe-side tuple counts. Build-side cost is included only when explicitly enabled.

**`adaptive2`**

Extends adaptive task decomposition with per-worker Chase-Lev deques.

- Tasks are distributed across worker-local deques.
- Workers process locally available tasks.
- Idle workers can attempt to steal tasks from other workers.
- A cost-based threshold is used to avoid stealing when the estimated remaining work does not justify the stealing overhead.

The stealing threshold accounts for estimated CAS and cache-migration costs.

The threshold is a heuristic intended to control unnecessary stealing, not an exact prediction of runtime cost.

---

## 3. Experimental Setup

### Hardware and Software Environment

The recorded benchmark environment is:

| Component | Configuration |
|---|---|
| CPU | Intel Core i9-14900HX |
| Execution environment | Ubuntu 22.04.5 LTS under WSL2 |
| Compiler | GCC 11.4.0 |
| Language standard | C++20 |
| Parallel framework | OpenMP |
| Available logical CPUs in WSL2 | 32 |
| Available memory | Approximately 15 GiB |
| OpenMP binding | `OMP_PROC_BIND=close` |
| OpenMP placement | `OMP_PLACES=cores` |

The proposal originally targeted a 4-core CPU with 8 GB RAM. The measurements currently recorded in the repository were collected on the above laptop and WSL2 environment. These should not be interpreted as measurements from the originally proposed 4-core hardware.

### Workload Configuration

The main count-only benchmark uses:

- Build relation size: 5 million tuples.
- Probe relation size: 5 million tuples.
- Zipf skew levels: \(z = 0.0, 0.5, 0.75, 1.0, 1.25, 1.5\).
- Thread counts: 1, 2, and 4.
- Scheduling methods: `static`, `paper`, `adaptive1`, and `adaptive2`.
- Warm-up iterations: 1.
- Measured repetitions: 10.
- Reported timing: median across measured repetitions.

All recorded benchmark rows in the main results dataset pass the correctness check (`correct = 1`).

The reproduction script also includes additional workload configurations, including probe-heavy and many-to-many workloads, as well as scheduling ablations.

---

## 4. Evaluation Metrics

The evaluation focuses on execution time, load imbalance, and scheduling overhead.

| Metric | Definition | Purpose |
|---|---|---|
| Probe time | Time spent in the probe phase | Measures scheduler performance in the phase affected by probe-side skew. |
| Total execution time | End-to-end join execution time | Captures the overall effect of scheduling decisions. |
| Load imbalance \(\lambda\) | \(\lambda = \frac{\max(T_i)}{\operatorname{avg}(T_i)}\) | Measures variation in worker completion times. |
| Planning time | Time spent generating and preparing adaptive tasks | Measures the overhead of adaptive scheduling decisions. |
| Task count | Number of generated tasks | Indicates task decomposition granularity. |
| Steal attempts | Number of attempted work-stealing operations | Measures stealing activity. |
| Successful steals | Number of successful work-stealing operations | Measures how often stealing transfers useful work. |

Here, \(T_i\) denotes the measured processing time associated with worker \(i\).

A value of \(\lambda\) close to 1 indicates balanced worker processing times. Larger values indicate greater imbalance.

**CPU utilization is not directly measured in the current benchmark results.** The reciprocal of the imbalance ratio, \(1/\lambda\), is not CPU utilization and should not be interpreted as a hardware utilization measurement.

### Speedup Terminology

Conventional parallel speedup is defined as:

\[
S_p = \frac{T_1}{T_p}
\]

where \(T_1\) and \(T_p\) are execution times from the same implementation using one and \(p\) workers, respectively.

Parallel efficiency is:

\[
E_p = \frac{S_p}{p}
\]

The repository also records a comparison against a separate sequential `std::unordered_map` reference implementation.

That comparison is a **ratio against the sequential reference**, not conventional parallel speedup. These two quantities should be reported separately.

---

## 5. Main Benchmark Results

The following table summarizes the recorded median measurements for the main count-only workload with 4 threads and 5 million tuples in each relation.

All times are in milliseconds.

| Skew \(z\) | Method | Probe Time | Total Time | \(\lambda\) |
|---:|---|---:|---:|---:|
| 0.00 | `static` | 8.487 | 55.524 | 1.0753 |
|  | `paper` | 7.579 | 56.454 | 1.0012 |
|  | `adaptive1` | 7.676 | 55.922 | 1.0013 |
|  | `adaptive2` | 7.671 | 55.585 | 1.0013 |
| 0.50 | `static` | 7.586 | 54.789 | 1.0546 |
|  | `paper` | 7.595 | 55.092 | 1.0013 |
|  | `adaptive1` | 7.544 | 54.729 | 1.0013 |
|  | `adaptive2` | 7.429 | 54.707 | 1.0013 |
| 0.75 | `static` | 8.652 | 54.615 | 1.0681 |
|  | `paper` | 7.951 | 54.248 | 1.0013 |
|  | `adaptive1` | 7.990 | 54.172 | 1.0011 |
|  | `adaptive2` | 7.957 | 54.292 | 1.0023 |
| 1.00 | `static` | 7.887 | 55.150 | 1.0716 |
|  | `paper` | 7.465 | 53.730 | 1.0006 |
|  | `adaptive1` | 7.553 | 54.886 | 1.0010 |
|  | `adaptive2` | 7.680 | 55.209 | 1.0071 |
| 1.25 | `static` | 5.580 | 54.612 | 1.2799 |
|  | `paper` | 4.694 | 53.994 | 1.0004 |
|  | `adaptive1` | 4.595 | 53.792 | 1.0003 |
|  | `adaptive2` | 4.910 | 53.805 | 1.0400 |
| 1.50 | `static` | 5.358 | 56.829 | 1.7704 |
|  | `paper` | 3.347 | 54.304 | 1.0015 |
|  | `adaptive1` | 3.237 | 54.831 | 1.0000 |
|  | `adaptive2` | 3.287 | 54.380 | 1.0730 |

### Observations

- Under low skew, all four methods have relatively similar end-to-end execution times.
- As skew increases, static scheduling exhibits greater load imbalance.
- The adaptive methods generally reduce worker imbalance, but improved balance does not always translate into the lowest end-to-end execution time.
- The `paper` baseline can outperform the adaptive methods in some configurations.
- The best-performing scheduler varies with the workload and skew level. No single adaptive method consistently wins across every configuration.

### Focus: \(z = 0.75\)

At the target skew level, the measured total execution times are:

| Method | Total Time |
|---|---:|
| `static` | 54.615 ms |
| `paper` | 54.248 ms |
| `adaptive1` | **54.172 ms** |
| `adaptive2` | 54.292 ms |

`adaptive1` achieves the lowest end-to-end time in this configuration, approximately 0.81% lower than `static`.

`adaptive2` is approximately 0.59% faster than `static`, while the paper-inspired baseline is approximately 0.67% faster.

These results show a small performance improvement at this skew level, rather than a consistent or large advantage for every adaptive scheduler.

### Focus: \(z = 1.5\)

At extreme skew:

- Static scheduling reaches a load imbalance ratio of approximately 1.77.
- `paper` reduces the imbalance ratio to approximately 1.00.
- `adaptive1` records the lowest probe time.
- `paper` records the lowest total execution time among the four methods.

This illustrates an important distinction: optimizing the probe phase does not necessarily minimize total execution time, since scheduling and planning overhead also contribute to the end-to-end result.

---

## 6. Additional Workloads and Ablation Experiments

In addition to the main count-only benchmark, the repository includes experiments for:

- Probe-heavy workloads.
- Many-to-many joins with output expansion enabled.
- Build-side skew.
- Adaptive scheduling and work-stealing ablations.
- Work-stealing threshold experiments.

The many-to-many workload is particularly relevant because output multiplicity can affect the amount of work performed by a partition. In this mode, the implementation traverses matching build-side payloads and computes a checksum.

The cost model can account for estimated match counts when expansion is enabled. However, the quality of this estimate and its effect on end-to-end performance should be evaluated separately from the count-only workload.

Results from different workload configurations should not be compared without accounting for relation sizes, skew parameters, thread counts, and output-expansion settings.

---

## 7. Correctness

Each parallel execution is validated against a sequential reference implementation using `std::unordered_map`.

The benchmark records a correctness field:

- `correct = 1`: the parallel result matches the sequential reference.
- `correct = 0`: the result does not match the reference.

All checked-in rows in the main `results.csv` dataset have `correct = 1`.

Correctness validation is performed independently of the performance comparison.

---

## 8. Build and Run

### Prerequisites

- C++20-compatible compiler.
- OpenMP support.
- GCC 11 or later recommended.
- Linux or WSL2 environment.
- Python 3 with the dependencies required by the analysis scripts.

### Compile

```bash
g++ -O3 -march=native -std=c++20 -fopenmp -Wall -Wextra \
    hashjoin.cpp -o hashjoin
```

### Quick Smoke Test

```bash
./hashjoin \
    --n 100000 \
    --zs 0,0.75,1.5 \
    --threads 1,2,4 \
    --reps 1 \
    --warmup 0 \
    --out smoke.csv
```

### Run the Main Benchmark

```bash
./hashjoin \
    --n 5000000 \
    --zs 0,0.5,0.75,1.0,1.25,1.5 \
    --threads 1,2,4 \
    --reps 10 \
    --warmup 1 \
    --out results.csv
```

### Run the Reproduction Suite

```bash
bash run_final.sh
```

The checked-in reproduction script covers the main count-only workload, probe-heavy workloads, many-to-many workloads, and scheduling ablations. Its configured experiment matrix contains 2,240 measured runs, excluding warm-up iterations.

### Analyze Results

Use the analysis scripts included in the repository to summarize the generated CSV files and produce plots.

```bash
python3 result_metrics.py results.csv
```

Check the repository for the specific plotting and analysis scripts available in the current version before running additional analysis commands.

---

## 9. Command-Line Arguments

| Argument | Description |
|---|---|
| `--n` | Number of tuples in the build relation \(R\). |
| `--ns` | Number of tuples in the probe relation \(S\). |
| `--zs` | Comma-separated Zipf skew values for the probe relation. |
| `--threads` | Number of OpenMP worker threads. |
| `--methods` | Scheduling methods to evaluate. |
| `--reps` | Number of measured repetitions. |
| `--warmup` | Number of warm-up iterations. |
| `--out` | Output CSV file path. |
| `--expand` | Enables many-to-many output expansion mode. |
| `--r_z` | Zipf skew for the build relation. |
| `--include_build` | Enables build-side cost in the adaptive cost estimate. |
| `--tau_ns` | Work-stealing threshold in nanoseconds. |
| `--task_overhead_ns` | Task creation and scheduling overhead parameter. |

For exact defaults and additional options, refer to the command-line argument definitions in `hashjoin.cpp`.

---

## 10. Limitations and Scope

The current evaluation has several limitations:

1. **Hardware scope:** The recorded measurements are from an Intel Core i9-14900HX laptop running Ubuntu through WSL2. They do not establish performance on every low-core CPU architecture.
2. **Paper baseline:** The `paper` scheduler is a simplified paper-inspired approximation, not an exact reproduction of the original Balkesen et al. implementation.
3. **Cost model:** The adaptive cost model is a first-order heuristic based on tuple counts and optional estimated output matches. Its predictive accuracy has not been established as a general runtime model.
4. **CPU utilization:** Hardware CPU utilization counters are not collected in the current benchmark dataset.
5. **Performance consistency:** Adaptive scheduling does not outperform every baseline in every workload configuration. Results depend on skew, workload type, and scheduling overhead.
6. **Statistical reporting:** The checked-in result summaries report medians and observed timing ranges. They should not be interpreted as paired confidence intervals unless such intervals are computed separately.

These limitations define the scope of the current implementation and provide directions for further evaluation.

---

## 11. References

Balkesen, C., Teubner, J., Alonso, G., and Özsu, M. T. (2013). *Main-Memory Hash Joins on Multi-Core CPUs: Tuning to the Underlying Hardware*. Proceedings of the IEEE International Conference on Data Engineering (ICDE).

