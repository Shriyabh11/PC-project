# Parallel hash-join scheduler benchmark

## 1. Overview

This C++20/OpenMP benchmark studies probe-phase scheduling for a radix hash
join under uniform and Zipf-skewed keys. Skew can make a few partitions much
larger than others, leaving threads that finish small partitions waiting for
the thread handling a straggler. It compares static partition assignment, a
paper-inspired fixed-chunk scheduler, cost-based shared-queue scheduling, and
cost-based work stealing.

The checked-in experiments show workload-dependent trade-offs, but no
scheduler is a consistent winner and these runs do not reproduce the base
paper's hardware experiment.

## 2. Methods

| Name | What it does | Function(s) |
|---|---|---|
| `static` | Assigns whole partitions to threads with OpenMP `schedule(static)`. | `run_static` |
| `paper` | Splits oversized partitions into fixed-size chunks and dynamically claims them from a shared task list. This is paper-inspired, **not an exact reproduction** of the paper's implementation. | `make_paper_tasks`, `run_queue` |
| `adaptive1` | Forms cost-based chunks using `C_i(1 - 1/k) > H(k)`, where `H(k) = k x task overhead`; `C_i` estimates S probe work, optionally includes R build work, and in sampled expand workloads includes estimated output work. It enforces a bounded task budget, sorts tasks largest-first, and dynamically claims them from a shared list. | `make_adaptive_tasks`, `run_queue` |
| `adaptive2` | Uses the adaptive tasks with per-worker Chase-Lev deques and cost-gated stealing. The gate uses `tau_steal = Cost(CAS) + Cost(cache migration)`. | `make_adaptive_tasks`, `run_steal` |

The `paper` scheduler's chunk size and oversized-partition threshold are
configurable with `--paper_chunk` and `--paper_factor`. Adaptive scheduler
parameters are also configurable in `hashjoin.cpp`.

## 3. Pipeline

For each workload and skew, the benchmark:

1. Generates seeded R and S relations. The default seeds are fixed in the
   source; S uses the selected Zipf skew and R uses `--r_z`.
2. Runs a sequential reference join and records its median time.
3. Partitions both relations in two radix passes of 5 bits each, producing
   1024 partitions.
4. Computes partition statistics, including the largest S partition divided
   by average S-partition size.
5. Builds a hash table for each R partition in a parallel static loop.
6. Runs the selected probe scheduler and records probe timing and scheduler
   statistics.
7. Checks match counts against the sequential reference. In expand mode, it
   also checks a payload checksum.

## 4. Workloads

- **Count-only:** default mode. `--n` sets the R tuple count; S has the same
  size unless `--ns` is supplied. S keys follow the selected `--zs` skew.
- **Probe-heavy:** `--ns` sets S larger than R, increasing the amount of probe
  work. `run_final.sh` includes a run with S four times the size of R.
- **Many-to-many / expand:** `--expand 1 --r_z Z` enables the expand workload
  and Zipf-skews R keys as well as S keys. Probing walks all matching R
  payloads, so work depends on the number of matches, not just the number of S
  tuples. The planner samples S partitions to estimate output size. Sampling
  is skipped when the largest-to-average S-partition ratio is below
  `--plan_gate` (default 1.5); `--samples` controls the per-partition sample
  limit.

## 5. Metrics

| Metric | Definition / formula | CSV column | Notes |
|---|---|---|---|
| End-to-end time | `partition_s + build_s + plan_s + probe_s` | `total_s` | Planning is charged to the method. |
| Probe-phase time | Time spent in the scheduled probe phase. | `probe_s` | Excludes partitioning, build, and task planning. |
| Speedup and parallel efficiency | `speedup = seq_s / total_s`; `efficiency = speedup / threads` | `seq_s`, `total_s`, `threads`; derived by analysis scripts | The sequential reference also gets faster under skew, so this speedup is confounded. It is meaningless in expand mode. |
| Relative improvement over static and paper | `100 x (T_base - T) / T_base`, for `T` equal to `total_s` or `probe_s` and base method `static` or `paper` | `analyze.py`: `impr_over_static_%` and `impr_over_paper_%` are total-time comparisons; `probe_impr_over_static_%` is probe-time vs static. `analyze_all.py`: `vs_paper_style_%` is total-time vs paper. | Probe-time improvement vs paper is not currently emitted by either analysis script. |
| Thread imbalance lambda | `max(T_i) / avg(T_i)`, where `T_i` is each thread's finish time in the probe phase | `lambda` | A value closer to 1 indicates more even thread finish times. |
| Partition imbalance | Largest S partition / average S partition | `max_avg_part` | Computed before probing. |
| Task and stealing statistics | Task count; planning time; steal attempts; successful steals; success ratio = `steals / steal_attempts` | `tasks`, `plan_s`, `steal_attempts`, `steals` | The success ratio is undefined when there are no attempts. |
| Correctness | Match count equals the sequential reference on every run; expand mode also checks the payload checksum | `matches`, `correct` | The checksum is checked internally but is not written to a separate CSV column. |
| Calibrated costs | `alpha`, `beta` in seconds per tuple; `tau_ns` and `overhead_ns` in nanoseconds | `alpha`, `beta`, `tau_ns`, `overhead_ns` | `alpha` estimates build cost and `beta` estimates probe cost; both are calibrated. The steal threshold and task overhead are calibrated by default and can be set with `--tau_ns` and `--task_overhead_ns`. |

The final experiment script uses one warm-up and ten measured repetitions per
configuration. `analyze.py` and `analyze_all.py` exclude warm-ups and aggregate
measurements by median. Method order rotates across repetitions to reduce
ordering bias.

## 6. Build and reproduce

Requires a C++20 compiler with OpenMP and Python 3 with the packages in
`requirements.txt`.

Build the benchmark:

```bash
g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
```

Quick correctness smoke test:

```bash
./hashjoin --n 100000 --zs 0,1.5 --threads 1,2 --reps 1 --warmup 1 --out smoke.csv
```

Run the complete experiment suite:

```bash
./run_final.sh
```

The script recompiles the binary, sets `OMP_PROC_BIND=close` and
`OMP_PLACES=cores`, and regenerates every `results*.csv` file listed below.
The script estimates 20-40 minutes for the run.

Install analysis dependencies and analyze one CSV or the whole collection:

```bash
python3 -m pip install -r requirements.txt
python3 analyze.py                 # defaults to results.csv
python3 analyze.py results_probeheavy.csv
python3 analyze_all.py
```

`analyze.py` writes `summary.csv` and plots in the current directory.
`analyze_all.py` reads all `results*.csv` files and writes an aggregate CSV,
report, and comparison plots under `analysis/`.

### Checked-in result files

| File | Experiment |
|---|---|
| `results.csv` | Count-only baseline; 5 million tuples per relation; skew sweep; 1, 2, and 4 threads. |
| `results_probeheavy.csv` | Probe-heavy count-only workload with S larger than R. |
| `results_expand_rz05.csv` | Expand workload with R skew `z=0.5`; S skew sweep. |
| `results_expand_rz075.csv` | Expand workload with R skew `z=0.75`; S skew sweep. |
| `results_ablation_rr.csv` | `adaptive1`/`adaptive2` comparison with round-robin initial task assignment. |
| `results_tau_1000.csv` | `adaptive2` steal-threshold sweep at 1000 ns. |
| `results_tau_20000.csv` | `adaptive2` steal-threshold sweep at 20000 ns. |
| `results_tau_500000.csv` | `adaptive2` steal-threshold sweep at 500000 ns. |
| `results_tau_5000000.csv` | `adaptive2` steal-threshold sweep at 5000000 ns. |

`run_final.sh` is the authoritative source for the exact parameters used to
regenerate these files.

## 7. Experimental setup

Fill in these values for the machine used to generate the results:

- CPU model: **TODO**
- Physical cores / logical threads: **TODO**
- RAM: **TODO**
- OS / WSL version: **TODO**
- `g++ --version`: **TODO**
- Thread affinity: `run_final.sh` sets `OMP_PROC_BIND=close` and
  `OMP_PLACES=cores`; confirm whether this was effective on the measured
  system: **TODO**

## 8. Design decisions and limitations

- Build and probe are separate measured phases in this benchmark; the base
  paper performs build and probe per partition.
- In count-only mode, hash-table entries hold a key and count, so the join
  counts matches rather than materializing output tuples. Expand mode keeps R
  payloads to walk matching tuples and verify a checksum.
- The adaptive split test uses `H(k) = k x per-task overhead`.
- The cost model omits the build term unless `--include_build 1` is enabled.
- The many-to-many workload and planning gate were added after the first
  results; they are extensions to the original count-only experiments.
- Work stealing did not show a consistent benefit in these experiments.
- Results come from one machine and use at most four threads. Thread counts
  above a machine's physical core count measure hyper-threading, not the
  intended core-scaling target.
- The `paper` method approximates the paper's fine-grained partition
  decomposition; it is not the original implementation. The published
  experiment and this benchmark use different hardware, relation sizes,
  thread counts, and timing scopes.

## 9. Results

See [`analysis/report.md`](analysis/report.md) and
[`analysis/all_experiments_summary.csv`](analysis/all_experiments_summary.csv)
for the analyzed results; `analyze_all.py` regenerates them from the CSV files.
