# Skew-Resilient Parallel Hash Joins via Adaptive Fine-Grained Scheduling

A C++20 / OpenMP implementation and empirical evaluation of skew-resilient parallel hash joins on multicore CPUs. This project investigates adaptive, overhead-aware work stealing and task decomposition to mitigate stragglers caused by foreign-key data skew.

---

## 1. Background & Methodology Alignment with Base Paper

This project is grounded in **Balkesen et al. (ICDE 2013)** (*"Main-memory hash joins on multi-core CPUs: Tuning to the underlying hardware"*) and follows the methodology outlined in the project slides:

1. **Two-Pass Radix Partitioning (5 + 5 bits $\to$ 1024 partitions):**
   - Both build relation $R$ and probe relation $S$ are partitioned into $2^{10} = 1024$ radix clusters using two passes of 5 bits. Each partition is sized to reside cleanly in CPU cache (L1/L2).
2. **Parallel Build Phase (Static OpenMP):**
   - For each partition $R_i$, an open-addressing hash table is constructed. Because $R$ keys are uniformly distributed by default, all 1024 hash tables have identical size ($\approx |R|/1024$ tuples). Build work is inherently balanced across threads.
3. **Parallel Probe Phase & Schedulers Compared:**
   - In real-world workloads, foreign keys in $S$ follow Zipf distributions ($z \in [0, 1.5]$). While $R$ is uniform, a few $S_i$ partitions become massive, causing stragglers during the probe phase.
   - We evaluate four schedulers for the probe phase:
     - **`static` (Baseline 1 - Paper Fig. 16a):** Statically assigns partitions to threads via `#pragma omp for schedule(static)`. Under skew, threads with small partitions finish early and idle at the barrier while straggler threads process huge partitions.
     - **`paper` (Baseline 2 - Paper Fig. 16b):** Balkesen et al.'s fine-grained task decomposition. Any partition exceeding $2\times$ the average size is broken into fixed-size chunks (`16,384` tuples) and dynamically claimed by worker threads from a shared list.
     - **`adaptive1` (Novelty - Shared Dynamic Queue):** Formulates workload cost $C_i = \alpha |R_i| + \beta |S_i| (+ \gamma \widehat{M}_i)$. Splits $S_i$ into $k$ chunks only when the theoretical reduction exceeds overhead: $C_i(1 - 1/k) > H(k) = k \cdot \tau_{overhead}$, bounded by a task budget $B = \text{budget\_per\_thread} \times P$. Tasks are dynamically claimed from a shared queue.
     - **`adaptive2` (Novelty - Work Stealing with Cost-Gated Threshold):** Distributes adaptive tasks to per-worker Chase-Lev lock-free deques using LPT (Longest Processing Time first) greedy placement. Workers pop tasks LIFO (hot cache). Idle workers steal tasks FIFO from victims only if the victim's estimated remaining work exceeds the cache/CAS migration threshold:
       $$\tau_{steal} = \text{Cost(CAS steal)} + \text{Cost(cache migration)}$$

---

## 2. The 6 Evaluation Metrics (from Slide 12)

As specified in the project proposal (Slide 12, *"Expected Outcomes and Evaluation"*), the benchmark tracks all 6 evaluation metrics:

| Metric | Definition & Formula | Slide Rationale & Meaning |
| :--- | :--- | :--- |
| **1. Execution Time** | Probe time ($T_{probe}$) & End-to-end time ($T_{total}$) in milliseconds | Primary objective: measures join throughput and latency reduction during the scheduled phase. |
| **2. Speedup** | $S_p = \frac{T_1}{T_p}$ (and relative speedup vs. `static`) | Measures parallel scaling as core count increases from 1 to 4 threads. |
| **3. Parallel Efficiency** | $E_p = \frac{S_p}{p} \times 100\%$ | Quantifies how effectively additional CPU cores translate into useful computation. |
| **4. Load Imbalance ($\lambda$)** | $\lambda = \frac{\max(T_i)}{\text{avg}(T_i)}$ where $T_i$ is thread finish time | Isolates the straggler effect. $\lambda = 1.0$ indicates perfect balance; $\lambda \gg 1.0$ indicates severe straggling. |
| **5. CPU Utilization** | $\text{CPU Util} = \frac{1}{\lambda} \times 100\%$ | Measures active worker CPU time vs. wasted idle time spent waiting at barriers. |
| **6. Scheduling Overhead** | Task count (`tasks`), steal attempts (`steal_attempts`), successful steals (`steals`), and planning time (`plan_s`) | Tests whether adaptive bounded budgeting controls scheduling overhead and verifies that the $\tau_{steal}$ gate prevents wasteful stealing. |

All parallel runs are verified against the single-threaded sequential ground truth (`correct == 1`).

---

## 3. Does $z = 0.75$ Outperform?

**Yes.** Across both count-only and output-expand workloads, the adaptive scheduling approach delivers clear performance gains at $z=0.75$:

### A. Count-Only Workload ($z = 0.75$, 4 Threads, 5M tuples)
- **`static`**: Probe = **8.65 ms**, $\lambda = 1.068$, CPU Util = **93.6%**
- **`paper`**: Probe = **7.95 ms**, $\lambda = 1.001$, CPU Util = **99.9%**, Tasks = **1029**
- **`adaptive1`**: Probe = **7.99 ms**, $\lambda = 1.001$, CPU Util = **99.9%**, Tasks = **1024**
- **`adaptive2`**: Probe = **7.96 ms**, $\lambda = 1.002$, CPU Util = **99.8%**, Tasks = **1024**, Steals = **10**

*Key Takeaway:* At $z=0.75$, `adaptive2` and `paper` both achieve an **8.1% probe time speedup** over `static`. Furthermore, `adaptive2` accomplishes this without creating redundant tasks (1024 tasks vs 1029 for paper).

### B. Many-to-Many Workload ($R$ skew $r_z = 0.75$, $S$ skew $z = 0.75$, 4 Threads)
When both relations are skewed and output matches vary widely across partitions:
- **`static`**: Probe = **21.97 ms**, $\lambda = 1.415$, CPU Util = **70.7%** (severe stragglers)
- **`paper`**: Probe = **19.48 ms**, $\lambda = 1.212$, CPU Util = **82.5%**
- **`adaptive2`**: Probe = **16.98 ms**, $\lambda = 1.067$, CPU Util = **93.7%**
- **`adaptive1`**: Probe = **16.12 ms**, $\lambda = 1.001$, CPU Util = **99.9%** (fastest)

*Key Takeaway:* In the many-to-many workload at $z=0.75$, **`adaptive` substantially outperforms BOTH `static` (26.6% faster) and `paper` (17.2% faster)**. The paper baseline's tuple-count heuristic fails to detect match explosion, whereas our cost model uses output sampling ($\gamma \widehat{M}_i$) to balance the heavy partitions.

### C. Behavior at High Skew ($z = 1.5$, 4 Threads, 5M tuples)
Under extreme skew (identical to Balkesen et al. Figure 16):
- **`static`**: Probe = **5.36 ms**, $\lambda = 1.770$, CPU Util = **56.5%** (threads spend 43.5% of their time idling!)
- **`paper`**: Probe = **3.35 ms**, $\lambda = 1.001$, Tasks = **1272** (creates 248 extra tasks)
- **`adaptive1`**: Probe = **3.24 ms**, $\lambda = 1.000$, Tasks = **1033** (only 9 extra tasks)
- **`adaptive2`**: Probe = **3.29 ms**, $\lambda = 1.073$, Tasks = **1033**

*Key Takeaway:* At $z=1.5$, `adaptive` is **39.5% faster than `static`** and slightly faster than `paper`, while generating **$27\times$ fewer extra tasks** thanks to the bounded budget ($C_i(1-1/k) > H(k)$).

---

## 4. Evaluation Plots for All 6 Metrics

Dedicated 6-metric comparison dashboards have been generated under [`analysis/plots/`](file:///c:/Shriya/PC_proj/analysis/plots):

| Skew Level | Plot File | Description |
| :--- | :--- | :--- |
| **All Skew Sweep (p=4)** | [`metrics_6_across_z_p4.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_across_z_p4.png) | All 6 metrics across $z \in [0.0, 1.5]$ for 4 threads. |
| **Focus Comparison (z=0.75)** | [`z075_focus_comparison.png`](file:///c:/Shriya/PC_proj/analysis/plots/z075_focus_comparison.png) | Deep-dive at $z=0.75$ comparing count-only and many-to-many expand workloads. |
| **$z = 0.0$ (Uniform)** | [`metrics_6_z_0_0.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_0_0.png) | 6-metric scaling across 1, 2, 4 threads under uniform keys. |
| **$z = 0.5$ (Low Skew)** | [`metrics_6_z_0_5.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_0_5.png) | 6-metric scaling across 1, 2, 4 threads at $z=0.5$. |
| **$z = 0.75$ (Target Skew)** | [`metrics_6_z_0_75.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_0_75.png) | 6-metric scaling across 1, 2, 4 threads at target skew $z=0.75$. |
| **$z = 1.0$ (Moderate Skew)**| [`metrics_6_z_1_0.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_1_0.png) | 6-metric scaling across 1, 2, 4 threads at $z=1.0$. |
| **$z = 1.25$ (High Skew)** | [`metrics_6_z_1_25.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_1_25.png) | 6-metric scaling across 1, 2, 4 threads at $z=1.25$. |
| **$z = 1.5$ (Extreme Skew)** | [`metrics_6_z_1_5.png`](file:///c:/Shriya/PC_proj/analysis/plots/metrics_6_z_1_5.png) | 6-metric scaling across 1, 2, 4 threads at $z=1.5$ (Balkesen Fig. 16). |

---

## 5. Build and Reproduction Instructions

### Prerequisites
- C++20 compiler with OpenMP support (`g++` 11+ or `clang++` 14+)
- Python 3.9+ with `pandas`, `matplotlib`, and `numpy` (install via `pip install -r requirements.txt`)
- Linux or Windows with WSL (Ubuntu 22.04 LTS recommended)

---

### One-Click Reproduction
To compile, run the full benchmark suite across all skew levels, and generate all metric analysis and plots in one command:

```bash
# On Linux / macOS / WSL:
bash run_all.sh

# On Windows PowerShell (via WSL):
wsl bash run_all.sh
```

---

### Step-by-Step Execution

#### 1. Quick Smoke Test (Takes ~2 seconds)
Verifies build environment, compiler flags, and mathematical correctness against the sequential reference:
```bash
g++ -O3 -march=native -std=c++20 -fopenmp -Wall -Wextra hashjoin.cpp -o hashjoin
./hashjoin --n 100000 --zs 0,0.75,1.5 --threads 1,2,4 --reps 1 --warmup 0 --out smoke.csv
```

#### 2. Run Full Main Benchmark (5M Tuples, All Skew Levels)
```bash
./hashjoin --n 5000000 \
           --zs 0,0.5,0.75,1.0,1.25,1.5 \
           --threads 1,2,4 \
           --reps 5 \
           --warmup 1 \
           --out results.csv
```

#### 3. Generate Summary & All 6-Metric Evaluation Plots
```bash
# Aggregates results into summary.csv and prints timing table
python3 analyze.py results.csv

# Generates all multi-panel 6-metric plots and z=0.75 focus comparison
python3 plot_all_6_metrics.py
```
All generated PNG figures are automatically saved to [`analysis/plots/`](file:///c:/Shriya/PC_proj/analysis/plots).

#### 4. Run Many-to-Many / Output-Heavy Skew Workload (Optional)
```bash
./hashjoin --n 1000000 --expand 1 --r_z 0.75 --zs 0,0.5,0.75,1.0 --threads 1,2,4 --reps 5 --warmup 1 --out results_expand_rz075.csv
```

---

### Command-Line Arguments Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--n <N>` | `5000000` | Number of tuples in build relation $R$. |
| `--ns <N>` | `0` (same as $R$) | Number of tuples in probe relation $S$ (set $> N$ for probe-heavy joins). |
| `--zs <z1,z2...>` | `0,0.5,0.75,1.0,1.25,1.5` | Comma-separated Zipf skew factors for probe keys in $S$. |
| `--threads <t1,t2...>` | `1,2,4` | Comma-separated thread counts to benchmark. |
| `--methods <m1,m2...>` | `static,paper,adaptive1,adaptive2` | Schedulers to evaluate. |
| `--reps <R>` | `5` | Number of timed repetitions per configuration. |
| `--warmup <W>` | `1` | Number of untimed warmup iterations to prime CPU caches. |
| `--out <path>` | `results.csv` | Output CSV path. |
| `--expand <0\|1>` | `0` | Enable many-to-many join mode (walks matching $R$ payloads and computes checksum). |
| `--r_z <Z>` | `0.0` | Zipf skew for build relation $R$ (used in many-to-many workloads). |
| `--include_build <0\|1>` | `0` | Include build cost $\alpha \|R_i\|$ in adaptive cost estimation ($C_i = \alpha \|R_i\| + \beta \|S_i\|$). |
| `--tau_ns <ns>` | `auto` | Work-stealing threshold $\tau_{steal}$ in nanoseconds (auto-calibrated by default). |
| `--task_overhead_ns <ns>`| `auto` | Task splitting overhead threshold $H(k) = k \cdot \tau_{overhead}$ (auto-calibrated). |

