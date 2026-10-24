# Skew-Resilient Parallel Hash Join

A C++20/OpenMP benchmark for parallel hash joins under uniform and Zipf-skewed
workloads. It compares four probe-phase schedulers:

- `static`: statically assigns whole partitions to threads.
- `paper`: splits large partitions into fixed-size tasks and dynamically claims them.
- `adaptive1`: creates cost-based tasks and dynamically claims them.
- `adaptive2`: uses per-worker deques and cost-gated work stealing.

The benchmark checks each method against a sequential reference and writes
measurements to CSV.

## Build and run

Requires a C++20 compiler with OpenMP support. On Linux or WSL:

```bash
g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
./hashjoin --n 5000000 --zs 0,0.5,0.75,1.0,1.25,1.5 \
  --threads 1,2,4 --reps 5 --warmup 1 --out results.csv
```

Run the full experiment suite with `./run_final.sh`. It takes roughly 20-40
minutes and regenerates the benchmark CSV files.

## Analyze results

Install the Python dependencies in the active environment:

```bash
python3 -m pip install -r requirements.txt
python3 analyze.py results.csv
```

The analysis script writes `summary.csv` and plot images to the current
directory. Pass a different benchmark CSV as the argument to analyze it.

## Benchmark data

The `results*.csv` files in the repository contain checked-in benchmark
outputs. The run scripts regenerate these files.
