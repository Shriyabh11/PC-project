#!/usr/bin/env bash
set -euo pipefail

export OMP_PROC_BIND=close OMP_PLACES=cores

{
  date --iso-8601=seconds
  uname -a
  g++ --version
  printf 'OMP_PROC_BIND=%s\nOMP_PLACES=%s\n' "$OMP_PROC_BIND" "$OMP_PLACES"
  if command -v lscpu >/dev/null 2>&1; then lscpu; fi
  if command -v free >/dev/null 2>&1; then free -h; fi
  if command -v lsb_release >/dev/null 2>&1; then lsb_release -a; fi
  if command -v perf >/dev/null 2>&1; then
    perf --version
  else
    echo "perf: not installed"
  fi
} > metrics_environment.txt

g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin

REPS=5
WARMUP=1
BASELINE_SKEWS=0,0.5,0.75,1.0,1.25,1.5,1.75
TARGET_SKEWS=0,0.5,0.75,1.0,1.25,1.5
BASELINE_THREADS=1,2,4,8
TARGET_THREADS=1,2,4
METHODS=static,paper,adaptive1,adaptive2

./hashjoin --n 5000000 --zs "$BASELINE_SKEWS" --threads "$BASELINE_THREADS" \
  --methods "$METHODS" --reps "$REPS" --warmup "$WARMUP" \
  --out results_metrics_baseline.csv

./hashjoin --n 16000000 --zs "$TARGET_SKEWS" --threads "$TARGET_THREADS" \
  --methods "$METHODS" --reps "$REPS" --warmup "$WARMUP" \
  --out results_metrics_target.csv

for granularity in 1 2 4 8; do
  ./hashjoin --n 5000000 --zs 1.5 --threads 1,2,4,8 \
    --methods adaptive1,adaptive2 --granularity "$granularity" \
    --reps "$REPS" --warmup "$WARMUP" \
    --out "results_metrics_granularity_${granularity}.csv"
done

for tau in 1000 20000 500000 5000000; do
  ./hashjoin --n 5000000 --zs 1.5 --threads 1,2,4,8 \
    --methods adaptive2 --init rr --tau_ns "$tau" \
    --reps "$REPS" --warmup "$WARMUP" \
    --out "results_metrics_tau_${tau}.csv"
done

python3 analyze_all.py
