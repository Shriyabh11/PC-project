#!/usr/bin/env bash
set -euo pipefail

g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
export OMP_PROC_BIND=close OMP_PLACES=cores

REPS=10
WARMUP=1
THREADS=1,2,4,8
METHODS=static,paper,adaptive1,adaptive2

run_candidate() {
  local name=$1
  local task_overhead=$2
  local granularity=$3
  local min_chunk=$4

  ./hashjoin --n 5000000 --zs 0.75 --threads "$THREADS" \
    --methods "$METHODS" --task_overhead_ns "$task_overhead" \
    --granularity "$granularity" --min_chunk "$min_chunk" \
    --reps "$REPS" --warmup "$WARMUP" \
    --out "tuning_z075_${name}.csv"
}

run_candidate auto_baseline auto 4 2048
run_candidate overhead_1000_g2 1000 2 512
run_candidate overhead_1000_g4 1000 4 512
run_candidate overhead_250_g2 250 2 512
run_candidate overhead_250_g4 250 4 512
run_candidate granularity_8 auto 8 2048
run_candidate granularity_16 auto 16 2048
run_candidate granularity_32 auto 32 2048
run_candidate granularity_64 auto 64 2048
