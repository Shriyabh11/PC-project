#!/usr/bin/env bash
# Extra experiments (run after run_all.sh). Each writes its own CSV; analyze with: python3 analyze.py <file>
set -e
g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin

# B. Probe-heavy workload: |S| = 4 x |R| (the probe phase is a bigger share of the total time)
./hashjoin --n 5000000 --ns 20000000 --zs 0,1.0,1.5 --threads 1,2,4 --reps 5 --warmup 1 --out results_probeheavy.csv

# C. Ablation: round-robin initial assignment, so stealing is needed to rebalance
./hashjoin --n 5000000 --zs 1.0,1.5 --threads 4 --reps 5 --warmup 1 --init rr \
           --methods adaptive1,adaptive2 --out results_ablation_rr.csv

# D. Sweep the steal threshold (nanoseconds) with round-robin start; compare against measured (auto)
for t in 1000 5000 20000 100000 500000 5000000; do
  ./hashjoin --n 5000000 --zs 1.5 --threads 4 --reps 5 --warmup 1 --init rr --tau_ns $t \
             --methods adaptive2 --out results_tau_$t.csv
done

# E. Many-to-many workload (R also skewed): probe cost depends on the number of matches, not only on |S_i|.
#    Speedup vs the sequential reference is not meaningful here (the reference is O(1) per tuple);
#    compare methods with each other (improvement over static / paper).
./hashjoin --n 1000000 --expand 1 --r_z 0.5 --zs 0,0.75,1.0,1.25,1.5 --threads 1,2,4 --reps 5 --warmup 1 --out results_expand_rz05.csv
./hashjoin --n 1000000 --expand 1 --r_z 0.75 --zs 0,0.5,0.75,1.0 --threads 1,2,4 --reps 5 --warmup 1 --out results_expand_rz075.csv