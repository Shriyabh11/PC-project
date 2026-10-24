#!/usr/bin/env bash
# Regenerates every result file used for the slides, all from the same version of the code.
# Plug in the laptop and close other programs first. Takes roughly 20-40 minutes.
set -e
g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
export OMP_PROC_BIND=close OMP_PLACES=cores

R=10   # measured repetitions per configuration (plus 1 warm-up)

# 1. Original workload (count-only join, |R| = |S| = 5M)
./hashjoin --n 5000000 --zs 0,0.5,0.75,1.0,1.25,1.5 --threads 1,2,4 --reps $R --warmup 1 --out results.csv

# 2. Probe-heavy variant (|S| = 4 x |R|)
./hashjoin --n 5000000 --ns 20000000 --zs 0,1.0,1.5 --threads 1,2,4 --reps $R --warmup 1 --out results_probeheavy.csv

# 3. Many-to-many workload (R also skewed): cost differs from partition size
./hashjoin --n 1000000 --expand 1 --r_z 0.5  --zs 0,0.75,1.0,1.25,1.5 --threads 1,2,4 --reps $R --warmup 1 --out results_expand_rz05.csv
./hashjoin --n 1000000 --expand 1 --r_z 0.75 --zs 0,0.5,0.75,1.0       --threads 1,2,4 --reps $R --warmup 1 --out results_expand_rz075.csv

# 4. Stealing ablation: round-robin start, so stealing has to rebalance
./hashjoin --n 5000000 --zs 1.0,1.5 --threads 4 --reps $R --warmup 1 --init rr --methods adaptive1,adaptive2 --out results_ablation_rr.csv

# 5. Steal-threshold sweep (nanoseconds), round-robin start
for t in 1000 20000 500000 5000000; do
  ./hashjoin --n 5000000 --zs 1.5 --threads 4 --reps $R --warmup 1 --init rr --tau_ns $t --methods adaptive2 --out results_tau_$t.csv
done

echo "Done. Upload: results.csv results_probeheavy.csv results_expand_rz05.csv results_expand_rz075.csv results_ablation_rr.csv results_tau_*.csv"
