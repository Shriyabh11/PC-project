#!/usr/bin/env bash
# ==============================================================================
# Full Benchmark Runner: Skew Sweep across Threads and 4 Scheduling Methods
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo " 1. Compiling Hash Join Benchmark (C++20, OpenMP, -O3)"
echo "=========================================================="
g++ -O3 -march=native -std=c++20 -fopenmp -Wall -Wextra hashjoin.cpp -o hashjoin

echo "=========================================================="
echo " 2. Running Skew Sweep Experiment"
echo "    Relations: |R| = |S| = 5,000,000 tuples"
echo "    Zipf Skew: z in {0, 0.5, 0.75, 1.0, 1.25, 1.5}"
echo "    Threads:   1, 2, 4"
echo "    Methods:   static, paper, adaptive1, adaptive2"
echo "=========================================================="
./hashjoin --n 5000000 \
           --zs 0,0.5,0.75,1.0,1.25,1.5 \
           --threads 1,2,4 \
           --reps 5 \
           --warmup 1 \
           --out results.csv

echo "=========================================================="
echo " 3. Running Summary Analysis"
echo "=========================================================="
python3 analyze.py results.csv

echo "=========================================================="
echo " 4. Generating All 6-Metric Evaluation Plots"
echo "=========================================================="
python3 plot_all_6_metrics.py

echo "=========================================================="
echo " Benchmark Run Complete!"
echo " Results saved in:  results.csv & summary.csv"
echo " Plots generated in: analysis/plots/"
echo "=========================================================="
