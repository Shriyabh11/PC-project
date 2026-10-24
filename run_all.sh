#!/usr/bin/env bash
# Full experiment: skew sweep x threads x 4 methods, 5 measured runs after 1 warm-up each.
set -e
g++ -O3 -march=native -std=c++20 -fopenmp hashjoin.cpp -o hashjoin
./hashjoin --n 5000000 --zs 0,0.5,0.75,1.0,1.25,1.5 --threads 1,2,4 --reps 5 --warmup 1 --out results.csv
python3 analyze.py results.csv