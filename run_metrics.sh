#!/usr/bin/env bash
set -euo pipefail

bash run_tuning_z075.sh
python3 analyze_all.py
