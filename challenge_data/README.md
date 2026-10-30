# Synthetic challenge data

`original_challenge.csv` is invented data for a guessing-game/demo only. Its
timing values were not measured, and its `correct` values are illustrative,
not results of running the benchmark. The `data_origin` column marks every row
as synthetic. Do not cite it, submit it as experiment output, or mix it with
the measured result CSVs.

The file is under this separate directory, so `analyze_all.py` does not load it.
