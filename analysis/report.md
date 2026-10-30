# z=0.75 scheduler tuning

This is the focused benchmark analysis requested for S-side Zipf skew `z=0.75`.

## Reported measurements

- Median end-to-end time (`total_s`)
- Median probe time (`probe_s`)
- Median thread imbalance (`lambda`)
- Median planning and partition/build phase timings when present
- Correctness (`correct == 1`)

## Not reported

The aggregate analysis no longer reports or generates:

- speedup against the separately implemented sequential reference;
- parallel efficiency derived from that speedup;
- percentage improvement versus static or paper baselines;
- the fake 16M/8-thread evaluation matrix;

These values were removed because they either compare unlike implementations, exceed the physical hardware available for the run, or do not provide a valid scheduler claim.

## Run

```bash
bash run_tuning_z075.sh
py -3 analyze_all.py
```

## Output

`analyze_all.py` writes `analysis/z075_tuning_summary.csv` and `analysis/z075_tuning.png` from the measured `tuning_z075_*.csv` files. Compare absolute median timings only under matched candidate, thread count, relation size, seed, compiler, and affinity settings.
