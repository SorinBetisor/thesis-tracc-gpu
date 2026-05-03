# CPU vs GPU greedy baseline (no JP): FATRAS full corpus + synthetic reference

**Date:** 2026-05-01  
**New FATRAS sweep:** `20260501_154244_fatras_cpu_gpu_baseline/`  
**Synthetic reference:** existing unified three-backend sweep `20260426_190931_unified_three_backend_sweep/` (baseline columns only; JP ignored here).

## Protocol

| Setting | Value |
|--------|-------|
| CPU binary | `/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver` |
| GPU binary | `/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda` |
| CPU args | `--backend=cpu --repeats=10 --warmup=3` |
| GPU args | `--repeats=10 --warmup=3 --determinism-runs=5` (baseline path only; **no** `--enable-jp`) |
| `LD_PRELOAD` | Same `libstdc++.so` as unified sweep (see `run.log` in this folder if present) |
| Inputs (FATRAS) | All `event_*.json` under `/user/sbetisor/data-work/data/fatras_csv_dumps/fatras_ttbar_mu*/` |

Per-event metrics are parsed from harness stdout into `summary.tsv`. **GPU baseline** columns (`base_*`) are GPU greedy output checked against CPU reference (`base_hash_match`, `baseline_track_overlap_vs_cpu`, duplicate rate post-resolution).

## FATRAS results (this run, N = 79 events)

| Pile-up (μ) | Events | Median CPU ms | Median GPU ms | Median CPU/GPU | Hash match (GPU vs CPU) |
|------------:|-------:|--------------:|--------------:|---------------:|------------------------:|
| 0 | 10 | 0.35 | 1.85 | 0.20 | 10/10 |
| 20 | 10 | 0.92 | 2.20 | 0.40 | 10/10 |
| 50 | 10 | 1.97 | 2.71 | 0.70 | 10/10 |
| 100 | 10 | 4.06 | 4.29 | 0.95 | 10/10 |
| 140 | 10 | 6.07 | 5.01 | 1.20 | 10/10 |
| 200 | 10 | 9.57 | 7.72 | 1.24 | 10/10 |
| 300 | 10 | 15.27 | 10.17 | 1.51 | 10/10 |
| 400 | 3 | 25.12 | 16.76 | 1.61 | 3/3 |
| 500 | 3 | 36.24 | 19.91 | 1.85 | 3/3 |
| 600 | 3 | 50.40 | 26.94 | 1.86 | 3/3 |

**Status:** All 79 rows in `summary.tsv` have `status=ok`. Every event has `base_hash_match=true`, `base_overlap=1`, `base_dup_post=0`, and determinism checks `det_base_pass=5`, `det_base_fail=0` (from parsed GPU logs).

**Interpretation (FATRAS):** For low μ and small candidate counts, fixed GPU launch / sync overhead makes the **CPU faster** (median CPU/GPU &lt; 1). From roughly μ ≈ 140 upward, median **GPU baseline** is faster than CPU on this hardware/config (median CPU/GPU &gt; 1), with the gap widening at μ 400–600.

Raw logs: `raw/<corpus>__<event>.cpu.txt` and `.gpu.txt`. Aggregated table: `summary.tsv`.

## Synthetic reference (unified sweep 2026-04-26, baseline only)

These rows come from `20260426_190931_unified_three_backend_sweep/summary.tsv` (`family=synthetic`). Only **CPU vs GPU baseline** timing and agreement are summarized; JP timings are not shown here.

| Corpus | CPU time (ms) | GPU baseline (ms) | CPU / GPU |
|--------|--------------:|------------------:|----------:|
| n500_low | 2.16 | 5.83 | 0.37 |
| n500_med | 2.47 | 11.65 | 0.21 |
| n500_high | 2.41 | 8.09 | 0.30 |
| n1000_low | 5.55 | 8.83 | 0.63 |
| n1000_med | 5.51 | 15.78 | 0.35 |
| n1000_high | 4.77 | 8.92 | 0.54 |
| n2000_low | 12.44 | 15.29 | 0.81 |
| n2000_med | 14.16 | 27.55 | 0.51 |
| n2000_high | 11.52 | 9.95 | 1.16 |
| n5000_low | 53.87 | 26.95 | 2.00 |
| n5000_med | 46.33 | 34.97 | 1.33 |
| n5000_high | 24.32 | 14.62 | 1.66 |
| n10000_low | 142.49 | 35.92 | 3.97 |
| n10000_med | 107.32 | 36.29 | 2.96 |
| n10000_high | 90.27 | 21.94 | 4.11 |

In that sweep, **GPU baseline hash vs CPU** is `true` for all 15 synthetic rows (`base_hash_match` in source TSV). (JP hash mismatches on some denser synthetics are unrelated to this baseline-only summary.)

## Paths

| Artifact | Path |
|----------|------|
| This FATRAS sweep (TSV + raw) | `/user/sbetisor/data-work/results/20260501_154244_fatras_cpu_gpu_baseline/` |
| Unified sweep (synthetic + older FATRAS incl. JP) | `/user/sbetisor/data-work/results/20260426_190931_unified_three_backend_sweep/` |
| FATRAS JSON dumps | `/user/sbetisor/data-work/data/fatras_csv_dumps/` |
