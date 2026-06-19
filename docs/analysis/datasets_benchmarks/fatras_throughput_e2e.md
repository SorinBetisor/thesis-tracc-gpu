# Fatras End-to-End Throughput — CPU vs GPU Greedy vs JP

**Date:** 2026-06-18  
**GPU:** Quadro GV100 (Nikhef Stoomboot)  
**Branch / commit:** `main` @ `c13cfbb4`  
**Raw data:** `/user/sbetisor/data-work/results/20260618_131900_rerun_fatras_throughput/`  
**Aggregate TSV:** `pileup_aggregate.tsv` (same directory)  
**Script:** `thesis/sorin-thesis-work/scripts/run_fatras_throughput_sweep.sh`

---

## What this measures

Prior Fatras tables reported **resolver-only latency** (`time_ms_mean`): GPU work with input already on device. This sweep adds **end-to-end per-event cost**:

\[
\text{e2e\_ms} = \text{time\_h2d\_ms} + \text{resolver\_ms\_mean} + \text{time\_d2h\_ms}
\]

| Backend | e2e components |
|---------|----------------|
| **CPU greedy** | resolver only (no H2D/D2H) |
| **GPU baseline** | H2D + greedy CUDA resolver + D2H |
| **JP** | same H2D + `--conflict-graph=jp` + D2H |

**Throughput** = `1000 / e2e_ms` events/s (single-event, no batch overlap).

Notes:
- H2D is timed **once per event** before warmup (cold-transfer model).
- Resolver times use 10 timed repeats + 3 warmup.
- One CUDA invocation per event runs baseline then JP on the same dump.
- All JP runs: `hash_match=true` (79/79 events).

---

## Results — mean per pileup level

Speedup columns are **GPU / JP** (>1× ⇒ JP faster).

| μ | n_cand | CPU (ms) | H2D (ms) | GPU res (ms) | GPU e2e (ms) | JP res (ms) | JP e2e (ms) | res speedup (mean) | e2e speedup (mean) | e2e speedup (median) |
|---|--------|----------|----------|--------------|--------------|-------------|-------------|-------------------:|-------------------:|---------------------:|
| 0 | 66 | 0.38 | 0.72 | 1.66 | 2.43 | 2.39 | 3.16 | 0.70× | 0.77× | 0.82× |
| 20 | 147 | 0.92 | 1.07 | 2.05 | 3.22 | 2.69 | 3.84 | 0.76× | 0.84× | 0.85× |
| 50 | 294 | 2.04 | 1.72 | 2.79 | 4.66 | 3.69 | 5.51 | 0.76× | 0.85× | 0.90× |
| 100 | 563 | 4.34 | 2.88 | 3.80 | 6.94 | 4.60 | 7.65 | 0.83× | 0.91× | 0.93× |
| 140 | 776 | 6.25 | 3.77 | 4.93 | 9.03 | 4.86 | 8.76 | 1.01× | 1.03× | 1.03× |
| 200 | 1115 | 9.93 | 5.23 | 7.33 | 12.98 | 7.34 | 12.70 | 1.00× | 1.02× | 1.03× |
| 300 | 1703 | 16.45 | 7.74 | 10.48 | 18.83 | 10.90 | 18.95 | 0.96× | 0.99× | **1.09×** |
| 400 | 2438 | 27.08 | 10.69 | 16.70 | 28.21 | 9.96 | 21.01 | **1.68×** | **1.34×** | 1.33× |
| 500 | 3110 | 38.36 | 13.73 | 21.16 | 35.87 | 12.03 | 25.95 | **1.76×** | **1.38×** | 1.35× |
| 600 | 3955 | 53.02 | 17.34 | 26.61 | 45.15 | 16.23 | 33.97 | **1.64×** | **1.33×** | 1.38× |

---

## Results — JP vs CPU (end-to-end)

| μ | CPU (ms) | JP e2e (ms) | JP vs CPU e2e |
|---|----------|-------------|---------------|
| 0 | 0.38 | 3.16 | 0.12× |
| 100 | 4.34 | 7.65 | 0.57× |
| 140 | 6.25 | 8.76 | 0.71× |
| 300 | 16.45 | 18.95 | 0.87× |
| 400 | 27.08 | 21.01 | **1.29×** |
| 500 | 38.36 | 25.95 | **1.48×** |
| 600 | 53.02 | 33.97 | **1.56×** |

JP beats CPU end-to-end from **μ ≈ 400** upward despite 11–17 ms H2D overhead.

---

## Interpretation

### 1. H2D dilutes resolver speedups

When H2D is ~8–17 ms, end-to-end JP-vs-GPU ratio is **always lower** than resolver-only ratio on the same data. Example at μ=300:

- Resolver mean: 0.96× (essentially tied)
- Median per-event resolver: **1.17×** (JP ahead on typical events)
- End-to-end mean: 0.99×; median per-event: **1.09×**

You cannot report 1.18× end-to-end at μ=300 if resolver mean speedup is ~1.0× — shared H2D caps the e2e gain.

### 2. Two crossover points

| Comparison | Crossover (approx.) |
|------------|---------------------|
| JP vs GPU greedy, resolver-only | **μ ≈ 140** |
| JP vs GPU greedy, end-to-end | **μ ≈ 140** (marginal), clear from **μ ≥ 400** |
| JP vs CPU, end-to-end | **μ ≈ 400** |

### 3. Where JP wins decisively

At μ=400–600, JP delivers **1.3–1.4× end-to-end** over GPU greedy and **1.3–1.6×** over CPU, with full hash agreement. This is the thesis-ready performance regime.

### 4. Low pileup: CPU still wins

Below μ≈100, CPU resolver latency (0.4–4 ms) is far below GPU fixed costs (H2D + launch). End-to-end makes the GPU story worse, not better.

---

## μ=300 — mean vs median

Two JP outlier events (003, 007) with slow JP runs pull the **mean** down. Typical events show JP ahead.

| Aggregation | GPU res (ms) | JP res (ms) | res speedup | e2e speedup |
|-------------|-------------|-------------|------------:|------------:|
| Mean | 10.48 | 10.90 | 0.96× | 0.99× |
| Median | 10.05 | 8.60 | **1.17×** | **1.09×** (median per-event) |

Report μ=300 as **borderline / neutral on mean, modest JP edge on median**, not as a clear win.

---

## Comparison with resolver-only table (20260426 unified sweep)

The earlier latency table used resolver-only means. At μ=300 that sweep reported GPU 10.72 ms / JP 10.99 ms (0.97×). The median there was GPU 10.29 / JP 8.78 (**1.17×**), consistent with this run.

This end-to-end sweep uses the same algorithm flags but adds transfer costs. High-pileup resolver speedups (1.6–1.8× at μ400–600) compress to 1.3–1.4× end-to-end.

---

## Files

```
/user/sbetisor/data-work/results/20260618_131900_rerun_fatras_throughput/
├── summary.tsv              # per-event: cpu, h2d, res, d2h, e2e, hash
├── pileup_aggregate.tsv       # per-corpus mean + median aggregates
├── mean_by_corpus.tsv         # events/s view (legacy aggregate from script)
├── raw_cpu/                   # per-event CPU benchmark stdout
├── raw_gpu/                   # per-event CUDA stdout (baseline + JP)
└── run_metadata.txt
```

Cross-references:
- [`synthetic_throughput_e2e.md`](synthetic_throughput_e2e.md) — same e2e protocol on synthetic low/med/high (n = 500–10000)
- `conflict_graph_results_mis_jp.md` — resolver-only JP vs CPU/baseline (20260426 sweep)
- `original_vs_final_comparison.md` — adaptive n_it + graph reuse vs April baseline
- `bottleneck_analysis.md` — H2D cost discussion (Section 7)

---

## Thesis-ready summary

> End-to-end Fatras benchmarking (H2D + resolver + D2H) shows JP matching GPU greedy from μ≈140 and beating both GPU greedy (**1.3–1.4×**) and CPU (**1.3–1.6×**) from μ≈400 upward. Below μ≈100, CPU wins because transfer and launch overhead dominate. At μ=300 the advantage is neutral on mean latency but visible on median; H2D (~8 ms) compresses resolver gains into smaller end-to-end margins. All 79 events pass `hash_match=true` against the CPU greedy reference.
