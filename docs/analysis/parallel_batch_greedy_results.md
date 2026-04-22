# Parallel Batch Greedy — Results Skeleton

**Status**: implementation landed on `thesis-novelty-parallel-batch` branch; this document is the **measurement protocol** and **reporting shell**.
Actual numbers will be filled in after the benchmark sweep is run on the same hardware as `physics_dataset_benchmark.md`.

**Hardware target** (to be confirmed at run time, must match the baseline numbers we compare against):
- GPU: NVIDIA Quadro GV100 (Stoomboot `wn-lot-001`), CUDA ≥ 12.x, driver ≥ 560.
- CPU: Intel Xeon (same node), single-threaded reference.
- Compiler / build flags: `-O3 -DNDEBUG`, traccc built with `TRACCC_BUILD_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=70`.

---

## 1. What this document will eventually contain

This is a deliberately scaffolded results document. It is committed together with the PBG implementation so the thesis has a stable URL for the experiment; the tables below are structured placeholders with `TBD` cells. Each `TBD` has a concrete command that produces the number.

Cross-references for the reader:
- Algorithm and correctness argument: `parallel_batch_greedy_design.md`.
- Baseline we compare against: `physics_dataset_benchmark.md` (same n grid, same synthetic generator).
- Why the CPU-identical `hash_match` no longer applies: `parallel_batch_greedy_design.md` Sec. "Determinism & quality claim".

---

## 2. Measurement protocol

### 2a. Inputs

Use the exact same five pre-resolver states as `bottleneck_analysis.md` and `physics_dataset_benchmark.md` so results are directly comparable:

| Source | n_candidates (approx) | Conflict density | How produced |
|---|---|---|---|
| Synthetic, physics-calibrated | 500, 1000, 2000, 5000, 10000 | `low` (primary), `med` (cross-check) | `benchmark_resolver_cuda --synth --n-candidates=<N> --conflict=<density>` |
| Fatras ODD ttbar pileup sweep | 56, 154, 307, 602, 821, 1167, 1770 | real | pre-serialized via `--dump-ambiguity-input` from Fatras pipeline, same dumps referenced in Sec. 14 of `bottleneck_analysis.md` |

Every point uses 20 timed repeats + 5 warmup iterations. Identical RNG seed across baseline and PBG runs so the two algorithms see the same candidate set.

### 2b. Algorithms under test

Three backends per input:

| Label | Binary | Flags |
|---|---|---|
| `cpu_greedy` (reference) | `benchmark_resolver` | default |
| `gpu_baseline` | `benchmark_resolver_cuda` | default |
| `gpu_pbg` | `benchmark_resolver_cuda` | `--parallel-batch --parallel-batch-window=<W>` |

Parallel-batch window sweep: `W ∈ {1024, 4096, 8192, 16384}`. `W=1` is a degenerate case that should reproduce the baseline removal order (reported once as a sanity check in Sec. 4b).

### 2c. Metrics

Timing (GPU-only path, resolver kernel region between NVTX markers):

| Metric | Source | Unit |
|---|---|---|
| `time_ms_mean`, `time_ms_std` | harness, 20 repeats | ms |
| `time_ms_median`, `time_ms_p95` | harness | ms |
| per-phase timings (`batch_identify`, `batch_apply`, `update_status`, `sort_updated_tracks`) | NVTX + nsys | ms |
| `n_outer_iterations` | harness (PBG only) | count |
| `avg_batch_size`, `max_batch_size` | harness (PBG only) | count |

Quality (both GPU backends compared against `cpu_greedy` on the same input):

| Metric | Definition | Expected for baseline | Expected for PBG |
|---|---|---|---|
| `hash_match` | byte-identical selected set vs CPU | `true` | **`false` by design** |
| `track_overlap_vs_cpu` | `|selected_gpu ∩ selected_cpu| / |selected_cpu|` | `1.000` | `≥ 0.95` hypothesis |
| `duplicate_rate_post` | fraction of distinct measurements shared by ≥ 2 accepted tracks | same as CPU | within `±0.01` of CPU |
| `n_selected` | size of accepted set | same as CPU | within `±1%` of CPU |

Hardware efficiency (from `ncu` on the two new kernels):

| Metric | Target kernel | Why we care |
|---|---|---|
| SM occupancy | `batch_identify_removals`, `apply_batch_removals` | validates parallelism claim |
| DRAM bandwidth utilization | `batch_identify_removals` | dominated by `tracks_per_measurement` reads |
| Atomic contention | `apply_batch_removals` | risk surface for the atomicMin / atomicSub design |
| L2 hit rate | `batch_identify_removals` | `claimed_by` reuse pattern |

### 2d. Harness invocation template

```bash
for N in 500 1000 2000 5000 10000; do
  ./benchmark_resolver_cuda \
    --synth --n-candidates=$N --conflict=low --repeats=20 --warmup=5 \
    --parallel-batch --parallel-batch-window=8192 \
    --log-batch-sizes=results/pbg_batch_sizes_n${N}.csv \
    --output=results/pbg_n${N}.json
done
```

Baseline rows are produced automatically by the same invocation — the harness runs both algorithms per input (see Sec. 4 of the implementation plan / `benchmark_resolver_cuda.cpp` `run_one` refactor).

---

## 3. Results tables (TBD until runs are collected)

### 3a. Timing — synthetic, `low` conflict density

| n | cpu_greedy (ms) | gpu_baseline (ms) | gpu_pbg, W=8192 (ms) | speedup PBG vs baseline | speedup PBG vs CPU |
|---|---|---|---|---|---|
| 500   | TBD | TBD | TBD | TBD× | TBD× |
| 1000  | TBD | TBD | TBD | TBD× | TBD× |
| 2000  | TBD | TBD | TBD | TBD× | TBD× |
| 5000  | TBD | TBD | TBD | TBD× | TBD× |
| 10000 | TBD | TBD | TBD | TBD× | TBD× |

### 3b. Timing — Fatras pileup sweep (real physics)

| pileup μ | n | cpu_greedy (ms) | gpu_baseline (ms) | gpu_pbg, W=8192 (ms) | GPU wins vs CPU? |
|---|---|---|---|---|---|
| 0   | 56    | TBD | TBD | TBD | TBD |
| 20  | 154   | TBD | TBD | TBD | TBD |
| 50  | 307   | TBD | TBD | TBD | TBD |
| 100 | 602   | TBD | TBD | TBD | TBD |
| 140 | 821   | TBD | TBD | TBD | TBD |
| 200 | 1167  | TBD | TBD | TBD | TBD |
| 300 | 1770  | TBD | TBD | TBD | TBD |

### 3c. Quality — overlap and duplicate rate

| n | track_overlap_vs_cpu (baseline) | track_overlap_vs_cpu (PBG) | duplicate_rate CPU | duplicate_rate PBG |
|---|---|---|---|---|
| 500   | TBD | TBD | TBD | TBD |
| 1000  | TBD | TBD | TBD | TBD |
| 2000  | TBD | TBD | TBD | TBD |
| 5000  | TBD | TBD | TBD | TBD |
| 10000 | TBD | TBD | TBD | TBD |

### 3d. Convergence — outer iteration count

| n | gpu_baseline iterations | gpu_pbg iterations | avg_batch_size | max_batch_size |
|---|---|---|---|---|
| 500   | TBD | TBD | TBD | TBD |
| 1000  | TBD | TBD | TBD | TBD |
| 2000  | TBD | TBD | TBD | TBD |
| 5000  | TBD | TBD | TBD | TBD |
| 10000 | TBD | TBD | TBD | TBD |

Per-iteration batch-size curves (`results/pbg_batch_sizes_n*.csv`) will be plotted in Sec. 4c.

---

## 4. Analysis (to write after runs)

### 4a. Crossover point against CPU greedy

Baseline today (from `physics_dataset_benchmark.md` Sec. 2a): GPU beats CPU only at n ≳ 1770. **Hypothesis for PBG** (restated from `novelty_improvements.md` Sec. 4a): crossover shifts down to the 500–1000 range.
This subsection will report the measured crossover for both `gpu_baseline` and `gpu_pbg`.

### 4b. Window sensitivity

Sweep `W ∈ {1, 1024, 4096, 8192, 16384}` at a single representative n (1770, Fatras μ=300) and a single conflict density (`low`). Expected shape:
- `W=1` recovers baseline behaviour (sanity check).
- Increasing `W` reduces `n_outer_iterations` until the per-iteration kernel overhead stops improving — pick the elbow as the default.

### 4c. Convergence curve

Plot `batch_size` vs outer iteration index for the largest synthetic n and the largest Fatras n. Expectation: monotone-decreasing staircase, settling at single-digit batches late in the loop.

### 4d. Quality regression budget

If `track_overlap_vs_cpu < 0.95` or `|duplicate_rate_PBG − duplicate_rate_CPU| > 0.02` at any configuration, PBG is considered a failed quality regression and the result will be reported honestly without recommending PBG as default. The hand-off criteria to Tier 2c (explicit conflict graph) in `conflict_graph_design.md` depend on how large the quality gap is here.

### 4e. Hardware efficiency

Summary table from `ncu` runs on `batch_identify_removals` and `apply_batch_removals` at n=10000 synthetic, `low` density. Checks against the two main risks from the design doc: atomic contention on `n_accepted_tracks_per_measurement` and serialization inside `claimed_by`.

---

## 5. Result artefacts

All raw outputs go under `thesis/sorin-thesis-work/results/<YYYYMMDD_HHMMSS>_pbg/`:
- `*.json` — one file per n, both backends in a single invocation.
- `pbg_batch_sizes_n*.csv` — per-iteration batch-size trace.
- `ncu_*.ncu-rep` — NCU reports for the two new kernels.
- `nsys_*.qdrep` — nsys traces, scoped to the NVTX resolver region.

A summary CSV (`summary.csv`) aggregated from the JSONs is produced by a small Python helper living in `thesis/sorin-thesis-work/scripts/` (to be added alongside the run data), mirroring the existing graph-reuse results pipeline.

---

## 6. Reproducibility checklist

Before publishing results into the thesis, verify:

- [ ] `git rev-parse HEAD` on `thesis-novelty-parallel-batch` matches the commit recorded in the JSON metadata.
- [ ] `cpu_greedy` numbers regenerated on the same node as `physics_dataset_benchmark.md` — not reused from that file.
- [ ] Every row has both `gpu_baseline` and `gpu_pbg` from the **same** harness invocation (same RNG seed, same input).
- [ ] `--log-batch-sizes` CSVs exist for every PBG row.
- [ ] `ncu` reports exist for at least the largest synthetic n.
- [ ] Quality budget (Sec. 4d) evaluated against the actual numbers, not against expected thresholds.
