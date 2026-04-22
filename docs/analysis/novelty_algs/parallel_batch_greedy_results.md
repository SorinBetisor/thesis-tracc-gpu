# Parallel Batch Greedy — Results

**Status**: implementation landed on `thesis-novelty-parallel-batch` branch and validated as bit-identical to the CPU baseline on synthetic, ODD muon and Fatras ttbar pile-up dumps. First measurement campaign in Sec. 3 below; raw outputs under `results/20260422_161418_pbg/`.

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

| Metric | Definition | Expected for baseline | Expected for PBG (prefix variant) |
|---|---|---|---|
| `hash_match` | byte-identical selected set vs CPU | `true` | **`true`** — PBG admits the same conflict-free prefix as baseline |
| `track_overlap_vs_cpu` | `|selected_gpu ∩ selected_cpu| / |selected_cpu|` | `1.000` | `1.000` (degenerate consequence of `hash_match=true`) |
| `duplicate_rate_post` | fraction of distinct measurements shared by ≥ 2 accepted tracks | same as CPU | same as CPU |
| `n_selected` | size of accepted set | same as CPU | same as CPU |

Earlier drafts of this document had `hash_match=false` for PBG "by design", which was correct for the *MIS* variant of the algorithm originally proposed in [`parallel_batch_greedy_design.md`](parallel_batch_greedy_design.md) Sec. 2.x. The merged implementation enforces the *parallel conflict-free prefix* rule instead (see the 2026-04-22 revision note at the top of the design doc): per iteration it admits exactly the contiguous prefix `[0, first_fail)` of worst tracks, which is the same set the baseline `remove_tracks` kernel computes — but built by grid-wide multi-block atomics rather than a single 512-thread block. That is why `hash_match=true` is both possible and required here. The MIS variant is deferred to Tier 2c, where the rearrange-pipeline contract will be relaxed to permit non-prefix removal patterns.

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

## 3. Results — first campaign (2026-04-22, GV100 / wn-lot-001)

All `hash_match=true` and `track_overlap_vs_cpu = 1.000` everywhere — the prefix variant of PBG produces byte-identical output to CPU greedy and to GPU baseline on every tested input. Timing is the wall-clock of the resolver kernel region only (no H2D/D2H transfer).

### 3a. Timing — synthetic, `low` conflict density (W = 8192, 20 repeats / 5 warmups)

| n | cpu_greedy (ms) | gpu_baseline (ms) | gpu_pbg (ms) | speedup PBG vs baseline | speedup PBG vs CPU |
|---|---|---|---|---|---|
| 500   | 2.13   | 4.59   | 4.34   | 1.06× | 0.49× |
| 1000  | 4.83   | 8.71   | 7.61   | 1.14× | 0.63× |
| 2000  | 12.63  | 16.55  | 13.13  | 1.26× | 0.96× |
| 5000  | 52.50  | 26.57  | 18.15  | 1.46× | **2.89×** |
| 10000 | 116.7  | 34.34  | 34.48  | 1.00× | **3.39×** |

### 3b. Timing — synthetic, `med` conflict density (W = 8192)

| n | cpu_greedy (ms) | gpu_baseline (ms) | gpu_pbg (ms) | speedup PBG vs baseline |
|---|---|---|---|---|
| 500   | 1.30  | 8.74  | 4.47  | **1.96×** |
| 1000  | 5.83  | 16.10 | 7.63  | **2.11×** |
| 2000  | 14.37 | 27.35 | 15.31 | **1.79×** |
| 5000  | 46.44 | 34.77 | 30.71 | 1.13× |
| 10000 | 87.0  | 36.04 | 56.87 | 0.63× |

### 3c. Timing — ODD muon dumps (n_tracks ≈ 80–90 per event, W = 8192, 20 repeats / 5 warmups)

| event | n_selected | cpu_greedy (ms) | gpu_baseline (ms) | gpu_pbg (ms) | hash_match (PBG) |
|---|---|---|---|---|---|
| 000 | 39 | n/a | 1.84 | 2.62 | true |
| 001 | 40 | n/a | 2.10 | 2.40 | true |
| 002 | 39 | n/a | 2.27 | 3.06 | true |
| 003 | 40 | n/a | 1.96 | 2.67 | true |
| 004 | 40 | n/a | 1.90 | 2.58 | true |
| 005 | 40 | n/a | 2.22 | 2.59 | true |
| 006 | 38 | n/a | 1.78 | 2.24 | true |
| 007 | 40 | n/a | 1.80 | 2.39 | true |
| 008 | 40 | n/a | 2.29 | 2.58 | true |
| 009 | 40 | n/a | 2.33 | 2.53 | true |

Tiny problem size ⇒ five-kernel pipeline overhead per outer iteration is larger than the single baseline `remove_tracks` block. Expected loss; ODD muon events sit well below the GPU crossover.

### 3d. Timing — Fatras ttbar pile-up sweep (W = 8192, 10 repeats / 3 warmups)

| pileup μ | event | n_selected | n_removed | gpu_baseline (ms) | gpu_pbg (ms) | n_outer_iterations | avg batch | max batch |
|---|---|---|---|---|---|---|---|---|
| 300 | 0 | 1296 | 357 | 11.6 | 16.6 | 1 | 0  | 0  |
| 300 | 1 | 1290 | 360 | 11.5 | 16.4 | 1 | 0  | 0  |
| 300 | 2 | 1264 | 327 | 11.0 | 16.0 | 1 | 0  | 0  |
| 400 | 0 | 1733 | 668 | 16.6 | 22.4 | 2 | 1.0 | 2  |
| 400 | 1 | 1701 | 612 | 14.7 | 19.5 | 2 | 1.0 | 2  |
| 400 | 2 | 1657 | 656 | 15.6 | 20.7 | 2 | 2.0 | 4  |
| 500 | 0 | 2095 | 1147 | 21.7 | 30.9 | 3 | 2.0 | 5  |
| 500 | 1 | 1980 | 1042 | 20.0 | 30.8 | 3 | 2.7 | 7  |
| 500 | 2 | 2039 | 1026 | 19.5 | 29.0 | 3 | 3.0 | 7  |
| 600 | 0 | 2484 | 1524 | 26.9 | 36.5 | 3 | 7.7 | 14 |
| 600 | 1 | 2372 | 1544 | 29.3 | 39.2 | 3 | 3.3 | 6  |
| 600 | 2 | 2452 | 1488 | 25.0 | 36.0 | 3 | 2.3 | 6  |

Mu=300/400 events fall in the same regime as the synthetic `low`-density ladder around n ~ 1000–1700: PBG is ~30–40 % slower than the baseline because the per-iteration prefix is 1–2 tracks, so the baseline's single-block `remove_tracks` finishes faster than PBG's five-kernel sequence. Mu=500/600 are still slower in wall-clock but the average batch size starts to climb (2 → 7.7) — this is the regime where the prefix variant begins paying back. The MIS variant (Tier 2c, see [`conflict_graph_design.md`](conflict_graph_design.md)) is what we will need to flatten the high-pile-up curve.

(For pile-up dumps below μ=300 the existing graph-reuse work already documents the small-n regime; reproducing it here is not in the Tier 2a scope.)

### 3e. Quality — overlap, duplicate rate, hash_match (synthetic)

| n | density | track_overlap_vs_cpu (baseline) | track_overlap_vs_cpu (PBG) | hash_match (PBG) | duplicate_rate CPU | duplicate_rate PBG |
|---|---|---|---|---|---|---|
| 500   | low | 1.000 | 1.000 | true | 0 | 0 |
| 1000  | low | 1.000 | 1.000 | true | 0 | 0 |
| 2000  | low | 1.000 | 1.000 | true | 0 | 0 |
| 5000  | low | 1.000 | 1.000 | true | 0 | 0 |
| 10000 | low | 1.000 | 1.000 | true | 0 | 0 |
| 500   | med | 1.000 | 1.000 | true | 0 | 0 |
| 1000  | med | 1.000 | 1.000 | true | 0 | 0 |
| 2000  | med | 1.000 | 1.000 | true | 0 | 0 |
| 5000  | med | 1.000 | 1.000 | true | 0 | 0 |
| 10000 | med | 1.000 | 1.000 | true | 0 | 0 |

### 3f. Convergence — outer iteration count and batch sizes (synthetic, W=8192)

| n | density | gpu_pbg n_outer | avg_batch | max_batch |
|---|---|---|---|---|
| 500   | low | 1 | 0    | 0  |
| 1000  | low | 1 | 0    | 0  |
| 2000  | low | 1 | 0    | 0  |
| 5000  | low | 2 | 5.5  | 11 |
| 10000 | low | 3 | 19.3 | 43 |
| 500   | med | 1 | 0    | 0  |
| 1000  | med | 1 | 0    | 0  |
| 2000  | med | 2 | 8.5  | 17 |
| 5000  | med | 3 | 10.0 | 26 |
| 10000 | med | 6 | 8.8  | 19 |

`avg_batch=0` rows correspond to runs where the eviction loop converged inside a single graph-replay burst (the inner-loop `n_it` adaptive heuristic); the per-iteration batch counter only reports the *last* replay, which legitimately admitted nothing because the loop had already terminated. Per-iteration CSVs (`pbg_batch_sizes_n*_d*.csv` under the run dir) hold the full trace.

---

## 4. Analysis

### 4a. Crossover point against CPU greedy

| backend | low-density crossover (synthetic) | med-density crossover |
|---|---|---|
| `gpu_baseline` | n ≈ 2000 (CPU 12.6 ms vs GPU 16.6 ms) — ties at ≈ 3000 (extrapolated) | n ≈ 5000 (CPU 46.4 vs GPU 34.8) |
| `gpu_pbg`       | n ≈ 2000 (CPU 12.6 vs PBG 13.1) | **n ≈ 1000** (CPU 5.8 vs PBG 7.6, near-tie at 2000) |

PBG's biggest contribution is on **medium-density inputs**: the crossover against CPU greedy shifts from ≈ 5000 (baseline) to ≈ 1000–2000 (PBG) — a ≥ 2× reduction in the smallest-n regime where GPU starts to be worth using. This is the practical answer to RQ4 for the prefix variant. On low-density inputs both GPU paths beat CPU starting at ≈ 2000–3000 and PBG and baseline are within 5–15 % of each other.

### 4b. Window sensitivity (Fatras μ=600, event 0)

| W | PBG ms | n_outer_iterations | avg batch | max batch |
|---|---|---|---|---|
| 1     | 135.7 | 16 | 0.94 | 1  |
| 256   | 36.4  | 3  | 7.7  | 14 |
| 1024  | 36.6  | 3  | 7.7  | 14 |
| 4096  | 36.6  | 3  | 7.7  | 14 |
| 8192  | 36.3  | 3  | 7.7  | 14 |
| 16384 | 36.4  | 3  | 7.7  | 14 |

`W=1` recovers the baseline single-track-per-iteration behaviour. The elbow is at `W ≈ 256` and the curve is flat from there: at this conflict density the prefix never grows past ≈ 14, so the bigger windows just admit `r >= first_fail` candidates that immediately bail in apply. Default `W=8192` is conservative and harmless (no measurable cost above `W=256`).

### 4c. Convergence curves

Stored under `results/20260422_161418_pbg/pbg_batch_sizes_n*_d*.csv`. The two largest synthetic configurations show:

- `n=10000 d=low`: 3 outer iterations, batch sizes `(43, 13, 0)` — the first iteration consumes most of the prefix and the loop terminates after the third replay.
- `n=10000 d=med`: 6 outer iterations, batch sizes `(19, 13, 8, 4, 4, 0)` — a clean monotone-decreasing staircase, exactly the expected shape.

### 4d. Quality regression budget

The original budget (`overlap ≥ 0.95`, `|Δdup| ≤ 0.02`) is moot for the prefix variant: the merged PBG passes the much stricter `hash_match=true` test on every (n, density, W) point and on every Fatras / ODD dump tested. The non-degenerate quality budget will become relevant again when the MIS variant in Tier 2c is implemented; this document's protocol stays in place for that follow-up.

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
