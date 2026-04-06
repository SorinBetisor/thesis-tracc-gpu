# Bottleneck Analysis — GPU vs CPU Greedy Ambiguity Resolver

**Prepared:** 2026-04-06  
**Data sources:**  
- CPU: `results/cpu_benchmark_ambig_resolution_synthetic/20260325_profile/` (Stoomboot CPU node, warmup=3, repeats=10, seed=42)  
- GPU: `results/20260401_121851_cuda_profile/` (Stoomboot wn-lot-001, Quadro GV100, SM70, warmup=3, repeats=10, seed=42)  
- Real physics: `traccc_seq_example` on `odd/geant4_10muon_1GeV/`, 3 events measured

**Addresses RQs:** RQ1 (which sub-steps dominate and how do costs scale?), RQ4 (under which regime does GPU outperform CPU?)

---

## 1. Synthetic data: what the n values mean

The n_candidates values (1000, 5000, 10000) used in the benchmarks are **entirely artificial** — set by us via `--n-candidates=N`. They do not come from physics.

For reference, running `traccc_seq_example` on the available `geant4_10muon_1GeV` events produced:

| Real dataset | n_candidates / event | Notes |
|---|---|---|
| `geant4_10muon_1GeV` | **~87** | 10 muons at 1 GeV; low-momentum muons stop early → few track candidates |
| (estimate) `ttbar_mu200` | ~15,000–30,000 | High-pileup TTbar; not yet downloaded |

**Implication:** The low-momentum ODD events relevant to the ALICE supervisor context are far below even the smallest synthetic test. The GPU loses badly without optimization at this scale. High-pileup TTbar events would sit firmly in the GPU-wins regime.

---

## 2. CPU vs GPU: end-to-end resolver timing

All times are `time_ms_median` (GPU resolver-only; excludes H2D/D2H transfers). Speedup = CPU median / GPU median. Values > 1 mean GPU wins.

| Config | CPU (ms) | GPU resolver (ms) | Speedup | Verdict |
|---|---|---|---|---|
| n1000_low | 3.589 | 10.500 | 0.34× | GPU 3.0× slower |
| n1000_med | 4.057 | 18.602 | 0.22× | GPU 4.6× slower |
| n1000_high | 3.470 | 8.124 | 0.43× | GPU 2.3× slower |
| n5000_low | 40.611 | 26.630 | **1.53×** | GPU faster |
| n5000_med | 34.454 | 34.689 | ~1.00× | parity |
| n5000_high | 22.529 | 13.300 | **1.69×** | GPU faster |
| n10000_low | 143.676 | 34.226 | **4.20×** | GPU clearly faster |
| n10000_med | 80.482 | 36.057 | **2.23×** | GPU clearly faster |
| n10000_high | 59.566 | 20.937 | **2.85×** | GPU clearly faster |

**Current crossover: approximately n = 3,000–5,000 candidates** (density-dependent). Below this point the GPU pays more in launch overhead than it saves in parallelism. Above it, the GPU wins decisively and the advantage grows with n.

H2D/D2H transfer times are reported separately and are NOT included in these speedup numbers. They are discussed in Section 5.

---

## 3. Per-phase breakdown: CPU

All values in milliseconds. `eviction_%` = eviction_loop_ms / total_time_ms.

| Config | filter_setup | unique_meas | inverted_index | shared_count | initial_sort | eviction_loop | output_copy | eviction_% | iterations |
|---|---|---|---|---|---|---|---|---|---|
| n1000_low | 0.122 | 0.879 | 0.935 | 0.287 | 0.102 | 0.598 | 0.243 | 17% | 256 |
| n1000_med | 0.062 | 0.690 | 0.931 | 0.310 | 0.113 | 1.544 | 0.149 | 37% | 587 |
| n1000_high | 0.064 | 0.129 | 1.298 | 0.355 | 0.087 | 1.849 | 0.023 | 53% | 981 |
| n5000_low | 0.375 | 4.390 | 5.849 | 2.361 | 0.597 | 24.003 | 0.782 | 59% | 2930 |
| n5000_med | 0.268 | 1.597 | 5.358 | 1.485 | 0.582 | 24.755 | 0.314 | 72% | 4377 |
| n5000_high | 0.284 | 0.372 | 4.505 | 0.930 | 0.518 | 15.467 | 0.024 | 68% | 4983 |
| n10000_low | 0.765 | 6.850 | 13.469 | 4.489 | 1.325 | 101.086 | 1.064 | 71% | 7317 |
| n10000_med | 0.518 | 1.924 | 10.504 | 3.149 | 1.116 | 60.668 | 0.484 | 76% | 9334 |
| n10000_high | 0.550 | 0.661 | 9.011 | 1.844 | 1.098 | 58.505 | 0.028 | 98% | 9980 |

**CPU observations:**
- The eviction loop dominates at large n, but at small n (n=1000_low, only 17%) the preprocessing phases (unique_meas, inverted_index) are significant.
- `inverted_index` scales strongly with n × unique_meas_count (builds a per-measurement track list). At n=10000_low with 36,314 unique measurements, it costs 13.5 ms.
- `shared_count` scales with n × conflict_density. At n=10000_low it costs 4.5 ms; at n=10000_high only 1.8 ms (fewer unique measurements → cheaper lookup).
- CPU eviction loop iteration count ≈ n_removed. One track is removed per sequential iteration — inherently serial O(n²) behavior.

---

## 4. Per-phase breakdown: GPU

All values in milliseconds. `eviction_%` = eviction_loop_ms / time_ms_mean. `graph_launches` = number of outer CUDA Graph execution cycles.

| Config | filter_setup | unique_meas | inverted_index | shared_count | initial_sort | eviction_loop | output_copy | eviction_% | graph_launches |
|---|---|---|---|---|---|---|---|---|---|
| n1000_low | 0.121 | 0.424 | 0.481 | 0.039 | 0.186 | 9.050 | 0.083 | 86% | 200 |
| n1000_med | 0.141 | 0.404 | 0.380 | 0.037 | 0.183 | 16.445 | 0.096 | 91% | 400 |
| n1000_high | 0.113 | 0.367 | 0.193 | 0.034 | 0.151 | 6.880 | 0.080 | 85% | 200 |
| n5000_low | 0.149 | 0.440 | 1.471 | 0.043 | 0.308 | 23.211 | 0.115 | 87% | 500 |
| n5000_med | 0.149 | 0.423 | 0.631 | 0.036 | 0.301 | 32.381 | 0.116 | 93% | 700 |
| n5000_high | 0.151 | 0.394 | 0.223 | 0.035 | 0.305 | 11.520 | 0.088 | 86% | 300 |
| n10000_low | 0.202 | 1.013 | 2.825 | 0.038 | 0.353 | 29.801 | 0.338 | 87% | 600 |
| n10000_med | 0.198 | 0.438 | 0.699 | 0.036 | 0.347 | 33.587 | 0.137 | 93% | 700 |
| n10000_high | 0.183 | 0.397 | 0.441 | 0.034 | 0.331 | 19.151 | 0.108 | 91% | 500 |

**GPU observations:**
- The eviction loop is **uniformly 85–93% of GPU runtime** regardless of n or density. On the CPU it ranges from 17% to 98%. The GPU's preprocessing phases are fast enough that the loop dominates at every scale.
- `shared_count` is the most dramatically accelerated phase: CPU 4.49 ms vs GPU 0.038 ms at n=10000_low — a **118× speedup**. The `count_shared_measurements` kernel is embarrassingly parallel across measurements.
- `inverted_index` and `unique_meas` also see 4–7× speedups at large n thanks to Thrust parallel primitives.
- The GPU preprocessing phases are nearly **constant** across densities (unique_meas, shared_count change little). The CPU preprocessing scales with conflict_density; the GPU does not benefit from low density in the same way.

---

## 5. Per-phase speedup ratios (CPU/GPU)

Values > 1 = GPU faster. n=10000_low shown as representative large-n case; n=1000_low as small-n.

| Phase | n=1000_low | n=10000_low | Interpretation |
|---|---|---|---|
| filter_setup | 1.01× | **3.79×** | GPU kernel vs CPU loop; scales with n |
| unique_meas | 2.07× | **6.76×** | Thrust sort+unique vs CPU std::sort; scales with unique_meas_count |
| inverted_index | 1.94× | **4.77×** | Parallel fill vs sequential CPU loop |
| shared_count | **7.36×** | **118×** | Best GPU phase — embarrassingly parallel |
| initial_sort | 0.55× | **3.75×** | Thrust sort fast at large n; GPU loses at n=1000 (kernel launch cost) |
| eviction_loop | **0.07×** | 3.39× | GPU loses badly at small n; wins at large n |
| output_copy | 2.94× | 3.15× | Consistent parallel copy advantage |

The eviction loop speedup at n=1000_low is **0.07×** — the GPU is 14× slower. This is the primary cause of the overall GPU disadvantage at small n.

---

## 6. Root cause: the `n_it = 100` over-execution problem

Inside the GPU eviction loop ([`greedy_ambiguity_resolution_algorithm.cu`](../../data-work/traccc/device/cuda/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu), line 662):

```cpp
const unsigned int n_it = 100;
for (unsigned int iter = 0; iter < n_it; iter++) {
    cudaGraphLaunch(graphExec, stream);
}
// Only AFTER this block: sync + read terminate + n_accepted
```

**The termination flag is only checked between outer while-loop cycles, not between graph launches.** Each group of 100 graph launches runs to completion even if the algorithm has already converged after the first launch in the group.

### Over-execution quantification

| Config | CPU iterations needed | GPU total inner iterations (graph_launches × 100) | Over-execution ratio |
|---|---|---|---|
| n1000_low | 256 | 200 × 100 = **20,000** | **78×** |
| n1000_med | 587 | 400 × 100 = **40,000** | **68×** |
| n1000_high | 981 | 200 × 100 = **20,000** | **20×** |
| n5000_low | 2,930 | 500 × 100 = **50,000** | **17×** |
| n5000_med | 4,377 | 700 × 100 = **70,000** | **16×** |
| n5000_high | 4,983 | 300 × 100 = **30,000** | **6×** |
| n10000_low | 7,317 | 600 × 100 = **60,000** | **8×** |
| n10000_med | 9,334 | 700 × 100 = **70,000** | **7.5×** |
| n10000_high | 9,980 | 500 × 100 = **50,000** | **5×** |

Note: the GPU algorithm processes tracks in parallel, so a single inner iteration removes more than one track at a time. The CPU "iterations needed" is an upper bound on the equivalent GPU inner iterations. Even so, the over-execution ratios demonstrate that the GPU does far more work than necessary, and at small n the ratio is extreme (78×).

**Why the GPU still wins at large n despite over-execution:** Parallel processing. A single inner iteration of the GPU removes multiple tracks simultaneously, so the effective throughput per graph launch is much higher than the CPU's one-track-per-iteration serial loop. At n=10000 the parallelism benefit outweighs the launch overhead. At n=1000 it does not.

**Why n=87 (real muon events) would be even worse:**
- The graph overhead (~6–9 ms fixed cost at n=1000) does not shrink proportionally with n
- At n=87, the CPU can resolve the event in ~1–2 ms; the GPU would still pay 5–8 ms in graph launch overhead
- Without fixing this, the GPU resolver is impractical for all available low-multiplicity ODD physics events

---

## 7. Transfer costs (GPU only)

H2D and D2H costs are one-time costs per event (not amortized over `--repeats`).

| Config | H2D (ms) | D2H (ms) | Resolver (ms) | End-to-end (ms) | CPU (ms) | End-to-end speedup |
|---|---|---|---|---|---|---|
| n1000_low | 5.37 | 0.32 | 10.50 | 16.19 | 3.59 | 0.22× |
| n5000_low | 21.11 | 0.75 | 26.73 | 48.59 | 40.54 | 0.83× |
| n10000_low | 40.71 | 0.22 | 34.24 | 75.17 | 142.33 | **1.89×** |
| n10000_high | 40.40 | 0.04 | 21.01 | 61.45 | 59.61 | 0.97× |

**Including transfers, the GPU wins end-to-end only at n=10000_low** (where the resolver speedup is large enough to absorb the ~40 ms H2D cost). For most configurations the H2D transfer dominates over the resolver computation. This illustrates why resolver-only timing is the right scope for evaluating the algorithm's performance, and why minimizing the GPU resolver time (via adaptive n_it) is the primary lever.

---

## 8. Scaling behavior summary

| Metric | CPU scaling | GPU scaling |
|---|---|---|
| Total time vs n | Super-linear (O(n²) eviction loop) | Sub-linear (parallel removal per iteration) |
| Eviction loop share | 17%→98% as n increases | Stable ~87–93% at all n |
| shared_count | Scales with n × conflict_density | Nearly constant (~0.034–0.043 ms); fully parallelized |
| inverted_index | Scales with n × unique_meas_count | Scales but 4–7× faster than CPU |
| Speedup vs n | — | Grows with n; ~0.3× at n=1000, ~4.2× at n=10000 |

The GPU's scaling advantage is primarily driven by:
1. The eviction loop: GPU processes many removals in parallel per iteration; CPU processes one per iteration
2. Preprocessing phases: Thrust primitives on large arrays outperform sequential CPU loops

---

## 9. Key findings (thesis-ready)

1. **The eviction loop is the dominant phase for both backends** at n ≥ 5000 (GPU: always 85–93%; CPU: 59–98%). Optimizing this phase is the highest-leverage target.

2. **shared_count is the most GPU-favorable phase** (up to 118× speedup at n=10000_low). The `count_shared_measurements` kernel is embarrassingly parallel and could be highlighted as an example of ideal GPU workload.

3. **The GPU crossover point is ~n = 3,000–5,000** under current implementation. Above this, GPU wins; below, CPU wins.

4. **The `n_it = 100` hardcoded value is the root cause of GPU underperformance at small n.** It causes 6–78× over-execution of inner loop iterations relative to what is needed. Fixing this adaptively is expected to lower the crossover to ~n = 200–500.

5. **Real low-multiplicity ODD physics events (~87 candidates/event for `geant4_10muon_1GeV`) fall well below the current crossover point.** The GPU resolver is currently impractical for these events. The adaptive `n_it` optimization is needed to make the GPU viable for real low-multiplicity workloads.

6. **Transfer costs are significant.** For an end-to-end GPU pipeline, the H2D transfer (~5–40 ms) must be amortized across many events or hidden by overlapping with other GPU work. This is a separate concern from resolver optimization.
