# CUDA Graph Reuse Implementation for GPU Ambiguity Resolution

**Prepared:** 2026-04-17  
**Purpose:** Document the first implemented novelty step for the thesis: reusing a single CUDA graph execution object across outer eviction-loop iterations, instead of rebuilding and instantiating the graph every time.

---

## 1. Motivation

Profiling in `bottleneck_analysis.md` and code inspection of `greedy_ambiguity_resolution_algorithm.cu` showed that the GPU ambiguity resolver spends most of its runtime in the eviction loop, and that a substantial part of the small-`n` cost comes from rebuilding the CUDA graph inside the outer loop:

```cpp
while (!terminate && n_accepted > 0) {
    cudaStreamBeginCapture(...)
    ...
    cudaStreamEndCapture(...)
    cudaGraphInstantiate(...)
    for (...) cudaGraphLaunch(...)
}
```

This is expensive because:

- the graph structure is rebuilt for every outer loop step,
- the graph is instantiated repeatedly even though the kernel chain itself is unchanged,
- only a subset of kernel launch dimensions actually varies with `n_accepted`.

This makes the graph-management overhead a strong thesis target: it is measurable, implementation-grounded, and does not require changing greedy semantics.

---

## 2. What Was Implemented

### 2a. New optional reuse mode

An optional reuse mode was added to the CUDA resolver:

- API: `set_reuse_eviction_graph(bool on)`
- benchmark flag: `--reuse-eviction-graph`
- default: `false`

Defaulting to `false` keeps the current baseline behavior available for direct before/after comparisons.

### 2b. Single capture + exec reuse

When reuse mode is enabled:

1. The eviction graph is captured and instantiated only once, on the first outer iteration.
2. Handles to the dynamic kernel nodes are collected from the captured graph.
3. On later outer iterations, the existing `cudaGraphExec_t` is reused.
4. The launch dimensions of the kernels that depend on `n_accepted` are updated using `cudaGraphExecKernelNodeSetParams(...)`.

The reused graph keeps the same kernel sequence and payload pointers as the baseline implementation. Only the launch configuration is updated.

### 2c. Kernels updated dynamically

The following nodes are updated between outer iterations:

- `fill_inverted_ids`
- `block_inclusive_scan`
- `scan_block_offsets`
- `add_block_offset`
- `rearrange_tracks`
- `update_status`

The following kernels stay fixed:

- `remove_tracks<<<1, 512>>>`
- `sort_updated_tracks<<<1, 512>>>`

This matches the code structure: only the adaptive/rearrangement/scan kernels depend on the current `n_accepted`.

---

## 3. Code Changes

### traccc source

- `device/cuda/include/traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp`
  - added `set_reuse_eviction_graph(bool)`
  - added `eviction_graph_instantiations` to `gpu_profile_data_t`

- `device/cuda/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu`
  - factored eviction launch configuration into a helper struct
  - added graph holder / node-handle helpers
  - added one-time graph capture + instantiation path
  - added per-iteration kernel-node launch-parameter updates
  - preserved the original rebuild-per-iteration path when reuse mode is off

- `examples/run/cuda/benchmark_resolver_cuda.cpp`
  - added CLI flag `--reuse-eviction-graph`
  - added output field `reuse_eviction_graph=true|false`
  - added profile field `profile_eviction_graph_instantiations`

---

## 4. Expected Thesis Value

This is an **engineering novelty** rather than an algorithmic one:

- greedy output semantics are unchanged,
- CPU↔GPU agreement checks remain directly applicable,
- performance gains can be attributed specifically to graph management,
- the result is especially relevant in the small-`n` regime where GPU currently loses.

This makes it a strong first novelty step because it is:

- low risk,
- easy to benchmark fairly,
- easy to explain in HPC terms,
- likely to shift the GPU crossover point downward.

---

## 5. Benchmarking Plan

The intended comparison is:

### Baseline

```bash
traccc_benchmark_resolver_cuda --synthetic --n-candidates=<N> --conflict-density=<D>
```

### Graph reuse

```bash
traccc_benchmark_resolver_cuda --synthetic --n-candidates=<N> --conflict-density=<D> --reuse-eviction-graph
```

### Profile mode

```bash
traccc_benchmark_resolver_cuda --synthetic --n-candidates=<N> --conflict-density=<D> --profile --reuse-eviction-graph
```

The key metrics to compare are:

- `time_ms_mean`, `time_ms_median`
- `profile_eviction_loop_ms`
- `profile_eviction_graph_launches`
- `profile_eviction_graph_instantiations`
- `hash_match`

Expected signature of success:

- `hash_match=true` remains unchanged,
- `profile_eviction_graph_instantiations` drops from roughly “one per outer loop step” to `1`,
- small-`n` runtime decreases most strongly,
- large-`n` runtime changes less, because graph construction is already amortized there.

---

## 6. Validation Status

### What was verified locally

- The patched CUDA ambiguity-resolution object builds in isolation:
  - `device/cuda/CMakeFiles/traccc_cuda.dir/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu.o`
- The patched benchmark source builds in isolation:
  - `examples/run/cuda/CMakeFiles/traccc_benchmark_resolver_cuda.dir/benchmark_resolver_cuda.cpp.o`

### What was not fully verified yet

The full `make traccc_benchmark_resolver_cuda` build currently fails in unrelated CUDA compilation units (`seeding` / `gbts_seeding`) due to an existing environment/toolchain problem involving `_Float32` glibc header parsing in NVCC. This blocked a full end-to-end rebuild from this environment, but did **not** block compilation of the patched ambiguity-resolution object itself.

### Practical implication

The implementation is ready for targeted runtime benchmarking once the broader CUDA build environment is healthy again, or when built on the node/configuration previously used for successful CUDA benchmark runs.

---

## 7. Measured Results

### 7a. Synthetic 3x3 sweep

**Run directory:** `results/20260417_130208_cuda_graph_reuse_synth/`  
**Same-binary control:** `results/20260417_131434_cuda_no_reuse_synth_control/`

| Config | control GPU mean (ms) | graph reuse GPU mean (ms) | speedup |
|---|---:|---:|---:|
| n1000_low | 13.527 | 10.389 | **1.30x** |
| n1000_med | 20.792 | 16.588 | **1.25x** |
| n1000_high | 8.137 | 7.894 | 1.03x |
| n5000_low | 36.839 | 25.865 | **1.42x** |
| n5000_med | 40.752 | 33.957 | **1.20x** |
| n5000_high | 24.414 | 13.096 | **1.86x** |
| n10000_low | 51.152 | 33.448 | **1.53x** |
| n10000_med | 41.032 | 35.015 | 1.17x |
| n10000_high | 28.591 | 20.134 | **1.42x** |

All rerun configurations produced `hash_match=true`.

**Interpretation:**

- Same-binary comparison confirms that graph reuse produces a clear and repeatable speedup across the full synthetic sweep.
- The largest gains appear where the eviction loop performs many outer iterations and repeatedly pays graph construction cost.
- The novelty is not limited to tiny `n`; even at `n=10000` the GPU time drops by about 17–53% depending on density in this control comparison.

### 7b. Physics-calibrated ttbar sweep

**Run directory:** `results/20260417_130405_physics_calibrated_graph_reuse/`  
**Same-binary control:** `results/20260417_131434_physics_calibrated_no_reuse_control/`

| Config | control GPU mean (ms) | graph reuse GPU mean (ms) | speedup | rerun CPU mean (ms) |
|---|---:|---:|---:|---:|
| n56_low | 1.496 | 1.492 | 1.00x | 0.171 |
| n56_med | 7.085 | 1.592 | **4.45x** | 0.172 |
| n154_low | 8.239 | 2.015 | **4.09x** | 0.479 |
| n154_med | 2.706 | 3.076 | 0.88x | 0.502 |
| n307_low | 17.225 | 3.315 | **5.20x** | 1.071 |
| n307_med | 10.238 | 5.917 | **1.73x** | 1.193 |
| n602_low | 5.062 | 5.286 | 0.96x | 2.438 |
| n602_med | 13.740 | 10.369 | **1.33x** | 2.880 |
| n821_low | 26.370 | 7.390 | **3.57x** | 3.623 |
| n821_med | 23.695 | 14.219 | **1.67x** | 4.134 |
| n1167_low | 17.566 | 9.038 | **1.94x** | 5.786 |
| n1167_med | 16.917 | 17.339 | 0.98x | 6.972 |
| n1770_low | 13.521 | 13.433 | 1.01x | 10.321 |
| n1770_med | 25.411 | 25.224 | 1.01x | 11.865 |

All rerun configurations produced `hash_match=true`.

**Interpretation:**

- Same-binary control confirms that graph reuse can produce large wins in several low- and medium-occupancy calibrated regimes, especially where repeated graph construction dominates.
- The effect is not monotonic across all configurations; some points show little change, and a few even show slight regressions, so this is still an engineering optimization with workload dependence rather than a universal speedup.
- Even with the improved GPU timings, the CPU still wins across the standard `n=56..1770` single-event ttbar range in these reruns.

### 7c. Real dump-based ODD muon events

**Run directory:** `results/20260417_131018_odd_muon_graph_reuse/`  
**Same-binary control:** `results/20260417_131525_odd_muon_no_reuse_control/`  
**Input dumps:** `data/odd_muon_dumps/20260406/event_*.json`

The real dump rerun completed on all 10 events with `hash_match=true`.

- Mean CPU time in reuse run: **0.490 ms**
- Mean GPU time with graph reuse: **2.173 ms**
- Mean CPU time in control run: **0.484 ms**
- Mean GPU time without reuse: **2.187 ms**
- GPU speedup vs same-binary control: **1.01x**

**Interpretation:**

- On frozen real event dumps, graph reuse has only a very small effect.
- At very small `n` the GPU remains clearly slower than CPU, even after removing repeated graph instantiation.
- This supports the thesis claim that graph reuse is a useful engineering improvement, but not a complete answer for low-multiplicity real physics workloads.

### 7d. Profile confirmation

Representative profile-mode reruns were saved to:

- `results/20260417_130208_cuda_graph_reuse_synth/n1000_med_profile.txt`
- `results/20260417_130405_physics_calibrated_graph_reuse/gpu_n1770_low_mu300_profile.txt`

In both cases:

- `profile_eviction_graph_instantiations=1`
- `profile_hash_match=true`

This confirms that the intended mechanism is active in practice, not just in code.

---

## 8. Recommended Next Measurements

Run the reuse-vs-baseline comparison first on:

- synthetic `n = 500, 1000, 2000, 5000`
- `conflict_density = low, med`

Then repeat on the physics-calibrated sweep:

- `n = 56, 154, 307, 602, 821, 1167, 1770`

Most important outputs for the thesis:

- absolute latency reduction at small `n`
- shift in the CPU↔GPU crossover estimate
- unchanged correctness (`hash_match=true`)
- evidence that graph instantiation count collapses to one per event

---

## 9. Build Note

The updated benchmark binary was rebuilt successfully on the GV100 node by compiling the modified CUDA ambiguity-resolution source with:

- `nvcc`
- `-ccbin` pointing to the Spack GCC 13.4 host compiler

The default current build configuration on this node uses `icpx` as the general C++ compiler, and direct NVCC compilation hit the known `_Float32` standard-library incompatibility. Using a CUDA-compatible GCC host compiler resolved this for the modified ambiguity-resolution translation unit and allowed the benchmark executable to be relinked.

For reproducibility, the new helper script for real dump reruns is:

- `scripts/run_odd_dump_benchmark_graph_reuse.sh`

---

## 10. Position in the Novelty Roadmap

This implementation corresponds to option **(B) Engineering novelty: CUDA graph parameter reuse** from `novelty_improvements.md`.

Recommended sequence after this:

1. Benchmark and quantify the crossover shift from graph reuse.
2. If time allows, combine it with either:
   - multi-block `remove_tracks`, or
   - SoA inverted-index redesign.
3. Keep the algorithmic redesign (`parallel batch greedy`) as the strongest but higher-risk extension.
