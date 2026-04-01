# GPU Benchmark Documentation — Greedy Ambiguity Resolver (CUDA)

**Scope:** GPU-side benchmarking of the CUDA greedy ambiguity resolution algorithm in traccc.  
**Thesis RQs addressed:**
- RQ1 — *Which sub-steps dominate runtime and how do costs scale with n_candidates and conflict density?*
- RQ4 — *Under which workload regimes does GPU ambiguity resolution outperform the CPU baseline?*

---

## What this benchmark measures

`traccc_benchmark_resolver_cuda` runs **only** the CUDA greedy ambiguity resolver in isolation.  
It does **not** run seeding, CKF, or track fitting.

The benchmark has three distinct timed regions:

| Region | What is timed | Metric key |
|--------|--------------|------------|
| H2D transfer | Host→device copy of measurements + tracks (once, before warmup) | `time_h2d_ms` |
| GPU resolver | `traccc::cuda::greedy_ambiguity_resolution_algorithm::operator()` + `stream.synchronize()`, repeated `--repeats` times | `time_ms_*` |
| D2H transfer | Device→host copy of result tracks (once, after timing loop) | `time_d2h_ms` |

The **main latency metric** (`time_ms_mean`, `events_per_sec`) covers only the GPU resolver call, excluding transfers. This is the correct scope for comparing GPU resolver performance against the CPU resolver. Transfer times are reported separately so end-to-end pipeline cost can also be computed.

A **CPU reference** resolver call is always made before timing. Its `output_hash` is compared to the GPU output hash — `hash_match=true|false` is printed every run.

---

## Binary location

```
$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda
```

Source: `traccc/examples/run/cuda/benchmark_resolver_cuda.cpp`

---

## Build requirements

The CUDA benchmark requires a CUDA-enabled traccc build. The CPU-only build on stbc-i2 does **not** produce this binary.

### First-time CUDA build (on GPU node)

```bash
# 1. SSH to the interactive GPU node
ssh gpu-int-mi50-gv100-lot001

# 2. Activate Spack environment
. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

# 3. Reconfigure with CUDA enabled
cd /data/alice/sbetisor/traccc/build
cmake -DTRACCC_BUILD_CUDA=ON -DTRACCC_BUILD_CUDA_UTILS=ON ..

# 4. Build only the GPU benchmark (fast; skips all other targets)
make traccc_benchmark_resolver_cuda -j$(nproc)
```

### Rebuild after source changes

```bash
cd /data/alice/sbetisor/traccc/build
make traccc_benchmark_resolver_cuda
```

---

## Input modes

Identical to the CPU benchmark. The same fixed seed (42) and density parameters are used, so CPU and GPU synthetic inputs are bit-for-bit identical.

### Synthetic mode (`--synthetic`)

| Parameter | Flag | Effect |
|-----------|------|--------|
| n_candidates | `--n-candidates=N` | Number of input track candidates |
| conflict_density | `--conflict-density=low\|med\|high` | Probability of hit sharing between candidates |

| Density | max_meas_id | Track length | Typical hit sharing |
|---------|-------------|--------------|---------------------|
| low     | 50 000      | 3–10 hits    | Large unique measurement universe; few conflicts |
| med     | 10 000      | 3–10 hits    | Moderate sharing |
| high    | 500         | 5–15 hits    | Dense sharing; most candidates conflict |

### Dump/replay mode (`--input-dump=<path>`)

Loads a pre-frozen JSON snapshot created by `traccc_seq_example --dump-ambiguity-input=<path>`.  
Identical to the CPU benchmark's dump mode — same input, same config, enabling a direct CPU vs GPU comparison.

---

## Standard metrics

Every run outputs the following (same key names as the CPU benchmark where applicable):

```
backend=gpu
n_candidates=<N>  n_selected=<K>  n_removed=<N-K>
time_ms_mean=<x>  time_ms_std=<x>  time_ms_median=<x>  time_ms_p95=<x>
events_per_sec=<x>
peak_memory_mb=<x>
output_hash=<hash>
time_h2d_ms=<x>
time_d2h_ms=<x>
cpu_hash=<hash>
gpu_hash=<hash>
hash_match=true|false
```

| Metric | Definition |
|--------|------------|
| `time_ms_*` | Wall-clock time of a single GPU resolver call (resolver + `stream.synchronize()`). Averaged over `--repeats` runs after `--warmup` warm-up calls. Excludes H2D/D2H transfers. |
| `events_per_sec` | `1000 / time_ms_mean`. GPU resolver throughput, excluding transfers. |
| `peak_memory_mb` | Peak host RSS (`getrusage`). Captures host-side allocation overhead. Does not reflect device memory usage (use `nvidia-smi` or Nsight for that). |
| `output_hash` | Hash of the sorted set of selected track measurement patterns. Equal to `gpu_hash`. |
| `time_h2d_ms` | Single-pass host→device transfer time for measurements + tracks (before warmup). |
| `time_d2h_ms` | Single-pass device→host transfer time for result tracks (after timing loop). |
| `cpu_hash` | Hash produced by the CPU reference resolver on the same input. |
| `gpu_hash` | Hash produced by the GPU resolver. Must equal `cpu_hash` for correctness. |
| `hash_match` | `true` if `gpu_hash == cpu_hash`. A `false` result means the GPU and CPU resolvers disagree on which tracks to select — a correctness failure. |

---

## Per-phase profiling

### `--profile` flag (CUDA event timers)

Pass `--profile` to run one extra resolver call with CUDA event timing inserted at each phase boundary. Output fields mirror the CPU benchmark exactly:

```
profile_filter_setup_ms=<x>
profile_unique_meas_ms=<x>
profile_inverted_index_ms=<x>
profile_shared_count_ms=<x>
profile_initial_sort_ms=<x>
profile_eviction_loop_ms=<x>
profile_output_copy_ms=<x>
profile_eviction_graph_launches=<N>
profile_unique_meas_count=<N>
profile_hash_match=true|false
```

Times are measured with `cudaEventRecord` / `cudaEventElapsedTime` at the existing stream-synchronization checkpoints in `greedy_ambiguity_resolution_algorithm.cu`. They capture **GPU kernel execution time only** — host-side work (e.g. the D2H copy needed to size the `tracks_per_measurement` buffer) that occurs between phase boundaries is not included. This is the correct scope for GPU compute analysis.

The CUDA Graph inside the eviction loop cannot be profiled at the individual-kernel level from C++ code; `profile_eviction_loop_ms` covers the entire graph-execution loop. For kernel-level breakdown inside the graph use nsys or ncu (see below).

**Phase mapping to CPU benchmark:**

| `profile_*_ms` field | GPU kernels included | CPU equivalent |
|----------------------|----------------------|----------------|
| `filter_setup_ms`    | `fill_vectors`       | `filter_setup` |
| `unique_meas_ms`     | Thrust sort/unique/reduce, `fill_unique_meas_id_map` | `unique_meas` |
| `inverted_index_ms`  | `fill_tracks_per_measurement`, `sort_tracks_per_measurement` | `inverted_index` |
| `shared_count_ms`    | `count_shared_measurements`     | `shared_count` |
| `initial_sort_ms`    | Thrust `transform` (rel_shared), `thrust::copy`, `thrust::sort` | `initial_sort` |
| `eviction_loop_ms`   | CUDA Graph loop (all eviction kernels × `eviction_graph_launches` launches) | `eviction_loop` |
| `output_copy_ms`     | `fill_track_candidates` | `output_copy` |

### NVTX range markers (nsys timeline)

NVTX ranges named `gpu_ar_filter_setup`, `gpu_ar_unique_meas`, `gpu_ar_inverted_index`, `gpu_ar_shared_count`, `gpu_ar_initial_sort`, `gpu_ar_eviction_loop`, `gpu_ar_output_copy` are inserted unconditionally in the algorithm. When the binary is run under `nsys profile --trace=cuda,nvtx`, these appear as labeled CPU-side ranges in the timeline, aligned with the corresponding GPU kernels.

### Nsight Systems + Nsight Compute profiling script

`scripts/run_resolver_profile_nsys.sh` automates both tools:

```bash
cd "$THESIS_REPO/scripts"

# Full profiling (nsys + ncu) for n=10000 med density:
./run_resolver_profile_nsys.sh --n-candidates=10000 --conflict-density=med

# nsys only (faster, no per-kernel kernel-metric overhead):
./run_resolver_profile_nsys.sh --nsys-only

# ncu only:
./run_resolver_profile_nsys.sh --ncu-only
```

Outputs are saved to `results/<timestamp>_profile_nsys/`:
- `n<N>_<density>.nsys-rep` — Nsight Systems timeline report
- `n<N>_<density>_ncu.ncu-rep` — Nsight Compute kernel metrics
- `n<N>_<density>_profile.txt` — `--profile` text output (same fields as above)

```bash
# Open Nsight Systems GUI
nsys-ui results/<timestamp>_profile_nsys/n10000_med.nsys-rep

# Print GPU trace summary as text
nsys stats --report gputrace results/<timestamp>_profile_nsys/n10000_med.nsys-rep

# Print ncu kernel summary
ncu --import results/<timestamp>_profile_nsys/n10000_med_ncu.ncu-rep --print-summary
```

---

## How the GPU algorithm works

The CUDA resolver (`traccc::cuda::greedy_ambiguity_resolution_algorithm`) implements the same greedy semantics as the CPU version but replaces sequential loops with parallel kernels. It is structured as two phases:

### Phase 1: Preprocessing (one-shot parallel kernels)

| Step | Kernel / operation | CPU equivalent |
|------|--------------------|----------------|
| Fill vectors | `fill_vectors` | `filter_setup` phase |
| Unique measurement count | Thrust `sort` + `unique_count` + `reduce_by_key` | `unique_meas` phase |
| Measurement ID map | `fill_unique_meas_id_map` | Building the binary-search lookup |
| Inverted index | `fill_tracks_per_measurement` | `inverted_index` phase |
| Sort buckets | `sort_tracks_per_measurement` | Part of `inverted_index` |
| Shared count | `count_shared_measurements` | `shared_count` phase |
| Relative shared | Thrust `transform` (divide) | `shared_count` phase |
| Initial sort | Thrust `sort` by `(rel_shared, pval)` | `initial_sort` phase |

### Phase 2: Eviction loop (CUDA Graph)

The eviction loop is compiled into a **CUDA Graph** and launched repeatedly. Each graph execution runs 100 iterations of:

| Kernel | What it does |
|--------|-------------|
| `remove_tracks` | Removes worst track(s); updates `tracks_per_measurement`, `n_shared`, `rel_shared`; marks updated tracks |
| `sort_updated_tracks` | Bitonic sort of the small set of updated track IDs |
| `fill_inverted_ids` | Builds a track_id → position-in-sorted_ids lookup |
| `block_inclusive_scan` | Block-wise prefix sum over `sorted_ids` (for insertion sort) |
| `scan_block_offsets` | Prefix sum of block offsets |
| `add_block_offset` | Completes the global prefix sum |
| `rearrange_tracks` | Insertion sort: moves updated tracks to their correct positions |
| `update_status` | Checks termination condition; updates `max_shared` and `n_accepted` |

The CUDA Graph avoids per-kernel launch overhead. The graph is re-instantiated when `n_accepted` drops enough to change the grid dimensions.

**Key difference from CPU:** the CPU eviction loop is strictly sequential (one track removed per iteration, O(n²) `std::find` reposition). The GPU loop uses a parallel insertion sort scheme — multiple tracks can be repositioned in parallel per iteration, reducing the effective iteration count.

---

## How to run

### Single GPU run

```bash
TRACCC_BIN=/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda

"$TRACCC_BIN" --synthetic --n-candidates=10000 \
  --conflict-density=low --repeats=10 --warmup=3
```

### Standard 3×3 sweep

```bash
cd "$THESIS_REPO/scripts"
./run_resolver_benchmark_sweep_cuda.sh
# outputs to results/<timestamp>_cuda/n{N}_{density}.txt
```

Override output directory:
```bash
RUN_ID=my_gpu_run_v1 ./run_resolver_benchmark_sweep_cuda.sh
```

### Dump/replay mode (real physics input)

```bash
# First generate the dump on a CPU node (or GPU node):
./build/bin/traccc_seq_example \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=1 \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --dump-ambiguity-input=ambiguity_input_event0.json

# Then benchmark on the GPU node:
"$TRACCC_BIN" --input-dump=ambiguity_input_event0.json \
  --repeats=10 --warmup=3
```

---

## CPU vs GPU comparison workflow

Since both benchmarks use the same input (same synthetic seed or same dump file) and report `output_hash`, CPU and GPU results can be compared directly:

```bash
# CPU result (from existing sweep)
grep "time_ms_median\|events_per_sec\|output_hash" \
  results/20260325_profile/n10000_low.txt

# GPU result (after running the sweep on a GPU node)
grep "time_ms_median\|events_per_sec\|output_hash\|hash_match\|time_h2d_ms" \
  results/<cuda_run_id>/n10000_low.txt
```

For a thesis-quality comparison, report:
- `time_ms_median` CPU vs GPU (resolver only)
- `time_ms_median + time_h2d_ms + time_d2h_ms` for end-to-end pipeline cost
- `hash_match=true` in all configurations (correctness evidence)

---

## Existing sweep results

No GPU sweep results yet — requires a CUDA build on a GPU node.

Planned reference configuration: Stoomboot `wn-lot-001` (NVIDIA Quadro GV100), same 3×3 synthetic sweep as the CPU baseline.

---

## Limitations of the current GPU benchmark

- **`--profile` measures phase boundaries, not individual kernels inside the CUDA Graph.** `profile_eviction_loop_ms` is a single bulk time. Use nsys/ncu for kernel-level breakdown within the loop.
- **No device memory reporting.** `peak_memory_mb` is host RSS only. Device peak memory usage must be measured externally (`nvidia-smi dmon` or `cudaMemGetInfo` instrumentation).
- **H2D/D2H timed as single passes.** For variance of transfer times, run with a larger `--repeats` and time the full transfer loop separately if needed.
- **CUDA Graph re-instantiation cost** is included in the first outer-loop iteration but not measured in isolation. At large `n_candidates`, this overhead is negligible relative to the total resolver time.
- **Single GPU stream.** No multi-stream or multi-GPU parallelism. One event per call.
- **CPU reference correctness check uses one pass** (1 warmup + 1 timed CPU call). It is not itself a determinism test; run the GPU benchmark multiple times and compare `gpu_hash` values across runs for a determinism check.
