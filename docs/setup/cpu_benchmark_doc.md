# CPU Benchmark Documentation — Greedy Ambiguity Resolver

**Scope:** CPU-side benchmarking of the greedy ambiguity resolution algorithm in traccc.  
**Thesis RQ addressed:** RQ1 — *Which sub-steps of ambiguity resolution dominate runtime and memory consumption, and how do these costs scale with n_candidates and conflict density?*

---

## What this benchmark measures

`traccc_benchmark_resolver` runs **only** the greedy ambiguity resolver in isolation.  
It does **not** run seeding, CKF, or track fitting.  
The timed region is a single call to `traccc::host::greedy_ambiguity_resolution_algorithm::operator()`.

This isolation is intentional: measuring the full reconstruction chain would obscure the resolver's contribution by mixing it with CKF/fitting variance and I/O costs.

---

## Binary location

```
$TRACCC_SRC/build/bin/traccc_benchmark_resolver
```

Source: `traccc/examples/run/cpu/benchmark_resolver.cpp`

---

## Input modes

### Synthetic mode (`--synthetic`)

Generates random track candidates in-process.  
Two parameters control the workload:

| Parameter | Flag | Effect |
|-----------|------|--------|
| n_candidates | `--n-candidates=N` | Number of input track candidates per event |
| conflict_density | `--conflict-density=low\|med\|high` | Probability of hit sharing between candidates |

Conflict density is implemented via the measurement ID space:

| Density | max_meas_id | Track length | Typical hit sharing |
|---------|-------------|--------------|---------------------|
| low     | 50 000      | 3–10 hits    | Few shared hits, large unique measurement universe |
| med     | 10 000      | 3–10 hits    | Moderate sharing |
| high    | 500         | 5–15 hits    | Dense sharing; most candidates conflict |

The random generator uses a fixed seed (42), so the synthetic workload is **fully reproducible**.

### Dump/replay mode (`--input-dump=<path>`)

Loads a pre-frozen JSON snapshot of the "just before resolution" state from a real traccc run.  
The snapshot is created by running `traccc_seq_example --dump-ambiguity-input=<path>`.  
The timed region and algorithm are identical to synthetic mode; only the input differs.

---

## Standard metrics

Every run outputs:

```
n_candidates=<N>  n_selected=<K>  n_removed=<N-K>
time_ms_mean=<x>  time_ms_std=<x>  time_ms_median=<x>  time_ms_p95=<x>
events_per_sec=<x>
peak_memory_mb=<x>
output_hash=<hash>
```

| Metric | Definition |
|--------|------------|
| `time_ms_*` | Wall-clock time of a single resolver call. Averaged over `--repeats` runs after `--warmup` warm-up calls. |
| `events_per_sec` | `1000 / time_ms_mean`. Inverse throughput. |
| `peak_memory_mb` | Peak RSS of the process (`getrusage` on Linux). Includes all allocations up to that point, not just resolver data structures. |
| `output_hash` | Hash of the sorted set of selected track measurement patterns. Used for determinism checks: same input across repeats must produce the same hash. |
| `n_selected` | Tracks kept after resolution. |
| `n_removed` | Tracks rejected (duplicates / conflicts). |

---

## Per-phase profiling (`--profile`)

When `--profile` is passed, a single additional resolver call is made (after 1 warm-up) with `std::chrono::high_resolution_clock` checkpoints between each internal phase.  
The profiling path reimplements the same algorithm logic and outputs an `output_hash` that is compared to the reference hash for correctness verification.

### Phase breakdown

```
profile_filter_setup_ms=<x>
profile_unique_meas_ms=<x>
profile_inverted_index_ms=<x>
profile_shared_count_ms=<x>
profile_initial_sort_ms=<x>
profile_eviction_loop_ms=<x>
profile_output_copy_ms=<x>
profile_eviction_iterations=<N>
profile_eviction_shared_updates=<N>
profile_unique_meas_count=<N>
profile_hash_match=true|false
```

| Phase | What it does | Data structures touched |
|-------|-------------|------------------------|
| `filter_setup` | Allocate `meas_ids`, `pvals`, `n_meas`; filter tracks below `min_meas_per_track` | `accepted_ids`, `meas_ids[i]` |
| `unique_meas` | Collect all unique measurement IDs from accepted tracks into a sorted vector | `unordered_set` → `vector<size_t>` + `std::sort` |
| `inverted_index` | Build `tracks_per_measurement[uidx]`: for each accepted track, map its measurements to sorted track lists | `lower_bound` into `unique_meas` (O(log U) per lookup), U = unique_meas_count |
| `shared_count` | For every accepted track, count how many of its measurements appear in more than one track; compute `rel_shared` | Same `lower_bound` pattern; O(N × L × log U) total |
| `initial_sort` | Sort `sorted_ids` descending by worst-first order `(rel_shared, -pval)` | `std::sort` on `accepted_ids.size()` elements |
| `eviction_loop` | Iteratively remove the worst track; update `tracks_per_measurement`, `n_shared`, `rel_shared`, and reorder `sorted_ids` | See sub-operations below |
| `output_copy` | Copy accepted tracks into the host output container | Linear scan of `accepted_ids` |

### Hot sub-operations inside the eviction loop

These are **not** individually timed but are identifiable bottleneck candidates for GPU work or algorithmic improvement:

| Sub-operation | Complexity per iteration | Notes |
|---------------|-------------------------|-------|
| `max_shared` scan | O(n_accepted) | Scans all remaining tracks each iteration |
| Binary search in `unique_meas` | O(log U) per measurement of removed track | U = unique measurement universe size |
| Erase from `tracks_per_measurement[uidx]` | O(|bucket|) | Sorted bucket; binary search + vector erase |
| `std::find` to reposition updated track in `sorted_ids` | **O(n_accepted)** | Linear scan — likely dominant as n_accepted stays large |
| `std::count` over `meas_ids[tid]` | O(L) per updated track, L = track length | Called once per bucket that drops to size 1 |

The `std::find` + `erase` + `insert` pattern for repositioning updated tracks in `sorted_ids` is O(n_accepted) and is expected to be the main per-iteration bottleneck at large `n_candidates`.

### `profile_eviction_iterations`

The actual number of times the "remove worst track" body executed.  This is typically close to `n_removed` but can differ if multiple tracks become conflict-free within a single iteration.

### `profile_eviction_shared_updates`

The number of times a track's `n_shared` was decremented (i.e., a measurement dropped from "shared by 2+ tracks" to "shared by exactly 1 track").  Higher conflict density typically produces more updates.

### `profile_unique_meas_count`

The size of the unique measurement universe.  For synthetic mode this correlates inversely with conflict density (low → ~50k, high → ~500).  This directly determines the size of `tracks_per_measurement` and the cost of all binary-search lookups.

---

## Existing sweep results (`results/20260224_022817/`)

3×3 CPU sweep run on Stoomboot (interactive node), CPU-only, no Condor:

| n_candidates | density | median (ms) | events/s | n_selected | peak RSS (MB) |
|--------------|---------|-------------|----------|------------|---------------|
| 1 000 | low  | 3.47 | 288 | 744  | 16.25 |
| 1 000 | med  | 3.94 | 254 | 413  | 13.75 |
| 1 000 | high | 3.38 | 297 | 19   | 12.75 |
| 5 000 | low  | 35.3 | 28.3 | 2 070 | 20.44 |
| 5 000 | med  | 31.2 | 32.0 | 623  | 16.15 |
| 5 000 | high | 22.1 | 45.5 | 17   | 14.98 |
| 10 000 | low | 115.8 | 8.6  | 2 683 | 23.98 |
| 10 000 | med | 70.0  | 14.4 | 666  | 19.64 |
| 10 000 | high | 59.8 | 16.7 | 20   | 18.78 |

**Observations from existing data (without phase timing):**

- Latency scales super-linearly with `n_candidates` (1k→10k is 10×, latency is 33× for low density).  
  This is consistent with the O(n_accepted²) worst-case for the `std::find` reposition step.
- At fixed `n_candidates`, lower conflict density → higher latency, because more tracks survive (larger `n_accepted` throughout the eviction loop → more expensive `max_shared` scans and `std::find` calls).
- Memory scales mildly with `n_candidates`; the dominant factor for RSS is the measurement universe size (`unique_meas_count`), which is largest for low density.

---

## How to run

### Standard sweep (reproduces existing results)

```bash
cd "$THESIS_REPO/scripts"
./run_resolver_benchmark_sweep.sh
# outputs to results/<timestamp>/n{N}_{density}.txt
```

### Sweep with per-phase profiling (new)

```bash
cd "$THESIS_REPO/scripts"
PROFILE=1 ./run_resolver_benchmark_sweep.sh
```

Each result file will contain the standard metrics followed by `profile_*` lines.

### Single profiling run

```bash
"$TRACCC_BIN" --synthetic --n-candidates=10000 \
  --conflict-density=low --repeats=10 --warmup=3 --profile
```

---

## How to rebuild after changes

```bash
cd "$TRACCC_SRC/build"
make traccc_benchmark_resolver
```

---

## Interpreting phase data for thesis RQ1

When the profiling sweep is available, look for:

1. **`profile_eviction_loop_ms` / total time**: If this ratio is >80%, the eviction loop dominates. Expected to be true at n_candidates ≥ 5000.
2. **`profile_inverted_index_ms` + `profile_shared_count_ms`**: These two phases together represent the preprocessing cost and scale with U (unique_meas_count). Expected to be non-negligible at low conflict density where U is large.
3. **`profile_eviction_iterations` vs `n_removed`**: A ratio close to 1 means one track removed per iteration (worst case). A lower ratio would indicate batch-like convergence.
4. **`profile_eviction_shared_updates` vs `profile_eviction_iterations`**: High ratio → many `std::find` calls per iteration → the reposition step is expensive.
5. **`profile_unique_meas_count` and memory**: Cross-reference with `peak_memory_mb` to see if `tracks_per_measurement` is a significant contributor (particularly at low density where U ≈ 50k).

---

## Limitations of the current CPU benchmark

- `--profile` is a **single-pass** measurement (1 warm-up + 1 timed pass). Phase times at sub-millisecond granularity (e.g., `initial_sort`, `output_copy`) will have relatively high noise.
- Peak RSS is measured **at the end** of the full benchmark run, not per phase. It captures the high-water mark across all warmup + timed + profiling calls.
- The eviction loop is timed as a **single monolithic phase**. Sub-operation costs (e.g., `std::find` vs `lower_bound` vs erase) require a separate instrumented build or a profiler (e.g., `perf`, `valgrind --tool=callgrind`).
- These are CPU-only results; GPU comparison requires enabling `--backend=gpu` (planned, see `docs/next_steps.md`).
