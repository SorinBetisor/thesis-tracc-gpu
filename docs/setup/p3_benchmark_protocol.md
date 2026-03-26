# P3 – Benchmark protocol for ambiguity resolution

**Goal:** Reproducible CPU vs GPU comparison of the ambiguity resolver with standardised metrics.

---

## What was implemented

### 1. Dump/load (traccc io)

- `write_ambiguity_input(path, track_container, config)` – writes track candidates + measurements to JSON
- `read_ambiguity_input(path, mr)` – loads them back into a `track_container`
- Used to freeze the state immediately before ambiguity resolution from a real traccc run

### 2. Dump option in traccc_seq_example

- `--dump-ambiguity-input=<path>` – dumps track candidates to JSON before running the resolver
- Lets you capture real physics-like input for later reproducible CPU vs GPU comparison

### 3. Resolver-only benchmarks

Two separate executables, one CPU and one GPU:

- `traccc_benchmark_resolver` (CPU): runs only the host greedy ambiguity resolver
- `traccc_benchmark_resolver_cuda` (GPU): runs only the CUDA greedy ambiguity resolver

Both support two input modes:
- `--input-dump=<path>`: load from JSON (frozen real data)
- `--synthetic`: generate random track candidates (for parameter sweeps)

Both report: latency (mean/std/median/p95), throughput, peak memory, output hash, n\_selected, n\_removed.

The CUDA binary additionally reports: `time_h2d_ms`, `time_d2h_ms`, `cpu_hash`, `gpu_hash`, `hash_match`.

### 4. Sweep scripts

- `run_resolver_benchmark_sweep.sh` – CPU 3×3 grid sweep
- `run_resolver_benchmark_sweep_cuda.sh` – GPU 3×3 grid sweep

Both sweep `n_candidates` × `conflict_density` and write one output file per configuration.

---

## Synthetic data generation

When `--synthetic` is used, the benchmark generates track candidates from scratch using a fixed random seed (seed=42). This means results are **fully reproducible** across runs and machines: the same seed always produces the same input.

### What is generated

For each of the `n_candidates` tracks:
- A **p-value** is drawn uniformly from [0, 1] — this represents the track quality score used for tie-breaking in the greedy resolver.
- A **track length** is drawn uniformly from `[min_length, max_length]` (density-dependent, see table below).
- That many **measurement IDs** are drawn uniformly from [0, `max_meas_id`] without replacement.

A separate **measurement collection** of size `max_meas_id + 1` is created, with each entry holding a sequential identifier (0 to `max_meas_id`). This satisfies the GPU resolver's internal requirement that measurement IDs form a contiguous range [0, n_meas-1].

### Conflict density settings

The three density levels control how many measurement IDs are available and how long tracks are. Lower `max_meas_id` means more tracks compete for the same hits → more conflicts → more work for the resolver.

| Density | `max_meas_id` | Track lengths | Total unique meas (approx) | Expected conflicts |
|---------|--------------|---------------|----------------------------|--------------------|
| `low`   | 50 000       | 3–10 (avg 6.5) | ~6 500 per 1k tracks      | sparse — few tracks share hits |
| `med`   | 10 000       | 3–10 (avg 6.5) | ~6 500 per 1k tracks      | moderate sharing |
| `high`  | 500          | 5–15 (avg 10)  | ~10 000 per 1k tracks     | dense — most hits shared by many tracks |

For `high` density: 1 000 tracks × avg 10 measurements = 10 000 measurements over only 500 IDs → each ID is claimed by ~20 tracks on average. Almost every track conflicts with many others. The CPU resolver must run ~1 eviction iteration per track removed.

### What the CPU profile columns mean

The CPU benchmark with `--profile` adds per-phase timing:

| Column | Phase |
|--------|-------|
| `profile_filter_setup_ms` | Building measurement ID → unique-ID map, filling meas\_ids per track |
| `profile_unique_meas_ms` | Counting and deduplicating unique measurement IDs |
| `profile_inverted_index_ms` | Building tracks-per-measurement index |
| `profile_shared_count_ms` | Counting shared measurements per accepted track |
| `profile_initial_sort_ms` | Initial sort of tracks by (rel\_shared, -pval) |
| `profile_eviction_loop_ms` | Greedy eviction loop (dominant for med/high density) |
| `profile_output_copy_ms` | Copying surviving tracks to output buffer |
| `profile_eviction_iterations` | Number of tracks removed one-by-one |
| `profile_eviction_shared_updates` | Number of surviving tracks whose shared-measurement count changed |
| `profile_unique_meas_count` | Total distinct measurement IDs seen across all accepted tracks |

---

## CPU results – 3×3 sweep (2026-03-25)

**Run ID:** `20260325_profile`
**Machine:** Nikhef Stoomboot CPU node
**Config:** warmup=3, repeats=10, seed=42

### Summary table

| Config | n\_cand | n\_selected | n\_removed | time\_mean\_ms | events/s | peak\_mem\_MB | eviction\_iters | dominant\_phase |
|--------|---------|------------|------------|----------------|----------|---------------|-----------------|-----------------|
| n1000\_low  | 1 000  | 744   | 256   | 3.58   | 279 | 16.0  | 256   | eviction 0.60 ms |
| n1000\_med  | 1 000  | 413   | 587   | 4.23   | 236 | 13.5  | 587   | eviction 1.54 ms |
| n1000\_high | 1 000  | 19    | 981   | 3.46   | 289 | 12.5  | 981   | eviction 1.85 ms |
| n5000\_low  | 5 000  | 2 070 | 2 930 | 40.54  | 25  | 20.2  | 2 930 | eviction 24.0 ms |
| n5000\_med  | 5 000  | 623   | 4 377 | 34.47  | 29  | 16.0  | 4 377 | eviction 24.8 ms |
| n5000\_high | 5 000  | 17    | 4 983 | 22.66  | 44  | 14.97 | 4 983 | eviction 15.5 ms |
| n10000\_low | 10 000 | 2 683 | 7 317 | 45.94  | 22  | — | 7 317 | eviction (majority) |
| n10000\_med | 10 000 | 666   | 9 334 | 80.27  | 12  | 19.3  | 9 334 | eviction 60.7 ms |
| n10000\_high| 10 000 | 20    | 9 980 | 59.61  | 17  | 18.0  | 9 980 | eviction 58.5 ms |

All CPU runs: `hash_match=true` (CPU is self-consistent across repeats).

### Key observations

- **Eviction loop dominates** for all configurations with more than a handful of removals. It scales approximately O(n\_removed) since each iteration removes one or a small batch of tracks.
- **High density is faster than medium for the same n\_candidates** (e.g. n1000\_high at 3.46 ms vs n1000\_med at 4.23 ms). This is because high-density workloads have fewer unique measurements (501 vs 4 754 for n=1000), making the inverted index and sorting phases cheaper, even though the eviction loop runs more iterations.
- **Memory footprint is modest** (13–20 MB) for all CPU configurations. Not a bottleneck.
- **Throughput drops sharply with n\_candidates** (from ~280 events/s at n=1000 to ~12 events/s at n=10000), consistent with super-linear scaling of the eviction loop.

---

## GPU results – 3×3 sweep (2026-03-26)

**Run ID:** `20260326_124050_cuda`
**Machine:** Nikhef Stoomboot GPU node `wn-lot-001`
**GPU:** NVIDIA Quadro GV100, SM 70
**Config:** warmup=3, repeats=10, seed=42, built with `-DCMAKE_CUDA_ARCHITECTURES=70`

### Summary table

| Config | n\_cand | n\_sel\_gpu | n\_sel\_cpu | n\_removed | time\_mean\_ms | h2d\_ms | d2h\_ms | hash\_match |
|--------|---------|------------|------------|------------|----------------|---------|---------|-------------|
| n1000\_low  | 1 000  | 1 000 | 744   | 0     | 7.00  | 8.67  | 0.42 | **false** |
| n1000\_med  | 1 000  | 1 000 | 413   | 0     | 6.87  | 4.97  | 0.41 | **false** |
| n1000\_high | 1 000  | 1 000 | 19    | 0     | 6.49  | 4.71  | 0.46 | **false** |
| n5000\_low  | 5 000  | 5 000 | 2 070 | 0     | 11.27 | 21.00 | 1.74 | **false** |
| n5000\_med  | 5 000  | 5 000 | 623   | 0     | 10.17 | 20.69 | 1.66 | **false** |
| n5000\_high | 5 000  | 1 800 | 17    | 3 200 | 10.96 | 20.36 | 0.75 | **false** |
| n10000\_low | 10 000 | 2 683 | 2 683 | 7 317 | 45.99 | 41.88 | 0.25 | **true**  |
| n10000\_med | 10 000 | 10 000 | 666  | 0     | 10.94 | 41.46 | 3.23 | **false** |
| n10000\_high| 10 000 | 10 000 | 20   | 0     | 9.93  | 41.58 | 2.07 | **false** |

### What `hash_match` means

The benchmark runs the CPU resolver on the same input and computes `cpu_hash` from the set of selected tracks (sorted measurement-ID patterns). It also computes `gpu_hash` from the GPU output. `hash_match=true` means the GPU produced the same selected set as the CPU reference. `hash_match=false` means the outputs differ — the GPU result is **incorrect**.

### GPU correctness summary

- 7 of 9 configurations: `n_removed=0` — the GPU returned all input tracks unchanged.
- 1 configuration (`n5000_high`): partial removal but wrong (1 800 vs correct 17 selected).
- 1 configuration (`n10000_low`): **correct** — matches CPU exactly.

### Transfer vs compute breakdown (n10000\_low, the only correct case)

| Phase | Time |
|-------|------|
| H2D transfer | 41.9 ms |
| GPU kernel (resolver) | ~4 ms (total – H2D – D2H) |
| D2H transfer | 0.25 ms |
| CPU reference time | 45.94 ms |

For the one valid data point, H2D dominates. The GPU kernel itself is faster than the CPU (≈4 ms vs ≈46 ms for the eviction logic alone), but the transfer overhead erases the gain. This is expected at these data sizes and is a known characteristic of the greedy resolver's working set.

---

## Known GPU algorithm bugs

The correctness failures are caused by bugs in the upstream traccc code. See `docs/context/gpu_bug.md` for the full investigation record. Summary:

1. **`terminate_device` never initialised to 0** — on calls 2–14 in our multi-call benchmark, reused device memory has `terminate_device=1` left over from the previous call. The `remove_tracks` kernel checks this at the top of every iteration and returns immediately if it is non-zero. Nothing is removed. Result: `n_removed=0`.

2. **`scanned_block_offsets_buffer` never set up** (line 488 of the algorithm sets up the wrong buffer) — the insertion sort that maintains track order after each removal receives garbage data. This corrupts the sorted order and contributes to wrong results in the `n5000_high` partial case.

3. **`cudaMemcpyHostToDevice` with a device source pointer** for `max_shared_device` initialisation — technically undefined but likely handled correctly by UVA on this system.

4. **Missing `stream.synchronize()` in the outer while loop** — race condition that can cause premature loop exit.

The upstream traccc test file (`tests/cuda/test_ambiguity_resolution.cpp`, lines 900–909) disables the large-scale CPU↔GPU comparison tests with the comment: *"The following tests are not working for some not fully understood reason. We are blaming the ambiguity resolution algorithm for now."*

**Implication for benchmarking:** Until these bugs are fixed, the GPU benchmark cannot produce valid CPU↔GPU comparison numbers for the standard sweep configurations. Only `n10000_low` provides a valid (but transfer-dominated) data point.

---

## Metrics (protocol)

| Metric | How |
|--------|-----|
| Latency | `std::chrono`; report mean, std, median, p95 over repeats |
| Throughput | events/s = 1000 / time\_ms\_mean |
| Peak memory (CPU) | `getrusage` RSS |
| Peak memory (GPU) | `cudaMemGetInfo` before and after resolver call |
| H2D / D2H transfer time | Separately timed in GPU benchmark |
| Output hash | Hash of sorted measurement-ID patterns per selected track (order-independent) |
| `hash_match` | CPU hash == GPU hash → correct output |
| Determinism | Same input N times → same output hash across all runs |

---

## Building the benchmarks

```bash
# CPU benchmark (standard build)
cd "$TRACCC_SRC/build"
cmake -S .. -B .
make traccc_benchmark_resolver

# GPU benchmark (CUDA build on a GPU node)
. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc
cd "$TRACCC_SRC/build"
cmake -S .. -B . -DTRACCC_BUILD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
make traccc_benchmark_resolver_cuda
```

---

## Running the sweeps

### CPU sweep

```bash
cd "$THESIS_REPO/scripts"
./run_resolver_benchmark_sweep.sh
```

Output: `results/<timestamp>/n{N}_{density}.txt`

### GPU sweep

```bash
# Must be on a GPU node (wn-lot-001 or similar)
cd "$THESIS_REPO/scripts"
./run_resolver_benchmark_sweep_cuda.sh
```

Output: `results/<timestamp>_cuda/n{N}_{density}.txt` plus `gpu_info.txt`

### Single run examples

```bash
# CPU, dump mode
./build/bin/traccc_benchmark_resolver \
  --input-dump=ambiguity_input_event0.json --repeats=10 --warmup=3

# CPU, synthetic
./build/bin/traccc_benchmark_resolver \
  --synthetic --n-candidates=10000 --conflict-density=med --repeats=10 --warmup=3

# GPU, synthetic
./build/bin/traccc_benchmark_resolver_cuda \
  --synthetic --n-candidates=10000 --conflict-density=low --repeats=10 --warmup=3
```

---

## Dump step (real physics input)

```bash
cd "$TRACCC_SRC"
export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"

./build/bin/traccc_seq_example \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=1 \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --dump-ambiguity-input=ambiguity_input_event0.json
```

---

## Reproducibility checklist

- [ ] Fix commit hash (traccc + thesis repo)
- [ ] Build: Release, same compiler/flags
- [ ] Run on dedicated nodes (interactive for dev, Condor for campaigns)
- [ ] Warmup (3 runs) + N repeats (≥ 10)
- [ ] Log: CPU model, GPU model, driver, CUDA version
- [ ] Store: raw output files + summary table
- [ ] Verify `hash_match=true` for all GPU configurations before reporting performance numbers
- [ ] Apply GPU bug fixes before running official benchmark campaign
