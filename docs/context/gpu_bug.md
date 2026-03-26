# GPU Ambiguity Resolver Correctness Failures – Investigation Record

**Date of investigation:** 2026-03-26
**Investigator:** Sorin Bețișor
**Environment:** Nikhef Stoomboot, `wn-lot-001`, NVIDIA Quadro GV100, CUDA 12.5, SM 70
**Binary:** `traccc_benchmark_resolver_cuda` (built with `-DCMAKE_CUDA_ARCHITECTURES=70`)
**Relevant source file:** `traccc/device/cuda/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu`

---

## 1. Observed behaviour

A 3×3 synthetic sweep was run on the GPU node (seed=42, warmup=3, repeats=10). Results are stored in `results/20260326_124050_cuda/`.

### GPU sweep results

| Config | n\_candidates | n\_selected\_gpu | n\_selected\_cpu | n\_removed | hash\_match |
|--------|--------------|-----------------|-----------------|------------|-------------|
| n1000\_low  | 1 000  | 1 000 | 744   | 0    | **false** |
| n1000\_med  | 1 000  | 1 000 | 413   | 0    | **false** |
| n1000\_high | 1 000  | 1 000 | 19    | 0    | **false** |
| n5000\_low  | 5 000  | 5 000 | 2 070 | 0    | **false** |
| n5000\_med  | 5 000  | 5 000 | 623   | 0    | **false** |
| n5000\_high | 5 000  | 1 800 | 17    | 3 200 | **false** |
| n10000\_low | 10 000 | 2 683 | 2 683 | 7 317 | **true**  |
| n10000\_med | 10 000 | 10 000 | 666  | 0    | **false** |
| n10000\_high| 10 000 | 10 000 | 20   | 0    | **false** |

**Pattern:** 7 out of 9 configurations produce `n_removed=0`, meaning the GPU resolver removes no tracks at all and returns all input candidates. One configuration shows partial but wrong removal. Only `n10000_low` produces correct results.

---

## 2. Confirmed code bugs in the upstream traccc GPU resolver

All four bugs are in `greedy_ambiguity_resolution_algorithm.cu`.

### Bug 1 — `terminate_device` never initialised to 0 (primary cause of `n_removed=0`)

**Line 430–431:**
```cpp
vecmem::unique_alloc_ptr<int> terminate_device =
    vecmem::make_unique_alloc<int>(m_mr.main);
```

`vecmem::make_unique_alloc` wraps `cudaMalloc`. CUDA does not zero-initialise device memory on reuse — only freshly OS-allocated pages are zeroed. In a multi-call benchmark (warmup=3, repeats=10, check=1 → 14 calls total), after the first successful call the algorithm sets `terminate_device=1` on device at convergence. When subsequent calls reuse that same device memory page, `terminate_device` starts at 1.

In `remove_tracks` (line 112–114 of that kernel):
```cpp
if (*(payload.terminate) == 1) {
    return;
}
```
Every one of the 512 threads immediately returns. `n_accepted_device` is never decremented. The while loop exits with the original `n_accepted`. `fill_track_candidates` copies all input tracks to output. Result: `n_selected = n_candidates`, `n_removed = 0`.

**Fix:** Add before the while loop:
```cpp
cudaMemsetAsync(terminate_device.get(), 0, sizeof(int), stream);
```

### Bug 2 — `scanned_block_offsets_buffer` never set up (corrupts insertion sort)

**Lines 486–488:**
```cpp
vecmem::data::vector_buffer<int> block_offsets_buffer{nBlocks_scan, m_mr.main};
m_copy.get().setup(block_offsets_buffer)->ignore();
vecmem::data::vector_buffer<int> scanned_block_offsets_buffer{nBlocks_scan, m_mr.main};
m_copy.get().setup(block_offsets_buffer)->ignore();   // ← wrong: should be scanned_block_offsets_buffer
```

`scanned_block_offsets_buffer` is allocated but the vecmem device-side metadata (capacity/size) is never initialised. The `scan_block_offsets` and `add_block_offset` kernels read/write through it with undefined behaviour. This corrupts the prefix-sum used by `rearrange_tracks` (the insertion sort that keeps the sorted order of accepted tracks up-to-date after each removal iteration). Over iterations, the sorted order of `sorted_ids` degrades. This is the likely primary cause of the partially wrong result in `n5000_high` and likely contributes to convergence issues in general.

**Fix:** Change the second `setup` call to:
```cpp
m_copy.get().setup(scanned_block_offsets_buffer)->ignore();
```

### Bug 3 — Wrong `cudaMemcpy` direction for `max_shared_device` initialisation

**Lines 439–440:**
```cpp
cudaMemcpyAsync(max_shared_device.get(), max_shared, sizeof(unsigned int),
                cudaMemcpyHostToDevice, stream);
```

`max_shared` is the raw device pointer returned by `thrust::max_element(thrust::device, n_shared_buffer.ptr(), ...)`. The source is device memory but `cudaMemcpyHostToDevice` is specified. The correct direction is `cudaMemcpyDeviceToDevice`. On this GV100 with CUDA 12.5 and UVA, the runtime likely auto-detects the pointer location and treats the transfer as D2D silently, so this may not cause observable failures on this system. It is nevertheless formally undefined behaviour.

**Fix:** Change to `cudaMemcpyDeviceToDevice`.

### Bug 4 — Race condition: missing `stream.synchronize()` in the outer while loop

**Lines 666–669:**
```cpp
cudaMemcpyAsync(&terminate, terminate_device.get(), sizeof(int),
                cudaMemcpyDeviceToHost, stream);
cudaMemcpyAsync(&n_accepted, n_accepted_device.get(), sizeof(unsigned int),
                cudaMemcpyDeviceToHost, stream);
// ← no stream.synchronize() here; while condition is checked immediately
```

The host-side `terminate` and `n_accepted` variables may be stale when the while condition is evaluated on the next iteration. This causes the outer loop to run additional no-op batches (harmless extra work) or, in edge cases, to exit with a stale `n_accepted` that does not reflect how many tracks were actually removed. This is the most likely contributing cause for `n5000_high` stopping at 3200 removals instead of the correct 4983.

**Fix:** Add `stream.synchronize()` (or `cudaStreamSynchronize(stream)`) after the two async copies before the while condition check. Note that this may significantly increase latency per outer iteration, which could be addressed with a double-buffering or callback pattern.

---

## 3. Evidence from the upstream traccc test suite

The file `traccc/tests/cuda/test_ambiguity_resolution.cpp` contains the following (lines 900–909):

```cpp
// The following tests are not working for some not fully understood reason.
// We are blaming the ambiguity resolution algorithm for now.
INSTANTIATE_TEST_SUITE_P(
    DISABLED_CUDAStandard, CUDAGreedyResolutionCompareToCPU,
    ::testing::Values(std::make_tuple(5u, 50000u,
                                      std::array<std::size_t, 2u>{1u, 10u},
                                      20000u, true),
                      std::make_tuple(5u, 50000u,
                                      std::array<std::size_t, 2u>{1u, 10u},
                                      20000u, false)));
```

The traccc developers themselves disabled these tests and acknowledged the algorithm is broken for larger workloads but could not identify why.

The test suites that **do pass** are:
- `CUDASimple`: 5 tracks, 10 IDs — trivial input
- `CUDASparse`: 5 000 tracks, 1 000 000 IDs — very sparse, few conflicts
- `CUDADense`: 5 000 tracks, 100 IDs — dense but short max_meas_id
- `CUDALong`: 10 000 tracks, 10 000 IDs, **track lengths 3–500** — very long tracks

**Key insight for `CUDALong` passing:** These tests use tracks up to 500 measurements long. With `bound=512` in `remove_tracks`, `n_tracks_to_iterate` becomes 1–2 (only 1–2 long tracks fit in shared memory per iteration). The broken `scanned_block_offsets_buffer` causes less damage when only 1–2 tracks are being repositioned at a time. Our benchmark uses short tracks (3–15 measurements), causing `n_tracks_to_iterate ≈ 32–79` — enough concurrent updates to fully expose the broken insertion sort.

**Key insight for why traccc's tests pass despite Bug 1:** The `CUDAGreedyResolutionCompareToCPU` test creates a fresh `resolution_alg_cuda` object for each event and calls it exactly once. With a single call per object lifetime, `terminate_device` gets a freshly OS-allocated (zeroed) page every time. Our benchmark calls the same algorithm object 14 times (3 warmup + 10 timed + 1 check), so calls 2–14 get reused device memory where `terminate_device=1` was left by the previous call.

---

## 4. Why `n10000_low` is the single passing case

`n10000_low` is the 7th configuration processed in our sweep script. Each configuration is a separate process invocation, so there is no cross-configuration memory reuse. Within the process, the warmup=3 calls run first.

The most likely explanation: for this specific size (10 000 tracks, 50 000 IDs, lengths 3–10), the CUDA allocator happens to return fresh OS-zeroed pages for `terminate_device` on the very first call. This call runs correctly, resolves all 7 317 conflicts, and sets `terminate_device=1`. However, the correctness check (the 14th call) uses the final `check_result_buf = gpu_resolver(device_input)` call. If that call also gets a zeroed allocation (e.g. because the allocator returned a genuinely fresh page at that point in the allocation sequence), it too produces correct output.

This explanation is consistent with the non-deterministic nature of the allocation pool: for other sizes the pool hands out a page that was previously left with `terminate_device=1`, causing immediate termination without any track removal.

An alternative explanation: for `n10000_low` the large working set forces the allocator to use new OS pages (which CUDA zeroes for security) rather than reusing pool pages.

Either way, the behaviour is allocation-pool-dependent and fundamentally non-deterministic.

---

## 5. What needs to be fixed

Minimum fix for correctness:

```cpp
// After allocating terminate_device and max_shared_device, before the while loop:
cudaMemsetAsync(terminate_device.get(), 0, sizeof(int), stream);

// Change line 488:
m_copy.get().setup(scanned_block_offsets_buffer)->ignore();  // was block_offsets_buffer

// Change line 440:
cudaMemcpyAsync(max_shared_device.get(), max_shared, sizeof(unsigned int),
                cudaMemcpyDeviceToDevice, stream);           // was HostToDevice

// After the while loop body, before reading terminate / n_accepted:
stream.synchronize();   // or cudaStreamSynchronize
```

After applying these fixes, the full 3×3 sweep should be rerun and `hash_match` should be true for all configurations before any thesis performance results are treated as valid.

---

## 6. Impact on thesis measurements

- **All 7 `n_removed=0` GPU results are invalid.** The GPU performed zero ambiguity resolution — it is simply returning all input tracks.
- **The `n5000_high` partial result (1 800 selected vs CPU 17) is also invalid.** Correct answer is 17.
- **Only `n10000_low` is a valid data point.** For this configuration: GPU mean latency 46 ms vs CPU mean latency 45 ms. The GPU does not show a speedup here; H2D transfer dominates (41.9 ms) and the actual GPU kernel time is a fraction of that.
- **Until the bugs are fixed, the benchmark can only report that the GPU resolver produces wrong output for 8 out of 9 tested configurations.**
- **For the thesis narrative**, this is valuable: you identified and localised specific upstream bugs that explain why the GPU resolver fails at scale, and you can demonstrate them in the context of the broader "GPU resolver is not yet production-ready" finding.

---

## 7. Files involved

| Path | Role |
|------|------|
| `traccc/device/cuda/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu` | Contains all four bugs |
| `traccc/device/cuda/src/ambiguity_resolution/kernels/remove_tracks.cu` | Reads `terminate_device`; immediately returns if it is non-zero |
| `traccc/tests/cuda/test_ambiguity_resolution.cpp` | Lines 900–909: upstream `DISABLED_CUDAStandard` with acknowledgement of the bug |
| `traccc/examples/run/cuda/benchmark_resolver_cuda.cpp` | Our benchmark harness (calls the resolver 14 times per config) |
| `results/20260326_124050_cuda/` | Full 3×3 GPU sweep results showing the failures |
| `results/cpu_benchmark_ambig_resolution_synthetic/20260325_profile/` | CPU 3×3 sweep results (all correct, all `hash_match=true`) |
