# Greedy Profiling - Where the GPU greedy resolver spends its time and how it is bound

---

## One-line answer

The GPU greedy resolver is **eviction-loop bound at the phase level and memory-latency bound at the kernel level**. It is not compute bound, not bandwidth bound, and not register bound. The eviction loop grows from ~58 % of runtime on the smallest ODD events to ~96 % on the largest synthetic inputs, and inside that loop the kernels run at 1.6-49 % achieved occupancy with sub-1 % DRAM utilisation while spending 11-72 % of their issue slots stalled on global-memory loads.

---

## Profiling (kernels)

Greedy phase and eviction-loop kernel breakdown across corpora

**(a) Major phases** - median share of cumulative profiled phase time, with the eviction-loop share (`% evict`) annotated on the right. Phases are grouped as: `Setup + sort` = filter_setup + initial_sort; `Measurement index` = unique_meas + inverted_index + shared_count; `Greedy eviction loop` = the iterative worst-first removal loop; `Output copy` = final device-to-host compaction.

**(b) Kernels inside the eviction loop** - median share of eviction-loop kernel time. Grouped as: `Remove tracks` = remove_tracks; `Rearrange` = rearrange_tracks; `Scan/offset` = block_inclusive_scan + scan_block_offsets + add_block_offset; `Sort/fill data` = sort_updated_tracks + fill_inverted_ids + update_status.

---

## (a) Major-phase breakdown (median % of profiled phase time)


| Corpus                          | Setup + sort | Measurement index | Greedy eviction loop | Output copy | % evict |
| ------------------------------- | ------------ | ----------------- | -------------------- | ----------- | ------- |
| ODD 1mu (n=10-11)               | 12.4         | 25.7              | 58.2                 | 3.5         | 58      |
| ODD 10mu (n=88-91)              | 7.3          | 15.1              | 75.5                 | 2.0         | 76      |
| Fatras mu=0-50 (n=65-230)       | 8.6          | 18.7              | 70.2                 | 2.5         | 70      |
| Fatras mu=100-300 (n=506-1745)  | 2.3          | 8.0               | 89.1                 | 0.7         | 89      |
| Fatras mu=400-600 (n=2313-3940) | 0.9          | 4.7               | 94.1                 | 0.3         | 94      |
| Synthetic n=500                 | 1.7          | 4.2               | 93.6                 | 0.4         | 94      |
| Synthetic n=1000                | 1.4          | 3.7               | 94.3                 | 0.4         | 94      |
| Synthetic n=2000                | 1.0          | 3.4               | 94.5                 | 0.5         | 95      |
| Synthetic n=5000                | 1.6          | 2.5               | 94.5                 | 0.2         | 95      |
| Synthetic n=10000               | 1.2          | 2.4               | 96.1                 | 0.2         | 96      |


**Reading of (a):**

- The eviction loop is the dominant phase everywhere and its share **grows monotonically with candidate count**: 58 % at n=10, 76 % at n=90, ~90 % by n=500, and ~96 % by n=10000.
- Preprocessing (`Measurement index`, i.e. building the per-measurement inverted index and shared-hit counts) is only significant at very low n. At n=10-11 it is 26 % of the time; by n=10000 it has collapsed to 2.4 %. This matches the earlier finding that on real low-multiplicity ODD events the inverted index, not the eviction loop, is the relative hotspot.
- `Setup + sort` and `Output copy` are never material (both under a few percent at any realistic n).

**Conclusion:** optimising the greedy resolver means optimising the eviction loop. Everything else is amortised away as soon as the event has more than a few hundred candidates.

---

## (b) Eviction-loop kernel breakdown (median % of eviction-loop kernel time)


| Corpus                          | Remove tracks | Rearrange | Scan/offset | Sort/fill data |
| ------------------------------- | ------------- | --------- | ----------- | -------------- |
| ODD 1mu (n=10-11)               | 24.4          | 13.1      | 32.1        | 30.3           |
| ODD 10mu (n=88-91)              | 29.9          | 13.3      | 28.7        | 28.3           |
| Fatras mu=0-50 (n=65-230)       | 23.2          | 12.9      | 32.9        | 31.2           |
| Fatras mu=100-300 (n=506-1745)  | 27.1          | 16.3      | 28.9        | 27.7           |
| Fatras mu=400-600 (n=2313-3940) | 32.7          | 20.6      | 23.7        | 23.0           |
| Synthetic n=500                 | 35.7          | 15.5      | 23.7        | 24.1           |
| Synthetic n=1000                | 35.5          | 17.8      | 23.3        | 24.0           |
| Synthetic n=2000                | 34.2          | 19.9      | 22.8        | 23.6           |
| Synthetic n=5000                | 33.2          | 21.9      | 22.2        | 23.1           |
| Synthetic n=10000               | 33.0          | 22.3      | 21.5        | 22.8           |


**Reading of (b):**

- `remove_tracks` is the single largest kernel and its share **rises with n** (24 % at small n up to ~35 % at n>=500). It is the computational hot path of the loop.
- `rearrange_tracks` grows the same way (13 % -> 22 %): as more tracks survive each batch, the compaction it performs gets heavier.
- The bookkeeping kernels (`Scan/offset` and `Sort/fill data`) together are the majority at small n (~62 % combined at n=10) but shrink toward ~45 % combined at large n as the actual removal/rearrange work takes over.
- No single kernel dominates. The loop is a **sequence of eight small kernels** re-launched every iteration, which is itself part of the binding story (launch overhead and repeated global-memory round trips).

---

## How the eviction loop is bound (kernel-level Nsight evidence)

Per-kernel Nsight Compute metrics for the eviction-loop kernels


| Kernel                 | occ % | block | DRAM % | IPC  | long-SB stall |
| ---------------------- | ----- | ----- | ------ | ---- | ------------- |
| `remove_tracks`        | 23.8  | 512   | 0.3    | 0.18 | 11 %          |
| `rearrange_tracks`     | 49.5  | 1024  | 0.4    | 0.13 | 36 %          |
| `block_inclusive_scan` | 6.2   | 128   | 0.6    | 0.03 | 45 %          |
| `update_status`        | 1.9   | 32    | 0.9    | 0.02 | 55 %          |
| `sort_updated_tracks`  | 22.6  | 512   | 0.1    | 0.08 | 25 %          |
| `fill_inverted_ids`    | 1.9   | 32    | 0.4    | 0.01 | 54 %          |
| `scan_block_offsets`   | 1.6   | 25    | 0.1    | 0.03 | 39 %          |
| `add_block_offset`     | 6.2   | 128   | 0.4    | 0.01 | 53 %          |


Four independent lines of evidence pin down what greedy is bound by:

1. **Latency bound, not bandwidth bound.** DRAM throughput is **under 1 % of peak** on every eviction-loop kernel, yet 11-72 % of warp-issue slots are lost to long-scoreboard stalls (warps blocked on a pending global-memory load). The problem is too few in-flight memory transactions, not saturated HBM. The irregular neighbour lookups through the `tracks_per_measurement` inverted index do not coalesce, so each load pays full latency that the kernels are too small to hide.
2. **Occupancy starved.** Achieved occupancy is 1.6-49.5 %, mostly under 25 %. The `nvidia-smi dmon` "85-93 % SM utilisation" number only means each SM has at least one resident warp; it does not mean the issue slots are used. Several bookkeeping kernels launch **32-thread (single-warp) or 25-thread blocks**, which caps occupancy near 2 % and directly explains their high stall fractions.
3. **Not compute bound.** IPC is 0.01-0.18 instructions per active cycle (peak is ~4 on Volta). The SMs are idle waiting on memory, not busy doing arithmetic.
4. **Not register bound.** All kernels use 16-32 registers per thread against a 64k-register / 64-warp SM budget, so registers never limit occupancy. The ceiling is block size and the memory-access pattern, not the register file.

---

## Why greedy scales (or doesnt scale) the way it does

- **The eviction loop is intrinsically serial in its outer structure.** Greedy removes the worst-scoring conflicting track, then recomputes shared-hit counts, and repeats. On the GPU this becomes one CUDA-graph batch per outer iteration; the termination flag is only read between batches. The loop therefore re-launches the eight-kernel compaction sequence hundreds of times, and each launch re-reads working-set arrays from global memory. This repeated latency-bound traversal, not any single expensive kernel, is what makes greedy expensive at scale (and is exactly the cost JP amortises by removing a whole independent set per batch).
- **At small n the balance flips to preprocessing.** For real low-multiplicity events (ODD, n~10-90) the loop finishes in a handful of iterations, so the fixed inverted-index construction becomes the relative hotspot. This is the regime where the GPU loses to the CPU outright.

