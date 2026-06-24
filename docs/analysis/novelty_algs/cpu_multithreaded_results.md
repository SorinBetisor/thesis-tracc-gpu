# CPU Multi-Threaded Ambiguity Resolution Results

**Date:** 2026-06-24
**Hardware:** AMD EPYC 7502P (32 physical cores, 64 logical threads), Stoomboot cluster
**Input:** FATRAS ttbar μ=600, `event_000.json` (n=4008 candidates, n_selected=2484)
**Harness:** `traccc_benchmark_resolver` (CPU), 20 timed repeats, 3 warmup

Cross-reference:
- GPU comparison: [`multi_stream_gpu_results.md`](multi_stream_gpu_results.md)
- Single-event GPU latency: [`conflict_graph_results_mis_jp.md`](conflict_graph_results_mis_jp.md)

---

## Motivation

GPU benchmarks compare against a single-threaded CPU baseline. To provide a fair
throughput comparison (GPU throughput vs. what the same hardware budget can do on CPU),
we add an OpenMP multi-threaded CPU path where N threads each run a fully independent
resolver on the same frozen input event — matching the GPU multi-stream model.

This establishes the CPU throughput ceiling that GPU implementations must reach to
justify GPU offloading in a real pipeline.

---

## Implementation

Added `--threads=N` flag to `traccc_benchmark_resolver` (CPU harness), building on
the existing `traccc::host::greedy_ambiguity_resolution_algorithm`. Key design choices:

- Each OpenMP thread allocates its own `vecmem::host_memory_resource` (wraps malloc,
  thread-safe) and its own resolver instance — zero shared mutable state.
- Outer loop is `#pragma omp parallel num_threads(N)` + `#pragma omp for schedule(dynamic)`
  over `repeats × N` work items, so load is balanced even if thread timing varies.
- Per-event wall times are captured per work item; aggregate throughput uses total
  wall time from the parallel region.
- Built against OpenMP via `find_package(OpenMP)` + `target_link_libraries(... OpenMP::OpenMP_CXX)`.

---

## Results — OpenMP throughput sweep (μ=600)

Single-thread baseline (no OpenMP): **1000 / mean_ms** ≈ **19.3 ev/s**

| Threads | OMP ev/s | Speedup vs 1 thread | Efficiency |
|--------:|---------:|--------------------:|-----------:|
| 1       | 19.3     | 1.00×               | 100%       |
| 2       | 37.3     | 1.93×               | 97%        |
| 4       | 74.4     | 3.85×               | 96%        |
| 8       | 164.1    | 8.50×               | 106%\*     |
| 16      | 287.9    | 14.9×               | 93%        |
| 32      | 564.4    | 29.2×               | 91%        |

\* >100% efficiency at 8 threads likely reflects NUMA effects — 8 threads fit within
a single NUMA domain and benefit from local memory; single-thread runs hit remote
pages intermittently.

---

## Comparison with GPU throughput at μ=600

| Backend                 | Events/sec | Speedup vs CPU 1T |
|-------------------------|-----------:|------------------:|
| CPU greedy, 1 thread    | 19.3       | 1.00×             |
| GPU baseline, 1 stream  | 38.8       | 2.01×             |
| GPU JP, 1 stream        | 68.5       | 3.55×             |
| GPU JP, 8 streams       | 80.7       | 4.18×             |
| CPU greedy, 4 threads   | 74.4       | 3.85×             |
| CPU greedy, 8 threads   | 164.1      | 8.50×             |
| CPU greedy, 16 threads  | 287.9      | 14.9×             |
| CPU greedy, 32 threads  | 564.4      | 29.2×             |

---

## Key observations

1. **Near-linear CPU scaling up to 32 threads** (91% efficiency at 32T). The CPU
   ambiguity resolver is embarrassingly parallel across events — no shared mutable
   state once input is frozen — so OpenMP scales cleanly.

2. **GPU JP 8-stream (80.7 ev/s) ≈ CPU 4-thread (74.4 ev/s)**. In pure aggregate
   throughput, the GPU JP approach (1 GPU stream per event) with 8 concurrent streams
   is roughly equivalent to 4 CPU threads running the greedy resolver. A 32-core CPU
   at full utilisation (564 ev/s) outperforms the GPU by ~7×.

3. **GPU advantage is in single-event latency, not throughput**. GPU JP delivers
   14.6 ms per event vs CPU's ~52 ms — a 3.5× latency speedup. In a real-time
   pipeline where latency (not throughput) is the bottleneck (e.g., HLT online
   trigger), the GPU wins decisively. For offline batch processing where throughput
   matters and many cores are available, a well-threaded CPU is competitive.

4. **Fair comparison caveats**: the CPU uses the sequential greedy algorithm (same
   code as the GPU baseline path). The GPU JP uses an algorithmically different
   approach (conflict-graph colouring) that finds the same solution faster. A fairer
   throughput comparison would pair CPU JP (if implemented) against GPU JP; the
   current numbers compare GPU JP against CPU greedy.

---

## Raw output (cluster)

Results produced via:
```
traccc_benchmark_resolver --input-dump=.../fatras_ttbar_mu600/event_000.json \
  --repeats=20 --warmup=3 --threads=<N>
```
on `wn-lot-001.nikhef.nl` (same node as GPU runs for fair hardware comparison).
