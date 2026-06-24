# GPU Multi-Stream Throughput Results

**Date:** 2026-06-24
**Branch:** `thesis-novelty-conflict-graph` (traccc-jp)
**Hardware:** NVIDIA Quadro GV100 (`wn-lot-001.nikhef.nl`, 32 GB, 80 SMs), CUDA 12.x
**Input:** FATRAS ttbar μ=600, `event_000.json` (n=4008 candidates, n_selected=2484)
**Harness:** `traccc_benchmark_resolver_cuda`, 20 timed repeats, 5 warmup

Cross-reference:
- Single-event latency results: [`conflict_graph_results_mis_jp.md`](conflict_graph_results_mis_jp.md)
- CPU multi-thread comparison: [`cpu_multithreaded_results.md`](cpu_multithreaded_results.md)

---

## Motivation

Single-event GPU benchmarks show JP at **68.5 ev/s** vs baseline greedy at **38.8 ev/s**
at μ=600. The question is whether additional CUDA streams (independent concurrent resolvers
sharing the same GPU) can further improve aggregate throughput — exploiting any idle SM
capacity left by a single stream.

---

## Method

Each binary invocation runs:
1. A baseline single-event timed block (warmup + 20 repeats) — single CUDA stream.
2. If `--streams=N` is passed (N > 1): a multi-stream block creates N independent
   `(stream, async_copy, resolver)` triples, dispatches them all, then synchronises.
   Wall-clock time covers N events in flight; throughput = N × 1000 / wall_ms.

All N streams read from the same read-only `device_input` buffer (uploaded once before warmup).
The speedup reported by the harness is `multi_stream_ev_s / single_event_ev_s` within
the same run (so both numbers are warm and directly comparable).

---

## Results

### Baseline greedy — multi-stream sweep

Single-event warmed latency (streams=1): **~25.8 ms** → **38.8 ev/s**

*(The first binary invocation showed 27.8 ms / 35.9 ev/s due to cold CUDA context; all
subsequent runs in the same session warm at ~25.7 ms.)*

| Streams | Multi-stream ev/s | Speedup vs 1-stream | Effective latency (ms) |
|--------:|------------------:|--------------------:|-----------------------:|
| 1       | 38.8 (single)     | 1.00×               | 25.8                   |
| 2       | 39.6              | 1.02×               | 50.5                   |
| 4       | 40.4              | 1.04×               | 98.9                   |
| 8       | 41.0              | 1.05×               | 195.4                  |

**Takeaway**: Baseline greedy at μ=600 is **compute-saturated at a single stream**. Adding
up to 8 concurrent streams yields only 5% extra throughput. The GV100's 80 SMs are
already nearly fully utilised by a single baseline event at this input size.

---

### Jones–Plassmann — multi-stream sweep

Single-event JP latency (streams=1): **14.6 ms** → **68.5 ev/s**

| Streams | Multi-stream ev/s | Speedup vs JP 1-stream | Speedup vs baseline 1-stream |
|--------:|------------------:|-----------------------:|-----------------------------:|
| 1       | 68.5 (single)     | 1.00×                  | 1.77×                        |
| 2       | 73.4              | 1.07×                  | 1.89×                        |
| 4       | 78.6              | 1.15×                  | 2.02×                        |
| 8       | 80.7              | 1.18×                  | **2.08×**                    |

**Takeaway**: JP scales better with additional streams than the baseline does, because
JP finishes each event faster (~14.6 ms vs ~25.8 ms), leaving more residual SM headroom
for concurrent streams. At 8 streams JP reaches **80.7 ev/s** — 2.08× the single-stream
baseline greedy throughput.

---

### Graph structure at μ=600 (consistent across all JP runs)

| Metric              | Value     |
|---------------------|-----------|
| Outer iterations    | 18        |
| Avg batch size      | 84.7      |
| Max batch size      | 787       |
| Avg vertices        | 2 601.8   |
| Avg edges           | 11 146.7  |
| Max edges           | 73 330    |

The conflict graph is **sparse** (≪ n² edges). JP identifies an independent set
per outer iteration; the 18 iterations confirm the same convergence as seen in
earlier multi-event sweeps (μ=600 consistently needs ~18 outer rounds).

---

## Combined throughput table (all backends, μ=600)

| Backend                 | Events/sec | Speedup vs CPU 1T | Speedup vs GPU baseline |
|-------------------------|-----------:|------------------:|------------------------:|
| CPU greedy, 1 thread    | 19.3       | 1.00×             | —                       |
| CPU greedy, 2 threads   | 37.3       | 1.97×             | —                       |
| CPU greedy, 4 threads   | 74.4       | 3.92×             | —                       |
| CPU greedy, 8 threads   | 164.1      | 6.49×             | —                       |
| CPU greedy, 16 threads  | 287.9      | 14.29×            | —                       |
| CPU greedy, 32 threads  | 564.4      | 22.0×             | —                       |
| GPU baseline, 1 stream  | 38.8       | 2.01×             | 1.00×                   |
| GPU baseline, 8 streams | 41.0       | 2.12×             | 1.05×                   |
| GPU JP, 1 stream        | 68.5       | 3.55×             | 1.77×                   |
| GPU JP, 2 streams       | 73.4       | 3.80×             | 1.89×                   |
| GPU JP, 4 streams       | 78.6       | 4.07×             | 2.02×                   |
| GPU JP, 8 streams       | 80.7       | 4.18×             | **2.08×**               |

*CPU thread results from OpenMP sweep; see [`cpu_multithreaded_results.md`](cpu_multithreaded_results.md).*

---

## Key observations

1. **GPU baseline is compute-saturated at 1 stream** at μ=600. CUDA multi-stream
   concurrency provides essentially no throughput gain (≤5%). Any throughput improvement
   must come from algorithmic reduction of per-event work (i.e., JP).

2. **JP reduces per-event GPU work enough** (~43% less wall time) to expose real SM
   headroom, allowing 2–8 streams to improve aggregate throughput by 7–18%.

3. **JP 8-stream (80.7 ev/s) vs CPU 32-thread (564 ev/s)**: the CPU throughput benchmark
   is an *embarrassingly parallel* workload where each thread runs a fully independent
   resolver with no shared state. GPU streams, sharing the same hardware, cannot replicate
   this scaling. For *single-event latency*, GPU JP (14.6 ms) is 3.4× faster than CPU
   (53 ms at μ=600); for *throughput at scale*, many-core CPUs with OpenMP have a large
   advantage when events can be processed in parallel.

4. **Selection quality is preserved**: all multi-stream JP runs produce `hash_match=true`
   (selection-identical to CPU reference) and `duplicate_rate_post=0`. Multi-streaming
   does not affect correctness because each stream resolver is fully independent.

---

## Raw output location

Results saved to cluster at:
`/data/alice/sbetisor/results/20260624_gpu_utilization_sweep/`

Files: `raw_{baseline,jp}_s{1,2,4,8}.txt`, `util_{baseline,jp}_s{1,2,4,8}.tsv`
(nvidia-smi dmon GPU utilization traces at 1-second granularity)
