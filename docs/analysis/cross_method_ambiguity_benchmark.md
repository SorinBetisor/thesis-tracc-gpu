# Ambiguity resolution: cross-method benchmark (FATRAS ttbar)

Resolver-only latency and throughput for five resolvers across the FATRAS
ttbar pileup range. Throughput = `1000 / resolver_ms` (single event, no
transfer, no clustering pre-pass counted for the graph resolvers).

## Measurement basis


| Method     | Basis                                                           |
| ---------- | --------------------------------------------------------------- |
| Greedy CPU | single thread, resolver only, 20 rep + 5 warmup                 |
| JP CPU     | single thread, resolver only, 20 rep + 5 warmup                 |
| Greedy GPU | on-device resolver only (mean of 10 events), GV100              |
| JP GPU     | on-device resolver only (mean of 10 events), GV100              |
| ML         | DBSCAN cluster + MLP infer, CPU onnxruntime (mean of 20 events) |


`n_cand` is the GV100 sweep's per-event mean (representative).
All greedy/JP runs are selection-identical to the CPU greedy reference
(`hash_match=true`, duplicate rate 0 post-resolution). ML uses a different grouping objective and is not selection-identical, but still duplicate free. When refering to "hash-match", it is computed taking single thread CPU greeedy's result json and hashing it, then comparing across methods.

## ML model

The ML resolver is ACTS' `duplicateClassifier`, a small MLP (8 inputs, hidden
layers 10-15-10, one sigmoid output, input normalisation baked into the graph)
with 426 trained parameters, taken from the ACTS MLAmbiguityResolution examples
(`duplicateClassifier.onnx`) and run here via CPU onnxruntime.

## Table 1: resolver latency (ms/event)


| μ   | n_cand | Greedy CPU | JP CPU | Greedy GPU | JP GPU | ML (cluster+infer) |
| --- | ------ | ---------- | ------ | ---------- | ------ | ------------------ |
| 0   | 66     | 0.43       | 0.48   | 1.66       | 2.39   | 10.84              |
| 20  | 147    | 1.15       | 1.24   | 2.05       | 2.69   | 31.22              |
| 50  | 294    | 2.60       | 2.91   | 2.79       | 3.69   | 82.97              |
| 100 | 563    | 6.01       | 7.01   | 3.80       | 4.60   | 171.86             |
| 140 | 776    | 8.83       | 9.81   | 4.93       | 4.86   | 240.70             |
| 200 | 1115   | 11.83      | 13.29  | 7.33       | 7.34   | 290.59             |
| 300 | 1703   | 20.16      | 22.48  | 10.48      | 10.90  | 329.80             |
| 400 | 2438   | 29.41      | 27.84  | 16.70      | 9.96   | 388.99             |
| 500 | 3110   | 36.17      | 20.81  | 21.16      | 12.03  | 389.58             |
| 600 | 3955   | 46.84      | 26.41  | 26.61      | 16.23  | 389.13             |


ML NN inference alone is 0.13 to 0.78 ms/event; the row above is clustering-bound.

## Table 2: throughput (events/s, resolver only)


| μ   | Greedy CPU | JP CPU | Greedy GPU | JP GPU | ML   |
| --- | ---------- | ------ | ---------- | ------ | ---- |
| 0   | 2329       | 2101   | 602        | 418    | 92.2 |
| 20  | 866        | 808    | 488        | 372    | 32.0 |
| 50  | 384        | 344    | 358        | 271    | 12.1 |
| 100 | 166        | 143    | 263        | 217    | 5.8  |
| 140 | 113        | 102    | 203        | 206    | 4.2  |
| 200 | 84.5       | 75.2   | 136        | 136    | 3.4  |
| 300 | 49.6       | 44.5   | 95.4       | 91.7   | 3.0  |
| 400 | 34.0       | 35.9   | 59.9       | 100    | 2.6  |
| 500 | 27.6       | 48.1   | 47.3       | 83.1   | 2.6  |
| 600 | 21.3       | 37.9   | 37.6       | 61.6   | 2.6  |


## Table 3: JP speedup (resolver latency ratio)


| μ   | JP CPU / Greedy CPU | JP GPU / Greedy GPU | JP GPU / Greedy CPU |
| --- | ------------------- | ------------------- | ------------------- |
| 0   | 0.90                | 0.69                | 0.18                |
| 20  | 0.93                | 0.76                | 0.43                |
| 50  | 0.89                | 0.76                | 0.70                |
| 100 | 0.86                | 0.83                | 1.31                |
| 140 | 0.90                | 1.01                | 1.82                |
| 200 | 0.89                | 1.00                | 1.61                |
| 300 | 0.90                | 0.96                | 1.85                |
| 400 | 1.06                | 1.68                | 2.95                |
| 500 | 1.74                | 1.76                | 3.01                |
| 600 | 1.77                | 1.64                | 2.89                |


## Table 4: multi-threaded throughput at μ=600 (OpenMP, events/s)

Event-parallel throughput: each thread runs an independent resolver on the same
frozen event (not single-event latency)


| threads | Greedy ev/s | JP ev/s | JP/Greedy |
| ------- | ----------- | ------- | --------- |
| 1       | 21.3        | 37.9    | 1.78      |
| 2       | 37.0        | 40.6    | 1.10      |
| 4       | 92.2        | 81.1    | 0.88      |
| 8       | 162.4       | 176.4   | 1.09      |
| 16      | 296.6       | 336.0   | 1.13      |
| 32      | 563.3       | 617.6   | 1.10      |


JP wins at 1 thread (1.78x, pure algorithmic gain from about 18 outer  
iterations vs hundreds for greedy). The advantage narrows to about 1.1x once  
both become memory-bandwidth-bound across threads. The 4-thread dip is L3  
contention on JP's neighbour lookups, which recovers at 8 threads and above.

## Hardware and HPC optimizations (tested at pileup 600)

Hardware: Quadro GV100 (Volta, sm_70, 80 SMs, 32 GB HBM2), input FATRAS ttbar
μ=600 (`event_000`, 4008 candidates)

### Multi-stream throughput (1 vs 8 CUDA streams)

Independent concurrent resolvers sharing one GPU. Throughput = events/s.


| streams | Greedy ev/s | JP ev/s | JP speedup vs greedy 1-stream |
| ------- | ----------- | ------- | ----------------------------- |
| 1       | 38.8        | 68.5    | 1.77x                         |
| 2       | 39.6        | 73.4    | 1.89x                         |
| 4       | 40.4        | 78.6    | 2.02x                         |
| 8       | 41.0        | 80.7    | 2.08x                         |


Baseline greedy is compute-saturated at one stream (only 5% gain out to 8  
streams). JP finishes each event in about 14.6 ms vs 25.8 ms, freeing SM  
headroom, so streams add 7 to 18%, reaching 80.7 ev/s (2.08x the single-stream  
greedy baseline).

### Nsight Compute profiling (single stream, μ=600)

Nsight Compute 2025.2.1 (last version supporting GV100). Kernel time and stall
breakdown for the dominant kernels:


| Kernel                          | calls | total µs | occupancy % | long-scoreboard stall % |
| ------------------------------- | ----- | -------- | ----------- | ----------------------- |
| `remove_tracks` (shared)        | 1000  | 31730    | 23.8        | 11                      |
| `rearrange_tracks` (shared)     | 1000  | 19340    | 49.5        | 36                      |
| `block_inclusive_scan` (shared) | 1000  | 5670     | 6.2         | 45                      |
| `update_status` (shared)        | 1000  | 5110     | 1.9         | 55                      |
| `sort_updated_tracks` (shared)  | 1000  | 4660     | 22.6        | 25                      |
| `build_conflict_coo` (JP)       | 36    | 3447     | 69.0        | 72                      |
| `apply_graph_removals` (JP)     | 36    | 2970     | 2.2         | 47                      |
| `graph_mis_propose` (JP)        | 36    | 858      | 2.5         | 38                      |


## Why score-based is excluded

Score-based (shared-hit clustering) accepts tracks in a single pass by a quality
score. That score is not derived from shared hits alone; it needs
detector-specific inputs:

- per-track fit-quality fields: χ², n.d.f., holes, outliers;
- per-detector-region hand-tuned weights for how much to trust hits/holes in
each part of a specific geometry.

These coefficients must be calibrated per detector geometry. traccc is
detector-agnostic and targets any geometry with no per-detector tuning, so
score-based only performs as well as a calibration someone has already produced
for that detector, historically an ATLAS shared-hit-clustering setting. That
calibration overhead is the likely reason it was dropped from traccc.

Consequence for this comparison: we do not have the calibration data to run it  
correctly. The only version runnable here is stripped and untuned, with every  
weight set to 0, which reduces the method to ranking by number of measurements.  
That still removes duplicates, but with all knobs at 0 it can also admit fake  
tracks into the cleaned set, so its selection is not trustworthy. It is  
therefore reported only as context, not in the head-to-head.

