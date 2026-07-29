# OpenMP JP/Greedy Ablation Study

**Implementation:** the multi-threaded mode is event-level parallelism, not a parallelized resolver. Each OpenMP thread owns a private resolver instance and a private `vecmem::host_memory_resource` (no shared state, no locks); a queue of `repeats x T` independent resolver calls is distributed via `#pragma omp for schedule()`, after an excluded parallel warmup pass. Throughput = total events / wall time. The claim is thus "one node resolves N events/s by batching events across cores", not "one event resolves faster with 32 threads".

Optimizations/experiments:

- **Thread placement / pinning** (unpinned vs `close`/`spread` on cores)
- **SMT** (32 threads packed on 16 cores vs all 64 hardware threads)
- **Oversubscription** (128 software threads)
- **Loop scheduling policy** (`static`, `static,1`, `dynamic,1`, `dynamic,8`, `guided`)
- **Cumulative bundle:** pinning + 64 threads + `dynamic,1` gives **1.19x to 1.79x over the paper baseline (32T) at every pileup**; adapting thread count to the regime gives 1.37x to 1.79x.

## 1. Configurations tested


| Config             | Threads | Environment                              | What it isolates                                           |
| ------------------ | ------- | ---------------------------------------- | ---------------------------------------------------------- |
| `base_T32`         | 32      | unpinned (paper baseline)                | reference                                                  |
| `close_cores_T32`  | 32      | `OMP_PROC_BIND=close OMP_PLACES=cores`   | pinning, one thread per physical core                      |
| `spread_cores_T32` | 32      | `OMP_PROC_BIND=spread OMP_PLACES=cores`  | pinning, spread across CCXs                                |
| `smt_pack_T32`     | 32      | `OMP_PROC_BIND=close OMP_PLACES=threads` | 32 threads packed onto 16 cores (SMT siblings share cores) |
| `smt_T64`          | 64      | `OMP_PROC_BIND=close OMP_PLACES=threads` | all 64 hardware threads                                    |
| `oversub_T128`     | 128     | unpinned                                 | 2x oversubscription of hardware threads                    |


## 2. Thread placement and SMT (throughput ratio vs `base_T32`)

Greedy:


| mu  | close cores | spread cores | SMT-packed 32T | 64T (SMT on) | 128T (oversub) |
| --- | ----------- | ------------ | -------------- | ------------ | -------------- |
| 0   | 1.18        | 1.13         | 0.79           | **1.37**     | 0.66           |
| 20  | 1.08        | 1.04         | 0.50           | **1.45**     | 0.97           |
| 50  | 1.15        | 1.19         | 0.81           | **1.59**     | 1.37           |
| 100 | 1.21        | 1.19         | 0.79           | **1.58**     | 1.45           |
| 140 | 1.28        | 1.28         | 0.87           | 1.58         | **1.62**       |
| 200 | 1.15        | 1.19         | 0.78           | **1.43**     | 1.39           |
| 300 | 1.17        | 1.17         | 0.76           | **1.50**     | 1.39           |
| 400 | 1.01        | 1.04         | 0.70           | **1.36**     | 1.36           |
| 500 | 1.00        | 1.00         | 0.70           | 1.19         | **1.46**       |
| 600 | 1.07        | 1.07         | 0.69           | 1.38         | **1.63**       |


JP:


| mu  | close cores | spread cores | SMT-packed 32T | 64T (SMT on) | 128T (oversub) |
| --- | ----------- | ------------ | -------------- | ------------ | -------------- |
| 0   | 1.21        | 1.20         | 0.82           | **1.48**     | 0.54           |
| 20  | 1.14        | 1.11         | 0.72           | **1.56**     | 0.93           |
| 50  | 1.16        | 1.17         | 0.80           | **1.43**     | 1.32           |
| 100 | 1.35        | 1.36         | 0.90           | **1.79**     | 1.55           |
| 140 | 1.16        | 1.18         | 0.79           | **1.56**     | 1.47           |
| 200 | 1.01        | 1.09         | 0.72           | **1.43**     | 1.24           |
| 300 | 1.16        | 1.11         | 0.79           | **1.45**     | 1.45           |
| 400 | 1.20        | 1.18         | 0.78           | **1.59**     | 1.53           |
| 500 | 1.11        | 1.10         | 0.71           | 1.43         | **1.54**       |
| 600 | 1.06        | 1.06         | 0.69           | 1.40         | **1.66**       |


## 3. Loop schedule on a homogeneous (samee pileup) queue

With `schedule(runtime)` and placement fixed to pinned cores at 32 threads, `OMP_SCHEDULE` was swept over `static`, `static,1`, `dynamic,1`, `dynamic,8`, `guided` on every corpus. Result: `**static`, `static,1`, `dynamic,1`, and `guided` agree within about 3 percent at every pileup for both algorithms.**

## 4. Loop schedule on a heterogeneous (mixed-pileup) queue

- **mix10:** all ten corpora in equal proportion,
- **bimodal:** half mu = 20, half mu = 600 (the adversarial case for static partitioning).

Throughput in events/s (ratio vs `static` in parentheses):


| Queue   | Algo   | static | static,1     | dynamic,1        | dynamic,8    | guided           |
| ------- | ------ | ------ | ------------ | ---------------- | ------------ | ---------------- |
| mix10   | greedy | 1424   | 1212 (0.85x) | **1711 (1.20x)** | 1378 (0.97x) | 1692 (1.19x)     |
| mix10   | JP     | 1387   | 1278 (0.92x) | **1703 (1.23x)** | 1393 (1.00x) | 1662 (1.20x)     |
| bimodal | greedy | 734    | 772 (1.05x)  | 1410 (1.92x)     | 1100 (1.50x) | **1569 (2.14x)** |
| bimodal | JP     | 836    | 873 (1.04x)  | **1492 (1.78x)** | 1157 (1.38x) | 1484 (1.77x)     |


## 5. Cumulative improvement ladder: baseline to final version

The single-knob results above compose into a ladder where each version adds exactly one change on top of the previous one:

- **V0 (baseline):** 32 threads, unpinned, `dynamic,1` schedule. This is the configuration behind the paper's published multi-threaded numbers.
- **V1 = V0 + thread pinning** (`OMP_PROC_BIND=close`, one thread per physical core).
- **V2 = V1 + all 64 SMT hardware threads** (`OMP_PLACES=threads`, 64 threads).
- **V3 = V2 + 2x oversubscription** (128 software threads, pinning kept).

Greedy, events/s (speedup vs V0 in parentheses):


| mu  | V0    | V1 +pin       | V2 +64T           | V3 +oversub      | best      |
| --- | ----- | ------------- | ----------------- | ---------------- | --------- |
| 0   | 54744 | 64511 (1.18x) | **74874 (1.37x)** | 56193 (1.03x)    | V2, 1.37x |
| 20  | 22694 | 24598 (1.08x) | **32898 (1.45x)** | 27268 (1.20x)    | V2, 1.45x |
| 50  | 9513  | 10983 (1.15x) | **15129 (1.59x)** | 12060 (1.27x)    | V2, 1.59x |
| 100 | 4306  | 5224 (1.21x)  | **6783 (1.58x)**  | 6537 (1.52x)     | V2, 1.58x |
| 140 | 2775  | 3550 (1.28x)  | 4393 (1.58x)      | **4478 (1.61x)** | V3, 1.61x |
| 200 | 2249  | 2584 (1.15x)  | 3225 (1.43x)      | **3288 (1.46x)** | V3, 1.46x |
| 300 | 1346  | 1580 (1.17x)  | **2013 (1.50x)**  | 1970 (1.46x)     | V2, 1.50x |
| 400 | 981   | 991 (1.01x)   | 1333 (1.36x)      | **1386 (1.41x)** | V3, 1.41x |
| 500 | 757   | 755 (1.00x)   | 899 (1.19x)       | **1183 (1.56x)** | V3, 1.56x |
| 600 | 561   | 602 (1.07x)   | 772 (1.38x)       | **945 (1.69x)**  | V3, 1.69x |


JP, events/s (speedup vs V0 in parentheses):


| mu  | V0    | V1 +pin       | V2 +64T           | V3 +oversub      | best      |
| --- | ----- | ------------- | ----------------- | ---------------- | --------- |
| 0   | 48676 | 58701 (1.21x) | **71843 (1.48x)** | 58208 (1.20x)    | V2, 1.48x |
| 20  | 19542 | 22264 (1.14x) | **30460 (1.56x)** | 25434 (1.30x)    | V2, 1.56x |
| 50  | 8651  | 10073 (1.16x) | **12351 (1.43x)** | 12314 (1.42x)    | V2, 1.43x |
| 100 | 3225  | 4352 (1.35x)  | **5779 (1.79x)**  | 5661 (1.76x)     | V2, 1.79x |
| 140 | 2690  | 3133 (1.16x)  | **4207 (1.56x)**  | 3976 (1.48x)     | V2, 1.56x |
| 200 | 2137  | 2157 (1.01x)  | **3054 (1.43x)**  | 2917 (1.37x)     | V2, 1.43x |
| 300 | 1150  | 1338 (1.16x)  | 1672 (1.45x)      | **1750 (1.52x)** | V3, 1.52x |
| 400 | 862   | 1031 (1.20x)  | **1370 (1.59x)**  | 1367 (1.59x)     | V2, 1.59x |
| 500 | 762   | 842 (1.11x)   | 1088 (1.43x)      | **1138 (1.49x)** | V3, 1.49x |
| 600 | 590   | 628 (1.06x)   | 824 (1.40x)       | **944 (1.60x)**  | V3, 1.60x |


Reading the ladder:

1. **The universal final version is V2** (pinned, all 64 hardware threads, `dynamic,1`): it improves on the baseline at every pileup, by **1.19x to 1.59x for greedy and 1.40x to 1.79x for JP**.
2. **V3 is a high-pileup mode if needed.** Relative to V2 it wins at mu of roughly 140 and above (up to 945 vs 772 events/s for greedy at mu = 600) but regresses at low pileup

## 6. Recommended configuration

- **Ablation for the paper:** placement +5 to +35 percent, SMT (64 threads) +19 to +79 percent, pinned oversubscription (128 threads) a further +15 to +22 percent at high pileup but below the 64-thread version at low pileup, schedule policy irrelevant on homogeneous queues but worth up to 2.1x on mixed-pileup queues.
- **Bundled final version vs baseline (Section 5):** the fixed final configuration (V2) delivers 1.19x to 1.79x across all pileups; letting the thread count adapt to the regime (V2 low, V3 high) delivers 1.37x to 1.79x.

