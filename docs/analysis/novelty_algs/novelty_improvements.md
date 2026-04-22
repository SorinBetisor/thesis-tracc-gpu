# Novelty & Improvement Directions: GPU Ambiguity Resolution

**Prepared:** 2026-04-08  
**Purpose:** Identify concrete novelty contributions for the thesis — from code-grounded incremental improvements to algorithmic alternatives.  
**Based on:** profiling data in `bottleneck_analysis.md`, source code analysis of `greedy_ambiguity_resolution_algorithm.cu` and `kernels/remove_tracks.cu`.

---

## 1. Why Greedy? Why It's the Problem

The greedy ambiguity resolver works as follows:

```
Sort tracks by score (rel_shared desc, p-value asc)
While (any track has rel_shared > threshold):
    Take the "worst" track (highest rel_shared)
    Remove it
    Update rel_shared of all tracks sharing a measurement with it
    Re-sort affected tracks
```

It is used in ACTS and traccc because:
- It is simple, deterministic, and reproducible.
- It gives well-understood quality properties (each removed track was the locally-worst choice).
- It has O(n log n + n_removals × avg_meas_per_track) complexity.

**The fundamental GPU problem:**  
The outer loop is a **data-dependent sequential chain**. Each removal changes the scores of neighboring tracks, which affects which track is worst in the next step. You cannot know which track to remove next until the current removal is fully propagated. This makes the outer loop inherently sequential — a structural barrier to GPU parallelism.

---

## 2. What the Profiling Data Shows

From `bottleneck_analysis.md` (GPU, all configs):

| Phase | Share of GPU time |
|---|---|
| `eviction_loop` | **85–93%** |
| `inverted_index` (preprocessing) | 3–9% |
| everything else | <5% |

The eviction loop itself consists of (per outer graph-replay cycle):

| Kernel | Launch config | Notes |
|---|---|---|
| `remove_tracks` | `<<<1, 512>>>` | **Single block. Maximum 512 threads. One SM.** |
| `sort_updated_tracks` | `<<<1, 512>>>` | Single block bitonic sort of updated tracks |
| `fill_inverted_ids` | `<<<blocks, warp>>>` | Scales with n_accepted — OK |
| `block_inclusive_scan` | `<<<blocks, threads>>>` | Prefix sum — multi-block, OK |
| `scan_block_offsets` | `<<<1, nBlocks_scan>>>` | Single block scan of block offsets |
| `add_block_offset` | `<<<blocks, threads>>>` | Multi-block, OK |
| `rearrange_tracks` | `<<<blocks, 1024>>>` | Multi-block, OK |
| `update_status` | `<<<blocks, warp>>>` | Multi-block, OK |

**Critical observation:** `remove_tracks` runs on a **single SM with 512 threads**. All the work of identifying which tracks to remove, sorting them, and propagating measurement status updates happens in one block. The rest of the GPU sits idle during this kernel.

**Critical observation 2:** The CUDA graph is **rebuilt from scratch every outer while-loop iteration** (`cudaStreamBeginCapture` → `cudaGraphInstantiate`). The rebuild is needed because `nBlocks_adaptive`, `nBlocks_rearrange`, and `nBlocks_scan` all depend on `n_accepted`, which decreases over time. The n_it sensitivity sweep showed that graph construction cost (not raw launch count) is the dominant overhead at small n.

---

## 3. Tier 1: Incremental GPU Improvements (lower risk, measurable)

### 3a. CUDA Graph Parameter Updates Instead of Rebuild

**Problem:** Each outer loop iteration calls `cudaStreamBeginCapture` + `cudaGraphInstantiate`. At n=1000 with 200 graph launches, this is 200 costly instantiations.

**Status update (2026-04-17):** A first implementation of this direction now exists behind an explicit benchmark flag (`--reuse-eviction-graph`). See `graph_reuse_implementation.md` for the code-level design, metrics, and validation status.

**Solution:** Build the graph once with **upper-bound kernel dimensions**, then use `cudaGraphExecKernelNodeSetParams()` (CUDA 11.1+) to update only the block/thread counts between outer iterations. `cudaGraphExecUpdate()` handles structural-compatible updates without re-instantiation.

**Expected impact:** Eliminate graph construction overhead entirely. From n_it sensitivity data, this is the dominant cost at small n (n < 2,000). Could push GPU break-even from n ≈ 3,000 down to n ≈ 1,000.

**Implementation complexity:** Medium. Requires storing `cudaGraphNode_t` handles for each of the 8 kernels and calling `cudaGraphExecKernelNodeSetParams` per outer iteration.

**Thesis angle:** First profiling paper to identify graph instantiation (not launch count) as the bottleneck in iterative GPU algorithms. Quantify cost reduction.

---

### 3b. Early Convergence Detection Without CPU Sync

**Problem:** The outer while loop requires a `m_stream.get().synchronize()` after every `n_it` graph replays to check `terminate` on the CPU. This is a CPU–GPU synchronization barrier that stalls the GPU pipeline.

**Solution:** Use one of:
1. **`cudaStreamAddCallback`** or **`cudaLaunchHostFunc`**: trigger a host-side callback when the graph completes, allowing the CPU to proceed without blocking.
2. **Persistent kernel with device-side loop**: Replace the outer while loop with a single persistent kernel that loops internally. Use `atomicLoad` on `terminate_device` instead of CPU sync. The GPU never goes idle between iterations.
3. **`cudaGraphConditionalNode`** (CUDA 12.4+): embed the convergence check inside the graph itself, allowing the graph to self-terminate without CPU involvement.

**Expected impact:** Eliminates pipeline stalls between graph replay batches. Most beneficial at small n where each batch removes few tracks and many sync round-trips are needed.

**Thesis angle:** Demonstrates how CPU–GPU synchronization patterns degrade throughput in iterative GPU algorithms. Matches RQ1 (which sub-steps dominate).

---

### 3c. Extend `remove_tracks` Beyond One Block

**Problem:** `remove_tracks<<<1, 512>>>` processes at most 512 measurements per invocation. For high-conflict tracks with many measurements, this is a bottleneck. More importantly, the single-block constraint means the entire removal phase uses one SM.

**Current code comment (line 370 in `.cu`):**
```
// @TODO: For the case where the measurement is shared by more than 1024
// tracks, the tracks need to be sorted again using thrust::sort
```
There is a documented TODO for the 1024 case. The 512 limit comes from the shared memory arrays `sh_buffer[512]`, `sh_meas_ids[512]`, `sh_threads[512]`, `sh_keys[512]` — all statically sized to 512.

**Solution:** Restructure `remove_tracks` into a two-pass multi-block kernel:
- Pass 1 (all blocks, parallel): Each block processes a slice of the `sorted_ids` array to identify its "removable" candidates, writing to a global scratch buffer.
- Pass 2 (reduction): Merge candidate sets from all blocks, resolve conflicts between candidates that share measurements, apply removals.

This is complex but directly addresses the core GPU utilization problem. Even a 4-block version would use 4× the SMs.

**Thesis angle:** Core GPU systems contribution. Profile before/after on both CUDA occupancy and wall-clock time.

---

### 3d. AoS → SoA Memory Layout for the Inverted Index

**Problem:** `tracks_per_measurement` and `track_status_per_measurement` are jagged vectors (logically AoS). In `remove_tracks`, accessing `tracks_per_measurement[unique_meas_idx]` and `track_status_per_measurement[unique_meas_idx]` for different measurements in parallel means each thread follows a pointer to a different heap region — scattered memory accesses, no cache coherence.

**Solution:** Reorder the data layout so that the primary access pattern is coalesced:
- Instead of `tracks_per_measurement[meas_idx][i]`, store as a flat array `tracks_per_meas_flat[offset[meas_idx] + i]` with a separate offset array.
- The offset array fits in L1/L2 cache; the flat track array is accessed sequentially within a measurement's block.

**Current state:** `flat_meas_ids_buffer` (the measurements per track, flattened) is already used for the unique measurement counting phase. The analogous structure for the inverted index (tracks per measurement) is still jagged.

**Expected impact:** At n=10,000 with 36,000 unique measurements, the inverted index building and lookup phases (`inverted_index`: 2.8 ms, `shared_count`: 0.04 ms GPU) — particularly the `fill_tracks_per_measurement` and `count_shared_measurements` kernels.

**Thesis angle:** Explicit AoS→SoA contribution. Directly measures memory bandwidth reduction using `ncu` Nsight Compute (bytes/second, L2 hit rate). Answers RQ2.

---

## 4. Tier 2: Algorithmic Redesign (medium risk, high impact)

### 4a. Parallel Batch Greedy ("Wavefront Removal")

**Core idea:** Instead of removing ONE track per iteration, identify a MAXIMAL SET of mutually non-conflicting "worst" tracks and remove them all in parallel.

**Algorithm:**
```
Sort tracks by score (worst first)
While (any track has rel_shared > threshold):
    batch = {}
    For each track t in sorted order (parallel scan):
        if rel_shared[t] > threshold:
            if no track already in batch shares a measurement with t:
                add t to batch   ← conflict-free parallel selection
    Remove all tracks in batch simultaneously
    Update rel_shared for all affected neighbors (parallel)
```

The parallel conflict-free selection is an **independent set problem** on the conflict sub-graph of "bad" tracks. A single parallel scan (with atomic conflict marking) approximates it.

**Why this helps:** Each iteration removes O(k) tracks instead of O(1), reducing the iteration count by factor k. The iteration count (currently n_removals = n_iterations) drops roughly to n_removals / avg_independent_set_size.

**Correctness question:** This no longer produces the same output as the sequential greedy. Two tracks that would be removed sequentially might now be "protected" by each other. This is acceptable because:
1. The greedy output is already an approximation (not the global optimum).
2. Quality metrics (duplicate rate, track efficiency) are what matter — these need to be measured.

**Thesis contribution:** Implement both variants. Show: (a) how much faster the parallel batch variant is, (b) how close the quality metrics are, (c) under which conflict density regimes quality diverges.

**Implementation sketch:**
```cuda
// New kernel: parallel_batch_identify
// Each thread i handles sorted_ids[i]
// Uses atomic CAS on a per-measurement "claimed" flag
// If all measurements of track i are unclaimed, mark them claimed and add i to batch
// Output: batch[] array of track IDs to remove this iteration
```

---

### 4b. Score Propagation / Iterative Relaxation

**Core idea:** Instead of hard removal, compute a soft "rejection score" for each track based on its neighbors, iteratively until convergence.

```
Initialize: weight[t] = 1 for all tracks (accepted)
For k = 1, 2, ..., until convergence:
    For each track t (parallel):
        conflict_pressure[t] = sum over measurements m shared with other accepted tracks:
                                  n_accepted_tracks_per_measurement[m] - 1
        if conflict_pressure[t] > threshold:
            weight[t] = 0  (soft reject)
    (no "re-sort" needed — all threads update simultaneously)
Output: accepted tracks = {t : weight[t] == 1}
```

This is similar to the Label Propagation algorithm for community detection, or Belief Propagation on a factor graph.

**Why GPU-friendly:** Every iteration is fully parallel (each thread updates one track independently). No sorting, no insertion sort, no sequential dependencies. The per-iteration cost is O(n × avg_measurements_per_track).

**Trade-off:** May require more iterations to converge than greedy. May produce slightly different (potentially better or worse) output than greedy.

**Thesis contribution:** This is the most GPU-native algorithm structure. Compare convergence speed, iteration count, and output quality against greedy. Particularly interesting for HL-LHC occupancy (high n).

---

### 4c. Conflict Graph Explicit Construction + Graph Algorithm

**Core idea:** Build an explicit track-track conflict graph (adjacency list) first, then run a graph algorithm on it.

**Phase 1 — conflict graph building (GPU, fully parallel):**
```
For each measurement m with n_accepted_tracks_per_measurement[m] > 1:
    For each pair (t_i, t_j) of tracks sharing m:
        Add edge (t_i, t_j) to conflict graph
```

This is O(Σ_m C(n_m, 2)) where n_m is the number of tracks sharing measurement m. For sparse conflicts this is small.

**Phase 2 — graph algorithm on conflict graph (multiple options):**
- **Greedy coloring**: assign each node a color such that no two adjacent nodes share a color. Each color class is an independent set that can be processed in parallel.
- **Maximum Weight Independent Set (MWIS) approximation**: find the largest set of mutually non-conflicting tracks with the highest combined score. This is the "ideal" answer but NP-hard in general; good approximations exist.
- **Vertex cover**: find the minimum set of tracks to remove such that every conflicting pair has at least one removed. This is equivalent to MWIS (complement).

**Why interesting:** The conflict graph is much smaller than the raw data (n nodes, edges proportional to conflict density). Running a graph algorithm on it is more cache-friendly than operating on the full measurement data.

**Thesis contribution:** Demonstrate that explicit conflict graph construction (fully parallel) amortizes the setup cost and enables faster graph-theoretic algorithms in the eviction phase.

---

### 4d. Hybrid CPU–GPU Split

**Core idea:** Use each processor where it is fastest.

```
Phase 1 (GPU): Build inverted index, compute rel_shared, identify all "obviously bad" tracks
               (rel_shared > 0.5, far above threshold) → remove them in one GPU batch
Phase 2 (CPU): The remaining tracks have low conflict density → run sequential greedy on CPU
               (CPU is faster at small-n sequential work)
```

From the profiling data:
- At n=1000 (typical real physics scale): CPU wins by 3–5×
- At n=10000: GPU wins by 2–4×

A hybrid could push the "effective n" seen by the CPU to always stay in the CPU-optimal regime (small n with few conflicts), while the GPU handles the heavy initial reduction.

**Thesis angle:** Empirically determine the optimal split threshold. Show that the hybrid always outperforms both pure CPU and pure GPU across all n.

---

## 5. Comparison: Why NOT the alternatives

| Algorithm | Why not (for this thesis) |
|---|---|
| **Hungarian / optimal matching** | NP-hard (exponential). Only feasible n < 30. |
| **ILP (Integer Linear Programming)** | NP-hard, requires external solver (Gurobi/CBC), not GPU-native. |
| **Simulated annealing** | Non-deterministic, hard to compare fairly, slow convergence. |
| **GNN-based selection** | Requires training data, separate ML pipeline. Thesis scope is too large. |
| **Pure `thrust::sort` per iteration** | Already considered and rejected in source code (comment at line 602). Too slow for sparse updates. |

---

## 6. Recommended Thesis Contribution Path

### Minimum viable contribution (low risk):
1. **Profile-guided diagnosis** (done): identify `remove_tracks<<<1,512>>>` and CUDA graph rebuild as bottlenecks.
2. **CUDA Graph parameter update** instead of rebuild (Tier 1a): measure reduction in construction overhead.
3. **AoS→SoA for inverted index** (Tier 1d): measure memory bandwidth improvement with Nsight Compute.

### Recommended contribution (medium risk, strong thesis):
1. Minimum viable (above).
2. **Parallel batch greedy** (Tier 2a): implement, benchmark, correctness analysis.
3. Show: at what n does the batch greedy GPU outperform sequential CPU? Does this extend the crossover to lower n?

### Strong contribution (higher risk, but potentially publishable):
1. All of the above.
2. **Score propagation / iterative relaxation** (Tier 2b): implement a pure GPU-native algorithm.
3. Show: at which conflict densities does quality diverge? Build a decision rule for which algorithm to use.

---

## 7. What to Benchmark for Each Improvement

For each proposed change, the benchmark protocol should measure:

| Metric | Tool | Purpose |
|---|---|---|
| `time_ms_mean` / `time_ms_std` | benchmark harness | Primary performance measure |
| Per-phase breakdown | profiling mode (`--profile`) | Confirm the right phase improved |
| `hash_match` vs CPU baseline | benchmark harness | Detect correctness regressions |
| Track selection overlap | new metric (% identical tracks) | Quality for approximate algorithms |
| Memory bandwidth (GB/s) | `ncu` | Confirm AoS→SoA improvement |
| SM occupancy (%) | `ncu` | Confirm multi-block improvement |
| Graph launch count | profiling | Confirm graph reuse improvement |
| n_accepted over iterations | new metric | Show batch removal convergence speed |

---

## 8. Answering the Thesis RQs with These Improvements

| RQ | Addressed by |
|---|---|
| **RQ1**: which sub-steps dominate, how do costs scale? | Section 2 above + bottleneck_analysis.md (done) |
| **RQ2**: does AoS→SoA reduce GPU memory traffic? | Tier 1d — direct measurement with ncu |
| **RQ3**: cost of enforcing determinism? | Parallel batch greedy produces non-identical but valid output → quantify quality vs speed tradeoff |
| **RQ4**: under which workloads does GPU outperform CPU? | CUDA graph reuse + batch greedy should push crossover from n≈3,000 down toward n≈500–1,000 |

---

## 9. Summary: Most Tractable Novelty for BSc Thesis

The single best combination of **novelty + measurability + feasibility** for a bachelor thesis is:

> **Parallel batch greedy** — replacing the sequential one-track-at-a-time eviction with a parallel wavefront that removes mutually non-conflicting worst tracks simultaneously — combined with **CUDA graph parameter reuse** to eliminate construction overhead.

Together these address both identified structural bottlenecks:
1. Too many outer iterations (batch greedy reduces iteration count).
2. Too much graph rebuild overhead per iteration (graph reuse eliminates it).

The thesis narrative is:
- We diagnosed why greedy is GPU-unfriendly: sequential outer loop + single-block removal kernel + per-iteration graph rebuild.
- We redesigned the removal step to exploit GPU parallelism.
- We measured the speedup and quality tradeoff rigorously.
- We established the n regime where the improved GPU resolver outperforms the CPU baseline.

This directly answers all four RQs and constitutes original engineering contribution.
