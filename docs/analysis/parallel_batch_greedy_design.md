# Parallel Batch Greedy: Design Document

**Prepared:** 2026-04-22
**Branch:** `thesis-novelty-parallel-batch`
**Scope:** Tier 2a from [novelty_improvements.md](novelty_improvements.md) Sec. 4a/6/9.
**Status:** Design locked, implementation pending.

---

## 1. Problem statement (recap)

The CUDA greedy ambiguity resolver in traccc runs an outer eviction loop that is data-dependent and serial: each iteration identifies the currently-worst track(s), removes them, propagates measurement state, re-sorts affected tracks, then decides whether to continue. The profiling in [bottleneck_analysis.md](bottleneck_analysis.md) showed that the eviction loop consumes 85-93% of GPU time and that within the loop the `remove_tracks<<<1,512>>>` kernel runs on a single SM with at most 512 threads.

Contrary to what `novelty_improvements.md` implies, the baseline `remove_tracks` kernel is **not strictly one-track-per-iteration**. Reading `device/cuda/src/ambiguity_resolution/kernels/remove_tracks.cu` (lines ~180-305) shows that every thread picks one of the worst tracks from the tail of `sorted_ids`, packs all that track's measurements into shared memory, and a collective sort + scan derives `n_removable_tracks = min_thread`: the length of the **longest prefix of the sorted tail such that no two tracks in that prefix share a measurement**. So the baseline already removes a small batch per iteration, but the batch size is bounded by:

- (a) the **first-conflict-stops-the-prefix rule**: as soon as track `k` in the sorted order shares a measurement with any earlier track, all tracks from index `k` onward are deferred to a later iteration, even if they are mutually independent among themselves.
- (b) the **single-block constraint**: no more than 512 candidate tracks are considered per kernel invocation because `sh_buffer[512]`, `sh_meas_ids[512]`, `sh_threads[512]`, `sh_keys[512]` are statically sized.

Tier 2a attacks both bounds.

---

## 2. Algorithm: Parallel Batch Greedy (PBG)

### 2.1 Informal description

Replace the "longest conflict-free prefix" rule with a priority-based independent-set construction over all tracks currently above the conflict threshold. Each candidate tries to claim its measurements by atomic priority. A candidate whose claims all succeed is admitted to this iteration's removal batch. The rest are deferred as before.

The priority is simply the track's position in `sorted_ids` counted from the tail: the worst track (highest `rel_shared`, worst tie-breaker by p-value) gets priority 0 and always wins. This preserves the greedy invariant "always remove the worst track first" while allowing additional, mutually-independent worst tracks to be removed in the same iteration.

### 2.2 Pseudocode

```
Input:  sorted_ids[0..n_accepted)     sorted worst-first (tail = worst)
        meas_ids[t]                   measurements of track t
        n_accepted_tracks_per_meas[m] (global, updated across iterations)
        rel_shared[t], threshold

Output: batch_ids[0..batch_size)      tracks to remove this iteration

Step 1 (init):
    claimed_by[m] := INT_MAX  for every unique measurement m

Step 2 (claim, parallel):
    for each candidate rank r in [0, n_candidates):
        t := sorted_ids[n_accepted - 1 - r]
        if rel_shared[t] <= threshold:  continue                // stable
        priority := r
        for each measurement m of t:
            if n_accepted_tracks_per_meas[m] > 1:
                atomicMin(&claimed_by[m], priority)             // reserve

Step 3 (confirm, parallel):
    for each candidate rank r in [0, n_candidates):
        t := sorted_ids[n_accepted - 1 - r]
        if rel_shared[t] <= threshold:  continue
        priority := r
        ok := true
        for each measurement m of t:
            if n_accepted_tracks_per_meas[m] > 1 and claimed_by[m] != priority:
                ok := false; break
        if ok:
            pos := atomicAdd(&batch_size, 1)
            batch_ids[pos] := t
```

Step 3 is the confirmation pass that resolves races between the claim writes: only the lowest-priority (best) candidate that touched each contested measurement retains the claim; any candidate whose claim was overwritten loses on that measurement and is not admitted this iteration.

### 2.3 Determinism

Priority is a function of `sorted_ids` only. `sorted_ids` is produced by a stable `thrust::sort` with the existing `track_comparator` (lex order over `rel_shared`, `pvals`). Ties between two tracks with identical `(rel_shared, pvals)` are broken by the stable sort's relative input order, which in turn is determined by the initial `pre_accepted_ids_buffer` population order (a `thrust::copy` from a deterministic device array). So, for the same input dump:

- every run produces the same `sorted_ids`,
- every run produces the same `claimed_by[]` end-state (atomicMin is associative/commutative),
- every run produces the same `batch_ids` (order may vary because `atomicAdd` packs in arrival order, but the **set** is identical; downstream code only consumes it as a set).

If future work requires a fixed `batch_ids` ordering, a subsequent `thrust::sort` on `batch_ids[0..batch_size)` by priority restores it. This is recorded as a config toggle but not planned for the first implementation because no downstream kernel depends on the order.

This is the story for RQ3 ("cost of enforcing determinism"): PBG is deterministic by construction, no extra synchronization is introduced.

### 2.4 Quality claim

PBG's output set per iteration is a **greedy independent set** in the track-track conflict graph restricted to the candidate window, with priority = rank in `sorted_ids`. This is a standard greedy MIS approximation. Formally:

- Let `W` be the set of worst tracks (`rel_shared > threshold`) in the current iteration.
- Let `C` be the conflict graph on `W` (edge t-t' if they share at least one still-contested measurement).
- Baseline `remove_tracks` returns the longest conflict-free prefix of `W` in priority order.
- PBG returns a greedy MIS of `C` in priority order.
- Every element of the baseline output is contained in the PBG output (the prefix is a sub-sequence of the priority-greedy MIS).

So PBG is a strict superset of baseline batch per iteration → strictly fewer outer iterations for the same input → identical removed-track sets on non-conflicting inputs → different removed-track sets only when a deferred-but-independent worst track can be rescued.

Because the full resolver output is the union of removals across iterations, PBG may produce a different final accepted-track set than the baseline. This is expected. Quality is evaluated via the metrics in Section 5, not via `hash_match`.

### 2.5 Complexity

Per outer iteration:

| phase | baseline | PBG |
|---|---|---|
| candidate window | ≤ 512 | up to `n_accepted` (multi-block) |
| claim/confirm | O(512 * avg_meas_per_track) | O(n_candidates * avg_meas_per_track) |
| batch size | 1 ≤ k ≤ prefix | k ≥ prefix |
| synchronization | 1 block, many `__syncthreads` | 2 grid-wide launches |

Expected outer iteration count:
- baseline: ~ `n_removals / avg_prefix_size` (often close to 1-4 per iteration based on current behavior).
- PBG: ~ `n_removals / avg_batch_size` with `avg_batch_size >= avg_prefix_size`. The gap depends on conflict density; under low density the two converge, under high density PBG wins most.

---

## 3. Data structures

All new allocations live inside the resolver and are sized once up front so the reuse path (if later combined with the graph-reuse branch) stays valid.

| name | type | size | lifetime |
|---|---|---|---|
| `claimed_by_buffer` | `vector_buffer<int>` | `meas_count` (number of unique measurements) | allocated once, `memset` to `INT_MAX` at the start of each outer iteration |
| `batch_ids_buffer` | `vector_buffer<unsigned int>` | `n_tracks` (upper bound) | allocated once |
| `batch_size_device` | `unique_alloc_ptr<unsigned int>` | 1 | cleared each iteration |
| `batch_sizes_log` | `vector_buffer<unsigned int>` | `max_iterations` (default 1024) | optional, enabled via `--log-batch-sizes` |

`claimed_by_buffer` fits in ~140 kB for 36 000 unique measurements (int per measurement). This is small compared to the existing jagged `tracks_per_measurement` buffers and does not disturb occupancy.

---

## 4. Kernel design

Two new kernels are added under `device/cuda/src/ambiguity_resolution/kernels/`. The existing baseline kernels (`sort_updated_tracks`, `fill_inverted_ids`, scan triplet, `rearrange_tracks`, `update_status`) are reused unchanged.

### 4.1 `batch_identify_removals`

**Launch config:** `<<< ceil(n_accepted / 256), 256 >>>`, grid-stride over candidates.

**Payload (fields new to this kernel marked with +):**

```
sorted_ids_view                      (existing)
rel_shared_view                      (existing)
meas_ids_view                        (existing)
meas_id_to_unique_id_view            (existing)
n_accepted_tracks_per_measurement_view (existing)
n_accepted                           (existing device scalar)
threshold                            (host scalar, baked into the kernel)
+ claimed_by_view
+ candidate_window_size               clamps how many tail entries we scan
```

**Body outline:**

```cuda
int r = blockIdx.x * blockDim.x + threadIdx.x;
if (r >= candidate_window_size) return;

unsigned n = *n_accepted;
if (r >= n) return;

unsigned t = sorted_ids[n - 1 - r];
if (rel_shared[t] <= threshold) return;     // no conflict, skip

for (uint16_t i = 0; i < meas_ids[t].size(); ++i) {
    unsigned m = meas_id_to_unique_id[ meas_ids[t][i] ];
    if (n_accepted_tracks_per_measurement[m] > 1) {
        atomicMin(&claimed_by[m], r);       // reserve at priority r
    }
}
```

No `__syncthreads()` is required: all inter-thread communication goes through `atomicMin` in global memory. `candidate_window_size` is set by the host to `min(n_accepted, BATCH_WINDOW_CAP)` where `BATCH_WINDOW_CAP` defaults to 8192; this keeps the claim phase bounded for very large inputs without changing correctness.

### 4.2 `apply_batch_removals`

This kernel plays the role of the *removal* portion of the baseline `remove_tracks`. It:

1. confirms each candidate's claim survived,
2. appends admitted tracks to `batch_ids`,
3. for every admitted track and every measurement of it, flips `track_status_per_measurement[...]` for the removed slot, atomically decrements `n_accepted_tracks_per_measurement`, detects the "last one standing" case that signals a neighbor whose `n_shared` needs updating, and enqueues that neighbor into the existing `updated_tracks_buffer` / `is_updated_buffer` / `track_count_buffer` machinery. This is byte-for-byte the same update semantics as the baseline.

**Launch config:** `<<< ceil(candidate_window_size / 128), 128 >>>`. We deliberately pick 128-thread blocks because each thread holds a per-track inner loop of average size `avg_meas_per_track` (~10 in realistic inputs), which gives us good occupancy on GV100.

**Body outline (pseudocode, NOT C++):**

```
r = global_thread_id
if r >= candidate_window_size: return
t = sorted_ids[*n_accepted - 1 - r]
if rel_shared[t] <= threshold: return

# --- confirm ---
ok = true
for m in meas_ids[t]:
    u = meas_id_to_unique_id[m]
    if n_accepted_tracks_per_measurement[u] > 1 and claimed_by[u] != r:
        ok = false; break
if not ok: return

# --- admit ---
slot = atomicAdd(batch_size, 1)
batch_ids[slot] = t

# --- apply removal (replicates the "remove_tracks" propagation) ---
for m in meas_ids[t]:
    u = meas_id_to_unique_id[m]
    # flip the removed track's status slot (binary search) - same as baseline
    # atomic decrement n_accepted_tracks_per_measurement[u]
    # when fetch_sub returns 2 (i.e. only one track remains on u), enqueue that track
    # into updated_tracks_buffer via atomicAdd on n_updated_tracks, and
    # atomicSub(n_shared[alive_trk], m_count_in_alive_trk)
    # recompute rel_shared[alive_trk] = n_shared[alive_trk] / n_meas[alive_trk]
```

Key invariant: because every measurement of an admitted track is claimed exclusively by that track, there is no race between admitted tracks on the same measurement. Admitted tracks *may* touch the same "alive" neighbor in two different measurements (the alive-neighbor's `n_shared` is decremented multiple times), which is why we use atomic sub and the existing `track_count` ref-count machinery unchanged.

### 4.3 Launch ordering inside the graph

Replace the single baseline `remove_tracks` node with:

```
  batch_identify_removals    (new)
  apply_batch_removals       (new)
  sort_updated_tracks        (unchanged)
  fill_inverted_ids          (unchanged)
  block_inclusive_scan       (unchanged)
  scan_block_offsets         (unchanged)
  add_block_offset           (unchanged)
  rearrange_tracks           (unchanged)
  update_status              (unchanged)
```

The capture/graph construction path in `greedy_ambiguity_resolution_algorithm.cu` (line ~555 onward) is gated on `m_parallel_batch`. When the flag is false, the baseline `remove_tracks` is emitted; when true, the two new kernels replace it.

A `cudaMemsetAsync(claimed_by_buffer.ptr(), 0xff, ...)` to reset `claimed_by` to `INT_MAX` is inserted at the head of the graph, and a `cudaMemsetAsync(batch_size_device.get(), 0, ...)` at the same point. Both are pinned into the captured graph so graph-replay handles them automatically.

---

## 5. Evaluation plan

### 5.1 Performance metrics

Captured by the extended `benchmark_resolver_cuda` harness:

| metric | rationale |
|---|---|
| `time_ms_mean / std` | primary speedup number |
| per-phase NVTX breakdown (`eviction_loop`, `batch_identify`, `batch_apply`) | confirm the right phase improved |
| outer iterations | direct measure of the "fewer iterations" claim |
| `avg_batch_size`, `max_batch_size` | tells us whether the algorithm is actually batching |
| SM occupancy on the two new kernels (Nsight Compute) | confirms multi-block scale-out |
| L2 hit rate, global mem throughput (Nsight Compute) | checks we are not bandwidth-bound |

### 5.2 Quality metrics

Because PBG's removed set is a greedy MIS not a prefix, `hash_match` vs CPU greedy is expected false. The new metrics:

- `track_overlap_vs_cpu = |selected_gpu ∩ selected_cpu| / |selected_cpu|`. Target ≥ 0.90 on the synthetic and fatras-real-dump inputs.
- `duplicate_rate_post_resolve = (# shared measurements across accepted tracks) / (# accepted tracks)`. Target ≤ CPU duplicate rate + small slack.
- `accepted_count_delta = |selected_gpu| - |selected_cpu|`. Expected slightly positive (PBG is less aggressive in rescuing independent worst tracks).
- `convergence_curve`: per-iteration `(iter, n_accepted, batch_size)` CSV, to visualize that the algorithm converges faster in iteration count.

### 5.3 Input set

Reuse the existing frozen pre-resolver dumps used by [bottleneck_analysis.md](bottleneck_analysis.md) and [physics_dataset_benchmark.md](physics_dataset_benchmark.md):
- synthetic: n ∈ {500, 1000, 2000, 5000, 10000}
- fatras real dump: one representative dump per occupancy level

### 5.4 Crossover hypothesis

From the novelty doc Sec. 4, current GPU resolver crosses CPU at n ≈ 3000. PBG should push this down. Success criterion for RQ4: crossover point moves to n ≤ 1500 on synthetic dumps.

---

## 6. Risks and open questions

### 6.1 Starvation of a single very-wide track

If the worst track has a measurement footprint so large that it keeps claiming enough measurements to block every other candidate, PBG collapses to baseline batch size 1. Empirically this is the pathological case but it is also precisely the input for which baseline already does batch = 1, so PBG is not *worse* than baseline, just not *better*. This is documented as expected behavior; no algorithmic mitigation is planned for this iteration.

### 6.2 Correctness regression vs the "last one standing" propagation

The trickiest part of the baseline `remove_tracks` kernel is the detection of a measurement that drops from 2 accepted tracks to 1: the surviving track must be enqueued into `updated_tracks_buffer` so its `rel_shared` gets recomputed. `apply_batch_removals` must replicate this faithfully. The design above uses the same atomic `fetch_sub` trick the baseline uses (`N_A == 1 + n_sharing_tracks` branch at line 439 of `remove_tracks.cu`). This is the single highest-risk implementation detail; the kernel must pass the track_overlap_vs_cpu ≥ 0.90 quality gate before merging.

### 6.3 Interaction with the configurable inner-loop iterations var (cherry-picked `5d2f0529` / `c4966f8d`)

The inner-loop iteration count is an orthogonal knob: it controls how many times the captured graph is replayed per outer rebuild. PBG changes the *outer* loop iteration count. The two compose. We document the combined sweep in the results skeleton but keep the default unchanged.

### 6.4 Combining with graph reuse (from the parallel branch)

Graph reuse assumes a fixed kernel topology with only launch-config updates. PBG adds two kernels to the topology. Reusing the PBG graph is therefore compatible in principle (the topology stays fixed across outer iterations) and is worth a second branch after PBG stands alone. Out of scope for this branch.

---

## 7. Deliverables on this branch

1. This design document (committed before any C++).
2. Public API toggle `set_parallel_batch_mode(bool)` + CLI flag `--parallel-batch`.
3. Two new kernels, gated by the toggle.
4. Harness metrics extension (overlap, duplicate rate, batch-size logging).
5. `parallel_batch_greedy_results.md` skeleton for the measurement campaign.
6. `conflict_graph_design.md` groundwork for Tier 2c.

---

## 8. Cross-references

- Bottleneck profile: [bottleneck_analysis.md](bottleneck_analysis.md)
- Prior novelty direction catalogue: [novelty_improvements.md](novelty_improvements.md) (see Sec. 4a for the seed of this design, and Sec. 7 for the metrics framework reused here)
- Graph-reuse parallel track: [graph_reuse_implementation.md](graph_reuse_implementation.md)
- Fatras real-dump validation: [fatras_real_dump_graph_reuse.md](fatras_real_dump_graph_reuse.md)
