# Parallel Batch Greedy: Design Document

**Prepared:** 2026-04-22
**Branch:** `thesis-novelty-parallel-batch`
**Scope:** Tier 2a from [novelty_improvements.md](novelty_improvements.md) Sec. 4a/6/9.
**Status:** Implemented and validated. PBG produces bit-identical results to the CPU baseline (`hash_match=true`) across all tested (n, density, window) configurations on synthetic inputs.

> **2026-04-22 revision — "parallel conflict-free prefix" semantics.** The first implementation followed Sec. 2.x verbatim and admitted a *priority-greedy maximal independent set* per iteration. This produced two kinds of failures: (i) crashes inside the unchanged downstream `rearrange_tracks` kernel for high-conflict inputs with `W ≥ 3`; (ii) `n_selected = 0` (overshoot to empty output) before the crash. Root cause: the downstream insertion-sort pipeline (`fill_inverted_ids` → `block_inclusive_scan` → `scan_block_offsets` → `add_block_offset` → `rearrange_tracks` → `update_status`) inherits an invariant from the baseline `remove_tracks` kernel — *the tracks removed in this iteration occupy a contiguous tail of `sorted_ids`*. A non-prefix MIS leaves "gaps" inside the live region of `sorted_ids`, the prefix-sum-driven insertion sort then computes wrong shifted indices, and the kernel writes out of bounds. Sections 2.4, 4.1, 4.2 and 4.3 below have been updated to describe the **conflict-free prefix** rule that the merged code actually implements: PBG admits ranks `[0, first_fail)` where `first_fail = min{ r : candidate r failed confirm }`. This keeps the rearrange-pipeline invariant intact while still allowing parallel multi-block scale-out beyond the baseline's 512-thread single-block prefix kernel. The original "MIS" design is retained as the Tier 2c target (rearrange-pipeline must be redesigned first); see [conflict_graph_design.md](conflict_graph_design.md).

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

Keep the baseline's "longest conflict-free prefix" rule but compute it in parallel across many blocks instead of inside a single 512-thread block. Each candidate (rank `r` from the tail of `sorted_ids`) tries to claim its still-contested measurements by atomic priority. A *confirm* pass then asks each candidate "did all your claims survive?". Any candidate whose claims did **not** all survive is the *first failure point* of this iteration's prefix; we record `first_fail = min{ r : candidate r failed confirm }` and the apply pass admits exactly the ranks `[0, first_fail)`.

The priority is simply the candidate's rank `r` (worst track has `r = 0` and always wins). The conflict-free-prefix rule is preserved by construction:

- Two ranks `r₁ < r₂` admitted by apply both passed confirm → both hold all their claims → no measurement is claimed by both → admitted ranks are pairwise measurement-disjoint.
- Apply admits a contiguous prefix of ranks → the removed tracks form the contiguous tail of `sorted_ids` → the unchanged rearrange/update_status pipeline sees the same shape it sees in the baseline.

The win over baseline `remove_tracks` is **not** a wider per-iteration batch (it is the same prefix); it is that the claim, confirm and apply passes all run with grid-wide parallelism (multi-block, up to `BATCH_WINDOW_CAP = 8192` candidates) instead of single-block ≤ 512 candidates, lifting bound (b) of Sec. 1. Bound (a) is intentionally left in place, because relaxing it requires the conflict-graph rewrite documented as Tier 2c.

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
    first_fail := n_candidates                                  // initialised by prologue
    for each candidate rank r in [0, n_candidates):
        t := sorted_ids[n_accepted - 1 - r]
        if rel_shared[t] <= threshold:  continue
        priority := r
        ok := true
        for each measurement m of t:
            if n_accepted_tracks_per_meas[m] > 1 and claimed_by[m] != priority:
                ok := false; break
        if not ok:
            atomicMin(&first_fail, r)                           // remember earliest failure

Step 4 (apply, parallel):
    for each candidate rank r in [0, n_candidates):
        if r >= first_fail:  continue                            // prefix admission
        t := sorted_ids[n_accepted - 1 - r]
        if rel_shared[t] <= threshold:  continue
        slot := atomicAdd(&batch_size, 1)
        batch_ids[slot] := t
        // ... measurement-level propagation, identical to baseline remove_tracks
```

Step 3 is the confirmation pass: only the lowest-priority (best) candidate that touched each contested measurement retains the claim; any candidate whose claim was overwritten loses on that measurement, fails confirm, and pushes its rank into `first_fail` via `atomicMin`. Step 4 admits exactly the conflict-free prefix `[0, first_fail)`. By construction the admitted set is measurement-disjoint and forms a contiguous tail in `sorted_ids`, so the unchanged downstream rearrange/update_status pipeline sees the same shape it does in the baseline path.

### 2.3 Determinism

Priority is a function of `sorted_ids` only. `sorted_ids` is produced by a stable `thrust::sort` with the existing `track_comparator` (lex order over `rel_shared`, `pvals`). Ties between two tracks with identical `(rel_shared, pvals)` are broken by the stable sort's relative input order, which in turn is determined by the initial `pre_accepted_ids_buffer` population order (a `thrust::copy` from a deterministic device array). So, for the same input dump:

- every run produces the same `sorted_ids`,
- every run produces the same `claimed_by[]` end-state (atomicMin is associative/commutative),
- every run produces the same `batch_ids` (order may vary because `atomicAdd` packs in arrival order, but the **set** is identical; downstream code only consumes it as a set).

If future work requires a fixed `batch_ids` ordering, a subsequent `thrust::sort` on `batch_ids[0..batch_size)` by priority restores it. This is recorded as a config toggle but not planned for the first implementation because no downstream kernel depends on the order.

This is the story for RQ3 ("cost of enforcing determinism"): PBG is deterministic by construction, no extra synchronization is introduced.

### 2.4 Quality claim

PBG's per-iteration output is **the longest conflict-free prefix of the candidate window in priority order** — exactly the same set the baseline `remove_tracks` kernel computes, but built by grid-wide multi-block atomics rather than a single-block bitonic + scan. Formally:

- Let `W` be the candidate window (the worst `min(n_accepted, BATCH_WINDOW_CAP)` tracks).
- Let `C` be the conflict graph on `W` (edge t-t' iff they share at least one still-contested measurement).
- Baseline `remove_tracks` returns the longest prefix of `W` (in priority order) that is mutually conflict-free.
- PBG returns the same set: it is the contiguous prefix `[0, first_fail)` where `first_fail` is the rank of the first candidate whose confirm failed.
- The two sets agree because (a) candidates with `r < first_fail` all passed confirm and are pairwise measurement-disjoint, and (b) candidate `first_fail` is exactly the first rank with a measurement collision against an earlier admitted candidate, which is the same termination criterion baseline uses.

**Consequence:** PBG produces *byte-identical* removed-track sets to baseline on every iteration, hence byte-identical final accepted-track sets, hence `hash_match=true` against both the CPU greedy and the GPU baseline. This is what we observe across the n ∈ {200, 1000, 5000, 20000} × density ∈ {low, med, high} × W ∈ {1, 4, 32, 256, 1024, 8192} synthetic ladder (Sec. 5). The original Sec. 2.4 ("PBG is a strict superset of baseline batch") only applies to the MIS variant retained for Tier 2c.

Note this means **PBG is currently only a kernel-architecture novelty**, not an algorithmic one: the per-iteration set is identical to baseline, so the only source of speedup is replacing the single-block bitonic+scan with grid-wide atomics. The wall-clock comparison in Sec. 5 is therefore measuring exactly that, on top of the unchanged downstream pipeline.

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
  cudaMemsetAsync(claimed_by, 0xff)   // reset claim slots to UINT_MAX
  batch_prologue              (new, single-thread: snapshots n_accepted, resets first_fail = window, batch_size = 0, ...)
  batch_identify_removals     (new, multi-block: claim phase only)
  batch_confirm               (new, multi-block: confirm + atomicMin into first_fail)
  apply_batch_removals        (new, multi-block: admit r < *first_fail, propagate)
  batch_commit                (new, single-thread: live n_accepted -= batch_size)
  update_rel_shared           (unchanged)
  sort_updated_tracks         (unchanged)
  fill_inverted_ids           (unchanged)
  block_inclusive_scan        (unchanged)
  scan_block_offsets          (unchanged)
  add_block_offset            (unchanged)
  rearrange_tracks            (unchanged)
  update_status               (unchanged)
```

Five new kernels replace the single baseline `remove_tracks`. The split lets every multi-block kernel see a frozen view of the world taken by the prologue:

- `batch_prologue` snapshots `n_accepted` into `n_acc_snapshot`. Identify, confirm and apply all read the snapshot, never the live `n_accepted`. Without this, the live `n_accepted` (which apply must decrement once per admitted track) drifts mid-grid and different blocks resolve `sorted_ids[n_acc - 1 - r]` to different track ids — that was the original cause of the high-conflict crashes before the snapshot/commit split was introduced.
- `batch_prologue` also writes `*first_fail = candidate_window_size`, so the apply admit predicate `r < *first_fail` is well-defined even if no candidate ever fails confirm (in which case the whole window is admitted).
- `batch_confirm` is split out of `apply_batch_removals` so that *every* candidate gets to push its rank into `first_fail` before *any* candidate decides whether to admit. If admit and confirm shared a kernel, an early-rank thread might read a stale `first_fail` and admit itself even though a higher-priority neighbor already failed and was waiting to lower the bound.
- `batch_commit` is a single-threaded post-apply kernel that subtracts `batch_size` from the live `n_accepted`. Splitting it out keeps the snapshot frozen during apply (otherwise `atomicSub`s on `n_accepted` race with apply's reads of the same address).

Both the `claimed_by` and `first_fail` resets are produced by, respectively, the head `cudaMemsetAsync` and the prologue, both captured into the CUDA graph so replay needs no host intervention.

The capture/graph construction path in `greedy_ambiguity_resolution_algorithm.cu` is gated on `m_parallel_batch`. When the flag is false, the baseline `remove_tracks` is emitted; when true, the five-kernel pipeline above replaces it.

### 4.4 Why not relax to non-prefix MIS today

The downstream insertion-sort pipeline (`fill_inverted_ids`, the `block_inclusive_scan` / `scan_block_offsets` / `add_block_offset` triplet, `rearrange_tracks`, `update_status`) was written for the baseline contract: *the iteration removes a contiguous tail of `sorted_ids`, leaving a contiguous live prefix that just needs the few "updated" tracks insertion-sorted into place*. In particular `rearrange_tracks` reads `sorted_ids[i]` and the prefix sums to compute a *shifted* destination `temp_sorted_ids[shifted_idx] = tid` — the shift is exactly the count of removed tracks at positions ≤ i, which is meaningful only when removed tracks are contiguous at the tail.

A non-prefix MIS would leave gaps inside the live region, the prefix sums no longer represent a tail-shift, and `rearrange_tracks` writes out of bounds (we observed this as `compute-sanitizer` flagged the `temp_sorted_ids[shifted_idx]` write at `rearrange_tracks.cu:220` for `n=1000 d=high w=3`).

Two options for unlocking the MIS variant in a follow-up branch:

1. Replace the insertion-sort pipeline with a generic compaction (`thrust::remove_if` on `sorted_ids` with an "is removed this iter" predicate, then re-sort the small dirty set). Cost: extra full-array pass; benefit: any per-iteration removal pattern works.
2. Sort the admitted batch by rank descending and walk the gap list to compact in-place. More surgery, smaller cost, only worth it if measurement shows the generic compaction dominates.

Both are out of scope here and are documented as the entry point for the Tier 2c branch (see `conflict_graph_design.md`).

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

The conflict-free-prefix variant of PBG (the one merged) admits the *same* per-iteration set as the baseline, so the primary correctness gate is `hash_match` against the CPU baseline. The auxiliary overlap/duplicate metrics are kept in the harness because they *must* still be sanity checks (`track_overlap_vs_cpu = 1`, `duplicate_rate_post_resolve = CPU duplicate rate`) and because they will be the primary quality metric for the Tier 2c MIS variant where `hash_match` will legitimately be false.

- `hash_match` vs CPU greedy. **Target: true** for every (n, density, W) on synthetic; verified bit-identical on the n × density × W ladder in Sec. 5.5.
- `track_overlap_vs_cpu = |selected_gpu ∩ selected_cpu| / |selected_cpu|`. Target = 1 (degenerate consequence of `hash_match=true`).
- `duplicate_rate_post_resolve = (# shared measurements across accepted tracks) / (# accepted tracks)`. Target = CPU duplicate rate.
- `accepted_count_delta = |selected_gpu| - |selected_cpu|`. Target = 0.
- `convergence_curve`: per-iteration `(iter, n_accepted, batch_size)` CSV, to visualize the per-iteration prefix length and the outer-iteration count, for direct comparison against the baseline single-block prefix.

### 5.3 Input set

Reuse the existing frozen pre-resolver dumps used by [bottleneck_analysis.md](bottleneck_analysis.md) and [physics_dataset_benchmark.md](physics_dataset_benchmark.md):
- synthetic: n ∈ {500, 1000, 2000, 5000, 10000}
- fatras real dump: one representative dump per occupancy level

### 5.4 Crossover hypothesis

From the novelty doc Sec. 4, current GPU resolver crosses CPU at n ≈ 3000. PBG should push this down. Success criterion for RQ4: crossover point moves to n ≤ 1500 on synthetic dumps.

### 5.5 Validation snapshot (2026-04-22)

After the prefix redesign all configurations on the synthetic ladder agree bit-for-bit with the CPU baseline. Times below are PBG vs baseline (mean over 2-3 repeats, 1 warmup, default n_it adaptive). All `hash_match=true`.

| n | density | W | baseline ms | PBG ms | outer iters | avg batch | max batch |
|---|---|---|---|---|---|---|---|
| 200 | high | 1 | 8.5 | 18.8 | 10 | 0.9 | 1 |
| 1000 | high | 4 | 8.7 | 34.4 | 8 | 2.25 | 4 |
| 1000 | high | 32 | 8.8 | 28.5 | 7 | 2.43 | 5 |
| 1000 | high | 8192 | 8.7 | 28.2 | 7 | 2.43 | 5 |
| 1000 | med | 32 | 15.4 | 7.0 | 1 | 0 | 0 |
| 1000 | med | 8192 | 15.9 | 7.4 | 1 | 0 | 0 |
| 5000 | med | 32 | 34.6 | 30.3 | 3 | 13.7 | 23 |
| 5000 | med | 8192 | 34.6 | 30.7 | 3 | 10.0 | 26 |
| 5000 | high | 8192 | 14.3 | 145.9 | 22 | 2.27 | 5 |
| 20000 | med | 8192 | 43.6 | 97.8 | 11 | 13.1 | 26 |
| 20000 | high | 8192 | 34.6 | 670.5 | 73 | 2.63 | 7 |

Two regimes are visible:

1. **Medium-density wins.** PBG is ~2× faster than baseline at n=1000 d=med (single outer iteration ends the loop early because the entire eviction prefix is admitted at once) and ~10% faster at n=5000 d=med. This is the primary intended speedup mode for Tier 2a: the parallel multi-block prefix amortizes well when the per-iteration prefix is long.
2. **High-density loses.** When the prefix is small (d=high, baseline already only removes 1-2 tracks per iter), PBG pays five kernel launches per outer iteration where baseline pays one, and the multi-block atomics buy nothing. This is the regime that Tier 2c (MIS variant + redesigned compaction) is designed to attack — see Sec. 4.4 above and `conflict_graph_design.md`.

The takeaway is that the prefix variant is a correctness-preserving stepping stone, not the speedup story by itself; it is the runway for the MIS variant once the rearrange-pipeline contract is relaxed.

---

## 6. Risks and open questions

### 6.1 Starvation of a single very-wide track

If the worst track has a measurement footprint so large that it keeps claiming enough measurements to block every other candidate, PBG collapses to baseline batch size 1. Empirically this is the pathological case but it is also precisely the input for which baseline already does batch = 1, so PBG is not *worse* than baseline, just not *better*. This is documented as expected behavior; no algorithmic mitigation is planned for this iteration.

### 6.2 Correctness regression vs the "last one standing" propagation

The trickiest part of the baseline `remove_tracks` kernel is the detection of a measurement that drops from 2 accepted tracks to 1: the surviving track must be enqueued into `updated_tracks_buffer` so its `rel_shared` gets recomputed. `apply_batch_removals` must replicate this faithfully. The design above uses the same atomic `fetch_sub` trick the baseline uses (`N_A == 1 + n_sharing_tracks` branch at line 439 of `remove_tracks.cu`). This is the single highest-risk implementation detail; the kernel must pass the track_overlap_vs_cpu ≥ 0.90 quality gate before merging.

**Status (2026-04-22):** the conflict-free-prefix variant passes the much stricter `hash_match=true` gate on every tested synthetic configuration (Sec. 5.5), which is the strongest possible correctness signal: byte-identical output to CPU greedy. This was achieved only after fixing two latent races/invariants that the original Sec. 2.x design did not anticipate:

- **n_accepted race in apply.** Original design had each apply thread `atomicSub(n_accepted, 1)` on admission. With multi-block apply, distinct blocks then resolve `sorted_ids[n_acc - 1 - r]` to different track ids mid-grid. Fix: snapshot `n_accepted` in a single-threaded `batch_prologue` kernel into `n_acc_snapshot`, have identify/confirm/apply read only the snapshot, defer the `n_accepted -= batch_size` decrement to a single-threaded `batch_commit` kernel that runs *after* apply.
- **rearrange-pipeline tail-contiguity invariant.** Original design admitted a non-prefix MIS, leaving gaps in `sorted_ids`'s live region. The unchanged downstream `rearrange_tracks` kernel then computed wrong shifted indices and wrote out of bounds. Fix: introduce the `batch_confirm` kernel that pushes the rank of the first failing candidate into a `first_fail` device scalar via `atomicMin`, and have apply early-return when `r >= *first_fail`. This restores the "removed tracks are a contiguous tail" invariant (= same shape baseline produces) at the cost of admitting only the conflict-free prefix instead of the MIS.

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
