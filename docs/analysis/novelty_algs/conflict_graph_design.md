# Explicit Conflict Graph for GPU Ambiguity Resolution — As-Built Design

**Prepared:** 2026-04-22
**Branch:** `thesis-novelty-conflict-graph`
**Status:** Implemented. Two algorithms (Luby-style MIS and Jones–Plassmann
greedy colouring, run as a one-round A/B) are wired through
`greedy_ambiguity_resolution_algorithm` on CUDA, are exposed on the harness
(`--conflict-graph={mis,jp,both}`), and are validated against the CPU
reference baseline across synthetic, ODD muon and Fatras ttbar pile-up dumps.
Runtime numbers and correctness are reported in `conflict_graph_results.md`.

> **2026-04-22 revision — "as-built".** This document supersedes the original
> design-only note. In particular: (i) the COO→CSR step uses
> `thrust::sort_by_key` + `thrust::lower_bound` directly and not a
> hand-written segmented scan; (ii) both Luby MIS and Jones–Plassmann are
> implemented as variants of a single *round* (same `propose`/`finalize`
> kernels, MIS iterates up to 32 rounds, JP exits after one); (iii) Stage 1
> compaction replaces the `rearrange_tracks` pipeline in graph mode.

Cross-reference:
- Runtime evidence responding to this design: `conflict_graph_results.md`.

---

## 1. Why this document exists

The baseline CUDA greedy resolver removes **one track per outer iteration**:
it finds the single worst track, evicts it, updates bookkeeping, and repeats.
This is a sequential structure that limits GPU parallelism — each outer
iteration uses essentially one block of work.

A natural GPU-friendly alternative is to find a **maximal independent set**
of removable tracks per outer iteration: a set of mutually non-conflicting
worst tracks that can all be evicted simultaneously, in parallel. To find
such a set efficiently we need explicit access to each track's conflict
neighbourhood, which requires **materializing the conflict graph**.

This document describes the design: build the conflict graph once per outer
iteration (COO → CSR, Sections 2–3) and run one of two classical parallel
graph algorithms over it (MIS or Jones–Plassmann, Section 4). The question
the companion results document answers is: *in which regimes does paying the
extra graph-construction cost pay off relative to the one-track-at-a-time
baseline?*

---

## 2. Explicit conflict graph — definition

Let `A ⊆ tracks` be the set of currently-accepted tracks with
`rel_shared > threshold`, and let
`M_A = { m : n_accepted_tracks_per_measurement[m] > 1 for tracks in A }`
be the *contested* measurements.

The **conflict graph** is `G = (V, E)` with `V = A` and

```
E = { (t_i, t_j) : ∃ m ∈ M_A . t_i, t_j ∈ tracks_per_measurement[m]
                                 ∧ accepted[t_i] ∧ accepted[t_j] }
```

Size bound, which the allocator uses to pre-size the COO buffers once up
front (no dynamic growth inside the outer loop):

```
|V|  ≤  |A|
|E|  ≤  Σ_{m ∈ M_A}  n_m · (n_m − 1)     // directed pairs, both orientations
        where n_m = |tracks_per_measurement[m] ∩ A|
```

Directed edges are emitted (and not just `n_m · (n_m − 1) / 2` unordered
pairs) because the MIS / JP kernels scan each vertex's own adjacency list
when deciding whether to enter the independent set — the reverse edge must
be present for that lookup to see both endpoints.

For the conflict densities in `bottleneck_analysis.md` (≤ 40% of
measurements contested, `n_m` typically ≤ 4), `|E|` is on the order of `|V|`
to a few times `|V|`. On real Fatras pile-up dumps the measured maximum
`|E|` across a whole resolver call never exceeded `56 k` even for μ = 500
events (see `conflict_graph_results.md` Sec. 3); on adversarial synthetic
dumps with `n_candidates = 5000` at high density it peaked at ~5 M. Both fit
comfortably in a pre-allocated buffer.

---

## 3. Phase 1 — parallel construction (COO → CSR)

### 3a. COO edge list — `build_conflict_coo`

One CTA per unique measurement `u`. Threads of the CTA:

1. Skip immediately if `n_accepted_tracks_per_measurement[u] ≤ 1` (fast
   reject — uncontested measurement contributes no edges).
2. Cooperatively gather the still-accepted members of
   `tracks_per_measurement[u]` into shared memory, guided by
   `track_status_per_measurement[u]`.
3. Emit the full directed pair list: for every `(i, j)` with `i ≠ j` over
   the gathered track ids, `atomicAdd` into a global `edge_count` and write
   `(src, dst) = (gathered[i], gathered[j])` into the COO buffers.

Source: `device/cuda/src/ambiguity_resolution/kernels/build_conflict_coo.cu`.

Worst-case storage is the Sec. 2 bound; the host side pre-allocates
`max_edges_ub = Σ_m n_m · (n_m − 1)` *once* from the initial
`unique_meas_counts` histogram, before the outer loop starts.

### 3b. Compaction to CSR — Thrust primitives

Host code after the kernel:

```cpp
cudaMemcpyAsync(&n_edges_host, edge_count_device, sizeof(unsigned),
                cudaMemcpyDeviceToHost, stream);
stream.sync();

thrust::sort_by_key(thrust_policy,
                    coo_src_buffer.ptr(),
                    coo_src_buffer.ptr() + n_edges_host,
                    coo_dst_buffer.ptr());

auto ci = thrust::counting_iterator<unsigned int>(0u);
thrust::lower_bound(thrust_policy,
                    coo_src_buffer.ptr(),
                    coo_src_buffer.ptr() + n_edges_host,
                    ci, ci + (n_tracks + 1u),
                    row_ptr_buffer.ptr());
```

Output layout after this step:
- `col_idx = coo_dst_buffer[0 .. n_edges)` — each vertex's neighbour list.
- `row_ptr[v]` = index of the first edge with `src = v`.
- `row_ptr[n_tracks] = n_edges`.

Reusing `coo_dst_buffer` as `col_idx` (rather than copying into a dedicated
array) saves the full edge-count allocation. The original `coo_src_buffer`
becomes the sort keys and is no longer needed after the `lower_bound` call.

**Design note — why Thrust and not a custom kernel.** An earlier draft of
this design (Sec. 3b of the pre-merge note) proposed a hand-written
segmented scan. In practice Thrust's radix sort + lower-bound produces the
same CSR layout with one-fifth the source-code surface, and the sort
dominates the graph-mode runtime by an order of magnitude less than MIS
rounds for the inputs we care about (see `conflict_graph_results.md` Sec.
3). The custom-kernel path was not pursued.

### 3c. Rebuild cadence

The conflict graph is rebuilt **every outer iteration**. Incremental
updates were not implemented: for the Fatras pile-up inputs the entire
graph build + CSR step measures at 0.3–0.5 ms per call, well below the
savings from collapsing the outer-iteration count, and incremental
bookkeeping is not on the critical path.

---

## 4. Phase 2 — candidate algorithms on the explicit graph

Both algorithms share the same two kernels —
`graph_mis_propose` and `graph_mis_finalize` — parameterized by a round
budget. The implementation is in
`device/cuda/src/ambiguity_resolution/kernels/graph_mis_round.cu`.

### 4a. Shared kernel structure

State per vertex (`mis_state_view`, initialized by `graph_mis_init`):
- `UNDECIDED` (0) — still a candidate.
- `IN_MIS` (1) — selected into the independent set; will be *removed* this
  outer iteration.
- `REMOVED_NEIGHBOR` (2) — neighbour of an `IN_MIS` vertex; stays in the
  accepted set but defers to the next outer iteration.

Each vertex also carries a deterministic priority `π(v) = inverted_ids[v]`,
which is the vertex's rank in `sorted_ids`. Higher priority = later in
sorted_ids = worse track (higher `rel_shared`).

`graph_mis_propose`:
- Early-exits if `mis_active[v] == 0` or state ≠ `UNDECIDED`.
- Scans `col_idx[row_ptr[v] .. row_ptr[v+1])`.
- A vertex qualifies as `IN_MIS` iff it is a *local maximum* in priority
  among still-`UNDECIDED` neighbours **and** it has at least one
  `UNDECIDED` neighbour.
- Also sets `*any_undecided = 1` if the vertex was reachable here, as the
  termination signal.

`graph_mis_finalize`:
- For each still-`UNDECIDED` vertex, if any neighbour is `IN_MIS`, mark
  itself `REMOVED_NEIGHBOR`.

**The "local maximum + has-undecided-neighbour" rule is the critical
correctness invariant.** Without it, an `UNDECIDED` vertex whose neighbours
all decided in earlier rounds (typically as `REMOVED_NEIGHBOR` survivors)
would vacuously pass the local-maximum test, become `IN_MIS`, and get
removed — even though it is a *good* track being rescued by its
neighbours' fates. That bug showed up in the first cut of the
implementation as over-removal in high-density synthetic inputs; see the
commit-log entry `"mis_propose: guard local-max on has_undecided_neighbor"`.

### 4b. Luby-style MIS — up to 32 rounds

```
repeat up to 32 times:
    any_undecided = 0
    graph_mis_propose   (device-side: local-max → IN_MIS, bumps any_undecided)
    graph_mis_finalize  (device-side: neighbours of IN_MIS → REMOVED_NEIGHBOR)
    copy any_undecided to host; break if 0
```

Output: a maximal independent set `I` containing the locally worst track
of every neighbourhood. We then call `apply_graph_removals` to flip
per-measurement bookkeeping for every vertex in `I`, Stage 1 compaction
drops them from `sorted_ids`, and the outer loop continues.

Determinism: tie-breaks on `(π(v), v)` lexicographically, where `π(v)`
is the same worst-first priority key used by the CPU greedy baseline, so
the overall resolver output is fully deterministic modulo the choice of
graph algorithm.

Reference: Luby, M. (1986). *A simple parallel algorithm for the maximal
independent set problem.*

### 4c. Jones–Plassmann greedy colouring — single round

Jones–Plassmann in its original form is a repeat-until-coloured loop:

```
while ∃ uncoloured v:
    for each uncoloured v in parallel:
        if π(v) > π(u) for all uncoloured neighbours u:
            color(v) = smallest color not used by already-coloured neighbours
```

For ambiguity resolution we do not need an *optimal* colouring; we need
one independent set per outer iteration, and the outer loop is exactly
the place where colour classes `C_1, …, C_χ` would be consumed anyway. So
the implementation runs **exactly one JP round**: call `propose`, call
`finalize`, do *not* iterate even if `any_undecided == 1`. The set of
`IN_MIS` vertices found in that single round is the JP "first colour
class" — it is an independent set by construction (`finalize` marks any
neighbour as `REMOVED_NEIGHBOR`), so apply + Stage 1 can consume it
directly.

In our kernel harness this is controlled by:

```cpp
const unsigned int max_rounds =
    (m_graph_algo == graph_algo_t::JP_COLOR) ? 1u : 32u;
for (unsigned int r = 0u; r < max_rounds; ++r) { /* propose, finalize */ }
```

Reference: Jones, M.T. and Plassmann, P.E. (1993). *A parallel graph
colouring heuristic.*

### 4d. Why both algorithms ship together

The A/B configuration (`--conflict-graph=both`) lets the resolver run MIS
and JP on the same input back-to-back; the harness reports per-algorithm
timings and quality metrics. The thesis uses this to disentangle *graph
construction* cost from *algorithm choice* and to argue why JP is the
better default on our real-data regime (see Sec. 4 of
`conflict_graph_results.md`).

| Regime (from measured data) | Winner | Why |
|---|---|---|
| Fatras μ=300..600 (real pile-up) | **JP** | low-density conflict graph, one JP round removes ~40–100 tracks; MIS spends 15–23 rounds for similar quality |
| Low-density synthetic | JP | same reasoning; JP is 1.2–1.5× faster than MIS for equal overlap |
| High-density synthetic (n ≥ 2000) | MIS | JP's single-round semantics leaves too many `REMOVED_NEIGHBOR`s undecided, forcing extra outer iterations; MIS converges in ~7–15 inner rounds and beats JP on wall clock |
| `n_candidates ≤ 100` (ODD muons) | ≈ tie | graph build dominates; both paths are within ± 5% of each other |

---

## 5. Stage 1 compaction — why the rearrange pipeline is bypassed

The baseline's `rearrange_tracks` + `update_status` pipeline assumes
"removed tracks occupy a contiguous tail of `sorted_ids`". The MIS (or
JP round) is not in general a prefix of the sorted worst-first ordering
— a locally-worst vertex halfway up the sorted list is still `IN_MIS`,
even though higher-rank vertices are not. Shoehorning that into the
prefix-removal kernel has two failure modes:

1. The baseline insertion-sort computes wrong shifted indices when there
   are "gaps" inside the live region, causing out-of-bounds writes in
   `rearrange_tracks`.
2. The single-block bitonic sort inside `sort_updated_tracks` assumes
   ≤ 512 updated entries; a graph-mode batch easily exceeds that.

Both symptoms are documented in the commit log. The graph-mode pipeline
replaces the whole tail with a generic **keep-mask → inclusive scan →
scatter → global sort** path:

```
fill_keep_flags        // keep_flag[i] = !is_removed[sorted_ids[i]]
thrust::inclusive_scan // prefix sums of keep_flag
compact_sorted_ids     // scatter survivors into temp_sorted_ids at new slots
                       // update n_accepted on device
cudaMemcpy temp_sorted_ids -> sorted_ids
thrust::sort(sorted_ids, trk_comp)   // global re-sort worst-first
```

The final `thrust::sort` is the safety net: it works for arbitrary batch
shapes, is O(n log n) per outer iteration, and on the inputs we care about
the total iteration count collapses by 2×–10× (see
`conflict_graph_results.md`), more than absorbing the extra sort cost.

After the sort, `is_updated_buffer` is zeroed and `max_shared_device` is
recomputed from scratch via `thrust::max_element` over `n_shared_buffer`,
because `apply_graph_removals` may have updated a number of survivors that
exceeds the assumptions of the incremental baseline bookkeeping.

Source: the graph-mode branch of
`greedy_ambiguity_resolution_algorithm.cu` (lines ~690–970 on the
`thesis-novelty-conflict-graph` branch) plus the two new kernels in
`device/cuda/src/ambiguity_resolution/kernels/{fill_keep_flags,compact_sorted_ids}.cu`.

---

## 6. Apply and rel_shared update

### 6a. `apply_graph_removals`

One thread per vertex; early-exits unless `mis_state[v] == IN_MIS`. For
the surviving `IN_MIS` vertices:

1. Walk the vertex's measurement list `meas_ids[v]`.
2. Decrement `n_accepted_tracks_per_measurement[u]` for each measurement,
   flip the per-vertex `track_status_per_measurement` entry to 0 (not
   accepted).
3. For each neighbour vertex `w` that was the *other* track on a now
   uncontested measurement, atomically decrement `n_shared[w]` and append
   `w` to `updated_tracks_buffer` if this is the first time it was
   flagged this iteration.
4. Set `is_removed[v] = 1`, **`n_shared[v] = 0`** (important: otherwise a
   later `thrust::max_element` over `n_shared_buffer` reads stale data
   from removed rows and the outer loop fails to terminate), and
   `atomicAdd(batch_size, 1)` for logging.

Source:
`device/cuda/src/ambiguity_resolution/kernels/apply_graph_removals.cu`.

### 6b. `update_rel_shared`

One warp per updated track; recomputes
`rel_shared[t] = float(n_shared[t]) / float(n_meas[t])`. Survivors whose
`n_shared` fell below threshold are now candidates for `is_removed` in
later outer iterations (or survive to completion).

---

## 7. Host-side orchestration — what the outer loop actually runs

The graph mode is selected via `set_conflict_graph_mode()` in
`greedy_ambiguity_resolution_algorithm.cu`. Unlike the baseline (which
captures a fixed CUDA graph once per resolver call), the graph mode is
**not captured into a CUDA graph**, because the COO→CSR conversion runs
host-side Thrust calls with data-dependent sizes. Each outer iteration is
a sequence of direct kernel launches interleaved with `thrust::*`,
separated by exactly two `cudaStreamSynchronize` calls per iteration
(one after `build_conflict_coo` to read `n_edges`, one at the end to
read `batch_size` and `max_shared`).

Graph-specific device scratch (`mis_priority_buffer`, `mis_active_buffer`,
`mis_state_buffer`, `coo_src_buffer`, `coo_dst_buffer`, `row_ptr_buffer`,
`edge_count_device`, `any_undecided_device`) is allocated once before the
loop starts with sizes bounded by Sec. 2.

**Termination rule**, checked on the host after each outer iteration:

```cpp
if (batch_host == 0 || max_shared_host == 0) { terminate = 1; }
```

`batch_host == 0` means the MIS/JP round found no removable vertex (graph
is an independent set at current threshold), `max_shared_host == 0` means
no track is contested any more. Either condition terminates cleanly.

---

## 8. Configuration surface

Host-side API on the resolver (`greedy_ambiguity_resolution_algorithm`):

```cpp
enum class graph_algo_t { NONE, LUBY_MIS, JP_COLOR };
void set_conflict_graph_mode(graph_algo_t);
graph_algo_t conflict_graph_mode() const;

// optional logging hooks (one entry per outer iteration)
void set_graph_batch_log(std::vector<unsigned int>* out);
void set_graph_size_log(std::vector<std::pair<unsigned int, unsigned int>>* out);
```

Benchmark harness (`benchmark_resolver_cuda`):

```
--conflict-graph=mis|jp|both
--log-graph-batches=<path.csv>
--log-graph-sizes=<path.csv>
```

`--conflict-graph=both` runs MIS and JP back-to-back on the same input
and emits `graph_mis_*` and `graph_jp_*` metric blocks, so a single
benchmark invocation produces the A/B numbers that feed
`conflict_graph_results.md`.

---


## 9. Design summary

| Property | Value |
|---|---|
| Graph representation | Explicit CSR, rebuilt every outer iteration |
| Output per outer iteration | MIS (Luby, up to 32 rounds) or first JP colour class (1 round) |
| Batch shape | Arbitrary per-vertex set — not required to be a sorted prefix |
| Downstream compaction | Keep-mask → inclusive scan → scatter → `thrust::sort` |
| Determinism tie-break | `sorted_ids` rank (same worst-first key as CPU greedy) |
| Extra device memory | `row_ptr[|V|+1] + col_idx[|E|] + mis_state[|V|]` (~2 MB on Fatras μ=600) |
| Target regime | Sparse conflict graphs (real detector pile-up, μ ≥ 200) |
| Numerical evidence | `conflict_graph_results.md` |

Both MIS and JP are measured under the same harness, with the same CPU
reference, on the same serialised input dumps — ensuring independently
reproducible numbers for each algorithm variant.
