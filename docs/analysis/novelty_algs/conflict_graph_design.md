# Explicit Conflict Graph — Tier 2c As-Built Design

**Prepared:** 2026-04-22
**Branch:** `thesis-novelty-conflict-graph` (grown from `thesis-novelty-parallel-batch`)
**Status:** Implemented. Two algorithms (Luby-style MIS and Jones–Plassmann
greedy colouring, run as a one-round A/B) are wired through
`greedy_ambiguity_resolution_algorithm` on CUDA, are exposed on the harness
(`--conflict-graph={mis,jp,both}`), and are validated against the CPU
reference baseline across synthetic, ODD muon and Fatras ttbar pile-up dumps.
Runtime numbers and correctness are reported in `conflict_graph_results.md`.

> **2026-04-22 revision — "as-built".** This document supersedes the original
> design-only note. Sections 1–6 kept the motivation and the bookkeeping
> argument; Sections 3, 4 and 7 were rewritten to match the merged code.
> In particular: (i) the COO→CSR step uses `thrust::sort_by_key` +
> `thrust::lower_bound` directly and not a hand-written segmented scan;
> (ii) both Luby MIS and Jones–Plassmann are implemented as variants of a
> single *round* (same `propose`/`finalize` kernels, MIS iterates up to 32
> rounds, JP exits after one); (iii) Stage 1 compaction replaces the
> `rearrange_tracks` pipeline in graph mode — the prefix-removal invariant
> from Tier 2a is intentionally relaxed here.

Cross-references:
- Companion algorithm in the same thesis chapter:
  `parallel_batch_greedy_design.md` (Tier 2a, also on this branch via
  `--parallel-batch`).
- Original proposal:
  [`novelty_improvements.md`](novelty_improvements.md) Sec. 4c.
- Runtime evidence responding to this design: `conflict_graph_results.md`.

---

## 1. Why this document exists

Parallel Batch Greedy (PBG) uses an **implicit** conflict graph:

- Nodes are the tracks in `sorted_ids` that are still accepted.
- Edges are "these two tracks share at least one measurement whose
  `n_accepted_tracks_per_measurement > 1`".
- The edges are never materialized; they are probed on the fly through the
  `claimed_by[]` array.

This is cheap and fits the existing kernel pipeline, but it has two
structural limits:

1. **Single-color pick per iteration.** Every PBG outer iteration produces
   *one* independent set (the conflict-free prefix). A single graph-colouring
   pass can produce χ independent sets in one shot, where χ is the chromatic
   number of the conflict sub-graph. For the conflict densities we care
   about χ is small.
2. **No global view.** Algorithms like Jones–Plassmann colouring and Luby
   MIS need neighbour-set access; PBG never materializes it.

Tier 2c **materializes** the conflict graph once per outer iteration and
runs a classical parallel graph algorithm over it. The question the
companion results document answers is: *in which regimes does paying the
extra graph construction cost actually pay off?*

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
updates (tracked in Sec. 5 of the pre-merge design note) were not
implemented: for the Fatras pile-up inputs the entire graph build + CSR
step measures at 0.3–0.5 ms per call, well below the savings from
collapsing PBG iterations, and incremental bookkeeping is not on the
critical path.

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

Determinism: tie-breaks on `(π(v), v)` lexicographically; the graph is
byte-identical to what PBG probes implicitly, so the overall resolver
output is fully deterministic modulo the choice of graph algorithm.

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
| `n_candidates ≤ 100` (ODD muons) | ≈ tie | graph build dominates; both paths are within PBG ± 5% |

---

## 5. Stage 1 compaction — why the rearrange pipeline is bypassed

Tier 2a preserved the baseline's "removed tracks occupy a contiguous
tail of `sorted_ids`" invariant so that the downstream
`rearrange_tracks` + `update_status` pipeline could stay untouched. In
Tier 2c the MIS (or JP round) is not in general a prefix of the
sorted worst-first ordering — a locally-worst vertex halfway up the
sorted list is still `IN_MIS`, even though higher-rank vertices are not.
Shoehorning that into a prefix-removal kernel has two failure modes:

1. The baseline insertion-sort computes wrong shifted indices when there
   are "gaps" inside the live region, causing out-of-bounds writes in
   `rearrange_tracks`.
2. The single-block bitonic sort inside `sort_updated_tracks` assumes
   ≤ 512 updated entries; a graph-mode batch easily exceeds that.

Both symptoms are documented in the commit log and in
`parallel_batch_greedy_design.md` Sec. 4.1–4.3. The Tier 2c pipeline
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

Identical to the Tier 2a kernel: one warp per updated track, recomputes
`rel_shared[t] = float(n_shared[t]) / float(n_meas[t])`. Survivors whose
`n_shared` fell below threshold are now candidates for `is_removed` in
later outer iterations (or survive to completion).

---

## 7. Host-side orchestration — what the outer loop actually runs

The dispatcher in
`greedy_ambiguity_resolution_algorithm.cu` chooses between three paths
based on `set_conflict_graph_mode()` and `set_parallel_batch_mode()`:

1. **Baseline**: `remove_tracks<<<1,512>>>` + captured CUDA graph.
   Unchanged from upstream.
2. **PBG (Tier 2a)**: `claim → confirm → apply → update_rel_shared → …`,
   captured into a CUDA graph once per resolver call.
3. **Graph (Tier 2c, this document)**: not captured into a CUDA graph,
   because the COO→CSR conversion runs host-side Thrust calls with
   data-dependent sizes. Each outer iteration is a sequence of direct
   kernel launches interleaved with `thrust::*`, separated by exactly two
   `cudaStreamSynchronize` calls per iteration (one after
   `build_conflict_coo` to read `n_edges`, one at the end to read
   `batch_size` and `max_shared`).

The device-side buffer layout is shared with the other modes: MIS /
graph-specific scratch (`mis_priority_buffer`, `mis_active_buffer`,
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

## 9. Summary for the thesis

Tiers 2a and 2c are two points on a single axis:

| Axis | Tier 2a (PBG) | Tier 2c (explicit graph) |
|---|---|---|
| Graph representation | implicit, probed through `claimed_by[]` | explicit CSR, rebuilt every outer iteration |
| Output per outer iteration | conflict-free *prefix* of sorted worst-first | MIS (Luby) or first JP colour class |
| Batch shape | contiguous tail of `sorted_ids` | arbitrary per-vertex set |
| Downstream pipeline | unchanged `rearrange_tracks` | Stage 1 compaction + `thrust::sort` |
| Determinism tie-break | `sorted_ids` rank | `sorted_ids` rank (same) |
| Extra memory | `claimed_by[|M_A|] + batch_ids[W]` | `row_ptr[|V|+1] + col_idx[|E|] + mis_state[|V|]` |
| When it wins | conflict graph already has large independent prefixes | conflict graph is dense or PBG iteration-bound |
| Numbers | `parallel_batch_greedy_results.md` | `conflict_graph_results.md` |

The chapter presents both algorithms measured under the same harness,
with the same CPU reference, on the same input dumps — so the thesis can
claim *independently reproducible* before/after numbers for each
algorithmic stage of the Tier 2 programme.
