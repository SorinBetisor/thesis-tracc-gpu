# Explicit Conflict Graph — Tier 2c Design Groundwork

**Scope**: design-only, not implemented on this branch.
This document belongs to the `thesis-novelty-parallel-batch` branch so that the thesis can present Parallel Batch Greedy (PBG, Tier 2a) and the *explicit* conflict graph path (Tier 2c) as parts of a single conceptual story, and so that a follow-up branch can implement 2c without re-deriving the algebra.

Cross-references:
- Algorithm currently implemented: `parallel_batch_greedy_design.md`.
- Original proposal: `novelty_improvements.md` Sec. 4c.
- Runtime evidence this design will respond to: `parallel_batch_greedy_results.md` Sec. 4d.

---

## 1. Why this document exists

Parallel Batch Greedy uses an **implicit** conflict graph:
- Nodes are the tracks in `sorted_ids` that are still accepted.
- Edges are "these two tracks share at least one measurement whose `n_accepted_tracks_per_measurement > 1`".
- The edges are never materialized; they are probed on the fly through the `claimed_by[]` array.

This is cheap and fits the existing kernel pipeline, but it has structural limits:
1. **Single-color pick per iteration.** Every outer iteration produces *one* independent set (the "batch"). A single graph-coloring pass can produce `χ` independent sets in one shot, where `χ` is the chromatic number of the conflict sub-graph. For conflict densities we care about, `χ` is small.
2. **Rediscovery cost.** Every iteration re-probes measurements through `tracks_per_measurement`. If the conflict graph changes slowly (only survivors of the dropped tracks alter it), we re-pay the probing cost each iteration.
3. **No global view.** Algorithms like Jones–Plassmann coloring and Luby MIS need neighbour-set access, which PBG never materializes.

Tier 2c **materializes** the conflict graph once per outer iteration (or even once per resolver call, see Sec. 5) and runs a classical parallel graph algorithm over it.

The question this doc exists to answer ahead of time: **under what measured PBG behaviour is the extra implementation cost justified?** Sec. 6 makes that explicit.

---

## 2. Explicit conflict graph — definition

Let `A ⊆ tracks` be the set of currently-accepted tracks with `rel_shared > threshold`. Let `M_A = { m : n_accepted_tracks_per_measurement[m] > 1 for tracks in A }` be the *contested* measurements.

The **conflict graph** is `G = (V, E)` with `V = A` and
`E = { (t_i, t_j) : ∃ m ∈ M_A such that t_i, t_j ∈ tracks_per_measurement[m] ∧ accepted[t_i] ∧ accepted[t_j] }`.

Each edge has a natural weight: the number of distinct contested measurements the two endpoints share (useful for prioritizing which conflict to resolve first, discussed in Sec. 4b).

Size bound:
```
|V|  ≤  |A|
|E|  ≤  Σ_{m ∈ M_A}  C(n_m, 2)
        where n_m = |tracks_per_measurement[m] ∩ A|
```
For the conflict densities in `bottleneck_analysis.md` (≤ 40% of measurements contested, `n_m` typically ≤ 4), `|E|` is on the order of `|V|` to a few times `|V|`. This is the structural reason this approach is tractable: the conflict graph is vastly smaller than the raw measurement incidence data.

---

## 3. Phase 1 — parallel construction (COO → CSR)

### 3a. COO edge list

One kernel:

```cuda
// Grid = one thread per (m, local pair index)
for each m in M_A in parallel:
    let T_m = tracks_per_measurement[m] restricted to accepted tracks
    for each unordered pair (t_i, t_j), i < j, in T_m:
        pos = atomicAdd(&edge_count, 1)
        src[pos] = t_i;  dst[pos] = t_j;  w[pos] = 1
```

Two practical details:
- `M_A` can be compacted up front with a parallel scan over `n_accepted_tracks_per_measurement`. We already build `tracks_per_measurement` in the baseline pipeline, so no new incidence structure is needed.
- Grid layout: use a warp (or small CTA) per contested measurement rather than one thread per pair, so warps with large `n_m` do not serialize. This matches the standard COO construction pattern used in nvGRAPH.

Bound on worst-case storage: `|E|_upper = Σ_m C(n_m, 2)`, with `n_m ≤ max_sharing_cap` (in traccc `max_sharing_cap = 5` is the default). Pre-allocate `|E|_upper` to avoid dynamic growth.

### 3b. Compaction to CSR

Edges need to be queried by source vertex in Phase 2. A single radix-sort of `(src, dst)` pairs followed by a segmented scan yields CSR offsets `row_ptr[|V|+1]`. Both primitives exist in CUB / Thrust and are the same building blocks used elsewhere in traccc, so no new dependency is introduced.

COO → CSR is a cost; it is amortized over one or more graph algorithm passes (see Sec. 5).

---

## 4. Phase 2 — candidate algorithms on the explicit graph

Two concrete algorithms are in scope. Both are inherently parallel and both are implementable with the CSR representation from Sec. 3b.

### 4a. Jones–Plassmann greedy coloring

Reference: Jones, M.T. and Plassmann, P.E. (1993). *A parallel graph coloring heuristic.*

Algorithm:
```
assign each v ∈ V a random priority π(v)
while ∃ uncoloured v:
    for each uncoloured v in parallel:
        if π(v) > π(u) for all uncoloured neighbours u:
            color(v) = smallest color not used by already-coloured neighbours
```

For the conflict graph, we do not need an optimal colouring; we need one partition of `V` into independent sets `C_1, ..., C_χ` such that every set is conflict-free. All tracks in a single color class can be treated in parallel by the existing propagation path (`update_status`, `rearrange_tracks`).

**Why this is attractive over PBG**: every color class is a whole batch, and one full colouring pass collapses multiple PBG outer iterations into a fixed number of color-class iterations (`χ` iterations instead of `n_removals / avg_batch_size`). If `χ` is small (conflict density is low), this is strictly fewer kernel boundaries.

**Determinism**: replace the random `π` with a deterministic priority equal to the track's `sorted_ids` rank. This preserves the same tie-breaking rule PBG already uses for RQ3, so the thesis story on determinism remains "lowest sorted-rank wins".

### 4b. Luby-style maximal independent set (MIS)

Reference: Luby, M. (1986). *A simple parallel algorithm for the maximal independent set problem.*

Algorithm:
```
I := ∅
while V ≠ ∅:
    for each v ∈ V in parallel:
        mark(v) = 1 with probability 1 / (2 * deg(v))
        for each marked v, if any neighbour is also marked:
            unmark the lower-priority one (deterministic tie-break via sorted_ids rank)
    add remaining marked vertices to I
    remove I and its neighbours from V
return I
```

Output is a single maximal independent set; one outer iteration of the resolver removes the complement (i.e. the neighbours), which is the set of tracks that *lose* every conflict. That is algorithmically closest to the current greedy semantics: the "worst" tracks in every neighbourhood are dropped, the rest survive.

**Trade-off vs 4a**: Luby converges in `O(log |V|)` rounds empirically but only produces *one* independent set per resolver iteration. Jones–Plassmann produces `χ` sets with one construction but is not trivially iterative-friendly if the graph changes between outer iterations.

### 4c. Which to pick

A short experimental table goes here once we have runtime data:

| Regime (from PBG results) | Likely better | Why |
|---|---|---|
| low conflict density, sparse graph | Jones–Plassmann | small `χ`, one pass resolves all |
| high conflict density, many pair overlaps | Luby MIS | avoids paying `χ` colouring cost on a dense graph |
| PBG `avg_batch_size` large already (≥ 50) | Stay on PBG | the implicit graph is already extracting most parallelism |
| PBG `avg_batch_size` small (≤ 5) with many outer iterations | Either 4a or 4b | PBG is iteration-bound and amortizing graph construction pays off |

---

## 5. Graph reuse across outer iterations

The conflict graph only changes when a track's accepted status flips. If the batch dropped `B` tracks in the previous iteration, the graph update is *local*: vertices in `B` are deleted, edges touching them are deleted, and no new edges appear.

Two engineering options:

1. **Rebuild from scratch every outer iteration.** Simpler. Pays `O(|E|)` per iteration.
2. **Incremental update.** Maintain the CSR across iterations, mark deleted edges lazily, rebuild only when the fraction of tombstones exceeds some budget (say 25%). Pays a small fraction of `O(|E|)` amortized.

Option 2 is the natural generalization of the graph-reuse idea already prototyped on `thesis-novelty-graph-reuse` — the cost structure is identical: amortize a one-time build over multiple launches. Sec. 4 of `graph_reuse_implementation.md` gives the existing reuse-budget machinery.

---

## 6. Hand-off criteria from PBG to Tier 2c

Do *not* implement 2c until PBG results are in. Decision rule, to be evaluated against the numbers that will populate `parallel_batch_greedy_results.md`:

**Implement Tier 2c if any two of the following hold**:

1. **Iteration bottleneck persists.** PBG `n_outer_iterations` at n = 1770 (Fatras μ=300) is still ≥ 0.5× the baseline's iteration count — i.e. PBG saved < 2× iterations. In that regime, compressing to `χ` classes is the remaining lever.
2. **Batch-size distribution is small with a long tail.** PBG `avg_batch_size < 10` while `max_batch_size > 100`. This says the implicit algorithm found a few good batches but is starving most of the time — an explicit colouring sees all classes at once.
3. **Quality regression is present but small.** `track_overlap_vs_cpu ∈ [0.90, 0.95]`. Explicit graph algorithms (especially Luby MIS with deterministic priority) give tighter control over *which* tracks survive, so the quality knob is richer than PBG's "threshold + window size".
4. **Atomic contention is the PBG hotspot.** `ncu` on `apply_batch_removals` shows stalls dominated by atomics on `n_accepted_tracks_per_measurement`. Explicit CSR traversal reads the graph without those atomics — the serialization moves from hardware atomics to algorithmic rounds, which is GPU-friendlier.

**Do *not* implement 2c if**:

- PBG already yields ≥ 2× over baseline and `track_overlap_vs_cpu ≥ 0.98`. In that regime the extra CSR build + graph-algorithm kernel is almost certainly slower than just running PBG with a larger window.
- `|E|_upper` at the largest n we care about exceeds device memory budget for the intermediate buffers (empirically ~400 MB with `max_sharing_cap = 5` and `n_input = 10000`; this is fine on a GV100 but should be checked on smaller cards).

---

## 7. What the follow-up branch would need to add

Strictly as a hand-off document, to save time when a future branch picks this up:

- New kernels under `device/cuda/src/ambiguity_resolution/kernels/`:
  - `build_conflict_coo.{cu,cuh}`
  - `coo_to_csr.{cu,cuh}` (or use CUB primitives directly without a new kernel)
  - `color_conflict_graph.{cu,cuh}` (Jones–Plassmann) and/or `mis_conflict_graph.{cu,cuh}` (Luby)
  - `apply_color_class.{cu,cuh}` — reuses the propagation path that `apply_batch_removals` already uses.
- New host member on `greedy_ambiguity_resolution_algorithm`:
  - `set_explicit_conflict_graph(bool)` with mutually-exclusive semantics vs `set_parallel_batch_mode(bool)`.
- No change to `ambiguity_io`; the same `--dump-ambiguity-input` corpus serves as regression input.
- Harness flag: `--explicit-conflict-graph` on `benchmark_resolver_cuda`, reusing the `run_one` lambda plus the existing `track_overlap_vs_cpu` / `duplicate_rate_post` metric code.

No C++ produced on this branch.

---

## 8. Summary for the thesis

Tier 2a and Tier 2c are two points on a single axis:

| Axis | Tier 2a (PBG, implemented) | Tier 2c (explicit graph, future) |
|---|---|---|
| Graph representation | implicit, probed through `claimed_by[]` | explicit CSR, rebuilt or incrementally updated |
| Output per outer iteration | one batch (one independent set) | `χ` independent sets (colouring) or one MIS |
| Kernel boundaries per resolver call | `n_removals / avg_batch_size` | `χ` (colouring) or `O(log |V|)` (Luby) |
| Determinism knob | `sorted_ids` rank tie-break | same `sorted_ids` rank tie-break |
| Extra memory | `claimed_by[|M_A|] + batch_ids[W]` | `row_ptr[|V|+1] + col_idx[|E|]` |
| When it wins | conflict graph has large independent sets | conflict graph is dense enough that colouring amortizes |

The thesis will present PBG measured, this document as motivation for a natural next step, and defer 2c implementation to a follow-up branch if the hand-off criteria in Sec. 6 are met.
