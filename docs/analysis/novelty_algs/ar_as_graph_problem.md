# Ambiguity resolution as a graph problem

Explains (1) how ambiguity resolution is recast as a graph problem, (2) what an independent set and a maximal independent set (MIS) are and why they are the right primitive, (3) how and why we materialise the conflict graph, and (4) how Jones-Plassmann computes an independent set in parallel. 

---

## 1. The problem, stated for graphs

Combinatorial track finding (CKF) emits a set of candidate tracks `T`. Each candidate `t` is defined by the set of detector measurements it uses, `M(t)`. Two candidates are *in conflict* when they claim the same measurement: a single physical hit cannot belong to two real particles, so at most one of any two measurement-sharing candidates should survive. Ambiguity resolution is the task of selecting a subset of `T` that is internally consistent (no measurement is over-shared beyond a threshold) while keeping the highest-quality candidates.

Track quality is summarised per candidate by a score derived from the fit (chi2, degrees of freedom, p-value) and the shared-hit fraction `R(t) = |S(t)| / |M(t)|`, where `S(t)` is the subset of `M(t)` shared with other candidates. A candidate violates the acceptance condition when `R(t)` exceeds a configured threshold. Resolution proceeds by removing violating candidates until none remain.

The key observation is that "which candidates conflict with which" is a **binary relation over pairs of candidates**, and a binary relation is a graph. Making that graph explicit lets us reason about, and parallelise, the removal decisions.

---

## 2. The conflict graph

Let `A subset T` be the currently accepted candidates (those passing the minimum-measurement filter and not yet removed), and let

```
M_A = { m : m is used by more than one candidate in A }
```

be the *contested* measurements. Define the **conflict graph**

```
G = (V, E),   V = A,
E = { (t_i, t_j) : exists m in M_A with t_i, t_j both using m }.
```

A vertex is a candidate; an edge joins two candidates that share at least one contested measurement. The graph is therefore a direct, lossless encoding of the ambiguity: everything the resolver needs to know about interference between candidates is expressed by adjacency in `G`.

Each vertex carries a deterministic **priority** `pi(v)`, equal to its rank in the worst-first sorted candidate order. Higher `pi` means a worse candidate (larger shared fraction / lower quality), i.e. a candidate more deserving of removal. The priority is the same key the sequential greedy baseline uses to choose what to evict, which is what lets the graph methods reproduce the greedy answer deterministically (Section 6).

---

## 3. Independent sets

The reason to build `G` is that it turns "which candidates can I safely remove at the same time" into a now graph question.

**Independent set.** A set of vertices `I subset V` is *independent* if no two of its members are adjacent: for all `u, v in I`, `(u, v) not in E`. Translated back to physics: an independent set of candidates is a set that pairwise shares no contested measurement.

**Why independence is exactly the safety condition for parallel removal.** The sequential greedy resolver removes one worst candidate, then recomputes the shared-hit counts of its neighbours before choosing the next victim. That recompute is what makes it correct: removing candidate `u` can lower a neighbour `w`'s shared fraction below threshold, so `w` should no longer be removed. If we remove a batch of candidates simultaneously, correctness is preserved **only if no removal in the batch would have changed the removal decision of another member of the batch**. Two candidates can influence each other's decision only through a shared measurement, i.e. only if they are adjacent in `G`. Therefore a batch that is an *independent set* can be removed in parallel and produce the same bookkeeping outcome as removing its members one by one. Independence is not a convenience; it is the precise condition under which batch removal is equivalent to sequential removal.

**Maximal vs maximum ( a previous confusion )**

- A **maximum** independent set is a largest independent set in `G`. Computing it is NP-hard, so it is not usable inside a per-iteration hot loop.
- A **maximal** independent set (MIS) is one that cannot be enlarged: every vertex not in the set is adjacent to some vertex in the set. Maximality is a local, greedily achievable property, computable in near-linear parallel work.

We only need maximality, for two reasons. First, maximality guarantees *progress*: a maximal set of removable worst candidates leaves no removable candidate that is non-adjacent to the batch, so each outer iteration makes as much conflict-free progress as the current graph allows. Second, we do not need the single largest batch; the outer loop runs again on the survivors with a freshly rebuilt graph, so anything left undecided this iteration is handled next iteration. Maximal, not maximum, is the correct and tractable target.

---

## 4. Why materialise the graph explicitly

The sequential greedy resolver already works with the conflict relation implicitly, but it only eve queries it one victim at a time. Its outer loop removes a single worst candidate per iteration, so it runs on the order of `N_removed` outer iterations, each re-launching the full bookkeeping and compaction pipeline. On the GPU that pipeline is a sequence of small kernels whose repeated launch is the dominant cost.

Materialising `G` changes the unit of removal from one candidate to one independent set. Instead of `O(N_removed)` outer iterations we perform roughly `O(log N_removed)` in practice, because each iteration evicts a whole conflict-free batch. This directly attacks the profiled bottleneck: the number of times the expensive compaction sequence is launched collapses. The explicit graph is what makes the batch decision possible, because computing an independent set requires scanning each vertex's conflict neighbourhood, which the implicit one-victim-at-a-time index does not expose cheaply.

The trade is a per-iteration graph-construction cost. This pays off precisely when conflict graphs are sparse and batches are large (real detector pile-up), and can lose when the graph is tiny (graph build dominates, e.g. ODD muons) or pathologically dense (batches shrink)

---

## 5. How the graph is built

The graph is rebuilt every outer iteration from the inverted index the resolver already maintains, and stored in Compressed Sparse Row (CSR) form for constant-time neighbour iteration.

1. **Emit edges (COO).** For each contested measurement `m in M_A`, every ordered pair of still-accepted candidates using `m` produces a directed edge. Uncontested measurements are skipped. Directed pairs (both orientations) are emitted so that each vertex's own adjacency list sees both endpoints when the independent-set kernel scans it. The worst-case edge count is bounded a priori by `sum_{m in M_A} n_m (n_m - 1)`, with `n_m` the number of accepted candidates on measurement `m`, so buffers are pre-sized once before the loop.
2. **Convert to CSR.** Sort the coordinate edge list by source vertex, then compute row offsets by lower-bounding vertex ids into the sorted keys. The result is `row_ptr[0..|V|]` and `col_idx[0..|E|)`, where `col_idx[row_ptr[v] .. row_ptr[v+1])` is the neighbour list of `v`.

For realistic pile-up the graph is sparse: contested measurements are a minority and `n_m` is small, so `|E|` is on the order of `|V|` to a few times `|V|`. This is what makes the per-iteration rebuild affordable.

---

## 6. Jones-Plassmann: computing an independent set in parallel

Jones-Plassmann (JP) is a parallel graph-colouring heuristic, and graph colouring and independent sets are two views of the same construction: each **colour class** of a valid colouring is, by definition, an independent set (no two same-coloured vertices are adjacent). JP builds a colouring one class at a time using vertex priorities, and the *first colour class it produces is exactly an independent set of local priority maxima* - which is what ambiguity resolution needs.

The JP rule, per round over the graph:

- A vertex enters the current independent set if its priority `pi(v)` is strictly greater than that of all its still-undecided neighbours (it is a *local maximum*) **and** it has at least one undecided neighbour.
- Every undecided vertex adjacent to a just-selected vertex is deferred (it is not selected now; it will be reconsidered later).

The local-maximum rule guarantees independence: if two adjacent vertices were both local maxima, each would have to out-rank the other, which is impossible. The extra "has at least one undecided neighbour" guard is a correctness invariant specific to ambiguity resolution: without it, an isolated or already-surrounded good candidate would vacuously pass the local-maximum test and be wrongly removed.

**One round per outer iteration.** Iterating the JP rule until no vertex is undecided produces a full colouring, whose first class is a *maximal* independent set. But the ambiguity resolver's outer loop is itself the place where successive colour classes would be consumed: after removing one independent set, it rebuilds the graph on the survivors and runs again. So we run **exactly one JP round per outer iteration**. That single round yields an independent set (correct to remove in parallel) that is not necessarily maximal, and the outer loop recovers any deferred vertices on the next rebuild. On sparse real-data graphs a single round already selects a large batch, which is why one-round JP is both correct and fast in the target regime.

**Determinism.** Priorities are the worst-first ranks used by the CPU greedy baseline, and ties break lexicographically on `(pi(v), v)`. The selected batch is therefore a deterministic function of the input, and the resolver reproduces the greedy selection exactly 