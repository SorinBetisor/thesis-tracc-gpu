# Synthetic conflict-density dataset

## 1. What the synthetic dataset is

A deterministic generator that produces track candidates with tunable size and conflict density. It is not physics per se: each candidate carries only a set of measurement (hit) IDs and a random quality score. Its single purpose is a clean scaling knob so we can isolate how each resolver scales with candidate count and conflict density without per-event physics variation.

### 1.1 Generation procedure

From the resolver benchmark generator:

1. A fixed RNG seed (`std::mt19937 gen(42)`) makes every event byte-reproducible.
2. A shared measurement pool of size `max_meas_id` is created; density is set by shrinking that pool.
3. For each of the `N` candidates:
  - draw a track length `L` uniformly from the density-dependent range,
  - draw a random p-value `~U[0,1]` (this becomes the resolver's quality score and removal priority),
  - draw `L` **distinct** measurement IDs uniformly from `[0, max_meas_id]`,
  - sort them and register that hit pattern as the candidate.

The conflict density then follows from `N x avg(L) / max_meas_id`: a smaller pool forces more candidates to collide on the same measurement, which densifies the conflict graph.

### 1.2 Density presets

Two presets are used for this study:


| Density | `max_meas_id` (hit pool) | Track length | Resulting conflict structure       |
| ------- | ------------------------ | ------------ | ---------------------------------- |
| low     | 50 000                   | 3-10         | Sparse; few candidates share a hit |
| med     | 10 000                   | 3-10         | Moderate sharing                   |


The generator also exposes a `high` preset (`max_meas_id = 500`, length 5-15) that produces a near-complete conflict graph. It is deliberately excluded from the results below; Section 4 explains the graph it creates, how future work could handle it, and why no physical detector reaches it.

### 1.3 What it can and cannot claim

It gives a controlled scaling curve and lets us push past a physical detector's candidate count, which exposes more independent work than real events do. It cannot claim physics realism: the conflict graph is uniform-random, so it does not reproduce the structured, sparse conflict topology a real CKF produces. It is a scaling instrument, nothing more.

## 2. GPU results (greedy vs JP)


| Density | n    | greedy GPU (ms) | JP GPU (ms) | JP/greedy | JP iters |
| ------- | ---- | --------------- | ----------- | --------- | -------- |
| low     | 500  | 5.47            | 2.22        | 2.46x     | 2        |
| low     | 1000 | 9.11            | 2.91        | 3.13x     | 4        |
| low     | 2000 | 15.48           | 4.79        | 3.23x     | 7        |
| low     | 5000 | 25.72           | 9.95        | 2.58x     | 12       |
| med     | 500  | 8.99            | 4.04        | 2.22x     | 8        |
| med     | 1000 | 15.36           | 5.34        | 2.87x     | 11       |
| med     | 2000 | 26.21           | 10.59       | 2.47x     | 21       |
| med     | 5000 | 33.33           | 29.93       | 1.11x     | 47       |


---

## 3. Why JP wins where it wins and loses where it loses

JP replaces greedy's one-removal-per-iteration loop with a batched removal: each outer round builds the conflict graph, finds a set of mutually non-conflicting candidates (a maximal-independent-set step), and removes them all at once. This trades a per-round cost (materialise the graph, scan neighbours, re-sort priorities) for far fewer rounds. Two conditions decide whether that trade pays off:

1. **Does the graph stay sparse as it grows? (in realistic physics datasets [Geant4 or FATRAS] the graph does stay sparse as pileup grows):** JP's round count is small only when a large independent set can be removed each round. At **low density** the graph is sparse and JP converges in a handful of rounds. As density rises the largest independent set per round shrinks, so the round count climbs (8-47 at medium density) and JP's per-round savings erode toward break-even (1.11x at medium n=5000). Each round still pays the full graph-materialise-and-scan cost, so a denser graph directly costs JP; pushed far enough this becomes the dense-graph pathology described in Section 4.
2. **The GPU makes each round cheap.** The batched removal maps onto thousands of threads, so the per-round graph work is nearly free and the collapse in round count is pure profit. This is why JP wins 2.5-3.2x at low density and up to medium n=2000 on the GV100, whereas greedy's inherently sequential one-track-at-a-time removal cannot use that parallelism.

---

## 4. The excluded high-density regime & why enough scaling

### What graph it creates

The `high` preset shrinks the hit pool to 500 IDs, so with 500-5000 candidates each measurement is claimed by roughly 20 or more tracks. The conflict graph becomes a dense, uniform-random graph whose average degree grows with the candidate count instead of staying bounded: almost every candidate conflicts with almost every other, approaching a complete graph.

### Why it breaks JP

On a near-complete graph the largest independent set per JP round collapses toward a single vertex, so JP removes only a handful of candidates per round and the number of outer rounds grows with n. Because every round still re-materialises and re-scans the whole graph, JP degenerates into near-sequential removal carrying full-graph overhead per step, and its GPU runtime blows up super-linearly while greedy stays in the tens of milliseconds. This is a property of the graph, not the device.

### How future work could mitigate it (if needed)

- **Density-adaptive fallback:** estimate the average degree or edge count cheaply when the graph is built, and fall back to greedy once it exceeds a threshold, so JP is only used where it pays off.
- **Stronger per-round removal:** a randomised colouring step, or accepting several independent colours per round, removes more vertices per round and cuts the round count on dense graphs.
- **Incremental graph updates:** avoid rebuilding the full CSR each round; update only the neighbourhood of removed vertices so the per-round cost falls even when many rounds are unavoidable.

### Why real data never reaches it

Real detector conflict graphs stay sparse even at high occupancy: shared hits are local, a track only overlaps a few geometric neighbours, so the average degree stays bounded and does not scale with candidate count. 

The high-density preset therefore maps to no physical geometry; it exists only to locate the boundary where the method breaks.

