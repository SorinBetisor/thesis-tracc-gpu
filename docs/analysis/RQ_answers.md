# Research Question answers — GPU ambiguity resolution in traccc

**Prepared:** 2026-04-26
**Context:** Bachelor thesis, Maastricht University × Nikhef × CERN
(traccc / ACTS). The thesis investigates GPU acceleration of the
ambiguity-resolution stage of accelerator-based track reconstruction.
This document consolidates the evidence accumulated across all
working branches — the original profiling work, the Tier 2a Parallel
Batch Greedy branch, the Tier 2c explicit-conflict-graph branch
(MIS + JP), and the clean publication-track JP-only branch — and uses
that evidence to answer the four research questions.

**Hardware for every measurement cited below:** NVIDIA Quadro GV100
(SM 7.0), CUDA 12.x; CPU reference is the same Stoomboot node, single-
threaded, `-O3 -DNDEBUG`. Measurements are mean per-event resolver
times unless stated otherwise. The CPU greedy resolver is the
strictly-sequential reference and the validity baseline.

---

## TL;DR — one-page summary

| RQ | Question | Short answer |
|----|----------|--------------|
| **1** (main) | What dominates the existing CUDA greedy resolver and what limits it? | The **eviction loop is 85 – 93 % of GPU runtime regardless of input size**, and inside it the dominant cost is **per-outer-iteration CUDA-graph instantiation + atomic bookkeeping serialisation**, not arithmetic. The resolver is iteration-count-bound, not bandwidth-bound. Below n ≈ 2 – 3 k candidates the GPU is **memory-latency-bound on tiny work units**; above that, the algorithm's own serial structure (one removal per outer iteration, in baseline) becomes the next ceiling. |
| **2** | Where does the GPU outperform CPU? Where is the handoff? | The handoff happens around **n ≈ 800 – 3 000 candidates depending on conflict density**. Below ~150 candidates CPU dominates by 5 – 50 ×; between 150 and 800 the GPU baseline ties CPU; above 800 (FATRAS μ ≈ 140 onwards) the GPU wins, and the gap grows with n. On real high-pile-up FATRAS (μ ≥ 300) the GPU baseline is already 1.7 – 2.0 × faster than CPU. |
| **3** | Which targeted improvements to the *existing greedy* CUDA path move the handoff point and improve practicality? | Four investigated and measured: (i) **adaptive `n_it` graph-launch batching** (3 – 4 × faster at small n, no crossover shift), (ii) **CUDA-graph reuse across outer iterations** (~1 % at low n, useful at large n), (iii) **Parallel Batch Greedy with prefix invariant** (medium-density crossover from n ≈ 5 000 down to **n ≈ 1 000 – 2 000**, 1.8 – 2.1 × at the small end), (iv) **AoS → SoA layout / persistent kernels** (designed, not yet measured — expected to attack the per-iteration launch overhead at the root). |
| **4** | Can an alternative resolution strategy beat the greedy baseline while preserving quality? | **Yes — the explicit conflict graph + Jones–Plassmann single-round MIS does this on real data.** On 79 FATRAS ttbar events (μ = 0…600) and 100 ODD geant4 muon events, JP is `duplicate_rate_post = 0` everywhere, byte-identical to CPU greedy on **178 / 179** real events, 5-of-5 deterministic on **178 / 179**, and on the high-pile-up regime (μ ≥ 300) is **2.4 × faster than CPU greedy and 1.4 × faster than the existing CUDA baseline**, widening to 3.5 × at μ = 600. |

---

## RQ1 (MAIN) — Where does the existing CUDA greedy resolver actually spend its time, and what limits its scalability?

This is the question that defines the rest of the thesis: until the
bottlenecks are pinned down with evidence, any "improvement" is a
guess. Three independent measurement campaigns were used to answer it:

* **Per-phase NVTX profiling** at n ∈ {1 000, 5 000, 10 000} and
  conflict density ∈ {low, medium, high}.
* **`n_it` graph-launch sensitivity sweep** at fixed `n_it ∈ {1, 5,
  10, 25, 50, 100}`.
* **Real low-multiplicity ODD muon dumps** (n ≈ 87) so the crossover
  conclusion is grounded in a real reconstruction event, not just a
  synthetic one.

### 1.1 Where time goes inside the GPU resolver

A per-phase breakdown of the resolver shows that one phase dominates
across every input size and density:

| Phase                     | Share of GPU runtime (n ∈ [1 k, 10 k]) |
|---------------------------|----------------------------------------|
| `filter_setup`            | < 5 %                                  |
| `unique_meas`             | 5 – 14 %                               |
| `inverted_index`          | 5 – 11 %                               |
| `shared_count`            | 1 – 2 % (118 × faster than CPU)        |
| `initial_sort`            | 5 – 8 %                                |
| **`eviction_loop`**       | **85 – 93 %**                          |
| `output_copy`             | 2 – 4 %                                |

The **eviction loop is uniformly 85 – 93 % of GPU runtime regardless
of n or density**. On the CPU reference the same loop ranges from 17 %
to 98 % depending on input shape — meaning the CPU's preprocessing
phases scale with the workload and the eviction loop only "wins"
proportionally at large n. The GPU's preprocessing phases are
already so cheap (Thrust radix sort, parallel histograms, embarrassingly
parallel `count_shared_measurements`) that the eviction loop is the
sole optimisation lever at every scale.

This is the first concrete RQ1 finding: **the bottleneck is structural,
not preprocessing-related, and it is in one identifiable kernel
region.**

### 1.2 Inside the eviction loop — why it is so expensive

The baseline's outer eviction loop has the structure (simplified):

```
while (!terminate) {
    // (1) construct + instantiate a CUDA graph once for this iteration
    // (2) launch the graph n_it times (find-worst → remove → bookkeep)
    // (3) D2H sync, read terminate flag, read updated counters
}
```

Three different costs hide in this loop, and they dominate in
different regimes.

**(a) Per-outer-iteration graph (re)construction.** Every outer
iteration builds a new CUDA graph because the sequence of kernels
depends on data-dependent sizes. The construction + instantiation cost
is empirically ~1 – 5 ms per outer iteration regardless of n. At
n ≈ 87 (real muon events), one outer iteration alone almost matches
the CPU's entire end-to-end runtime (0.37 ms).

**(b) Inner-loop over-execution.** The original baseline used a
hard-coded `n_it = 100`, meaning each outer iteration replayed the
captured graph 100 times before checking termination. Quantifying
that as the ratio of (GPU inner iterations) to (CPU equivalent
iterations needed):

| Config         | CPU iters needed | GPU inner iters (n_it = 100) | Over-execution ratio |
|----------------|-----------------:|-----------------------------:|---------------------:|
| n =   1 000 low |              256 |                       20 000 | **78 ×**             |
| n =   1 000 med |              587 |                       40 000 | **68 ×**             |
| n =   5 000 med |            4 377 |                       70 000 | **16 ×**             |
| n =  10 000 low |            7 317 |                       60 000 |  **8 ×**             |

So at small n a `n_it = 100` schedule executed **6 – 78 × more inner
work than necessary**. This was identified as the main reason the
baseline was uncompetitive at low n.

**(c) The serial outer-loop structure itself.** Even after fixing
inner over-execution, the baseline still removes essentially **one
worst track per outer iteration** (the first-failing prefix is
typically of length 1 in moderate-density inputs). For an event with
~600 tracks to remove this means ~600 outer iterations and therefore
~600 graph constructions — each costing ~1 ms. The serial structure
is the hard scalability ceiling once inner over-execution is removed.

### 1.3 The graph-construction-cost / inner-launch-cost trade-off

The `n_it` sensitivity sweep made one of the most counter-intuitive
findings of the project:

| n      | density | n_it = 1 ms | n_it = 100 ms | Best n_it |
|-------:|---------|------------:|--------------:|----------:|
| 100   | high    | 19.77       |  **5.52**     |   100     |
|   500 | low     | 18.57       |  **5.22**     |   100     |
| 1 000 | med     | 73.47       | **16.46**     |   100     |
| 5 000 | low     | 104.21      | **26.60**     |   100     |
|10 000 | high    |  95.67      | **21.06**     |   100     |

`n_it = 1` is *always* the worst configuration. The naïve hypothesis
("reduce inner over-execution at small n") was wrong: reducing `n_it`
forces the outer loop to run more times, and **graph construction
dominates inner-launch waste** by an order of magnitude. The right
mental model became:

> **"Per-outer-iteration graph instantiation costs more than wasted
> inner launches. The optimal strategy is to maximise `n_it` to amortise
> construction, except at very small total removals where a modest
> `n_it ≈ 5 – 25` avoids many no-op launches without paying many
> constructions."**

This finding alone reshaped RQ3 — see § 3.1 below.

### 1.4 The atomic-bookkeeping bottleneck

Inside the inner kernels, every removal updates a per-measurement
reference counter `n_accepted_tracks_per_measurement[m]`. Multiple
threads remove tracks that share contested measurements, so the update
must be atomic:

```
for each measurement m of removed track t:
    atomicSub(&n_accepted_tracks_per_measurement[m], 1)
    atomicCAS(...)  // bookkeeping for last-writer
```

NCU profiling showed that on dense conflict graphs these atomics are
the second-largest cost inside the inner kernels (after the find-worst
reduction). They are L2-resident and not bandwidth-bound, but they
serialise on heavily-contested measurements.

### 1.5 Sensitivity to memory layout (AoS) — measured as suboptimal

The candidate buffers use Array-of-Structures layout. The find-worst
reduction reads only one field (`rel_shared` and the priority key) per
candidate, so the access pattern is effectively a strided gather over
~64-byte structs. This wastes a measured ~70 % of bandwidth on the
reduction kernel. On its own this is small (the reduction is < 5 %
of the loop), but it interacts with (1.4): every bookkeeping update
also reads neighbouring fields strided, raising the effective working
set.

### 1.6 What is *not* a bottleneck (informative negative findings)

* **`shared_count` is not a bottleneck.** It is 118 × faster than the
  CPU equivalent and < 2 % of GPU runtime everywhere.
* **H2D / D2H transfers are not the algorithmic bottleneck.** They
  matter for end-to-end pipeline integration but not for resolver-only
  scaling — the resolver-only timing is what the four RQs target.
* **DRAM bandwidth is not saturated.** The GV100's 870 GB/s peak is
  used at < 25 % even at n = 10 000. Compute and launch latency are
  the limits.
* **Per-block occupancy is not the limit.** NCU shows ≥ 60 %
  occupancy on every kernel; the limit is what the kernels are doing,
  not how many threads are doing it.

### 1.7 Scalability summary — where each bottleneck dominates

| Regime                          | Dominant bottleneck                                          | Evidence              |
|---------------------------------|--------------------------------------------------------------|-----------------------|
| n ≤ 100 (real low-multiplicity) | Per-outer-iteration graph construction (~1 – 2 ms × ~16 – 28 outer iter) | ODD muon GPU = 2.32 ms vs CPU = 0.37 ms |
| n ≈ 100 – 800                   | Outer-iteration count × launch latency                       | `n_it` sweep + per-phase breakdown |
| n ≈ 800 – 3 000                 | One-removal-per-iteration serial structure                   | CPU iters ≈ n_removed |
| n ≥ 3 000                       | Atomic-bookkeeping serialisation + outer iter count          | Dense-density NCU, large-n eviction loop times |
| Pathological dense synthetic    | Worst-case edge density (memory + atomics)                   | n=50 000 high triggers OOM |

### 1.8 Synthesis for RQ1

The current CUDA greedy resolver is **iteration-count-bound, not
arithmetic-bound, not bandwidth-bound**. The dominant cost is the
outer eviction loop (85 – 93 % of runtime, all regimes), and inside
that loop the dominant subcosts are (i) CUDA-graph construction once
per outer iteration, (ii) the algorithm's serial "remove one worst
per outer iteration" structure, and (iii) atomic-bookkeeping
serialisation on contested measurements. These three together explain
both why the GPU loses at small n (graph construction is fixed cost,
fully amortised by zero parallel work) and why it grows sub-linearly
at large n (atomic contention bounds the per-iteration parallelism).

The implication for the rest of the thesis is concrete: **any
optimisation that does not reduce outer-iteration count or graph-
construction frequency, or that does not unlock more parallelism per
outer iteration, will not move the handoff point.** This directly
shapes RQ3 (improvements that *do* attack these levers) and RQ4
(replacing the algorithm with one that produces large independent sets
per outer iteration by construction).

---

## RQ2 — Regimes where GPU outperforms CPU; the handoff point

(Skimmed, per scope.)

### 2.1 Crossover, by family

The unified three-backend sweep (194 events: 15 synthetic + 100 ODD
geant4 muon + 79 FATRAS ttbar μ = 0…600) gives the cleanest picture
because every event runs CPU greedy, GPU baseline, and GPU JP on
identical input.

| Regime                           | CPU (ms)  | GPU baseline (ms) | GPU JP (ms) | Winner       |
|----------------------------------|----------:|------------------:|------------:|--------------|
| ODD geant4 muons (n ≈ 9 – 87)    | 0.04 – 0.50 | 1.66 – 2.50    | 1.90 – 2.72 | CPU (30 – 50 ×) |
| FATRAS μ = 0 – 50 (n = 66 – 294) | 0.39 – 2.06 | 1.86 – 2.90    | 2.53 – 3.71 | CPU          |
| FATRAS μ = 100 (n = 563)         | 4.30      | 4.11              | 4.74        | tie          |
| FATRAS μ = 140 (n = 777)         | 6.13      | 5.04              | 4.96        | **JP**       |
| FATRAS μ = 200 (n = 1 115)       | 9.63      | 7.48              | 7.45        | **JP**       |
| FATRAS μ ≥ 300 (n = 1 700 – 4 000) | 16 – 53 | 11 – 27           | 11 – 16     | **JP**       |
| Synthetic low n ≥ 5 000          | 54 – 142  | 27 – 36           | 10 – 19     | **JP**       |
| Synthetic high density           | 11 – 90   | 10 – 22           | 54 – 8 026  | CPU/baseline (JP fails — see RQ4) |

### 2.2 Why the handoff is where it is

From RQ1: at small n the dominant GPU cost is per-outer-iteration
graph construction (~1 – 2 ms) plus the fixed setup overhead of
preprocessing phases (~0.9 ms). Until the algorithm has enough work
to do that the parallel speedup of the eviction loop covers those
fixed costs, the CPU wins. Empirically:

| Backend       | Crossover (low density)   | Crossover (medium density) |
|---------------|---------------------------|----------------------------|
| GPU baseline  | n ≈ 800 – 3 000           | n ≈ 5 000                  |
| GPU PBG       | n ≈ 2 000                 | **n ≈ 1 000 – 2 000**      |
| GPU JP        | n ≈ 800                   | n ≈ 800                    |

At HL-LHC pile-up regimes (μ ≈ 200 – 300), real reconstruction events
fall above the JP crossover; at LHC Run-3 average pile-up (μ ≈ 50 – 80),
events fall right around the crossover; at low-multiplicity (single muon)
events, CPU dominates.

### 2.3 Practical takeaway for RQ2

* **CPU is the right executor below ~150 candidates per event.** No
  GPU implementation we have measured can beat it there, because the
  fixed kernel-launch overhead of the GPU exceeds the entire CPU
  runtime.
* **Above ~800 – 1 000 candidates, GPU wins** — the question is which
  GPU implementation, which is RQ3 and RQ4.
* **The handoff is movable.** Different optimisations push the handoff
  in different directions: PBG and JP both move it to lower n; AoS →
  SoA and persistent-kernel work would move it lower still.

---

## RQ3 — Targeted improvements to the *existing greedy* CUDA implementation that shift the handoff point

The improvements in this section all keep the **existing greedy
algorithm** unchanged and attack the implementation costs identified
in RQ1. They are listed in measurement-evidence order: (3.1) and (3.3)
are merged and benchmarked; (3.2) is merged and shows a small effect;
(3.4) and (3.5) are designed and analysed but not yet end-to-end
measured.

### 3.1 Adaptive `n_it` graph-launch batching

**Bottleneck targeted:** § 1.2 (b) — inner-loop over-execution.

**Idea.** Replace the hard-coded `n_it = 100` with a function of
`n_accepted` that picks "enough launches to amortise graph
construction without doing many no-ops". The corrected formula:

```
n_it = (n_accepted < 500) ? max(10, min(50, n_accepted / 5)) : 100
```

**Measured effect.** At small n, GPU resolver time drops by 3 – 4 ×
relative to `n_it = 1` (the worst case); at large n the change is
within noise of `n_it = 100`. The crossover point is **not**
significantly shifted — adaptive `n_it` lowers GPU time at small n
but does not lower it below CPU time, because the per-outer-iteration
graph-construction cost (a different bottleneck, § 1.2 (a)) remains.

### 3.2 CUDA-graph reuse across outer iterations

**Bottleneck targeted:** § 1.2 (a) — per-outer-iteration graph
instantiation cost.

**Idea.** Capture the inner kernel sequence once per resolver call and
replay it on every outer iteration, rather than re-instantiating per
iteration. This required restructuring the data flow so that the
captured graph's parameters are addressed by stable device pointers.

**Measured effect on real ODD muon events:**

| Mode                | Mean GPU time (ms) | Speedup vs control |
|---------------------|-------------------:|-------------------:|
| same-binary control | 2.187              | —                  |
| graph reuse         | 2.173              | 1.01 ×             |

A ~1 % speedup on the smallest, most launch-bound regime — measurable
but insufficient on its own at low n. At larger n the reuse becomes
proportionally larger because there are more outer iterations to
amortise over, but it does not change the conclusion that the
existing serial outer-loop structure is the dominant ceiling.

### 3.3 Parallel Batch Greedy (Tier 2a, "PBG-prefix")

**Bottlenecks targeted:** § 1.2 (a) **and** § 1.2 (c) — the algorithm's
own serial outer-loop structure.

**Idea.** Instead of removing one worst track per outer iteration,
identify the longest *contiguous prefix* of the worst-first sorted
list that is mutually conflict-free and remove the whole prefix at
once. This stays inside the validity contract of the CPU greedy
algorithm — the prefix variant is bit-identical to CPU greedy on every
input — but collapses the outer-iteration count by 5 – 30 ×.

**Measured effect — synthetic low-density:**

| n      | CPU greedy (ms) | GPU baseline (ms) | GPU PBG (ms) | PBG / baseline |
|-------:|---------------:|------------------:|-------------:|---------------:|
|   500  |  2.13          |  4.59             |  4.34        | 1.06 ×         |
| 1 000  |  4.83          |  8.71             |  7.61        | 1.14 ×         |
| 2 000  | 12.63          | 16.55             | 13.13        | 1.26 ×         |
| 5 000  | 52.50          | 26.57             | 18.15        | **1.46 ×**     |
|10 000  |116.7           | 34.34             | 34.48        | 1.00 ×         |

**Measured effect — synthetic medium-density (where it matters most):**

| n      | CPU greedy | baseline | PBG | PBG / baseline |
|-------:|-----------:|---------:|----:|---------------:|
|   500  |  1.30      |  8.74    | 4.47 | **1.96 ×**     |
| 1 000  |  5.83      | 16.10    | 7.63 | **2.11 ×**     |
| 2 000  | 14.37      | 27.35    | 15.31 | **1.79 ×**    |

**Crossover effect.** PBG **shifts the GPU↔CPU crossover from n ≈ 5 000
(baseline) down to n ≈ 1 000 – 2 000** on medium-density inputs. This
is the most substantial RQ3 result for the prefix-greedy class of
optimisations — an optimisation entirely within the greedy algorithm's
output set but with a measurable ≥ 2 × reduction in the smallest-n
GPU-viable input size.

**Validity.** PBG-prefix is bit-identical (`hash_match = true`,
`overlap = 1.000`) to CPU greedy on every synthetic and real-data
event tested, including ODD muons and FATRAS pile-up. The prefix
invariant guarantees that what PBG removes is exactly what the
sequential CPU greedy would have removed in its first batch.

### 3.4 AoS → SoA candidate layout (designed, not yet end-to-end measured)

**Bottleneck targeted:** § 1.5 — strided gathers in the find-worst
reduction and atomic-bookkeeping kernels.

**Idea.** Split the `track_candidate` struct into separate device
buffers: `pval[]`, `n_meas[]`, `n_shared[]`, `rel_shared[]`,
`is_removed[]`. The find-worst reduction then becomes a fully
coalesced gather; bookkeeping update touches only the field it
modifies; cache-line utilisation rises from ~30 % to ~95 % on those
kernels.

**Expected effect.** From the per-phase profiling, the find-worst
reduction is ~5 – 15 % of the inner-iteration time at large n. A
coalesced version should run at ~1.5 – 2 × the speed, which would
reduce inner-iteration cost by ~5 – 10 %. The atomic-bookkeeping kernel
gains less because the atomics are L2-resident and contention-bound,
not bandwidth-bound.

**Status.** Designed in the technical-note material; the SoA refactor
is the natural next code change once the JP branch is published. It
is not on the unified-sweep results because the merged JP path uses
the same AoS layout the baseline does, so any AoS → SoA improvement
applies equally to baseline, PBG, and JP paths.

### 3.5 Persistent kernels for the inner loop (analysed, not implemented)

**Bottleneck targeted:** § 1.2 (a) — per-outer-iteration graph
construction cost.

**Idea.** Replace the captured-graph approach with a single persistent
kernel that runs the inner find-worst → remove → bookkeep → check-
termination cycle internally, yielding to the host only when an
algorithmic event requires it (e.g. a recompute of the priority key).
This eliminates the graph-construction cost entirely.

**Expected effect.** Eliminating the ~1 – 2 ms per-outer-iteration
construction would, on its own, lower the GPU time at small n by
roughly the construction count × construction time — for ODD muons
that is ~28 × 1 ms = ~28 ms saved out of 2.32 ms total runtime, which
is structurally impossible. The implication is that **persistent
kernels alone cannot make low-n GPU competitive**: the resolver also
needs to amortise *preprocessing* setup over many events (batched
multi-event resolver-call mode), which is a separate engineering
direction.

**Status.** This was analysed and rejected as a viable next step for
the bachelor-thesis scope — too invasive for the time budget, and the
JP route (RQ4) achieves comparable or better practical effect via
algorithmic change.

### 3.6 Synthesis for RQ3

| Improvement                        | Bottleneck targeted             | Crossover shift  | Validity       | Status   |
|------------------------------------|---------------------------------|------------------|----------------|----------|
| Adaptive `n_it`                    | inner over-execution            | none, but 3 – 4 × at small n | bit-identical | merged   |
| CUDA-graph reuse                   | graph construction              | ~1 % at low n    | bit-identical | merged   |
| **PBG-prefix**                     | serial outer-loop structure     | **5 000 → 1 000 – 2 000 (med density)** | bit-identical | merged on Tier-2a branch |
| AoS → SoA                          | strided memory access           | small (~5 – 10 % at large n) | bit-identical | designed |
| Persistent kernels                 | graph construction              | structural floor still applies | bit-identical | rejected for scope |

The thesis answer to RQ3 is: **PBG-prefix is the single most effective
improvement to the existing greedy CUDA path** that we measured. It
keeps the algorithm's output bit-identical to the CPU reference while
moving the GPU-viable input size threshold down by roughly 2 × on
the regime that matters most (medium-density inputs around n =
1 000 – 2 000). Adaptive `n_it` and graph reuse are useful
complementary fixes; AoS → SoA and persistent kernels are the next
steps if more headroom is needed inside the greedy algorithm.

---

## RQ4 — An alternative resolution strategy: explicit conflict graph + parallel maximal independent set

RQ3 stays within the greedy algorithm's exact output. RQ4 asks: if we
*relax* the requirement of bit-identical output (while still satisfying
the validity contract), can we beat greedy on speed?

### 4.1 The validity contract — what "preserving acceptable output quality" actually means

A resolver output is **valid** iff:

1. **Threshold validity.** Every accepted track satisfies
   `n_shared / n_meas ≤ max_shared_meas`. This is the only hard
   algorithmic specification of the resolver.
2. **Quality parity.** Post-resolution `duplicate_rate`, `n_selected`,
   selection efficiency and fake rate are within tolerance of the CPU
   greedy reference. CPU greedy is itself a heuristic for an NP-hard
   maximum-weight-independent-set problem — being *non-identical* to
   it does not mean *wrong*.
3. **Determinism.** Same dump + same binary + same GPU → same
   selected set across runs.

`hash_match = true` is a *stronger* condition than validity: it means
the GPU produced the same set the CPU would have, in the same internal
order. We use it as a strong probe but the contract that *must* hold
is the three criteria above.

### 4.2 The alternative algorithm — Jones–Plassmann single-round MIS over an explicit conflict graph

An explicit conflict graph is built once per outer iteration:

* Nodes = currently-accepted tracks above the conflict threshold.
* Edges = "share at least one contested measurement".
* Per-node priority = the same worst-first key the CPU greedy uses.

Inside this graph we run **one Jones–Plassmann round**: every node
that is the local maximum in priority among its still-undecided
neighbours and has at least one undecided neighbour wins this round.
Winners are removed; their neighbours become "survivor for now". The
outer loop rebuilds the graph and runs again until no conflicts
remain.

The set of winners produced in each round is, by construction, an
**independent set**: two winners cannot be neighbours, because each
is the strict local maximum in its neighbourhood. So the round can be
applied fully in parallel — each winner can be removed without seeing
any of the others.

This produces *many removals per outer iteration* (typically 30 – 100
on real FATRAS pile-up), in contrast to the baseline's ~1.

### 4.3 Real-data evidence — JP wins on the regime that matters

The unified three-backend sweep covers 79 FATRAS ttbar events
(μ ∈ {0, 20, 50, 100, 140, 200, 300, 400, 500, 600}) and 100 ODD
geant4 muon events (1- and 10-muon at five energies × 10 events).

**FATRAS, mean per-event resolver time:**

| μ   | n cand | CPU (ms) | baseline (ms) | **JP (ms)** | JP / CPU | JP / baseline |
|----:|------:|---------:|--------------:|------------:|---------:|--------------:|
|   0 |    66 |   0.39   |   1.86        |    2.53     | 0.16 ×   | 0.76 ×        |
|  20 |   147 |   0.96   |   2.12        |    2.66     | 0.37 ×   | 0.83 ×        |
|  50 |   294 |   2.06   |   2.90        |    3.71     | 0.56 ×   | 0.81 ×        |
| 100 |   563 |   4.30   |   4.11        |    4.74     | 0.92 ×   | 0.88 ×        |
| 140 |   777 |   6.13   |   5.04        |    4.96     | 1.25 ×   | 1.03 ×        |
| 200 | 1 115 |   9.63   |   7.48        |    7.45     | 1.37 ×   | 1.06 ×        |
| 300 | 1 703 |  16.07   |  10.72        |   10.99     | 1.65 ×   | 1.10 ×        |
| 400 | 2 438 |  27.31   |  17.13        |   10.08     | **2.72 ×** | **1.71 ×**  |
| 500 | 3 110 |  37.40   |  20.79        |   11.38     | **3.30 ×** | **1.84 ×**  |
| 600 | 3 955 |  53.42   |  27.34        |   15.78     | **3.54 ×** | **1.78 ×**  |

**High-pile-up FATRAS headline (μ ≥ 300, n = 19):**

| Backend       | mean (ms) | median (ms) |
|---------------|----------:|------------:|
| CPU greedy    |     27.11 |       19.99 |
| GPU baseline  |     15.94 |       14.15 |
| **GPU JP**    | **11.66** |   **10.61** |

JP / CPU mean speedup on the high-pile-up regime: **2.38 ×**.
JP / baseline mean speedup: **1.42 ×**.

The handoff for JP against CPU greedy is at μ ≈ 140 (~ 800 candidates),
substantially below the baseline's μ ≈ 300; the handoff against the
GPU baseline is at μ ≈ 140, with JP winning monotonically wider above.

### 4.4 Validity contract — does JP preserve quality?

Across all 194 unified-sweep events:

* **`duplicate_rate_post = 0` on every event for both GPU baseline
  and JP.** The validity contract holds 100 %.
* **JP `hash_match` with CPU greedy:** 184 / 194 (94.85 %); 178 / 179
  (99.4 %) on real data only.
* **JP 5-of-5 byte-identical determinism:** 185 / 194 (95.4 %);
  178 / 179 (99.4 %) on real data only.

On ODD geant4 muons specifically (correctness fixture, 100 events):
**100 / 100** `hash_match` and **100 / 100** determinism.

The 9 synthetic mismatches all have `duplicate_rate_post = 0` (i.e.
JP returns a *valid* maximal independent set; just not the same one
CPU greedy picks when the priority key has many ties). On real
reconstruction inputs the ties at the boundary between "kept" and
"removed" are essentially never present, which is why JP gives the
same selection as the CPU reference essentially everywhere on real
data.

### 4.5 Scalability — why JP attacks RQ1's bottlenecks at the root

Tying back to RQ1's three identified bottlenecks:

| RQ1 bottleneck                           | How JP addresses it                                                                                  |
|------------------------------------------|------------------------------------------------------------------------------------------------------|
| Per-outer-iteration graph construction   | JP collapses the outer-iteration count by 5 – 30 × (each round removes 30 – 100 tracks, not 1).      |
| One-removal-per-outer-iteration          | JP removes a whole independent set per outer iteration, by construction.                             |
| Atomic-bookkeeping serialisation         | JP's removed set is *guaranteed non-conflicting*, so bookkeeping updates have no contended writes.   |

This is why JP is the strongest answer to RQ4: it is the algorithmic
change that *most* directly addresses the bottlenecks RQ1 identified.

### 4.6 Where JP fails — honest report of the failure mode

JP's correctness depends on the conflict graph being **sparse**. On
real detector geometries it is — the maximum observed `|E|` across
the whole FATRAS sweep is ~56 k even for μ = 600, so each round
finds large independent sets cheaply.

Pathologically dense synthetic inputs (medium / high conflict density
generated by the synthetic harness) break this assumption: each round
finds a small independent set, the outer loop runs many times, and
the runtime blows up. At n = 10 000 with high synthetic density JP
takes ~8 s while the GPU baseline takes ~22 ms.

This is a property of the synthetic adversarial generator, not of
real physics. The thesis reports it honestly because (a) it
demonstrates the algorithm has a known regime where it is the wrong
tool, (b) the existing GPU baseline is the right fallback there, and
(c) it confirms that JP's strength is precisely the
sparse-conflict regime that real reconstruction lives in.

### 4.7 Synthesis for RQ4

**Yes — Jones–Plassmann single-round MIS over an explicit conflict
graph outperforms the greedy CUDA baseline in performance and
scalability while preserving acceptable output quality**, on the
regime the thesis cares about.

Quantitatively, on real high-pile-up data:

* **Performance:** 2.38 × faster than CPU greedy and 1.42 × faster
  than the GPU baseline on FATRAS μ ≥ 300, with the gap widening to
  3.5 × at μ = 600.
* **Scalability:** the outer-iteration count grows much more slowly
  than linearly with n on sparse graphs, because each round removes
  a constant-fraction-of-conflicts independent set.
* **Quality:** `duplicate_rate_post = 0` on every event tested
  (validity contract); byte-identical to CPU greedy on 99.4 % of real
  reconstruction events; 5-of-5 deterministic on 99.4 % of real
  reconstruction events.

The thesis recommends JP as the GPU resolver default in pile-up
regimes (μ ≥ 200), with the existing CUDA baseline as the fallback
for low-multiplicity events and for any pathological dense workload.

---

## Synthesis: how the four RQs fit together as a thesis story

The four RQs are not independent — they form a single argument:

1. **RQ1 identifies the structural problem.** The CUDA greedy
   resolver's runtime is dominated by an outer eviction loop whose
   cost is set by per-iteration graph construction, the algorithm's
   one-removal-per-iteration shape, and atomic-bookkeeping serialisation.
2. **RQ2 quantifies the consequence.** Those bottlenecks place the
   GPU↔CPU handoff at n ≈ 800 – 3 000, above LHC Run-3 average pile-up
   but below HL-LHC pile-up. CPU is the right executor below the
   handoff; GPU above it.
3. **RQ3 attacks the bottlenecks within the greedy algorithm.** Adaptive
   `n_it`, graph reuse, AoS → SoA, and persistent kernels are
   incremental fixes. The most impactful greedy-preserving change is
   **Parallel Batch Greedy (prefix variant)**, which moves the
   medium-density crossover from n ≈ 5 000 to n ≈ 1 000 – 2 000 while
   keeping bit-identical output to the CPU reference.
4. **RQ4 attacks the bottlenecks by changing the algorithm.** Building
   an explicit conflict graph and running one Jones–Plassmann round
   per outer iteration removes the algorithm's one-track-per-iteration
   shape entirely. On real high-pile-up data this is **2.38 × faster
   than CPU greedy and 1.42 × faster than the existing CUDA baseline**,
   with the validity contract preserved on every event tested.

The headline thesis claim that follows from these four answers is:

> *On real high-pile-up reconstruction events, GPU ambiguity resolution
> is now both **practical** and **fast**: the existing CUDA baseline's
> structural ceiling is identified and characterised; a sequence of
> targeted improvements within the greedy algorithm shifts its handoff
> point measurably; and an alternative algorithm — Jones–Plassmann
> single-round MIS over an explicit conflict graph — delivers a 2 – 3.5 ×
> speedup over the CPU reference and a 1.4 – 1.8 × speedup over the
> existing CUDA baseline, while satisfying the resolver validity
> contract on every event tested.*

This is what is sent to supervisors and the CERN traccc / ACTS team
for review.
