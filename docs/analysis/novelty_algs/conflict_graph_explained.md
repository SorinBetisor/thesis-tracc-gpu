# Tier 2c — Plain-English Explanation

**Prepared:** 2026-04-22
**Purpose:** a plain-English companion to the more technical
[`conflict_graph_design.md`](conflict_graph_design.md) and
[`conflict_graph_results.md`](conflict_graph_results.md). This document
explains, without GPU jargon, how the two Tier 2c algorithms work and how
they compare to the CPU greedy baseline on real and synthetic data.

---

## 1. How MIS and JP work, intuitively

Both algorithms solve the same underlying problem: **pick a set of
tracks to remove where no two removed tracks share a measurement** (so
the removals don't interfere with each other and can happen in
parallel).

We represent the problem as a **graph**:

- Every **track** becomes a **node**.
- Two tracks are connected by an **edge** if they share at least one
  contested measurement (i.e. they are "in conflict").
- Every node has a **priority**: its rank in the worst-first sorted
  list. Higher priority = worse track = more likely to be removed.

### 1a. Luby-style MIS (Maximal Independent Set) — iterative "tournaments"

Think of MIS as a **tournament between neighbours**, repeated in rounds:

1. Every node looks at its still-undecided neighbours.
2. A node *wins* the round if it has the highest priority (worst track)
   among all its undecided neighbours **and** has at least one
   undecided neighbour.
3. Winners get marked `IN_MIS` — these will be removed.
4. Any node that has a winning neighbour gets marked `REMOVED_NEIGHBOR`
   — it survives this round but is not chosen now.
5. Nodes whose neighbours are all decided stay undecided and wait for
   the next outer iteration.
6. Repeat until no undecided node remains (or we hit the 32-round
   budget).

So MIS keeps running mini-tournaments until everyone has a verdict.
Each round shrinks the "undecided" pool. Typical behaviour on real data:
MIS finishes in 5–15 rounds per outer iteration.

*Analogy:* imagine everyone in a queue shouting their score. In each
round, whoever shouts the highest score in their immediate
neighbourhood wins, their neighbours step back, and we go again with
whoever is left.

### 1b. Jones–Plassmann (JP) — just one round of the same tournament

JP is structurally identical to MIS but **we stop after a single round**.

Why does that work? Because that one round already produces an
independent set by construction (MIS winners never neighbour each
other). It is just a *smaller* set — we get fewer removals per outer
iteration. But the outer loop runs again next iteration anyway, so the
remaining undecided tracks get another shot with a freshly rebuilt
graph.

*Analogy:* instead of running the tournament until it resolves, we take
one snapshot of "who is winning right now", remove those, and come back
next iteration to redo the whole thing on the survivors.

### 1c. Why JP tends to be faster on real data

- Building the graph and doing one propose/finalize pass is cheap.
- On sparse conflict graphs (like real Fatras pile-up), a single round
  already finds ~40–100 winners — about the same as MIS finds in 10+
  rounds.
- Fewer internal rounds → less wall-clock time per outer iteration.
- MIS only pulls ahead when the graph is very dense and JP's
  single-round view leaves too many undecideds, forcing many extra
  outer iterations.

---

## 2. What "overlap" means

All the GPU algorithms are judged against the **CPU greedy baseline**,
which is the strictly sequential reference: it picks the single worst
track, removes it, updates bookkeeping, picks the next worst, and so
on. It is slow but defines the "correct" answer. Two metrics measure
how close each GPU backend gets:

- **`hash_match` (byte-identical)**: `true` if the GPU backend selects
  *exactly the same set of tracks* as the CPU baseline, in the same
  internal order. A hash of the selected-track ids is compared directly
  with the CPU hash. This is the strictest possible check.
- **`track_overlap_vs_cpu` (overlap ratio)**: the fraction of tracks
  that are common to both the GPU and the CPU selections. Computed as

  ```
  overlap = |S_gpu ∩ S_cpu| / |S_cpu|
  ```

  where `S_gpu` and `S_cpu` are the sets of accepted track ids. An
  overlap of `1.0` means every CPU-selected track is also in the GPU
  selection (and vice-versa, since both selections have the same size).
  An overlap of `0.95` means 5% of the selected tracks differ — the GPU
  kept some tracks the CPU dropped, and dropped some tracks the CPU
  kept. Those "swaps" still leave a valid (conflict-free) selection,
  they just ordered the tie-breaks differently.

`hash_match = true` implies `overlap = 1.0`, but not vice-versa:
overlap of 1.0 with `hash_match = false` means the selected *sets* are
identical but the *internal ordering* differs (for example, the sort
produced by `thrust::sort` stabilizes ties differently from the
baseline's incremental insertion sort). For the physics that is
indistinguishable — both cases accept the same tracks.

An overlap close to 1.0 (say ≥ 0.99) is normally considered
indistinguishable from CPU for downstream tracking quality; values
below ~0.95 start to show up as measurable differences in efficiency
and fake rate.

---

## 3. Master comparison table

All times are mean ms of 5 timed repeats (2 warm-ups) on the same
node. CPU greedy is single-threaded (`traccc_benchmark_resolver
--backend=cpu`); GPU backends are from `traccc_benchmark_resolver_cuda`
on an NVIDIA Quadro GV100. "Overlap" is the minimum `overlap_vs_cpu`
across the events in that row (so the reported number is the
worst-case agreement with CPU in that group); JP and MIS columns show
JP/MIS overlap respectively. Cells marked **bold** are the fastest per
row.

### 3a. Real pile-up (Fatras ttbar) and ODD muons

| Dataset | n tracks | CPU greedy (ms) | GPU baseline (ms) | MIS (ms) | JP (ms) | MIS overlap | JP overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fatras μ=300 | 1 856 | 19.06 | 15.20 | 13.83 | **10.40** | 1.000 | **1.000** |
| Fatras μ=400 | 2 438 | 27.36 | 16.94 | 16.15 | **10.03** | 1.000 | **1.000** |
| Fatras μ=500 | 3 110 | 38.28 | 20.61 | 17.97 | **12.29** | 0.9995 | **1.000** |
| Fatras μ=600 | 3 955 | 52.81 | 26.76 | 23.87 | **15.20** | 0.9987 | **1.000** |
| ODD 10-muon | 87 | **0.52** | 2.39 | 2.56 | 2.47 | 1.000 | **1.000** |

Headline:

- **JP matches CPU byte-for-byte on every Fatras event tested** (12/12
  `hash_match = true`, overlap = 1.000).
- **JP is 1.8× – 3.5× faster than CPU greedy** on Fatras pile-up
  (19.06 ms → 10.40 ms at μ=300, up to 52.81 ms → 15.20 ms at μ=600).
- **JP is 1.46× – 1.76× faster than the CUDA baseline** on the same
  events.
- **MIS** is byte-identical through μ ≤ 400 and drops by ≤ 0.13% at
  higher pile-up. Still slightly faster than the CUDA baseline.
- **ODD muon events are too small to benefit from GPU** — CPU greedy
  at 0.52 ms beats everything. This is expected: the whole resolver
  only has ~87 candidates, which is less than a single thread-block of
  work on the GPU.

### 3b. Synthetic (adversarial stress test)

These inputs are produced by the synthetic generator with deliberately
elevated conflict densities (`--conflict-density=low|med`). They
exceed anything produced by a real detector geometry and are reported
as a stress test, not as the target regime.

| Dataset | n tracks | CPU greedy (ms) | GPU baseline (ms) | MIS (ms) | JP (ms) | MIS overlap | JP overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| synth low n=500    |    500 | **2.11** | 5.60  |  3.00 |  2.20 | 1.000 | **1.000** |
| synth low n=1000   |  1 000 | **4.83** | 9.46  |  4.18 |  3.04 | 0.995 | **1.000** |
| synth low n=2000   |  2 000 | 12.54 | 15.20 |  8.52 | **7.84** | 0.978 | 1.000 |
| synth low n=5000   |  5 000 | 52.31 | 26.86 | 12.62 | **10.15** | 0.834 | 0.967 |
| synth low n=10000  | 10 000 | 168.34 | 34.38 | **18.07** | 18.51 | 0.659 | 0.910 |
| synth med n=500    |    500 | **2.42** |  9.44 |  5.36 |  4.16 | 0.942 | 0.996 |
| synth med n=1000   |  1 000 | **5.48** | 16.20 |  7.88 |  5.44 | 0.855 | 0.998 |
| synth med n=2000   |  2 000 | 13.98 | 27.41 | 11.21 | **10.77** | 0.649 | 0.970 |
| synth med n=5000   |  5 000 | 45.92 | 35.06 | **22.40** | 28.75 | 0.387 | 0.881 |
| synth med n=10000  | 10 000 | **104.97** | 36.14 | 43.72 | 78.33 | 0.266 | 0.854 |

Three things to notice:

1. **Small-n synthetic is CPU-dominated**: for n ≤ 1000 the single-
   threaded CPU is faster than any GPU backend. Launch overhead and
   per-iteration kernel latency outweigh the work.
2. **JP is the fastest GPU backend on low-density synthetic through
   n = 5000**, with overlap ≥ 0.967.
3. **Quality degrades on dense synthetic at n ≥ 5000**. Both MIS and
   JP make ordering decisions that diverge from CPU's strictly-
   one-at-a-time semantics: MIS over-removes when many vertices
   simultaneously qualify as locally worst, JP under-removes when its
   single-round snapshot leaves many `REMOVED_NEIGHBOR` tracks
   undecided. In the densest cases (med density, n ≥ 5000) overlap
   drops below 0.9 and the GPU backends no longer faithfully
   reproduce the CPU reference. **This is a property of the adversarial
   generator, not of the real physics regime.**

---

## 4. The big-picture story

| Regime | Matches CPU exactly? | Fastest backend | Speedup vs CPU |
|---|---|---|---|
| Real Fatras pile-up (μ=300–600), **JP** | **Yes (100%)** | JP | **1.8× – 3.5×** |
| Real Fatras pile-up, MIS | Yes for μ ≤ 400; overlap ≥ 0.9987 above | JP (not MIS) | 1.1× – 1.3× |
| ODD single muons | Yes (100%) | CPU (events too small for GPU) | n/a |
| Synthetic low-density, n ≤ 2000 | JP yes (hash or overlap ≥ 0.995) | JP | ~1× – 1.6× |
| Synthetic low-density, n ≥ 5000 | JP overlap 0.91–0.97 | JP / MIS | 2.8× (n=5000) – 9.3× (n=10000) |
| Synthetic medium-density, n ≥ 5000 | Overlap 0.27–0.88 (stress test) | MIS | 2.0× – 2.4× |

**Bottom line:** on the data that actually matters for the thesis —
real physics pile-up events — **Jones–Plassmann produces the same
answers as CPU greedy while being about 1.8× – 3.5× faster** than the
single-threaded CPU baseline, and **1.5× – 1.8× faster than even the
existing CUDA baseline** (which was already faster than CPU).

This is the Tier 2c result the thesis leads with. MIS is kept as an
A/B option to show that the round-count choice matters, and to
demonstrate a graceful degradation pattern under adversarial stress;
JP is the recommended default.

Raw numbers underlying this table:

- CPU timings: `results/20260422_171612_conflict_graph/cpu_timings.txt`
- GPU Fatras: `results/20260422_171612_conflict_graph/fatras_sweep.txt`
- GPU ODD: `results/20260422_171612_conflict_graph/odd_sweep.txt`
- GPU synthetic: `results/20260422_171612_conflict_graph/synthetic_sweep.txt`
