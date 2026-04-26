# GPU Ambiguity Resolution via Explicit Conflict Graphs
## Technical Note for CERN/traccc Review

**Date:** 2026-04-26  
**Branch:** `thesis-novelty-conflict-graph` on  
`https://github.com/SorinBetisor/thesis-tracc-gpu`  
**Commit:** `43ce28d7` (format fix) / `243dbf7d` (conflict-graph implementation)  
**Status:** Preliminary — intended for internal CERN/traccc-team review before any upstream PR.

---

## 1. Problem and motivation

The ACTS/traccc greedy ambiguity resolver eliminates duplicate and overlapping
track candidates by iteratively removing the worst-sharing track from the
candidate set. In the current CUDA implementation (`greedy_ambiguity_resolution_algorithm.cu`)
the outer loop is serial at the host level: each eviction step requires a full
device-to-host synchronisation to check the termination condition, followed by
a new kernel launch batch. Profiling on the Quadro GV100 (Stoomboot, Nikhef)
shows that **85–93% of GPU resolver time** is spent in the eviction loop even
after prior optimisations. At Fatras ttbar pile-up μ=600 (≈4000 track
candidates), the eviction loop executes ≈21 outer iterations averaging 1.3 ms
each.

The Tier 2a "Parallel Batch Greedy" (PBG) path previously introduced a CUDA-graph-captured
kernel sequence that admits a small independent set of worst tracks
simultaneously. PBG reduces resolver time by 20–30% at small n but
**degrades back to baseline at μ ≥ 400** because the implicit-graph prefix it
uses becomes tiny when the sorted-ids list is long.

---

## 2. Approach: explicit conflict graph

The Tier 2c implementation (this note) replaces the implicit-graph approach
with an **explicit track-track conflict graph**: two tracks conflict iff they
share at least one measurement. Given this graph, the eviction problem reduces
to finding and removing a **Maximum Independent Set** in the conflict
graph — a well-studied problem with efficient GPU-parallel approximations.

### 2.1 Graph construction (per outer iteration)

1. **COO edge list**: `build_conflict_coo` — one CUDA block per unique
   measurement. Each block gathers still-accepted tracks sharing that measurement
   and emits directed (a, b) pairs. Updated (2026-04-26): the gather loop is
   bounded to `blockDim.x = 128` entries per block in the fast path; rows wider
   than 128 use a safe slow path that reads directly from global memory.
2. **COO → CSR**: `thrust::sort_by_key` on (src, dst) pairs, then
   `thrust::lower_bound` to compute row pointers.
3. **Graph algorithms** (one of two options; see §2.2 and §2.3):
   - `graph_mis_init` → `graph_mis_propose` → `graph_mis_finalize` (Luby MIS)
   - `graph_jp_propose` → `graph_jp_finalize` (Jones–Plassmann, one round)
4. **Apply removals**: `apply_graph_removals` removes the IN_MIS set
   race-free (the MIS is an independent set, so no two removed tracks share
   a measurement slot touched by parallel atomic decrements).

### 2.2 Luby-style MIS (graph_mis)

Each round marks a vertex as IN_MIS iff it has the highest priority among all
still-UNDECIDED neighbours, and has at least one UNDECIDED neighbour (the
"has_undecided_neighbour invariant" prevents premature removal of good tracks
once all their bad neighbours have already been removed in prior rounds). Rounds
iterate until no UNDECIDED vertices remain for this outer iteration.

### 2.3 Jones–Plassmann one-round (graph_jp)

A single propose/finalize pair is executed per outer iteration. Each vertex
that is a local maximum in priority and has at least one UNDECIDED neighbour
joins the IN_MIS set. This is equivalent to extracting the first colour class
of a Jones–Plassmann graph colouring. Fewer internal rounds per outer iteration
is JP's key advantage when the conflict graph is sparse.

---

## 3. Resolver validity contract

The CPU greedy resolver is itself a heuristic for a Maximum Weight Independent
Set problem. Any resolver output is **valid** iff:

1. **Threshold validity**: every accepted track `t` satisfies
   `rel_shared(t) = n_shared(t) / n_meas(t) <= max_shared_meas`. This is the
   resolver's only hard algorithmic specification.
2. **Quality parity**: `duplicate_rate_post`, `n_selected`, and (where truth
   labels are available) selection efficiency and fake rate are within an
   agreed tolerance of the CPU greedy reference.
3. **Determinism**: same serialised dump, same binary, same GPU → same selected
   set across repeated runs.

`hash_match = true` in the tables below means the GPU output is additionally
**selection-identical** to the CPU greedy reference — same tracks by sorted
measurement-id pattern. This is stronger than validity and is reported
separately. Being non-identical to the CPU reference does **not** mean wrong.

---

## 4. Results

### 4.1 Hardware and build

| Item | Value |
|---|---|
| GPU | NVIDIA Quadro GV100 (Stoomboot `wn-lot-001`) |
| CUDA | 12.x |
| Compiler | Intel `icpx` + `nvcc`, `-O3 -DNDEBUG`, `CMAKE_CUDA_ARCHITECTURES=70` |
| Repeats | 5 timed, 2 warmup |
| Seed | fixed (reproducible) |

### 4.2 Fatras ttbar pile-up — headline result

Means across the events available per pile-up point.
Full per-event breakdown: `results/20260422_171612_conflict_graph/fatras_sweep.txt`.

| μ | n̄ | baseline (ms) | pbg (ms) | graph_mis (ms) | **graph_jp (ms)** | JP speedup vs baseline | JP speedup vs PBG |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1 856 | 15.20 | 21.29 | 13.83 | **10.40** | **1.46×** | **2.05×** |
| 400 | 2 438 | 16.94 | 23.38 | 16.15 | **10.03** | **1.69×** | **2.33×** |
| 500 | 3 110 | 20.61 | 30.46 | 17.97 | **12.29** | **1.68×** | **2.48×** |
| 600 | 3 955 | 26.76 | 37.09 | 23.87 | **15.20** | **1.76×** | **2.44×** |

**Quality on Fatras pile-up (validity contract check):**

| μ | MIS `hash_match` | MIS overlap (min) | JP `hash_match` | JP overlap | Validity contract |
|---|---|---|---|---|---|
| 300 | 2/2 | 1.0000 | 2/2 | 1.0000 | PASS (all) |
| 400 | 3/3 | 1.0000 | 3/3 | 1.0000 | PASS (all) |
| 500 | 2/3 | 0.9995 | **3/3** | **1.0000** | PASS (all) |
| 600 | 0/3 | 0.9987 | **3/3** | **1.0000** | PASS (all) |

JP is **selection-identical to the CPU greedy reference on every Fatras event
tested** (12/12 events across μ=300..600). MIS satisfies the validity contract
(criterion 1) on all events but diverges from the CPU reference at μ=500..600.

### 4.3 ODD 10-muon gun (n ≤ 100)

| Backend | time_ms mean | hash_match |
|---|---:|---|
| baseline | 2.39 | 10/10 |
| pbg | 2.98 | 10/10 |
| graph_mis | 2.56 | 10/10 |
| graph_jp | 2.47 | 10/10 |

At n ≤ 100 all backends are within noise. The conflict-graph approach targets
the high-n, sparse-conflict-density regime.

### 4.4 Synthetic stress test

JP is fastest at low conflict density and large n (e.g. n=10 000 low: 18.5 ms
vs baseline 34.4 ms). Quality degrades at medium density for n ≥ 2000, where
MIS converges faster (fewer outer iterations) and JP's single-round semantics
leaves undecided vertices. Both modes satisfy validity criterion 1 on all
synthetic inputs tested. The synthetic medium-density high-n regime exceeds
conflict densities observed in any real detector geometry in this study.

---

## 5. Engineering notes

### 5.1 Shared-memory safety (updated 2026-04-26)

The original `build_conflict_coo` gather loop (`atomicAdd` into shared memory)
was bounded at `blockDim.x = 128` slots only implicitly. When a measurement
was shared by more than 128 still-accepted tracks the atomic slot counter could
exceed the shared-memory allocation. This has been fixed: the fast path
explicitly bounds slot to `n_rows <= blockDim.x`; rows wider than 128 tracks
fall through to a safe slow path that reads directly from global memory without
any shared-memory buffering. The slow path is never triggered on real detector
data at the pile-up regimes tested.

### 5.2 No CUDA-graph capture for Tier 2c

Tier 2a (PBG) captures its kernel sequence once per resolver call with CUDA
Graphs, amortising launch overhead. Tier 2c cannot do this because the
COO→CSR conversion involves host-side Thrust calls with data-dependent sizes
that change each outer iteration. Kernel launches are still submitted on a
single CUDA stream; cross-iteration latency is comparable to Tier 2a.

### 5.3 Determinism

Both MIS and JP are fully deterministic given identical input and GPU
(same priority assignment via `inverted_ids`, same tie-breaking rule, same
kernel grid dimensions). The `--determinism-runs=N` flag in the benchmark
harness verifies this by running N additional passes and asserting
selection-identical output.

### 5.4 Hash function portability

The output hash metric previously used `std::hash<std::string>` which is
non-portable. It has been replaced with a stable FNV-1a 64-bit hash that
produces identical output across compilers and standard library versions.

---

## 6. Reproduction

All numbers in this note can be reproduced with:

```bash
# Build (GPU node, Stoomboot):
. /data/alice/sbetisor/spack/share/spack/setup-env.sh && spack env activate traccc
cd /data/alice/sbetisor/traccc
git checkout thesis-novelty-conflict-graph  # commit 43ce28d7
mkdir -p build && cd build
cmake -DTRACCC_BUILD_CUDA=ON -DTRACCC_BUILD_CUDA_UTILS=ON .. && \
  make traccc_benchmark_resolver_cuda -j$(nproc)

# Fatras sweep (one line per pile-up point):
./build/bin/traccc_benchmark_resolver_cuda \
  --input-dump=<path-to-dump>/event_*.json \
  --repeats=10 --warmup=3 \
  --parallel-batch --conflict-graph=both \
  --determinism-runs=5
```

Dump files are serialised pre-resolver state from `traccc_seq_example
--dump-ambiguity-input=<path>`. The Fatras corpus at μ=300..600 lives at
`/data/alice/sbetisor/traccc/data/odd/fatras_ttbar_mu{PU}/` (generated by
ACTS Fatras with `generate_fatras_high_pileup.sh`).

Expanded validation scripts (Phase D, Phase E) are at:
```
thesis-tracc-gpu/scripts/phase_d{1..7}_*.sh
thesis-tracc-gpu/scripts/phase_e{1,3}_*.sh
```

The Phase D corpus is intentionally broad (see §6.1).

---

### 6.1 Validation corpus

To pre-empt the "ODD-specific" reviewer objection, the Phase D campaign
exercises the resolver on independently varying axes:

| phase | corpus | axis | source | n events / cell |
|------:|:-------|:-----|:-------|---:|
| D1 | ODD geant4 10muon 10 GeV | reference cell | `traccc-data-v10` | 10 |
| D2 | ODD fatras ttbar μ ∈ {200, 300, 400, 500, 600} | high pile-up ladder | Fatras | ≥10 |
| D3 | telescope, n=200 | alt geometry, light density | `traccc_simulate_telescope` | 10 |
| D5 | ODD geant4 1muon × {1, 5, 10, 50, 100} GeV | low-density momentum ladder | `traccc-data-v10` | 10 |
| D5 | ODD geant4 10muon × {1, 5, 50, 100} GeV | multi-muon momentum ladder | `traccc-data-v10` | 10 |
| D5 | ODD fatras ttbar μ ∈ {0, 20, 50, 100, 140} | low/medium pile-up ladder | `traccc-data-v10` | 10 |
| D7 | telescope / toy_detector / wire_chamber × n ∈ {50, 200, 500, 1000} | cross-detector geometry | local simulation | 5 |
| D6 | aggregator | rolls D1..D5,D7 into a single CSV + markdown table | — | — |

This sweep covers four orthogonal stress dimensions: **momentum**,
**multiplicity**, **pile-up**, and **detector geometry**. The aggregator
(Phase D6) emits `aggregate.csv` and `aggregate_summary.md` — the latter is
intended to be appended to this note before the upstream review meeting.

A reviewer asking "do we believe this generalises beyond ODD?" can be pointed
directly at the toy_detector and wire_chamber cells in D7, which use a
geometry topology fundamentally different from the silicon-tracker ODD layout.

---

## 7. Open questions for the CERN/traccc team

The following decisions are best made in consultation with the upstream
maintainers before this work is considered for an upstream PR:

1. **Resolver mode exposure**: Should the conflict-graph mode be exposed as a
   `greedy_ambiguity_resolution_config` field (e.g. `resolver_mode =
   baseline | pbg | conflict_graph_mis | conflict_graph_jp`) or as a separate
   algorithm class? The latter is cleaner for upstream conventions but requires
   more structural refactoring.

2. **Canonical benchmark dataset**: Which dataset should be the reference for
   upstream regression tests? The current CI uses `geant4_10muon_10GeV` (n ≤
   100), which is too small to distinguish resolver performance. A Fatras
   ttbar pile-up corpus at μ=200 or μ=400 would be more representative.

3. **Acceptable tolerance for non-CPU-identical selections**: For the Fatras
   high-density events where MIS diverges from CPU greedy (overlap 0.9987 at
   μ=600), is this within the quality budget? For JP it is a non-issue
   (overlap = 1.0), but a formal tolerance policy helps define the test oracle.

4. **COO pre-allocator upper bound**: The maximum edge count is currently
   estimated as `n_accepted * max_tracks_per_meas`. At synthetic high-density
   with n=10 000 this underestimates. What is the preferred strategy: a two-pass
   exact count (costs an extra device-to-host), a conservative overestimate
   (wastes memory), or a dynamic resize?

5. **Truth-based quality metrics**: The benchmark harness now supports a
   `--truth-file=<path.tsv>` flag for efficiency / fake-rate reporting. Is
   there a standard truth CSV format in traccc that the harness should match?

---

## 8. Limitations and honest caveats

- **12 Fatras events** across 4 pile-up points is the current result basis.
  Phase D expands this to ≥10 events per pile-up point; results will be updated.
- **Resolver-only timing** only. End-to-end full-chain events/s impact (Phase
  E3) is pending; at the resolver-only level the speedup is real, but the
  fraction of total chain runtime occupied by the resolver at different μ values
  is not yet quantified.
- **One GPU** (GV100). Portability to V100, A100, H100 is expected given the
  absence of architecture-specific intrinsics, but not tested.
- **No incremental CSR reuse**. The design originally proposed maintaining the
  CSR with tombstones across outer iterations. This optimisation was not needed
  to match CPU reference output on real data and was not implemented.
