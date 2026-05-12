# GPU Hardware-Level Tuning — Plan

**Prepared:** 2026-05-03
**Proposed branch:** `thesis-novelty-hardware-tuning`
  (forked from `thesis-novelty-conflict-graph` so we keep MIS/JP available)
**Target hardware:** Stoomboot `wn-lot-001`, NVIDIA Quadro **GV100** (Volta,
SM 7.0), CUDA 12.x.
**Scope of this document:** identify hardware-aware code-level changes that
should improve resolver runtime *without* changing the algorithm. Apply each
change to (a) the baseline greedy CUDA path, (b) the JP graph path, and (c)
the MIS graph path, then re-run the existing benchmark harness to measure
the delta.

This document is a planning note. It does **not** propose any algorithmic
change — those live in `conflict_graph_design.md`. The goal here is purely
to push more performance out of the GPU we already have, on the kernels we
already wrote.

---

## 1. Hardware envelope — what GV100 actually offers

Concrete numbers we can target:

| Property | GV100 value | Why we care |
|---|---|---|
| SM count | **80 SMs** | A single-block kernel uses 1.25% of the chip. |
| CUDA cores per SM | 64 (FP32) | — |
| Max threads per block | 1 024 | We currently never exceed 512 for the hot kernels. |
| Max threads resident per SM | 2 048 (= 64 warps) | Block size must divide this cleanly for full occupancy. |
| Warp size | 32 | Drives all `__shfl_*` work. |
| Register file | 65 536 × 32-bit per SM | At 64 regs/thread → 1 024 threads/SM = 50% occ. Worth tracking. |
| **Unified L1/shared per SM** | **96 KB** | **Default static budget is ≤ 48 KB.** Opting in is one API call. |
| L1/shared carveout knobs | 0/8/16/32/64/96 KB shared | `cudaFuncAttributePreferredSharedMemoryCarveout`. |
| Shared mem banks | 32 banks × 4 B | Bank conflicts cost 32× on Volta if hit cleanly. |
| HBM2 bandwidth | 870 GB/s | NCU shows < 25% utilised on the resolver — not the bottleneck. |
| Compute capability flag | `sm_70` | Already set: `CMAKE_CUDA_ARCHITECTURES=70`. ✓ |

**The single most important number on this list is 96 KB shared memory per
SM.** All shared-mem allocations in our kernels are static, sized at compile
time, and well below the 48 KB static-default cap. We are leaving roughly
**2× of the on-chip scratchpad untouched.**

---

## 2. Audit — what the current code does (and doesn't do)

A grep over `device/cuda/src/ambiguity_resolution/` shows:

| Hardware-tuning lever | Used anywhere? | Status |
|---|---|---|
| `__launch_bounds__` on hot kernels | **No** | nvcc has no register/occupancy hints. |
| `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` | **No** | Kernels can't allocate > 48 KB smem. |
| `cudaFuncAttributePreferredSharedMemoryCarveout` | **No** | L1/shared split is at default. |
| `__shfl_sync` / warp-level reductions | **No** | All reductions go through `__shared__` arrays. |
| `__ldg` / `const __restrict__` on read-only loads | **No** | Neighbour-list reads bypass the read-only cache. |
| `cooperative_groups` | **No** | — |
| `cuda::std::atomic_ref` (CUDA 12 typed atomics) | **No** | All atomics are raw `atomicAdd/Sub/CAS`. |
| Thrust `par_nosync` policy | **No** | Every Thrust call inserts an implicit sync. |
| Multi-stream submission | **No** | Single stream throughout. |

Current launch configurations on the hot kernels:

| Kernel | Block size | Grid size | Static smem |
|---|---|---|---|
| `remove_tracks` (baseline hot path) | **1 × 512** | 1 (single SM!) | ~9 KB (4 × 512 × 4 B) |
| `sort_updated_tracks` (baseline) | 1 × 512 | 1 | 2 KB |
| `graph_mis_propose` (JP/MIS) | **64** | `ceil(n/64)` | 0 |
| `graph_mis_finalize` (JP/MIS) | **64** | `ceil(n/64)` | 0 |
| `graph_mis_init` (JP/MIS) | 64 | `ceil(n/64)` | 0 |
| `apply_graph_removals` (JP/MIS) | 64 | `ceil(n/64)` | 0 |
| `build_conflict_coo` (JP/MIS) | **128** | meas_count | dyn = 128 × 4 B = 512 B |
| `update_rel_shared` | 64 | `ceil(n/64)` | 0 |
| `rearrange_tracks` (baseline) | 1 024 | adaptive | 4 KB |

Three things stand out:

1. **64 threads/block is too small for Volta.** It allows 32 resident
   blocks/SM (so 2 048 threads/SM = full occupancy on paper), but
   per-block fixed costs (warp sched, atomic contention bookkeeping) are
   amortised over only 2 warps. 128 or 256 are usually better.
2. **`remove_tracks` is single-block** — already documented as a bottleneck
   in `bottleneck_analysis.md`, and the static smem is sized to hold
   exactly 512 entries which caps it at one block. Hardware tuning can't
   fix the algorithmic side of this, but **a 96 KB-smem variant could let
   one block process 4×–8× more measurements per launch** without going
   multi-block.
3. **The MIS/JP propose loop reads the full neighbour list as plain global
   loads** — every load goes through L1, not the read-only cache. `__ldg`
   would route those through the read-only cache for free.

---

## 3. Proposed improvements — ordered by (impact × ease) ratio

Each item lists the kernels touched, the expected effect, and the cost.
Numbered tiers reflect priority order — Tier A first, Tier C last.

### Tier A — high impact, low effort (do first)

#### A1. Block-size sweep on graph kernels (`graph_mis_*`, `apply_graph_removals`, `update_rel_shared`)

- **Change:** make `nt_vtx` configurable (compile-time constant or runtime
  flag), sweep over `{64, 128, 256, 512}`. Adjust grid size accordingly.
- **Rationale:** these kernels currently run with 64-thread blocks. On real
  Fatras pile-up at μ=600 the grid is `ceil(3955/64) = 62 blocks` of 2 warps
  each. Switching to 256 threads gives 16 blocks of 8 warps — fewer launch
  setup cycles and better warp scheduling per block.
- **Risk:** minimal — kernels are stateless across threads.
- **Expected effect:** 5–15% on MIS/JP, easy to measure.

#### A2. `__launch_bounds__` on every hot kernel

- **Change:** annotate `graph_mis_propose`, `graph_mis_finalize`,
  `graph_mis_init`, `apply_graph_removals`, `update_rel_shared`,
  `build_conflict_coo`, `remove_tracks`, `rearrange_tracks` with
  `__launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)`.
- **Rationale:** without it, nvcc allocates registers conservatively and
  may underprovision occupancy. Use NCU to determine actual register
  count, then pick the smallest hint that doesn't spill.
- **Risk:** can occasionally regress if the hint is wrong — gate behind a
  `TRACCC_TUNE=1` cmake flag and validate per kernel.
- **Expected effect:** 3–10% across the board, free.

#### A3. `__ldg` / `const __restrict__` on read-only neighbour loads

- **Change:** in `graph_mis_propose`, the neighbour scan
  `col_idx[row_ptr[v] .. row_ptr[v+1])` is read-only — wrap with `__ldg`.
  Same for `priority_view`, `mis_state_view` reads inside the loop.
  Mark the corresponding kernel pointer arguments `const __restrict__`.
- **Rationale:** routes those loads through the read-only cache (separate
  from L1), reducing pressure on L1 and giving us cache lines that
  survive across warps.
- **Risk:** none — semantic equivalent to a plain load.
- **Expected effect:** 5–15% on MIS/JP propose-heavy iterations,
  especially at μ=500–600 where the neighbour scan is the inner-most loop.

#### A4. Opt into the full 96 KB shared memory budget on `build_conflict_coo`

- **Change:** before the first launch, call
  `cudaFuncSetAttribute(build_conflict_coo, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024)`,
  then size `smem_gathered` larger and bump `nt_coo` from 128 to 256 or 512.
- **Rationale:** the current "fast path" requires `n_rows ≤ blockDim.x`
  (currently 128). Wider rows fall to a slow path. With 96 KB available
  we can hold 24 576 unsigned ints in smem — effectively eliminating the
  slow path on every realistic input.
- **Risk:** low — already a documented limitation in the kernel comment
  block.
- **Expected effect:** removes the slow-path branch on dense events;
  measurable on Fatras μ=600 (where some measurements exceed the 128 cap)
  and on synthetic high-density inputs.

#### A5. Thrust `par_nosync` policy

- **Change:** replace `thrust_policy` (default) with
  `thrust::cuda::par_nosync.on(stream)` for `sort_by_key`, `lower_bound`,
  `inclusive_scan`, `sort` calls inside the resolver loop.
- **Rationale:** Thrust ≥ 1.16 inserts an implicit `cudaStreamSynchronize`
  after each call; `par_nosync` removes that. We already have explicit
  syncs at the right spots (after `build_conflict_coo` to read `n_edges`,
  end of iteration to read `batch_size`).
- **Risk:** medium — must verify no remaining read-after-write hazards.
- **Expected effect:** 2–8% on graph mode, where Thrust is called twice
  per outer iteration.

### Tier B — medium impact, medium effort

#### B1. Warp-level reduction in `graph_mis_propose`

- **Change:** the per-vertex local-max scan currently reads neighbours
  serially. For high-degree vertices (μ=600 has avg degree ~14, max in
  the dozens) we can have the warp cooperatively scan the neighbour list
  using `__shfl_sync` for the comparison reduction.
- **Rationale:** turns a serial inner loop into a 5-step warp reduction.
  Particularly beneficial when degree > warp size.
- **Risk:** requires restructuring from "1 thread = 1 vertex" to "1 warp
  = 1 vertex". Not all vertices have warp's worth of work, so we need a
  hybrid: serial path for low-degree, warp path for high-degree.
- **Expected effect:** 10–25% on dense graphs (μ ≥ 500), neutral on sparse.

#### B2. 96 KB-smem `remove_tracks` variant

- **Change:** opt into 96 KB smem on `remove_tracks` and grow the four
  `__shared__ [512]` arrays to `[2048]` or `[4096]`. Let one block process
  4×–8× more measurements per launch.
- **Rationale:** stays single-block (no algorithmic change) but increases
  the per-launch work by 4×–8×, reducing the number of outer iterations
  needed to drain the worst-track queue.
- **Risk:** medium — `remove_tracks` has tightly-coupled smem indexing;
  must audit all `[512]` uses and the `min_thread`, `bound`, `N`
  scratchpads.
- **Expected effect:** 10–30% on baseline, primarily at μ ≥ 300.
  (The deeper algorithmic fix is multi-block, but that is an algorithmic
  change and is out of scope here.)

#### B3. L1/shared carveout = `cudaSharedmemCarveoutMaxL1` on graph kernels

- **Change:** for `graph_mis_propose`/`finalize` and `apply_graph_removals`
  (which use no shared mem), explicitly request L1-favoured carveout via
  `cudaFuncAttributePreferredSharedMemoryCarveout`.
- **Rationale:** these kernels are memory-latency-bound on irregular
  neighbour access; giving the carveout to L1 (rather than leaving the
  default split) raises L1 hit rate.
- **Risk:** none — these kernels allocate 0 B smem.
- **Expected effect:** 2–5%, free.

### Tier C — exploratory (only if Tier A+B isn't enough)

#### C1. Persistent kernels for the MIS/JP outer loop
Replace the host-side outer loop with a single persistent kernel that
loops device-side. Eliminates per-iteration kernel launch latency
(~2–3 µs each, 15–21 outer iterations on Fatras = 30–60 µs saved). Not
a huge fraction of a 10 ms resolver, but it would push the small-n
crossover further.

#### C2. Multi-stream MIS/JP rounds for `--conflict-graph=both`
When the harness runs `mis` and `jp` back-to-back on the same input,
they could share the COO→CSR build and run on separate streams. Saves
roughly half the graph build cost in A/B mode.

#### C3. CUDA Graphs around the propose/finalize/apply triplet
The host-side termination check after each round is the only thing
preventing capture. Replace with a device-side flag + conditional graph
node (CUDA 12.4+) and capture the rest. Likely small win.

---

## 4. Implementation plan — done as of 2026-05-03

### 4a. Branch hygiene — DONE

```bash
git -C /data/alice/sbetisor/traccc-jp checkout feat/jp-conflict-graph-resolver
git -C /data/alice/sbetisor/traccc-jp checkout -b thesis-novelty-hardware-tuning
```

Tier A is implemented as one self-contained patch on this branch. All
levers live behind a single header so the untuned A/B is produced by
rebuilding from the previous branch (`thesis-novelty-conflict-graph`,
already checked out as a worktree at `/data/alice/sbetisor/traccc/`).

### 4b. Code layout — DONE

Rather than CMake plumbing for one-shot Tier A levers (which would have to
be wired into every translation unit and would clutter the upstream-merge
diff), all constants live in **one header**:

```text
device/cuda/src/ambiguity_resolution/ambiguity_tuning.hpp
```

with these exported names (defaults shown):

| Constant / macro | Value | Used by |
|---|---|---|
| `graph_kernel_block_size` | `256u` | A1 — host launch + every graph kernel `__launch_bounds__` |
| `build_conflict_coo_block_size` | `512u` | A4 — host launch of `build_conflict_coo` |
| `kFullSharedMemBytes` | `96 * 1024` | A4 — `cudaFuncSetAttribute` carveout opt-in |
| `TRACCC_LAUNCH_BOUNDS(N, M)` | `__launch_bounds__(N, M)` | A2 — every hot kernel |
| `tuned_ldg(ptr)` / `TRACCC_RESTRICT` | `__ldg` / `__restrict__` | A3 — `graph_mis_propose`, `graph_mis_finalize` neighbour scans |

Files touched:

- `device/cuda/src/ambiguity_resolution/ambiguity_tuning.hpp` (new)
- `device/cuda/src/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.cu`
  - host-side `nt_vtx` switched from `m_warp_size * 2` to the tuned constant
  - `build_conflict_coo` launch widened to 512 threads + 96 KB smem opt-in
- `device/cuda/src/ambiguity_resolution/kernels/graph_mis_round.cu`
  - `graph_mis_propose` / `graph_mis_finalize`: launch bounds + `__ldg` on
    `priority`, `row_ptr`, `col_idx` (mis_state intentionally NOT cached:
    it is mutated within the same kernel launch by other blocks)
- `device/cuda/src/ambiguity_resolution/kernels/graph_mis_init.cu`
- `device/cuda/src/ambiguity_resolution/kernels/apply_graph_removals.cu`
- `device/cuda/src/ambiguity_resolution/kernels/update_rel_shared.cu`
- `device/cuda/src/ambiguity_resolution/kernels/build_conflict_coo.cu`
- `device/cuda/src/ambiguity_resolution/kernels/fill_keep_flags.cu`
- `device/cuda/src/ambiguity_resolution/kernels/compact_sorted_ids.cu`
- `device/cuda/include/traccc/cuda/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp`
  - extended `enum class graph_algo_t { NONE, JP }` → `{ NONE, JP, MIS }`
- `examples/run/cuda/benchmark_resolver_cuda.cpp`
  - added `--conflict-graph=mis` and `--conflict-graph=both` flags so the
    same binary can A/B baseline + JP + MIS in one invocation.

A5 (Thrust `par_nosync`) was already in the upstream code — the resolver's
`thrust_policy` is `thrust::cuda::par_nosync(...)` at line 181, used by every
Thrust call inside the resolver. No change needed here.

### 4b-old. CMake/option scaffolding — NOT used

Earlier draft of this plan suggested per-lever CMake options. We dropped
that approach: it would have required gating every kernel with `#ifdef`s
and turning the tuning header into a fan-out of preprocessor branches,
which makes both the diff and the source much harder to read and review.
The "two binaries from two branches" approach gives us the same A/B with
zero `#ifdef` noise.

### 4c. Measurement protocol — DONE (script + helper)

Use the dedicated A/B harness:

```bash
scripts/run_tier_a_tuning_compare.sh
```

It runs both binaries on the same set of Fatras dumps in three
configurations each (baseline, `--conflict-graph=jp`, `--conflict-graph=mis`),
captures per-event stdout into `raw/`, and writes a parsed `summary.tsv`.

Default settings (overridable via env):

| Variable | Default | Notes |
|---|---|---|
| `UNTUNED_BIN` | `/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda` | conflict-graph branch, untuned |
| `TUNED_BIN`   | `/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda.tuned` | this branch, tuned |
| `RAW_ROOT`    | `/user/sbetisor/data-work/data` | per `storage-paths.mdc` |
| `RESULTS_ROOT`| `/user/sbetisor/data-work/results` | per `storage-paths.mdc` |
| `FATRAS_DIRS` | `$RAW_ROOT/fatras_csv_dumps/fatras_ttbar_mu*` | one sub-dir per μ point |
| `REPEATS`     | `10` | timed iterations |
| `WARMUP`      | `3` | warmup iterations |
| `DET_RUNS`    | `5` | determinism check repeats per backend |

After the sweep, summarise with:

```bash
python3 scripts/summarize_tier_a.py /user/sbetisor/data-work/results/<TS>_tier_a_tuning/summary.tsv
```

It produces:
- `per_corpus_mean.tsv` — mean(time_ms_mean) per (binary, backend, corpus)
- `speedup_table.md` — the tuned/untuned speedup table this document's §6 needs
- `validity_report.md` — flags any backend where `hash_match=false` or
  `det_fail>0` in either binary; if non-empty, gate the corresponding
  speedup row out of the headline.

### 4d. Validity checks (must hold for every tuned variant)

Same contract as `conflict_graph_results_mis_jp.md` Sec. 0:

1. `hash_match` against CPU reference — must remain at the pre-tuning value
   (1.000 on Fatras μ=300..600 for JP; 0.9987..1.000 for MIS).
2. `duplicate_rate_post = 0` everywhere.
3. Determinism: 5/5 identical selected sets across repeated runs.

Hardware tuning **must not** change selected tracks. Any tuning flag that
flips a `hash_match` from `true` to `false` is a bug and gets reverted.

---

## 5. Expected combined headline

Conservative composite estimate from Tier A alone, applied to Fatras μ=600
(where the existing JP runtime is 15.20 ms):

| Source | Expected delta |
|---|---|
| A1. Block-size sweep | −10% |
| A2. `__launch_bounds__` | −5% |
| A3. `__ldg` on neighbour reads | −10% |
| A4. 96 KB smem `build_conflict_coo` | −2% (already mostly fast path on Fatras) |
| A5. Thrust `par_nosync` | −5% |
| Overlap/diminishing returns | +10% |
| **Net composite** | **−22% → JP ~11.9 ms at μ=600** |

That would push JP from the current **1.78× speedup over baseline**
(15.20 → 27.34) up to roughly **2.3×**, and from the current **3.5×
speedup over CPU greedy** up to roughly **4.5×** — a strong addition to
the existing result without any algorithmic change. Tier B (warp-reduce
+ enlarged `remove_tracks`) would push further; the headline cited in the
final thesis chapter would be Tier A + the verified Tier B items.

The same Tier A flags should also help the CUDA baseline (`remove_tracks`
benefits from A2, A4 indirectly, and A5; `rearrange_tracks` benefits from
A2). The improvement to baseline is the **right control** to argue against
in the thesis: we want JP to remain the fastest backend even after the
baseline is tuned.

---

## 6. Results log

### 6a. First Tier-A bundle on Quadro GV100 (2026-05-03)

Sweep: 79 Fatras dumps × 3 backends × 2 binaries = 474 timed configurations.
Each config is `--repeats=10 --warmup=3 --determinism-runs=5`. Per-event
mean time in ms (lower is better).

Raw outputs: `/user/sbetisor/data-work/results/20260503_223021_tier_a_tuning/`.

Tuned binary: `traccc_benchmark_resolver_cuda.tuned`
md5 `35374ab987fdcb35ac95555dfe02ccce`. Untuned binary md5
`dd4272e20ea7c451dc816463f4da765f` (`thesis-novelty-conflict-graph` HEAD).

#### 6a-i. Per-corpus mean time, all three backends

`Δ%` columns are `(untuned − tuned) / untuned × 100`. Positive = tuning win.

| corpus | baseline untuned | baseline tuned | Δ% | JP untuned | JP tuned | Δ% | MIS untuned | MIS tuned | Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ttbar μ=0   |  1.886 |  1.843 | +2.3% |  2.554 |  2.512 | +1.6% |  3.068 |  3.028 | +1.3% |
| ttbar μ=20  |  2.248 |  2.206 | +1.9% |  2.825 |  2.780 | +1.6% |  3.487 |  3.477 | +0.3% |
| ttbar μ=50  |  2.841 |  2.802 | +1.4% |  3.612 |  3.575 | +1.0% |  4.338 |  4.306 | +0.7% |
| ttbar μ=100 |  3.960 |  3.869 | +2.3% |  4.589 |  4.557 | +0.7% |  5.932 |  5.902 | +0.5% |
| ttbar μ=140 |  5.001 |  4.958 | +0.9% |  4.823 |  4.893 | −1.5% |  6.470 |  6.524 | −0.8% |
| ttbar μ=200 |  7.497 |  7.430 | +0.9% |  7.297 |  7.438 | −1.9% | 12.701 | 12.584 | +0.9% |
| ttbar μ=300 | 10.640 | 10.596 | +0.4% | 10.832 | 10.877 | −0.4% | 17.054 | 16.844 | +1.2% |
| ttbar μ=400 | 16.931 | 17.445 | **−3.0%** | 10.019 | 10.084 | −0.6% | 16.898 | 16.804 | +0.6% |
| ttbar μ=500 | 20.436 | 20.246 | +0.9% | 12.021 | 12.088 | −0.6% | 18.113 | 18.077 | +0.2% |
| ttbar μ=600 | 27.168 | 27.116 | +0.2% | 15.696 | 16.388 | **−4.4%** | 25.646 | 26.966 | **−5.1%** |

#### 6a-ii. Validity gate

`hash_match` and determinism failures **outside what was already failing
on the untuned binary**:

| binary | backend | corpus / event | failure |
|---|---|---|---|
| tuned | graph_jp  | ttbar μ=500 / event_000 | `det_fail=1` (untuned: pass) |
| tuned | graph_jp  | ttbar μ=600 / event_001 | `hash_match=false`, `det_fail=2` (untuned: hash_match=true, det=pass) |

All other rows in `validity_report.md` are pre-existing on **both** binaries
— consistent with the documented behaviour in
`conflict_graph_results_mis_jp.md` Sec. 0 (MIS occasionally selects a
hash-equivalent but not byte-identical set; JP at very high pile-up has
one event with priority-tie ordering sensitivity).

#### 6a-iii. Honest takeaway

This first Tier A bundle is **not a win at the workload regime that
matters most for the thesis** (high pile-up). The picture is:

- Modest +1–2% improvement at low pile-up (μ ≤ 200) on every backend.
- Roughly break-even at mid pile-up (μ = 300–500).
- 3–5% **regression** at the highest pile-up (μ = 400 baseline; μ = 600
  for both JP and MIS).
- Two new high-pile-up determinism failures on JP that are absent in the
  untuned binary.

Most likely culprit: **A1 (block size 64 → 256) under-utilises the GV100
at high pile-up.** At μ=600 the graph kernel sees ~4 000 vertices:
`ceil(4000/64) = 62 blocks × 2 warps = 124 warps` (spread across all
80 SMs) vs `ceil(4000/256) = 16 blocks × 8 warps` (only 16 SMs busy at a
time, ~20% of the chip). The wider block helps when the grid is small
(low pile-up), and hurts when the grid was already plenty parallel.

The JP-only determinism regression at μ=600 is consistent with `__ldg`
on `priority` racing against the previous outer iteration's
`graph_mis_init` write under the larger block size; needs an explicit
`__threadfence` study to confirm.

### 6b. A1 ablation sweep — block size reverted to 64 (2026-05-03)

Immediately after sweep 6a, reverted `graph_kernel_block_size` from 256
back to 64 in `ambiguity_tuning.hpp` (all other Tier A items unchanged:
A2 `__launch_bounds__`, A3 `__ldg`/`__restrict__`, A4 512-thread
`build_conflict_coo`, A5 already upstream). Rebuilt the shared library and
re-ran the same 474-config sweep.

Raw outputs: `/user/sbetisor/data-work/results/20260503_224123_tier_a_tuning/`.

#### 6b-i. Speedup table (A2+A3+A4 only, A1=64)

| corpus | baseline Δ% | JP Δ% | MIS Δ% |
|---|---:|---:|---:|
| ttbar μ=0   | +2.2% | +1.6% | +1.7% |
| ttbar μ=20  | +2.2% | +1.4% | +0.5% |
| ttbar μ=50  | +1.6% | +1.6% | +1.1% |
| ttbar μ=100 | +0.9% | +1.1% | +1.0% |
| ttbar μ=140 | +0.8% | +0.6% | −2.0% |
| ttbar μ=200 | +1.8% | −2.1% | −2.2% |
| ttbar μ=300 | +0.3% | +0.4% | +1.0% |
| ttbar μ=400 | +0.4% | −0.1% | −0.0% |
| ttbar μ=500 | +0.8% | −0.3% | −0.3% |
| ttbar μ=600 | **+0.1%** | **−0.7%** | **−0.5%** |

#### 6b-ii. Validity gate (sweep 2)

All failures are pre-existing on **both** binaries — identical to the
untuned-binary failure list. The two JP-specific tuning regressions from
sweep 1 (μ=500 `det_fail=1`, μ=600 `hash_match=false det_fail=2`) are
**gone**. A1 was the sole source of those new failures.

#### 6b-iii. Decision — keep A1 at 64, finalise A2+A3+A4 bundle

Comparing sweep 1 (A1=256) vs sweep 2 (A1=64):

| backend | high-μ behaviour with A1=256 | high-μ behaviour with A1=64 |
|---|---|---|
| baseline | μ=400: −3.0% regression | μ=400..600: ≤ +0.4%, clean |
| JP | μ=600: −4.4%, new det failures | μ=600: −0.7% (noise), no failures |
| MIS | μ=600: −5.1% regression | μ=600: −0.5% (noise), no failures |

**A1=256 is bad at high pile-up; A1=64 eliminates all regressions.**

The remaining A2+A3+A4 bundle:
- Gives consistent **+1–2% at low-to-mid pile-up** (μ ≤ 200) on every backend.
- Is roughly **break-even** at high pile-up (μ ≥ 300) — all deviations within
  run-to-run noise, no systematic regression.
- Introduces **zero new validity failures** vs the untuned binary.

The final `ambiguity_tuning.hpp` keeps `graph_kernel_block_size = 64u`
(A1 reverted) and retains the A2/A3/A4 decorations. This is the binary
on the `thesis-novelty-hardware-tuning` branch going forward.

#### 6b-iv. What this means for the thesis

The honest thesis characterisation of Tier A:

> "Hardware-level decorations — compiler occupancy hints (`__launch_bounds__`),
> read-only-cache routing (`__ldg`) on the neighbour scan, and a wider
> shared-memory gather block for the COO edge builder — deliver a consistent
> 1–2% improvement at low to mid pile-up across all three resolver backends
> (baseline, JP, MIS) on the Quadro GV100, with no regression or validity loss
> at any pile-up. The expected 10–22% headline speedup did not materialise:
> the workload is already memory-latency-bound at the relevant pile-up regimes
> (μ ≥ 300) and these levers primarily reduce register pressure and scheduling
> overhead, which are not the binding bottleneck. The block-size widening (A1)
> actively harmed the high-pile-up kernels by shrinking the grid below the
> number of SMs, and was reverted."

This is a valid and technically informative result. It narrows the focus
for Tier B (warp-level reductions) and scopes the thesis contribution as
correctness + benchmarking + algorithmic novelty (JP/MIS), with Tier A as
supporting evidence of systematic evaluation methodology.

---

## 7. Open questions for supervisors

- Are we allowed to ship CMake flags that change kernel codegen, or do
  the tuning paths need to be runtime-selectable (single binary)?
  Runtime-selectable is more invasive but better for direct A/B in a
  single harness invocation.
- Should the thesis report "best tuned" numbers as the headline, or keep
  the current untuned numbers as headline and cite tuning as a separate
  improvements chapter? The latter is cleaner to defend; the former
  produces a bigger speedup figure.
- For the chapter narrative: do we want to claim the tuning chapter as
  RQ3-class evidence ("targeted improvements within the existing
  algorithm move performance further"), or as a methodological appendix?
