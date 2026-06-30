# GPU Profiling — Kernel-level Nsight Compute analysis

**Date:** 2026-06-30
**Hardware:** Quadro GV100 (Volta, sm_70, 80 SMs, 32 GB HBM2), driver 580.159.04, CUDA 12.5
**Tool:** Nsight Compute 2025.2.1 (`/opt/nvidia/nsight-compute/2025.2.1/ncu`)
**Input:** FATRAS ttbar μ=600, `event_000.json`, 4008 candidates → 2484 selected / 1524 removed
**Binary:** `/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda`

Cross-reference:
- Aggregate SM-utilisation traces: [`multi_stream_gpu_results.md`](multi_stream_gpu_results.md)
- Algorithm walkthrough: [`jp_explained.md`](jp_explained.md), [`conflict_graph_design.md`](conflict_graph_design.md)

> **Why this doc exists.** The `nvidia-smi dmon` numbers we have been quoting (85–93 % "SM utilisation") only measure whether each SM has *at least one resident warp*. They do not tell us how saturated those SMs are, what is stalling the warps, or whether the kernels are bandwidth- or latency-bound. This doc replaces those handwavy numbers with kernel-level Nsight Compute metrics.

---

## Method

Two ncu runs at s=1 (single stream), one per backend:

```bash
NCU=/opt/nvidia/nsight-compute/2025.2.1/ncu     # 2026.1 dropped GV100
METRICS="sm__warps_active.avg.pct_of_peak_sustained_active, \
         launch__registers_per_thread, \
         launch__block_size, launch__grid_size, \
         gpu__time_duration.sum, \
         dram__bytes.sum.per_second, \
         gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed, \
         l1tex__t_sector_hit_rate.pct, lts__t_sector_hit_rate.pct, \
         smsp__inst_executed.avg.per_cycle_active, \
         smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct, \
         smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct, \
         smsp__warp_issue_stalled_no_instruction_per_warp_active.pct"

$NCU --csv --log-file ncu_jp_s1.csv --metrics "$METRICS" --target-processes all \
     $BIN --input-dump=event_000.json --repeats=1 --warmup=0 --conflict-graph=jp --streams=1

$NCU --csv --log-file ncu_g_s1.csv  --metrics "$METRICS" --target-processes all \
     $BIN --input-dump=event_000.json --repeats=1 --warmup=0                  --streams=1
```

Wall time per run: ~11–14 min (ncu kernel replay × ~1000 launches per kernel × ~30 kernels). Raw CSVs (~100k rows each) saved on the cluster at `/tmp/ncu_*.csv`.

**Note on GV100 + ncu:** the spack-env `ncu` is 2026.1.1, which dropped Volta support. Must use `/opt/nvidia/nsight-compute/2025.2.1/ncu`. ncu ≥ 2025.3 also drops GV100.

---

## Headline results — shared compaction kernels (dominate runtime in both backends)

These kernels are launched by both greedy and JP, with near-identical per-kernel metrics. Total per-launch time and call count drive each backend's total runtime, so they explain *why* JP wins.

| Kernel | calls | tot µs | occ % | regs | block | DRAM % | L1 hit | L2 hit | IPC | long-SB stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `remove_tracks`       | 1000 | 31 730 | 23.8 | 32 | 512  | 0.3 | 39.9 | 17.5 | 0.18 | 11 % |
| `rearrange_tracks`    | 1000 | 19 340 | **49.5** | 28 | 1024 | 0.4 | 49.5 | 85.6 | 0.13 | 36 % |
| `block_inclusive_scan`| 1000 |  5 670 |  6.2 | 16 | 128  | 0.6 | 15.9 | 81.8 | 0.03 | 45 % |
| `update_status`       | 1000 |  5 110 | **1.9** | 20 |  32  | 0.9 | 10.0 | 77.6 | 0.02 | **55 %** |
| `sort_updated_tracks` | 1000 |  4 660 | 22.6 | 28 | 512  | 0.1 | 81.7 | 32.6 | 0.08 | 25 % |
| `fill_inverted_ids`   | 1000 |  4 310 |  1.9 | 16 |  32  | 0.4 |  3.6 | 89.0 | 0.01 | 54 % |
| `scan_block_offsets`  | 1000 |  4 310 |  1.6 | 16 |  25  | 0.1 |  0.0 | 47.5 | 0.03 | 39 % |
| `add_block_offset`    | 1000 |  4 200 |  6.2 | 16 | 128  | 0.4 | 59.3 | 66.0 | 0.01 | 53 % |

Columns:
- **occ %** — achieved warps active (`sm__warps_active.avg.pct_of_peak_sustained_active`)
- **regs** — registers per thread (`launch__registers_per_thread`)
- **block** — threads per block (`launch__block_size`)
- **DRAM %** — % of peak HBM bandwidth sustained
- **L1 / L2 hit** — sector hit rates
- **IPC** — instructions per active cycle per SM partition
- **long-SB stall** — % of warp-issue slots stalled on long-scoreboard (waiting on a global-memory load result)

## JP-only kernels (added by `--conflict-graph=jp`)

| Kernel | calls | tot µs | occ % | regs | block | DRAM % | L1 hit | L2 hit | IPC | long-SB stall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `build_conflict_coo`    | 36 | 3 447 | **69.0** | 28 | 512 | 0.6 | 91.6 | 94.2 | 0.21 | 72 % |
| `apply_graph_removals`  | 36 | 2 970 |  2.2     | 32 |  64 | 0.4 | 41.1 | 48.7 | 0.03 | 47 % |
| `graph_mis_propose`     | 36 |   858 |  2.5     | 16 |  64 | 0.2 | 51.9 | 40.5 | 0.04 | 38 % |
| `graph_mis_init`        | 36 |   133 |  3.1     | 16 |  64 | 1.1 |  2.4 | 64.3 | 0.01 | 39 % |

Total JP-only kernel time: **7.4 ms** out of a JP iteration that takes ~24 ms — the rest is the shared compaction sequence.

---

## Key findings

### 1. The dmon "SM utilisation" headline number is misleading
`nvidia-smi dmon` reports the fraction of time each SM has ≥1 resident warp, not the fraction of issue slots actually used. Our dmon traces show 85–93 % for greedy_s1/s8 and 89 % / 61–84 % for JP_s1/s8.

**Achieved occupancy is 1.6 – 49.5 %, mostly under 25 %.** The SMs are "busy" by the dmon definition but starved of warps to actually issue from.

### 2. The workload is memory-latency bound, not bandwidth bound
- DRAM throughput is **<1 % of peak** on every kernel.
- Long-scoreboard stalls (warps waiting on global-memory loads) consume **11 – 72 % of warp-issue slots**.

This means the GPU isn't running out of HBM bandwidth — it's running out of *in-flight* memory transactions. The access pattern (irregular neighbour lookups via the `tracks_per_measurement` inverted index) is too random to coalesce well and the kernels are too small to hide the latency through warp parallelism.

### 3. Register pressure is *not* the limiter
All kernels use 16–32 registers/thread. On GV100 each SM has 64k registers and 64 warps max; the register budget allows full occupancy. The constraint is elsewhere — block size and L1/SMEM, not registers.

### 4. Many kernels launch with tiny blocks (25 – 64 threads)
`update_status` (32), `fill_inverted_ids` (32), `scan_block_offsets` (25), `graph_mis_*` (64). A 32-thread block is exactly **one warp**. Even with 80 blocks resident across 80 SMs that's 1 warp/SM ≈ 3 % occupancy on a chip that can hold 64 warps/SM. This is the dominant occupancy ceiling on `update_status` (1.9 %), `fill_inverted_ids` (1.9 %), `scan_block_offsets` (1.6 %).

The block sizes were chosen to match the typical per-iteration work item (one track or one measurement); they're not tuned for warp-level parallelism on GV100.

### 5. JP wins by amortising the shared compaction sequence, not via fast JP-only kernels
The JP-only kernels (`graph_mis_*`, `build_conflict_coo`, `apply_graph_removals`) only contribute 7.4 ms of the total. The bulk (~90 %) is in the shared compaction pipeline (`remove_tracks`, `rearrange_tracks`, scans).

Greedy walks that compaction pipeline **once per track removed** (~1500 calls). JP walks it **once per batch** (~36 calls — the 36 here is the call count under `--launch-count`-limited ncu replay; in production it's ~18 outer iterations, each triggering the compaction sequence ~2× via prologue+commit). The result is fewer launches of the most expensive shared kernels.

### 6. `build_conflict_coo` is the one kernel that uses the GPU well
- 69 % achieved occupancy
- 91.6 % L1 hit / 94.2 % L2 hit
- IPC 0.21 (vs 0.02 – 0.13 elsewhere)

It's the only "regular" kernel in the AR pipeline — it iterates over a dense candidate × measurement array. If we ever extend JP with more graph-construction work, doing it here amortises cheaply.

---

## What this means for optimisation

In priority order:

1. **Block-size tuning of the small-launch kernels.**
   `update_status` and `fill_inverted_ids` currently launch at 32 threads/block. Raising to 256 (8 warps/block) without changing the algorithm would lift occupancy on these from ~2 % toward 25–50 % and reduce long-scoreboard stalls (more warps per SM ⇒ more latency hiding).
   *Estimated effort: low (one block-size constant per kernel).* *Estimated win: 1.5–3× on the affected kernels, which dominate JP runtime.*

2. **Kernel fusion of the compaction sequence.**
   Each outer iteration currently launches `scan_block_offsets → add_block_offset → block_inclusive_scan → sort_updated_tracks → remove_tracks → rearrange_tracks → fill_inverted_ids → update_status`. Each launch has ~5–10 µs of overhead and a separate global-memory round trip for its inputs/outputs. Folding 2–3 of these into a single kernel (where the algorithm permits) reduces both launch overhead and DRAM traffic.
   *Estimated effort: medium.* *Estimated win: 1.2–1.5× on the shared compaction time, which dominates both backends.*

3. **CUDA graph reuse** (already partially explored in [`graph_reuse_implementation.md`](graph_reuse_implementation.md)).
   Reduces launch overhead but doesn't help the memory-latency problem; complementary to fusion.

4. **Algorithmic: increase batch size of JP.**
   Currently ~18 outer iterations at μ=600. If we relaxed the strict "every priority maximum" rule (e.g. Luby-style randomised colouring) we might pack more removals per iteration and call the compaction pipeline fewer times. Risk: solution quality drift.

The **block-size fix is the obvious first thing** — it costs almost nothing and should be measurable in a single Stoomboot run.

---

## Caveats

- **s=1 only in this pass.** Stream-8 profiling is queued (`/tmp/ncu_*_s8.csv` once that run completes); will be appended here. The hypothesis to confirm with s=8: long-scoreboard stalls rise faster on JP than on greedy because JP's lookup pattern hits less coalescing, making per-warp memory latency worse when multiple events queue on the same SMs.
- **Single event.** Per-event variance not characterised. Numbers above use `event_000` from FATRAS μ=600.
- **`--launch-count`-limited replay.** ncu caps the number of launches it profiles per kernel (1000 here). For very-high-iteration greedy this samples a representative window, not the entire run.
- **Greedy total per-event latency in this run is ~26 ms (1 stream).** All percentages are averaged across the kernel's launches; per-launch variability not shown.

---

## Reproduction

```bash
ssh gpu-int-mi50-gv100-lot001
source /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc
export LD_LIBRARY_PATH=/data/alice/sbetisor/traccc-jp/build/lib64:$LD_LIBRARY_PATH

NCU=/opt/nvidia/nsight-compute/2025.2.1/ncu
BIN=/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda
DUMP=/data/alice/sbetisor/data/fatras_csv_dumps/fatras_ttbar_mu600/event_000.json

# (see METRICS string at the top of this doc)
$NCU --csv --log-file ncu_jp_s1.csv --metrics "$METRICS" \
     "$BIN" --input-dump="$DUMP" --repeats=1 --warmup=0 --conflict-graph=jp
$NCU --csv --log-file ncu_g_s1.csv  --metrics "$METRICS" \
     "$BIN" --input-dump="$DUMP" --repeats=1 --warmup=0
```

Parsing script (Python, in-line in the analysis session): groups CSV rows by short kernel name and averages each metric across all launches.
