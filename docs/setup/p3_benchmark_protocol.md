# P3 – Benchmark protocol for ambiguity resolution

**Goal:** Reproducible CPU vs GPU comparison of the ambiguity resolver with standardized metrics.

---

## What was implemented

### 1. Dump/load (traccc io)

- `write_ambiguity_input(path, track_container, config)` – writes track candidates + measurements to JSON
- `read_ambiguity_input(path, mr)` – loads them back into a `track_container`
- Used to freeze the “right before ambiguity resolution” state from a real traccc run

### 2. Dump option in traccc_seq_example

- `--dump-ambiguity-input=<path>` – dumps track candidates to JSON before running the resolver
- Lets you capture real physics-like input for later CPU vs GPU comparison

### 3. Resolver-only benchmark (`traccc_benchmark_resolver`)

- Runs **only** the greedy ambiguity resolver (no seeding, CKF, fitting)
- Two input modes:
  - `--input-dump=<path>`: load from JSON (frozen real data)
  - `--synthetic`: generate random track candidates (for sweeps)
- Reports: latency, throughput, peak memory, output hash, n_selected/n_removed

### 4. Sweep script (`run_resolver_benchmark_sweep.sh`)

- Runs a 3×3 grid: n_candidates × conflict_density
- Writes one output file per configuration (e.g. `n10000_med.txt`)

---

## What the result files mean

**File names** (e.g. `n10000_med.txt`, `n1000_high.txt`):

- `n10000` = 10 000 track candidates
- `n1000` = 1 000 track candidates
- `low` / `med` / `high` = conflict density (how much hit sharing between candidates)

**Conflict density** (synthetic mode):

| Density | max_meas_id | Track length | Effect |
|---------|-------------|--------------|--------|
| low     | 50 000      | 3–10         | Few candidates share hits → fewer conflicts |
| med     | 10 000      | 3–10         | Moderate sharing |
| high    | 500         | 5–15         | Many candidates share hits → more conflicts, more work for resolver |

**Execution:** CPU only, run directly on the cluster (interactive or batch). No Condor, no GPU yet.

**Example output** (`n10000_med.txt`):

```
n_candidates=10000   ← input: 10k track candidates
n_selected=666       ← output: 666 tracks kept after resolution
n_removed=9334       ← 9334 duplicates/conflicts removed
time_ms_mean=70.43   ← mean latency per “event” (one resolver call)
time_ms_median=70.47
time_ms_p95=71.09
events_per_sec=14.2  ← throughput
peak_memory_mb=19.5664
output_hash=1434612... ← for determinism checks (same input → same hash)
```

**Interpretation:** Higher n_candidates and higher conflict density → more work → longer latency. The sweep lets you study scaling (RQ1, RQ4).

---

## Overview

The benchmark harness has two modes:

1. **Microbenchmark (primary):** Resolver-only with frozen or synthetic input. Answers RQ1–RQ4.
2. **End-to-end (secondary):** Full chain via `traccc_throughput_mt` / `traccc_seq_example_cuda`.

---

## Dump step (base test case)

Run `traccc_seq_example` and dump the ambiguity-resolution input before resolution:

```bash
cd "$TRACCC_SRC"
export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"

./build/bin/traccc_seq_example \
  --detector-file=geometries/odd/odd-detray_geometry_detray.json \
  --material-file=geometries/odd/odd-detray_material_detray.json \
  --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
  --digitization-file=geometries/odd/odd-digi-geometric-config.json \
  --use-acts-geom-source=1 \
  --input-directory=odd/geant4_10muon_10GeV/ \
  --input-events=10 \
  --dump-ambiguity-input=ambiguity_input_event0.json
```

This writes `ambiguity_input_event0.json` with track candidates and measurements.

---

## Building the benchmark

From traccc source (after P2 baseline build):

```bash
cd "$TRACCC_SRC/build"
cmake -S .. -B .
make traccc_benchmark_resolver
```

## Resolver-only benchmark

```bash
./build/bin/traccc_benchmark_resolver --input-dump=ambiguity_input_event0.json \
  --backend=cpu --repeats=10 --warmup=3
```

Synthetic sweep:

```bash
./build/bin/traccc_benchmark_resolver --synthetic --n-candidates=10000 \
  --conflict-density=med --backend=cpu --repeats=10 --warmup=3
```

Options:
- `--input-dump=<path>`: Load from JSON dump
- `--synthetic`: Generate synthetic data
- `--n-candidates=N`: For synthetic (1k–50k)
- `--conflict-density=low|med|high`
- `--backend=cpu|gpu`
- `--repeats=N` (default 10)
- `--warmup=N` (default 3)

---

## Metrics (protocol)

| Metric | How |
|--------|-----|
| Latency | `std::chrono`; report mean, std, median, p95 |
| Throughput | events/s |
| Peak memory | `getrusage` RSS (CPU); `cudaMemGetInfo` (GPU) |
| Output hash | Hash of sorted measurement-ID patterns per selected track |
| Conflict check | No shared hits remain after resolution |
| Determinism | Same input 20–50 times → same output hash |

---

## CPU validation sweep (3×3)

**How it runs:** CPU only. Direct execution on the cluster (interactive or batch). No Condor, no GPU.

From thesis repo:

```bash
cd "$THESIS_REPO/scripts"
./run_resolver_benchmark_sweep.sh
```

Sweeps: `n_candidates` ∈ {1k, 5k, 10k} × `conflict_density` ∈ {low, med, high}. Outputs go to `results/<runID>/n{N}_{density}.txt`, where `<runID>` defaults to a timestamp (e.g. `20250224_143022`). Override with `RUN_ID=myrun` or `OUTDIR=/path` when invoking the script.

---

## GPU runs (Condor) – planned

```bash
cd "$THESIS_REPO/scripts"
mkdir -p logs
condor_submit run_resolver_benchmark_gpu.submit
```

Requires traccc built with `-DTRACCC_BUILD_CUDA=ON` and `benchmark_resolver` extended to support `--backend=gpu`. Until then, the GPU script falls back to CPU.

---

## Reproducibility checklist

- [ ] Fix commit hash (traccc + thesis)
- [ ] Build: Release, same compiler/flags
- [ ] Run on dedicated nodes (interactive for dev, Condor for campaigns)
- [ ] Warm-up (3 runs) + N repeats (≥10)
- [ ] Log: CPU model, GPU model, driver, CUDA version
- [ ] Store: raw outputs + JSON/CSV summary + plots
