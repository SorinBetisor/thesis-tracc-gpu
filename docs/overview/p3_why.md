# P3 – Benchmark protocol: Why

## Rationale

Reproducible CPU vs GPU comparison of ambiguity resolution requires a benchmark protocol. The full chain (P2) includes seeding, CKF, and fitting; variance in those stages would obscure the performance of the resolver alone. P3 isolates the ambiguity resolution step and freezes its input so that CPU and GPU backends receive identical data. This enables fair comparison and supports the research questions on scaling, throughput, and the crossover point where GPU outperforms CPU.

---

## Key contribution

The state immediately before ambiguity resolution is serialised to disk and reloaded for benchmarking. CPU and GPU runs consume the same track candidates and measurements. This isolates ambiguity resolution and removes upstream variance from seeding and track finding. Fair comparison requires identical inputs.

---

## Definitions and context

**Ambiguity resolution** is the step that selects a subset of track candidates when multiple candidates share measurements (hits). A greedy algorithm iteratively picks the best-scoring track and removes conflicting candidates until no shared hits remain. The workload scales with the number of candidates and the degree of conflict (how many candidates share hits).

**Track candidates** are the input to the resolver: tracks found by CKF that may overlap. Each candidate is a sequence of measurement IDs. A conflict exists when two candidates share at least one measurement.

**Conflict density** (in synthetic mode) controls how often candidates share measurements. A smaller measurement ID space or longer tracks increases the probability of overlap. The synthetic generator uses three levels: low (max_meas_id=50000, few shared hits), med (max_meas_id=10000), and high (max_meas_id=500, many shared hits). Higher density typically means more work for the resolver, though the number of surviving tracks (n_selected) also affects runtime.

**n_candidates** is the number of track candidates per event. The sweep varies this (1k, 5k, 10k) to study scaling. Larger candidate sets increase both memory and compute.

**Output hash** is a hash of the sorted measurement-ID patterns of the selected tracks. It serves as a correctness and determinism check: identical input must produce identical output across repeated runs.

---

## What was implemented

| Artifact | Purpose |
|----------|---------|
| Dump/load (traccc io) | Serialise track candidates and measurements to JSON; reload into a track container. Freezes real physics-like input for fair CPU vs GPU comparison. |
| Dump option in traccc_seq_example | Writes the candidate set to disk immediately before resolution. Captures a realistic event from the full chain. |
| Resolver-only benchmark | Runs the greedy resolver in isolation. Two modes: load from dump (real frozen event) or synthetic (controlled sweeps over n_candidates and conflict_density). Reports latency (mean, median, p95), throughput (events/s), peak memory (RSS), output hash, n_selected, n_removed. |
| Sweep script | Executes a 3×3 grid: n_candidates × conflict_density. Writes one result file per configuration. |

The combination provides both a real fixed test case (from dump) and a synthetic stress test suite for scaling studies.

---

## Benchmark protocol and metrics

The protocol defines what is measured and how. These metrics will appear in the thesis experiments section.

| Metric | Definition |
|--------|------------|
| Latency | Wall-clock time per resolver call. Reported as mean, median, p95 across repeats after warmup. |
| Throughput | Events per second (inverse of mean latency). |
| Peak memory | Resident set size (RSS) on CPU; GPU memory via cudaMemGetInfo when GPU backend is enabled. |
| Correctness | Conflict check: no two selected tracks share a measurement. |
| Determinism | Output hash stable across repeated runs on the same input. |

Dump and load are not timed in the resolver microbenchmark; they serve only to freeze inputs. The timed region is the resolver call itself.

---

## Early CPU validation result

A 3×3 sweep was run: n_candidates ∈ {1000, 5000, 10000} × conflict_density ∈ {low, med, high}. Execution was CPU-only, directly on the cluster (interactive or batch), without Condor or GPU. Results are stored under `results/<runID>/` (runID defaults to a timestamp).

| n_candidates | conflict_density | ms/event (median) | events/s |
|--------------|-------------------|-------------------|----------|
| 1000 | low | 3.49 | 287 |
| 1000 | med | 3.95 | 253 |
| 1000 | high | 3.41 | 294 |
| 5000 | low | 35.4 | 28.2 |
| 5000 | med | 31.3 | 31.8 |
| 5000 | high | 22.3 | 44.9 |
| 10000 | low | 115.8 | 8.6 |
| 10000 | med | 70.5 | 14.2 |
| 10000 | high | 57.4 | 17.4 |

Interpretation: increasing n_candidates increases work and latency, as expected. Conflict density affects how many tracks survive resolution; higher density often yields more conflicts early, so fewer survivors and sometimes less work per event. The high-density cases can therefore show lower latency than low-density at the same n_candidates. Overall, the scaling behaviour is consistent with expectations for RQ1 and RQ4. Technical details and result file formats are in `docs/setup/p3_benchmark_protocol.md`.

---

## What follows

See `docs/next_steps.md` for concrete next steps: GPU backend enablement, CPU vs GPU base test case, and scaling sweeps on GPU nodes.
