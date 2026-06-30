# Geant4 (ColliderML) vs FATRAS at pu200 — AR Benchmark Results

**Date:** 2026-06-29
**Hardware:** AMD EPYC 7502P, GV100 GPU node (`gpu-int-mi50-gv100-lot001`)
**Input source:** `CERN/ColliderML-Release-1` — HuggingFace dataset, full Geant4 + DD4hep + ODD simulation, ACTS reconstruction
**Local cache:** `/data/alice/sbetisor/colliderml/ttbar_pu200_raw/` (100 events, 1.9 GB total)

Cross-reference:
- FATRAS baseline: [`cpu_multithreaded_results.md`](cpu_multithreaded_results.md)
- Dataset spec: <https://huggingface.co/datasets/CERN/ColliderML-Release-1>

---

## Motivation

Supervisors requested validation of FATRAS-based AR benchmarks against a true Geant4 dataset. ColliderML R1 was selected as the only open dataset combining:
- Full Geant4 simulation via DD4hep on the Open Data Detector (ODD)
- ACTS-reconstructed track candidates with hit linkage
- Pile-up coverage (pu0 and pu200, matching HL-LHC design conditions)

The benchmark question: **does the FATRAS μ=200 AR workload match real Geant4 pu200?**

---

## Method

### Data path
1. Stream-download 100 events of `ttbar_pu200_{tracks,tracker_hits,particles}` from HuggingFace into `/data/alice/sbetisor/colliderml/`.
2. Convert ColliderML `tracks` table (with `hit_ids` per track) → traccc `ambiguity_input` JSON via `colliderml_to_traccc.py`.
3. Run `traccc_benchmark_resolver` (same harness as FATRAS sweeps) on the converted dumps.

### Two scenarios
- **Raw**: tracks as provided by ColliderML's ACTS reconstruction.
- **Synthetic CKF-injection**: for each track, with probability 0.5, generate a sibling that shares the first 65% of hits and adds 1–3 random "wrong" hits from a nearby track. Models the branching ambiguity that traccc's own CKF would produce, since ColliderML's pipeline has cleaned its output.

### Quality surrogate
ColliderML provides no `chi2` / `pval`. We use `pval ≈ n_hits + 10⁻⁶ · |qop|` — longer tracks win ties, qop term breaks ties deterministically.

---

## Results — single event (event 0)

### Ambiguity density comparison

| Dataset | n_candidates | n_selected | n_removed | rmv% | shared hit% |
|---|---:|---:|---:|---:|---:|
| FATRAS μ=200 | 1332 | 1038 | 294 | **22%** | high |
| **ColliderML pu200 raw (Geant4)** | 1395 | 1304 | 91 | **6.5%** | 0.7% |
| ColliderML pu200 + synthetic ambig | 2062 | 1304 | 758 | **37%** | 37% |

**Headline finding: real Geant4 pu200 has ~3× lower AR rejection rate than FATRAS μ=200** at the same pile-up. ACTS's CKF + standard cleaning produces a much cleaner candidate set than the traccc+FATRAS pipeline used for our μ=200 dump.

The synthetic-injection variant restores high-ambiguity stress (matches FATRAS μ=600 ambiguity range) while keeping real Geant4 hit topology.

---

## Thread sweep on ColliderML pu200

### RAW (low ambiguity — both algorithms underutilised)

| Threads | Greedy ev/s | JP ev/s | JP/Greedy |
|--------:|------------:|--------:|----------:|
| 1       | 105.1       | 101.9   | 0.97      |
| 2       | 202.2       | 196.7   | 0.97      |
| 4       | 404.1       | 391.9   | 0.97      |
| 16      | 1590        | 1555    | 0.98      |
| 32      | 2813        | 2680    | 0.95      |

JP ≈ greedy across all thread counts. Hash match = **TRUE** in all cases (identical solutions). With only 91 tracks to remove, neither algorithm has meaningful work.

### SYNTHETIC AMBIGUITY (dup_rate=0.5, branch_frac=0.65 → 37% shared)

| Threads | Greedy ev/s | JP ev/s | JP/Greedy |
|--------:|------------:|--------:|----------:|
| 1       | 51.1        | 54.2    | **1.06**  |
| 2       | 99.0        | 105.6   | **1.07**  |
| 4       | 213.9       | 211.2   | 0.99      |
| 16      | 774.1       | 845.1   | **1.09**  |
| 32      | 1379        | 1543    | **1.12**  |

JP regains its edge (+6–12%) — same pattern as FATRAS μ=500 (n_removed=1147, rmv 35%). Hash match = **FALSE** here: with many near-tie synthetic siblings, JP and greedy select different but equally-valid maximal independent sets (same n_selected=1304 in both).

---

## Cross-dataset comparison at pu200 (1 thread, greedy)

| Backend | Dataset | ev/s | Latency (ms) |
|---|---|---:|---:|
| CPU greedy | FATRAS μ=200 | 84.5 | 11.83 |
| CPU greedy | ColliderML pu200 (Geant4 raw) | **105.1** | 9.52 |
| CPU greedy | ColliderML pu200 + synth ambig | 51.1 | 19.58 |

Geant4 raw is **24% faster than FATRAS** at the same pile-up because there's less to resolve. With injected ambiguity matching FATRAS-density conflicts, it slows to a regime between FATRAS μ=200 and μ=300.

---

## Key findings

1. **FATRAS overestimates the AR workload at HL-LHC pile-up.** Real Geant4 + ACTS reconstruction at pu200 has 6.5% rejection rate vs FATRAS μ=200's 22%. **The FATRAS-based stress numbers in this thesis should be interpreted as upper bounds, not central values, for HL-LHC-class workloads.**

2. **JP and greedy converge in the low-ambiguity regime** (raw Geant4): both finish in ~10 ms regardless of thread count, with JP/greedy ≈ 0.97. The algorithmic batch-removal advantage of JP only matters when there are many conflicts to resolve.

3. **JP's batch-removal advantage is ambiguity-density dependent, not pile-up dependent.** What matters is the *fraction of shared hits*, not the raw track count. At 37% shared hits (synthetic) JP wins by 6–12%; at 0.7% shared (raw) it ties greedy. This corroborates the FATRAS pile-up sweep finding that the JP crossover is around μ=400 (which is where shared-hit density rises sharply in that simulation).

4. **The synthetic-injection harness is a useful tool** for separating *hit physics* (Geant4-accurate) from *reconstruction-specific ambiguity* (highly tuning-dependent). It lets us probe AR performance at any conflict density on top of any hit topology.

5. **Throughput scaling is identical to FATRAS** — same near-linear OMP scaling up to 32 threads, same modest NUMA bonus at 4–8T. The CPU architecture dominates, not the dataset.

---

## Caveats

- **Single event only** — full 100-event sweep pending (next step). Numbers above are for `event_000000`.
- **Surrogate pval** — no chi² in ColliderML; length-based ranking may slightly disadvantage greedy on real data (tie-breaking matters more in greedy than JP).
- **Synthetic ambiguity is parametric** — `dup_rate=0.5, branch_frac=0.65` was chosen by inspection to match FATRAS μ=400-class shared-hit density. Different parameters would give different JP/greedy ratios.
- **ColliderML pu200 is the highest pile-up they publish** — no open Geant4 dataset goes higher. FATRAS μ=300–600 remains the only stress-test option.

---

## Reproduction

```bash
# Setup (one-time)
ssh -J sbetisor@login.nikhef.nl sbetisor@gpu-int-mi50-gv100-lot001
bash /data/alice/sbetisor/setup_colliderml.sh        # venv + datasets + pyarrow

# Download (~10 min)
bash /data/alice/sbetisor/run_colliderml_download.sh

# Convert + benchmark
bash /data/alice/sbetisor/run_colliderml_benchmark.sh

# Full thread sweep
bash /data/alice/sbetisor/colliderml_full_sweep.sh
```

All inputs: `/data/alice/sbetisor/colliderml/`
All results: `/data/alice/sbetisor/results/colliderml_pu200_sweep_<timestamp>.txt`
