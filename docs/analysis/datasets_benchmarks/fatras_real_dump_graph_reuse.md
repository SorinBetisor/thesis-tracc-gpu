# Real Fatras ttbar Dump Benchmarks with CUDA Graph Reuse

**Prepared:** 2026-04-19  
**Purpose:** Summarise the first real high-pileup Fatras ttbar dump-based ambiguity-resolution benchmarks, including CPU vs GPU behavior and the measured effect of CUDA graph reuse.

---

## 1. What was benchmarked

This benchmark uses **real ODD Fatras ttbar events** generated locally at:

- `mu=400`
- `mu=500`
- `mu=600`

with **3 events per pileup point**.

The resulting CKF candidate files (`event*-tracks_ckf.csv`) were converted into resolver-input JSON dumps and then benchmarked with:

- CPU resolver: `traccc_benchmark_resolver`
- GPU resolver: `traccc_benchmark_resolver_cuda`
- GPU resolver + graph reuse: `traccc_benchmark_resolver_cuda --reuse-eviction-graph`

### Result directory

Primary run:

- `results/20260419_193151_fatras_real_graph_reuse/`

Key files:

- `results/20260419_193151_fatras_real_graph_reuse/summary_table.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu400_no_reuse/summary.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu400_reuse/summary.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu500_no_reuse/summary.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu500_reuse/summary.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu600_no_reuse/summary.txt`
- `results/20260419_193151_fatras_real_graph_reuse/mu600_reuse/summary.txt`

### Scripts used

- `scripts/generate_fatras_high_pileup.sh`
- `scripts/convert_fatras_ckf_csv_to_dumps.py`
- `scripts/run_benchmark_from_dumps.sh`
- `scripts/run_fatras_ttbar_graph_reuse.sh`

---

## 2. Important methodological note

These are **real CKF candidate sets from Fatras ttbar events**, but the dumps were **not** produced via `traccc_seq_example --dump-ambiguity-input`.

That direct route was blocked because the locally generated Fatras directories contain populated `measurements.csv` and `tracks_ckf.csv`, but effectively empty `cells.csv`, so rerunning the full front of the chain through `traccc_seq_example` produced empty events.

Instead, the benchmark used a conversion step:

1. read `event*-tracks_ckf.csv`
2. extract the candidate measurement lists from `Measurements_ID`
3. reconstruct a resolver-style `pval` from `(chi2, ndf)` using a Wilson-Hilferty chi-square survival approximation
4. write the standard ambiguity-input JSON format

So the benchmark measures the resolver on **real high-pileup CKF candidates**, with a **reconstructed p-value proxy** rather than the original in-memory p-value from a native pre-resolver dump.

This is important to state explicitly in the thesis.

---

## 3. Aggregate results

### 3a. Mean timings

| pileup `mu` | mean `n_candidates` | CPU mean (ms) | GPU mean, no reuse (ms) | GPU mean, reuse (ms) | graph reuse speedup |
|---|---:|---:|---:|---:|---:|
| 400 | 2437.7 | 27.4615 | 18.9623 | 16.5642 | **1.145x** |
| 500 | 3109.7 | 37.8286 | 20.1616 | 20.0363 | 1.006x |
| 600 | 3954.7 | 54.0376 | 26.2859 | 26.1387 | 1.006x |

All aggregate runs show `hash_all_match=true`.

### 3b. GPU advantage vs CPU

Using:

```text
GPU advantage (%) = (CPU - GPU) / CPU × 100
```

the mean GPU advantage is:

| pileup `mu` | GPU advantage vs CPU, no reuse | GPU advantage vs CPU, reuse |
|---|---:|---:|
| 400 | **30.95% faster** | **39.68% faster** |
| 500 | **46.70% faster** | **47.03% faster** |
| 600 | **51.36% faster** | **51.63% faster** |

### 3c. Graph reuse benefit

Using:

```text
Graph reuse benefit (%) = (GPU_no_reuse - GPU_reuse) / GPU_no_reuse × 100
```

the mean graph-reuse benefit is:

| pileup `mu` | graph reuse benefit |
|---|---:|
| 400 | **12.65%** |
| 500 | **0.62%** |
| 600 | **0.56%** |

---

## 4. Per-event breakdown

### 4a. `mu=400`

| Event | `n_candidates` | CPU (ms) | GPU no reuse (ms) | GPU reuse (ms) | GPU faster than CPU (reuse) | graph reuse benefit |
|---|---:|---:|---:|---:|---:|---:|
| event 0 | 2655 | 31.1672 | 21.5130 | 18.7519 | 39.83% | 12.83% |
| event 1 | 2345 | 24.8458 | 17.9793 | 15.6894 | 36.85% | 12.74% |
| event 2 | 2313 | 26.3715 | 17.3946 | 15.2514 | 42.17% | 12.32% |

### 4b. `mu=500`

| Event | `n_candidates` | CPU (ms) | GPU no reuse (ms) | GPU reuse (ms) | GPU faster than CPU (reuse) | graph reuse benefit |
|---|---:|---:|---:|---:|---:|---:|
| event 0 | 3242 | 40.1994 | 21.3205 | 21.1761 | 47.32% | 0.68% |
| event 1 | 3022 | 36.0510 | 19.6635 | 19.5415 | 45.79% | 0.62% |
| event 2 | 3065 | 37.2353 | 19.5007 | 19.3912 | 47.92% | 0.56% |

### 4c. `mu=600`

| Event | `n_candidates` | CPU (ms) | GPU no reuse (ms) | GPU reuse (ms) | GPU faster than CPU (reuse) | graph reuse benefit |
|---|---:|---:|---:|---:|---:|---:|
| event 0 | 4008 | 54.6355 | 26.0185 | 25.9266 | 52.55% | 0.35% |
| event 1 | 3916 | 54.1623 | 28.2279 | 27.9683 | 48.36% | 0.92% |
| event 2 | 3940 | 53.3150 | 24.6113 | 24.5212 | 54.01% | 0.37% |

---

## 5. Main interpretation

### 5a. We now have a real dataset regime where GPU wins

This is the key result:

- at `mu=400`, `mu=500`, and `mu=600`, the GPU resolver is already faster than the CPU on real high-pileup Fatras ttbar-derived dumps
- the advantage is substantial, not marginal:
  - about **31%** at `mu=400`
  - about **47%** at `mu=500`
  - about **51%** at `mu=600`

This is the clearest real-data-backed demonstration so far that the GPU ambiguity resolver becomes worthwhile at sufficiently high occupancy.

### 5b. The crossover picture is consistent with the earlier synthetic analysis

The measured candidate counts are:

- `mu=400` → mean `n ≈ 2438`
- `mu=500` → mean `n ≈ 3110`
- `mu=600` → mean `n ≈ 3955`

These sit directly in and above the previously estimated crossover region (`n ≈ 2500–4000`), so the new real-event results align well with the synthetic and physics-calibrated narrative.

### 5c. Graph reuse helps most near the crossover

Graph reuse is most effective at `mu=400`:

- mean GPU time improves from `18.96 ms` to `16.56 ms`
- this is a **12.65%** speedup

At higher occupancies (`mu=500`, `mu=600`), graph reuse is still correct and still slightly beneficial, but the effect becomes very small (`~0.6%`).

This suggests:

- near the crossover, graph-management overhead is still visible
- deeper in the GPU-favorable regime, the runtime is increasingly dominated by the actual resolver work rather than graph instantiation overhead

### 5d. The next bottleneck is no longer graph instantiation

Once `mu` reaches `500–600`, reuse contributes very little, while the GPU already has a large lead over the CPU.

That strengthens the earlier conclusion that the next major ceiling is the structure of the eviction loop itself, especially:

- `remove_tracks<<<1,512>>>`
- the sequential greedy dependency chain inside the eviction phase

---

## 6. Thesis-ready takeaway

> On real high-pileup ODD Fatras ttbar-derived CKF candidate dumps, the CUDA ambiguity resolver on the Quadro GV100 outperforms the CPU baseline once the ambiguity-resolution input reaches roughly `n ≈ 2400+` candidates per event. In the new dump-based measurements, the GPU is about **31% faster at `mu=400`**, **47% faster at `mu=500`**, and **51% faster at `mu=600`**. CUDA graph reuse remains correctness-preserving and gives a **meaningful extra gain near crossover** (`12.65%` at `mu=400`), but its effect becomes negligible at larger occupancies (`~0.6%` at `mu=500–600`), indicating that graph instantiation is no longer the dominant cost there.

---

## 7. Recommended wording caveat for the thesis

If this section is cited in the thesis, it should say that:

- the events are **real Fatras ttbar full-chain outputs**
- the ambiguity inputs were built from **CKF candidate CSV outputs**
- the resolver `pval` was **reconstructed from `(chi2, ndf)`**
- therefore these are **real high-occupancy CKF candidate benchmarks**, but not native `--dump-ambiguity-input` snapshots produced directly by `traccc_seq_example`

That wording keeps the claim strong while staying fully accurate.
