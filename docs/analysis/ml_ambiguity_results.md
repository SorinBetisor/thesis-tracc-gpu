# ML ambiguity resolution on the FATRAS ttbar events

**Date:** 2026-07-17
**Runner:** `/user/sbetisor/data-work/scripts/ml_ambiguity_runner.py`
**Model:** ACTS `duplicateClassifier.onnx` (MLP, 8 raw inputs, `Normalise` baked in), run via `onnxruntime 1.23.2` (CPU).
**Data:** `/data/alice/sbetisor/traccc/data/odd/fatras_ttbar_mu*/event*-tracks_ckf.csv`
— the **same** events the traccc greedy / JP / SB resolvers were benchmarked on.
**Raw results:** `/user/sbetisor/data-work/results/20260717_011511_ml_ambiguity_fatras/`

## What this is

A faithful port of ACTS' `MLAmbiguityResolution/ambiguity_solver_full_chain.py`:
DBSCAN over `(eta, phi)` + shared-hit sub-clustering, then the MLP scores every
track and the highest-scoring track per cluster is kept. The only deviations
from upstream are (a) loading the ONNX via onnxruntime instead of the absent
`.pt` checkpoint, and (b) pointing it at our FATRAS CSVs.

### Feature contract (verified against `CsvTrackWriter.cpp`)

The MLP consumes exactly 8 raw features, in CSV column order:

```
nStates, nMeasurements, nOutliers, nHoles, ndf, chi2/ndf, eta, phi
```

`seed_id` and `Measurements_ID` (current writer's rename of the old `Hits_ID`)
are dropped, matching the older schema the model was trained on.

## Results (aggregate per mu)


| mu  | events | tracks | dup_in % | eff (good) % | eff (particle) % | dup_out % | fake_out % | cluster ms/ev | infer ms/ev |
| --- | ------ | ------ | -------- | ------------ | ---------------- | --------- | ---------- | ------------- | ----------- |
| 20  | 20     | 3083   | 8.99     | 99.29        | 99.75            | 0.04      | 0.00       | 31.1          | 0.13        |
| 50  | 20     | 6146   | 12.40    | 98.96        | 99.85            | 0.04      | 0.00       | 82.8          | 0.14        |
| 100 | 20     | 12042  | 17.46    | 98.31        | 99.82            | 0.02      | 0.00       | 171.7         | 0.19        |
| 200 | 20     | 23356  | 22.52    | 97.67        | 99.80            | 0.04      | 0.00       | 290.3         | 0.28        |
| 300 | 20     | 35399  | 27.03    | 96.82        | 99.79            | 0.03      | 0.00       | 329.4         | 0.37        |
| 400 | 3      | 7312   | 30.05    | 95.91        | 99.69            | 0.02      | 0.00       | 388.5         | 0.53        |
| 600 | 3      | 11863  | 37.81    | 94.58        | 99.63            | 0.07      | 0.00       | 388.4         | 0.78        |


## Reading of the numbers

- **Duplicate suppression is excellent:** input duplicate rate rises from ~9%
(mu20) to ~38% (mu600); ML drives the *output* duplicate rate to ~0% across
the whole range. This is the metric that is directly comparable to
greedy / JP / SB.
- **Efficiency degrades gracefully with pileup:** good-track efficiency
99.3% → 94.6%; particle-reconstruction efficiency stays ~99.6–99.8%.
- **Fakes are ~0** because FATRAS ttbar has essentially no fakes — the fake-rate
axis is not discriminating on this sample (as expected).
- **Cost is dominated by clustering, not the network.** NN inference is tiny
(0.13–0.78 ms/event) and scales mildly with track count; DBSCAN +
Python sub-clustering is 30–390 ms/event and dominates. This is the key
ML-specific caveat for any latency comparison.

## Positioning for the paper

- ML belongs as a **reference / context column**, *not* in the strict
selection-identity head-to-head with greedy/JP. Its grouping objective
(DBSCAN in direction space + hit sharing) differs from the shared-measurement
conflict graph, so it is not a drop-in replacement — same situation as SB.
- Report ML's **quality** (duplicate suppression, efficiency) on the identical
FATRAS events. Do **not** put its ms/event into the GPU latency race: the
headline time is Python clustering overhead, and the model itself is CPU
onnxruntime, not the C++/CUDA path.
- If a latency figure for ML is wanted, quote **NN inference alone**
(sub-millisecond) separately from clustering, exactly as ACTS' own script
splits them.

## Cross-method comparison (same FATRAS events)

Greedy/JP numbers from
`/user/sbetisor/data-work/results/20260618_131900_rerun_fatras_throughput/pileup_aggregate.tsv`
(GV100, commit `c13cfbb4`); ML numbers from this run. Candidate counts match
per event (e.g. mu600 ≈ 3955/ev in both), confirming identical events.

**Method matrix.** Greedy exists on CPU + GPU; JP is a GPU-only CUDA
conflict-graph path (verified: no `core/` host JP on any branch, flag only in
`examples/run/cuda/benchmark_resolver_cuda.cpp`); ML here is CPU onnxruntime.


|        | CPU                  | GPU                |
| ------ | -------------------- | ------------------ |
| Greedy | reference            | yes                |
| JP     | n/a (does not exist) | yes (CUDA)         |
| ML     | yes (onnxruntime)    | not available here |


### Table 1 — Latency (ms/event) & throughput (events/s, in parens)


| μ   | cand/ev | Greedy CPU  | Greedy GPU (res) |     | JP GPU (res) |     | ML infer    | ML total (cluster+infer) |
| --- | ------- | ----------- | ---------------- | --- | ------------ | --- | ----------- | ------------------------ |
| 20  | 147     | 0.92 (1087) | 2.05             |     | 2.69         |     | 0.13 (7692) | 31.2 (32)                |
| 50  | 294     | 2.04 (490)  | 2.79             |     | 3.69         |     | 0.14 (7194) | 83.0 (12)                |
| 100 | 563     | 4.34 (230)  | 3.80             |     | 4.60         |     | 0.19 (5136) | 171.9 (5.8)              |
| 200 | 1115    | 9.93 (101)  | 7.33             |     | 7.34         |     | 0.28 (3614) | 290.6 (3.4)              |
| 300 | 1703    | 16.45 (61)  | 10.48            |     | 10.90        |     | 0.37 (2667) | 329.8 (3.0)              |
| 400 | 2438    | 27.08 (37)  | 16.70            |     | 9.96         |     | 0.53 (1890) | 389.0 (2.6)              |
| 600 | 3955    | 53.02 (19)  | 26.61            |     | 16.23        |     | 0.78 (1281) | 389.1 (2.6)              |


- Greedy CPU = resolver only (no transfer). GPU columns: `res` = resolver-only,
`e2e` = H2D + resolver + D2H. ML `total` = DBSCAN clustering + NN inference
(CPU); ML `infer` = network only.
- Full ML pipeline is ~7–30× slower than greedy/JP (clustering-bound). NN
inference alone is sub-ms but does no conflict resolution by itself.

### Table 2 — Truth-based quality (%), same events

Greedy/JP select identically to the CPU greedy reference (`hash_match=true`, all
79 events) which we established equals truth, so their duplicate-out / efficiency
columns are the truth values themselves.


| μ   | dup in | Greedy/JP dup out | ML dup out | Greedy/JP eff | ML eff (good) | ML eff (particle) | fake out (all) |
| --- | ------ | ----------------- | ---------- | ------------- | ------------- | ----------------- | -------------- |
| 20  | 8.99   | ≡ truth           | 0.04       | ≡ truth       | 99.29         | 99.75             | 0.00           |
| 50  | 12.40  | ≡ truth           | 0.04       | ≡ truth       | 98.96         | 99.85             | 0.00           |
| 100 | 17.46  | ≡ truth           | 0.02       | ≡ truth       | 98.31         | 99.82             | 0.00           |
| 200 | 22.52  | ≡ truth           | 0.04       | ≡ truth       | 97.67         | 99.80             | 0.00           |
| 300 | 27.03  | ≡ truth           | 0.03       | ≡ truth       | 96.82         | 99.79             | 0.00           |
| 400 | 30.05  | ≡ truth           | 0.02       | ≡ truth       | 95.91         | 99.69             | 0.00           |
| 600 | 37.81  | ≡ truth           | 0.07       | ≡ truth       | 94.58         | 99.63             | 0.00           |


