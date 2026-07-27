# JP ambiguity resolution — correctness finding (FATRAS ttbar, GV100)

Date: 2026-07-17
Node: wn-lot-001.nikhef.nl (NVIDIA Quadro GV100)
Repo: /data/alice/sbetisor/traccc-jp  (build/bin/traccc_benchmark_resolver_cuda)
Data: /data/alice/sbetisor/data/fatras_csv_dumps/fatras_ttbar_mu{0..600}/event_000.json
Truth source: /data/alice/sbetisor/traccc/data/odd/fatras_ttbar_mu*/event000000000-tracks_ckf.csv

## Summary

The GPU Jones–Plassmann (JP) ambiguity resolver is **correct in two independent senses**:

1. **Reference correctness (vs CPU greedy):** JP reproduces the CPU greedy
   selection **byte-for-byte at every pileup point** (mu = 0…600):
   `hash_match=true`, selection Jaccard = 1, 0 cpu-only / 0 gpu-only tracks,
   and deterministic (3/3 determinism runs pass). GPU greedy also matches.
   The plain maximal-independent-set variant (`graph_mis`) diverges at
   mu >= 500 (hash_match=false, though still valid + dup-free); JP does not.
   => MIS dropped from scope; JP is the faithful GPU path.

2. **Truth correctness (vs the CKF `particleId` barcode):** with per-track
   truth files built from the CKF `particleId` + `good/duplicate/fake`
   columns, JP shows:
   - `fake_rate = 0` at all pileup (no fakes survive)
   - `duplicate_rate_post = 0` at all pileup (no measurement shared by >=2
     selected tracks — all combinatorial duplicates removed)
   - particle coverage (selected / distinct true particles) **>= 99%** at all
     pileup (100% up to mu=50).

## Data

Reference-correctness sweep:
`/user/sbetisor/data-work/results/20260717_001721_gpu_correctness_sweep/gpu_correctness.log`

Truth-validation sweep:
`/user/sbetisor/data-work/results/20260717_002112_jp_truth_validation/`
(`jp_truth.log` + `truth_files/`)

| mu | true particles P | input tracks N | dup in input | JP selected | fake_rate | dup_rate_post | coverage (sel/P) |
|----|------|------|------|--------|-----------|---------------|------|
| 0   | 63   | 74   | 11   | 63   | 0 | 0 | 100.0% |
| 20  | 167  | 185  | 18   | 167  | 0 | 0 | 100.0% |
| 50  | 336  | 382  | 46   | 336  | 0 | 0 | 100.0% |
| 100 | 641  | 791  | 150  | 640  | 0 | 0 | 99.84% |
| 140 | 826  | 1036 | 210  | 823  | 0 | 0 | 99.64% |
| 200 | 1040 | 1332 | 292  | 1038 | 0 | 0 | 99.81% |
| 300 | 1458 | 2030 | 572  | 1453 | 0 | 0 | 99.66% |
| 400 | 1825 | 2655 | 830  | 1812 | 0 | 0 | 99.29% |
| 500 | 2114 | 3242 | 1128 | 2095 | 0 | 0 | 99.10% |
| 600 | 2505 | 4008 | 1501 | 2484 | 0 | 0 | 99.16% |

## Caveats / honest notes

- The benchmark's built-in `selection_efficiency` field (0.66 at mu=600) is
  **track retention** (`selected_matched / (good+duplicate input tracks)`), not
  particle-level efficiency. It falls by design as duplicates are (correctly)
  removed. The physics-meaningful efficiency is the `coverage = sel/P` column
  above (>= 99%).
- At high pileup JP selects slightly fewer tracks than there are true particles
  (2484 vs 2505 at mu=600 → ~0.8% of particles lose their track). This is an
  **inherent property of hit-based greedy resolution** (two real particles
  sharing hits at high occupancy → one is dropped), NOT a JP defect: JP
  reproduces CPU greedy bit-for-bit.
- Not yet measured: whether any true particle keeps **two** hit-disjoint
  surviving tracks (`dup_rate_post=0` only rules out measurement-sharing
  survivors). Requires emitting selected track→particle mapping for an exact
  per-particle efficiency / duplicate-survivor count.
- FATRAS ttbar is nearly fake-free (0 fakes except 2 at mu=600). Fake rate is
  not discriminating on this sample; the Geant4 corpus (`data/odd_dumps/geant4_*`)
  would be needed for meaningful fake studies.
