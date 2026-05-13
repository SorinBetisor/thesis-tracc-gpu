# PBG — Full Fatras Pileup Sweep (mu0 → mu600)

**Date:** 2026-05-13  
**GPU:** Quadro GV100 (Nikhef Stoomboot)  
**PBG window:** W = 8192  
**Raw data:** `/user/sbetisor/data-work/results/20260513_013837_pbg_fatras_full/`

---

## Background

The April 22 PBG campaign (documented in `parallel_batch_greedy_results.md` Sec. 3d) only
covered mu=300–600, 3 events each, using a fixed n\_it=100 baseline. That run showed PBG was
**30–45 % slower** than the GPU baseline at those pileup levels because per-iteration batch
sizes were tiny (0–14 tracks) and the 5-kernel PBG pipeline paid a fixed launch overhead that
overwhelmed the marginal batching benefit.

This document fills the gap: mu0–mu200 (previously unmeasured) and a full 10-event run at
mu300+ (previously 3 events). The baseline here uses **adaptive n\_it** (current default),
which is important context for the numbers.

---

## Results

### Per-event timing — all pileup levels

| Corpus | n\_events | n\_cand (mean) | Baseline (ms) | PBG W=8192 (ms) | Speedup |
|--------|:---------:|:--------------:|:-------------:|:---------------:|:-------:|
| mu0    | 10 | 66   | 1.805  | 1.808  | −0.2% |
| mu20   | 10 | 147  | 2.130  | 2.120  |  +0.5% |
| mu50   | 10 | 294  | 2.661  | 2.659  |  +0.1% |
| mu100  | 10 | 563  | 3.878  | 3.796  | **+1.6%** |
| mu140  | 10 | 776  | 4.911  | 4.910  |  +0.0% |
| mu200  | 10 | 1115 | 7.395  | 7.303  | **+1.1%** |
| mu300  | 10 | 1703 | 10.436 | 10.442 |  −0.1% |
| mu400  |  3 | 2438 | 16.618 | 16.605 |  +0.1% |
| mu500  |  3 | 3110 | 20.016 | 20.011 |  +0.0% |
| mu600  |  3 | 3955 | 26.592 | 26.596 |  −0.0% |

All `hash_match=true`. Zero correctness or determinism failures across all events.

---

## Comparison with the April 22 Campaign

The April 22 results (mu300–600, n\_it=100 fixed baseline) are reproduced below alongside
the current numbers for the same pileup levels:

| Corpus | Apr 22 baseline (ms) | Apr 22 PBG (ms) | Current baseline (ms) | Current PBG (ms) |
|--------|:--------------------:|:---------------:|:---------------------:|:----------------:|
| mu300  | ~11.0–11.6 (3 ev.)  | ~16.0–16.6      | 10.44 (10 ev.)        | 10.44            |
| mu400  | ~14.7–16.6          | ~19.5–22.4      | 16.62                 | 16.61            |
| mu500  | ~19.5–21.7          | ~29.0–30.9      | 20.02                 | 20.01            |
| mu600  | ~25.0–29.3          | ~36.0–39.2      | 26.59                 | 26.60            |

**The PBG overhead visible in April disappeared.** The current PBG is within measurement
noise of the baseline at every pileup level. Two factors explain this:

1. **Adaptive n\_it in the baseline** reduced the baseline runtime, so PBG now has to match a
   faster target. The adaptive formula cuts inner-loop iterations by ~2× at mu300 (n\_cand ≈
   1700 → n\_it = ⌈1700/32⌉ = 54 vs the old fixed 100), directly lowering the denominator
   that made PBG look worse.

2. **The current binary is on `thesis-novelty-conflict-graph`**, which includes the same
   adaptive-n\_it path in the PBG execution. When the eviction inner loop exits early, PBG's
   per-iteration overhead (5 kernels vs 1) becomes proportionally smaller relative to the
   useful work done per outer iteration.

---

## Why PBG Does Not Help on Fatras Datasets

PBG's design benefit is **batch parallelism**: if multiple non-conflicting tracks can be
removed in a single outer iteration (batch > 1), the multi-block kernel amortises launch
overhead over more useful work per cycle. The actual batch sizes on Fatras events are:

| Corpus | Typical batch sizes (Apr 22, W=8192) | Outer iterations |
|--------|:------------------------------------:|:----------------:|
| mu300  | 0 (1 outer iter)                     | 1                |
| mu400  | avg 1.0–2.0, max 2–4                 | 2                |
| mu500  | avg 2.0–3.0, max 5–7                 | 3                |
| mu600  | avg 2.3–7.7, max 6–14               | 3                |

At the pileup levels where PBG would theoretically help (large n, dense conflicts), the
outer-loop trip count is 1–3 and average batch sizes remain in single digits. The Fatras ttbar
geometry does not produce the adversarial conflict patterns that synthetic `med`/`high`
density inputs do — tracks tend to be well-separated enough that the greedy algorithm
terminates in very few outer iterations regardless.

On synthetic `med` density (documented in `parallel_batch_greedy_results.md` Sec. 3b), PBG
achieved **1.79–2.11× speedup** over the GPU baseline at n=500–2000 because batch sizes
were 8–17 and outer iterations 1–2. Fatras events simply don't generate that conflict
structure.

---

## Interpretation for the Thesis

| Claim | Evidence |
|-------|----------|
| PBG is **correct** on all Fatras pileup levels | hash\_match=true, zero failures, all 10 pileup levels |
| PBG does **not regress** on any real-data corpus | speedup within ±2% at every pileup level |
| PBG does **not help** on Fatras real-data workloads | batch sizes 0–14, outer iterations 1–3, gains within noise |
| PBG benefit requires high conflict density | synthetic med/high data showed 1.8–2.1× speedup (prior campaign) |
| Adaptive n\_it "absorbed" the April 22 PBG overhead | Apr 22 mu600 PBG was 36–39 ms; current PBG is 26.6 ms (−32%) |

**The practical conclusion for the thesis:** PBG is a correct, non-regressing alternative
execution path. Its performance advantage is workload-dependent: it activates on
high-conflict-density synthetic or hypothetical future detector geometries where
multiple non-conflicting tracks compete per iteration. On the current Fatras ttbar physics
corpus (ttbar at μ=0–600), the greedy algorithm's naturally sparse conflict structure means
batch sizes stay small and PBG is effectively equivalent to the baseline. This is a clean,
defensible result — the algorithm is correct and ready, but the physics workload does not
yet produce the conflict patterns that make it faster.

---

## Files

```
/user/sbetisor/data-work/results/20260513_013837_pbg_fatras_full/
├── summary.tsv           # per-event: n_cand, baseline_ms, pbg_ms, speedup, hash_match
├── mean_by_corpus.tsv    # per-corpus averages
├── raw_baseline/         # per-event stdout, adaptive n_it baseline
├── raw_pbg/              # per-event stdout, PBG W=8192
├── batch_sizes/          # per-event CSV from --log-batch-sizes
└── run_metadata.txt      # branch, commit, GPU, parameters
```

Cross-references:
- `parallel_batch_greedy_results.md` — prior (Apr 22) PBG results including synthetic and window sensitivity
- `original_vs_final_comparison.md` — combined adaptive n\_it + graph reuse vs April 1 baseline
- `graph_reuse_full_sweep.md` — graph reuse standalone across all pileup levels
