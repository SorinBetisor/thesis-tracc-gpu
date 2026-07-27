# Datasets - What we benchmark on, what each represents, and why ttbar

**Scope:** the three input families used for every ambiguity-resolution benchmark in this thesis (synthetic, ODD Geant4 muon, ODD Fatras ttbar), what each is meant to represent, what it actually contains, and the reason ttbar is the primary physics dataset.

All inputs are consumed as frozen pre-resolver dumps (the state immediately before ambiguity resolution) so CPU and GPU see byte-identical candidates. The resolver only ever sees a set of track candidates plus their measurement (hit) memberships; the datasets differ in how those candidates are produced and therefore in candidate count and conflict structure.

---

## Summary


| Family           | Origin                                    | What it represents                     | Candidates/event (n)     | Conflict structure             | Role in thesis                                       |
| ---------------- | ----------------------------------------- | -------------------------------------- | ------------------------ | ------------------------------ | ---------------------------------------------------- |
| Synthetic        | Randomly generated, seed=42               | A controlled scaling knob, not physics | 100 - 50 000 (we set it) | Tunable (low/med/high density) | Isolate scaling and find the GPU/CPU crossover       |
| ODD Geant4 muon  | Geant4 full sim, single/10 isolated muons | Clean, low-occupancy events            | 10 - 91                  | Sparse, few shared hits        | Real-physics floor; the low-n regime where GPU loses |
| ODD Fatras ttbar | Fatras fast sim + Pythia8, tunable pileup | Production-like LHC occupancy          | 65 - 3 940 (mu 0 - 600)  | Realistic, grows with pileup   | Primary physics dataset; spans the crossover         |


---

## 1. Synthetic

**What it represents.** A controlled dial for problem size and conflict density. It is deliberately not physics. Its only job is to produce a clean scaling curve so we can answer "at what candidate count does the GPU resolver beat the CPU" without confounding it with per-event physics variation.


| Density | `max_meas_id` | Track length   | Effect                                              |
| ------- | ------------- | -------------- | --------------------------------------------------- |
| low     | 50 000        | 3-10 (avg 6.5) | Sparse; few candidates share hits                   |
| med     | 10 000        | 3-10 (avg 6.5) | Moderate sharing                                    |
| high    | 500           | 5-15 (avg 10)  | Dense; each hit claimed by ~20 candidates at n=1000 |


**What it is good for and its limits.** It gives the controlled crossover curve (GPU wins above n ~= 2000-3000). It cannot claim physics realism: the conflict graph is uniform-random branching, which does not reproduce the long chains and varying branch depths a real CKF produces. It is a scaling instrument, nothing more.

---

## 2. ODD Geant4 muon

**What it represents.** The clean, low-occupancy end of real physics. These are OpenDataDetector (ODD) events with Geant4 full simulation of a small fixed number of isolated muons at a fixed momentum. Isolated high-quality tracks produce very few combinatorial duplicates, so the ambiguity resolver has almost nothing to do. This is the realistic floor of the problem.

**What it contains.** Two multiplicities at five energies each:

- `geant4_1muon_{1,5,10,50,100}GeV`: n ~= 10-11 candidates/event.
- `geant4_10muon_{1,5,10,50,100}GeV`: n ~= 88-91 candidates/event.

Each candidate carries its full track state and measurement memberships. Because these are real detector hits, measurement IDs are sparse, non-contiguous detector indices and must be renumbered to `[0, N-1]` before the GPU resolver can use them.

**What it is good for and its limits.** It is the real-physics evidence that the GPU resolver loses at low multiplicity: on `geant4_10muon_1GeV` (n ~= 87) the GPU is ~6x slower than the CPU. Limit: by construction the muon events can never stress the resolver, because isolated muons do not generate the dense shared-hit conflicts that ambiguity resolution exists to solve.

---

## 3. ODD Fatras ttbar (primary dataset)

**What it represents.** Production-like LHC occupancy with a tunable pileup dial. ttbar is the standard busy-final-state benchmark for tracking, and overlaying pileup (`mu`) lets us scan candidate count continuously from a nearly empty event up through and beyond the GPU/CPU crossover, all with the same physics process and detector.

**What it contains.** Generated with ACTS v44 using Fatras fast simulation plus Pythia8 ttbar, written as CSV in the same schema as the official Geant4-based ODD datasets, then converted to pre-resolver dumps. Pileup levels mu in {0, 20, 50, 100, 140, 200, 300, 400, 500, 600}, 10-20 events per level.

Scaling is approximately linear in pileup:


| mu  | mean n_measurements | mean n_CKF_tracks | mean n_ambi_tracks | mean n_seeds |
| --- | ------------------- | ----------------- | ------------------ | ------------ |
| 0   | 1 653               | 56                | 51                 | 871          |
| 20  | 8 033               | 154               | 140                | 4 802        |
| 50  | 18 116              | 307               | 269                | 14 041       |
| 100 | 35 228              | 602               | 496                | 30 856       |
| 140 | 47 798              | 821               | 656                | 43 263       |
| 200 | 67 325              | 1 167             | 904                | 62 517       |
| 300 | 98 845              | 1 770             | 1 291              | 93 463       |


(measurements ~= 330 x mu, CKF tracks ~= 5.9 x mu, seeds ~= 312 x mu). The resolver operates on the final high-quality CKF candidates, not the raw seeds: strict CKF cuts (hit count, chi2/ndf, outlier fraction) keep only ~1.9 % of seeds, which is why candidate counts sit in the tens-to-thousands range rather than the tens-of-thousands of seeds.

**What it is good for and its limits.** It is the dataset that carries the crossover argument in real physics: the synthetic crossover at n ~= 2000-3000 maps to mu ~= 340-510 in Fatras ttbar, which brackets HL-LHC design pileup (mu ~= 200) and peak pileup (mu ~= 300). The one caveat is that Fatras is fast simulation, not Geant4, so the physics content is qualitatively similar but not identical to a full Geant4 ttbar sample. We used Fatras because the official Geant4 ttbar ODD sample was not accessible (CERN mirror returning 503), and Fatras let us generate the full pileup sweep locally in minutes.

---

## 4. Why we focused on ttbar

1. **It is the field-standard tracking benchmark.** ttbar produces a dense, high-multiplicity final state, which is exactly the regime where ambiguity resolution does real work. Reporting on ttbar makes the results directly comparable to the ACTS/traccc tracking literature.
2. **Pileup is a continuous occupancy knob.** Overlaying mu from 0 to 600 sweeps candidate count smoothly across the GPU/CPU crossover using one process and one detector. No other available dataset lets us vary occupancy this way while staying physical.
3. **Its occupancy maps onto the deployment targets.** The mu range covers standard LHC (mu ~= 50-80, CPU wins), HL-LHC (mu ~= 200-300, approaching and crossing break-even), and by extrapolation Pb-Pb heavy-ion occupancy (mu >> 1000, GPU wins decisively). This is what turns a microbenchmark into a statement about when GPU ambiguity resolution is worth deploying.

