# Next steps

Concrete actions to move from P3 (benchmark protocol ✅) to GPU integration and RQ experiments.

---

## Immediate

### 1. GPU backend enablement

- [ ] Build traccc with CUDA: `-DTRACCC_BUILD_CUDA=ON`
- [ ] Make `--backend=gpu` actually run the GPU path in `traccc_benchmark_resolver`
- [ ] Add GPU memory metric (`cudaMemGetInfo` or allocator stats)

### 2. CPU↔GPU base test case

- [ ] Choose 1–3 dumped events from the standard dataset (e.g. geant4_10muon_10GeV)
- [ ] Run CPU resolver on dump (reference)
- [ ] Run GPU resolver on same dump
- [ ] Compare hashes + metrics (latency, throughput, memory)

### 3. Scaling sweeps

- [ ] Synthetic grid expanded (e.g. up to 50k candidates)
- [ ] Identify crossover point (where GPU wins over CPU)
- [ ] Run via Condor on GPU (and CPU) nodes

---

## Later

- End-to-end GPU chain (`traccc_seq_example_cuda`)
- Plots and tables for thesis experiments chapter
- Fix commit hashes for reproducibility
