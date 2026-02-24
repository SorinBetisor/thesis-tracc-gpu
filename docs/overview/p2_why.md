# P2 – Baseline: Why

## Rationale

A reproducible CPU baseline is a prerequisite for any GPU comparison. Claims about GPU acceleration require a reference: the CPU performance of the same workload must be known. P2 establishes that traccc can be built and that the full reconstruction chain runs end-to-end on CPU. Without this baseline, later phases cannot attribute performance differences to the ambiguity resolver or to GPU vs CPU execution.

---

## Definitions and context

**traccc** is the track reconstruction software in the ACTS ecosystem, designed for accelerator-based (GPU) execution. The sequential example (`traccc_seq_example`) runs the full chain on a single CPU thread: read cells, clusterization, spacepoint formation, seeding, track finding (CKF), ambiguity resolution, and track fitting. It serves as the reference implementation for correctness and as the baseline for performance comparison.

**host-fp32** is a CMake preset for CPU builds with single-precision floating point and ROOT enabled. It produces binaries that run on the host without CUDA. The preset, together with documented cmake flags, ensures reproducibility across sessions and machines.

**Spack** is a package manager used when system packages (e.g. TBB, Boost) are unavailable. The cluster does not provide sudo; Spack installs dependencies into a user-controlled directory. The traccc Spack environment pins versions of TBB, Boost, ROOT, and ACTS, avoiding ABI mismatches that arise when different parts of the stack use incompatible library versions (e.g. nlohmann_json).

**ODD** (Open Data Detector) is a simplified geometry used for development and validation. The standard test data set (`odd/geant4_10muon_10GeV`) contains simulated events from this geometry.

---

## What was achieved

| Component | Achievement | Purpose |
|-----------|-------------|---------|
| Spack environment | traccc dependencies installed and activated | Controlled, reproducible dependency stack without system packages. |
| CMake configuration | host-fp32 preset with Spack libs and nlohmann_json alignment | Build reproduces across sessions; ABI issues avoided. |
| traccc_seq_example | Full chain runs to completion | Baseline correctness and performance established. |
| Documented commands | Exact cmake and run commands in setup doc | Reproducibility for thesis and future work. |

---

## Outcome

The CPU baseline is established. The ambiguity resolution step can be isolated and benchmarked in isolation. Technical steps are in `docs/setup/p2_baseline.md`.

---

## What follows

P3 adds a benchmark protocol for ambiguity resolution so CPU and GPU can be compared on identical inputs. See `docs/setup/p3_benchmark_protocol.md`.
