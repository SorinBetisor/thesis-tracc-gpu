# P1 – Infrastructure: Why

## Rationale

Before any work on traccc or ACTS can begin, the execution environment must be validated. The thesis requires compilation and execution of both CPU and GPU code on the Nikhef Stoomboot cluster. Without a verified environment, builds may fail in opaque ways, and batch submissions may not reach the intended nodes. P1 establishes that the cluster can compile, run, and submit jobs, and that a reproducible working layout exists for subsequent phases.

---

## Definitions and context

**Stoomboot** is the HTCondor-based batch system at Nikhef. Jobs are submitted from a login node and executed on worker nodes. The login node must not be used for compilation or long-running tasks; interactive CPU nodes (stbc-i1, stbc-i2, stbc-i3) are reserved for development, builds, and IDE connections. GPU nodes are split into interactive (wn-lot-001, for short sanity checks) and batch-only (wn-lot-002 and beyond, accessed via Condor).

**el9** refers to the Enterprise Linux 9 container image used on worker nodes. Compatibility with this image determines whether transferred binaries and scripts run correctly. A GLIBC mismatch between the login node and workers can cause silent failures when the wrong executable is transferred.

**$HOME** on the cluster has a small quota. Large repositories (ACTS, traccc) and build artifacts are stored under `/data/alice/sbetisor` to avoid quota exhaustion. The thesis repository remains in $HOME as a lightweight collection of scripts, documentation, and submit files.

---

## What was validated

| Component | Validation | Outcome |
|-----------|------------|---------|
| SSH and interactive CPU nodes | Connectivity and IDE Remote-SSH | Development and builds can proceed on stbc-i2. |
| HTCondor CPU job | Minimal C program submitted via Condor | Job ran on wn-sate-079; el9 image, gcc, and logging verified. Output: n=1000000, sum=2999997. |
| GLIBC mismatch | Transfer of /bin/bash as executable | Failed; fix: use wrapper script or disable executable transfer. |
| CUDA on wn-lot-001 | nvcc, nvidia-smi, minimal kernel | CUDA 12.5 selected; compiler and runtime functional. |
| Storage layout | Repos under /data, thesis in $HOME | Layout documented; migration to /project planned when available. |

---

## Outcome

The infrastructure is ready for traccc: cluster reachable, CPU and GPU jobs runnable, storage layout established. Technical steps for reproduction are in `docs/setup/p1-infrastructure-setup.md`.

---

## What follows

P2 uses this infrastructure to build and run traccc on CPU. See `docs/setup/p2_baseline.md`.
