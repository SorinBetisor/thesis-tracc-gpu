# Infrastructure & environment validation

## Goal of this phase

Goal: verify that CPU batch jobs can be compiled and executed on Nikhef Stoomboot
and establish a working directory layout for ACTS/traccc development.

## Storage

- No writable /project area yet.
- Temporary workspace: /data/alice/sbetisor.
- $HOME used only for lightweight thesis repo and configuration files.

## HTCondor CPU sanity check

- Submitted a minimal C program via HTCondor.
- Job executed on worker node (wn-sate-079).
- Compilation and execution performed inside job sandbox.
- Output verified: n=1000000, sum=2999997.
- Confirms:
  - el9 image usable
  - gcc available on worker nodes
  - file transfer + logging works correctly

Submit file: `compile_and_run.submit`

## Known issues & fixes

- GLIBC mismatch occurred when /bin/bash was transferred as executable.
- Cause: login-node bash incompatible with worker container.
- Fix: use wrapper script (run.sh) as executable, or disable executable transfer.

## Repository layout decision

- ACTS and traccc cloned into /data/alice/sbetisor as sibling repositories.
- Thesis repo remains lightweight and independent.
- This layout will be migrated to /project/<group> once available.
