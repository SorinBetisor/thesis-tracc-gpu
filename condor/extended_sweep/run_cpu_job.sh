#!/usr/bin/env bash
# Condor worker script — CPU resolver benchmark, one config per job.
# Called with: run_cpu_job.sh <n_candidates> <conflict_density> <outdir>
set -euo pipefail

N_CANDIDATES="$1"
DENSITY="$2"
OUTDIR="$3"

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver"

mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/n${N_CANDIDATES}_${DENSITY}.txt"

echo "host=$(hostname) n_candidates=$N_CANDIDATES density=$DENSITY"

"$TRACCC_BIN" --synthetic \
    --n-candidates="$N_CANDIDATES" \
    --conflict-density="$DENSITY" \
    --repeats=10 --warmup=3 --profile \
    2>&1 | tee "$OUTFILE"
