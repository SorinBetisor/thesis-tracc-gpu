#!/usr/bin/env bash
# Condor worker — single n_it sensitivity config.
# Called with: run_n_it_job.sh <n_candidates> <density> <n_it> <outdir>
set -euo pipefail

N_CANDIDATES="$1"
DENSITY="$2"
N_IT="$3"
OUTDIR="$4"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda"
mkdir -p "$OUTDIR"

OUTFILE="$OUTDIR/n${N_CANDIDATES}_${DENSITY}_nit${N_IT}.txt"
echo "host=$(hostname) n=$N_CANDIDATES density=$DENSITY n_it=$N_IT"

"$TRACCC_BIN" --synthetic \
    --n-candidates="$N_CANDIDATES" \
    --conflict-density="$DENSITY" \
    --n-it="$N_IT" \
    --repeats=10 --warmup=3 \
    2>&1 | tee "$OUTFILE"
