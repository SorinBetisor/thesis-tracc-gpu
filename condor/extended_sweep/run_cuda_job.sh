#!/usr/bin/env bash
# Condor worker script — GPU resolver benchmark, one config per job.
# Called with: run_cuda_job.sh <n_candidates> <conflict_density> <outdir>
set -euo pipefail

N_CANDIDATES="$1"
DENSITY="$2"
OUTDIR="$3"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda"

mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/n${N_CANDIDATES}_${DENSITY}.txt"

echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "n_candidates=$N_CANDIDATES density=$DENSITY"

# One warm-up call before the timed run
"$TRACCC_BIN" --synthetic --n-candidates=1000 --conflict-density=med \
    --repeats=1 --warmup=1 > /dev/null 2>&1 || true

"$TRACCC_BIN" --synthetic \
    --n-candidates="$N_CANDIDATES" \
    --conflict-density="$DENSITY" \
    --repeats=10 --warmup=3 --profile \
    2>&1 | tee "$OUTFILE"
