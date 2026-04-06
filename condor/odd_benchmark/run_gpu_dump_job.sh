#!/usr/bin/env bash
# Condor worker — GPU resolver benchmark on one event dump file.
# Called with: run_gpu_dump_job.sh <dump_path> <outdir>
set -euo pipefail

DUMP_PATH="$1"
OUTDIR="$2"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda"
mkdir -p "$OUTDIR"

BASENAME="$(basename "$DUMP_PATH" .json)"
OUTFILE="$OUTDIR/${BASENAME}_cuda.txt"

echo "host=$(hostname)"
echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "dump=$DUMP_PATH"

"$TRACCC_BIN" --input-dump="$DUMP_PATH" --repeats=10 --warmup=3 --profile \
    2>&1 | tee "$OUTFILE"
