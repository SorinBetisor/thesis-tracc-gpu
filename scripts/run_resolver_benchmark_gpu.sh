#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_BIN="${TRACCC_BIN:-}"
if [[ -z "$TRACCC_BIN" ]]; then
  TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
  TRACCC_BIN="$TRACCC_SRC/build/bin/traccc_benchmark_resolver"
fi

echo "=== GPU Resolver benchmark ==="
echo "host=$(hostname)"
nvidia-smi || true
echo ""

"$TRACCC_BIN" --synthetic --n-candidates=10000 --conflict-density=med \
  --backend=gpu --repeats=10 --warmup=3 2>&1 || {
  echo "Note: GPU backend requires traccc built with TRACCC_BUILD_CUDA=ON"
  echo "Falling back to CPU..."
  "$TRACCC_BIN" --synthetic --n-candidates=10000 --conflict-density=med \
    --backend=cpu --repeats=10 --warmup=3 2>&1
}
