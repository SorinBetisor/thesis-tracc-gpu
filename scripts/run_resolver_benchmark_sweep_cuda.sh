#!/usr/bin/env bash
# GPU resolver benchmark sweep — requires a CUDA build and a GPU node.
# Run this on wn-lot-001 (interactive) or via HTCondor (batch).
# Produces the same output format as run_resolver_benchmark_sweep.sh (CPU)
# plus extra GPU fields: backend, time_h2d_ms, time_d2h_ms, cpu_hash, gpu_hash, hash_match.
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_BIN="${TRACCC_BIN:-}"
if [[ -z "$TRACCC_BIN" ]]; then
  TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
  TRACCC_BIN="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
fi

if [[ ! -x "$TRACCC_BIN" ]]; then
  echo "traccc_benchmark_resolver_cuda not found at $TRACCC_BIN"
  echo "Build steps on a GPU node:"
  echo "  . /data/alice/sbetisor/spack/share/spack/setup-env.sh"
  echo "  spack env activate traccc"
  echo "  cd /data/alice/sbetisor/traccc/build"
  echo "  cmake -DTRACCC_BUILD_CUDA=ON -DTRACCC_BUILD_CUDA_UTILS=ON .."
  echo "  make traccc_benchmark_resolver_cuda -j\$(nproc)"
  exit 1
fi

THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_cuda}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$RUN_ID}"
mkdir -p "$OUTDIR"

echo "=== GPU Resolver benchmark sweep (3x3: n_candidates x conflict_density) ==="
echo "Host:    $(hostname)"
echo "Binary:  $TRACCC_BIN"
echo "Output:  $OUTDIR"
echo ""

# GPU info
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader 2>/dev/null | head -1 | tee "$OUTDIR/gpu_info.txt" || true
echo ""

# One silent warmup call to bring the GPU to steady state before measuring
"$TRACCC_BIN" --synthetic --n-candidates=1000 --conflict-density=med \
  --repeats=1 --warmup=1 > /dev/null 2>&1 || true

for n in 1000 5000 10000; do
  for density in low med high; do
    outfile="$OUTDIR/n${n}_${density}.txt"
    echo "Running n_candidates=$n conflict_density=$density -> $outfile"
    "$TRACCC_BIN" --synthetic --n-candidates="$n" \
      --conflict-density="$density" --repeats=10 --warmup=3 \
      2>&1 | tee "$outfile"
    echo ""
  done
done

echo "=== Sweep complete ==="
echo "Results: $OUTDIR"
