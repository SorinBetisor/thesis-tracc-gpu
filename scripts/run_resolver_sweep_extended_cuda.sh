#!/usr/bin/env bash
# Extended GPU resolver benchmark sweep — 10 n_candidates points × 3 densities = 30 configs.
# Mirrors run_resolver_sweep_extended.sh for the CPU so results are directly comparable.
# Must run on a CUDA-capable node (e.g. wn-lot-001 interactive, or submit via Condor).
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
    exit 1
fi

THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_extended_cuda}"
OUTDIR="${OUTDIR:-$THESIS_REPO/results/$RUN_ID}"
mkdir -p "$OUTDIR"

PROFILE="${PROFILE:-1}"
PROFILE_FLAG=""
[[ "$PROFILE" == "1" ]] && PROFILE_FLAG="--profile"

echo "=== Extended GPU resolver sweep (10×3: n_candidates × conflict_density) ==="
echo "Binary:  $TRACCC_BIN"
echo "Output:  $OUTDIR"
echo "Profile: ${PROFILE_FLAG:-off}"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null | head -1 | tee "$OUTDIR/gpu_info.txt" || true
echo ""

# Warm-up GPU before any timed run
"$TRACCC_BIN" --synthetic --n-candidates=1000 --conflict-density=med \
    --repeats=1 --warmup=1 > /dev/null 2>&1 || true

for n in 100 500 1000 2000 3000 5000 7500 10000 20000 50000; do
    for density in low med high; do
        outfile="$OUTDIR/n${n}_${density}.txt"
        echo "n_candidates=$n conflict_density=$density -> $outfile"
        "$TRACCC_BIN" --synthetic --n-candidates="$n" \
            --conflict-density="$density" --repeats=10 --warmup=3 \
            ${PROFILE_FLAG} \
            2>&1 | tee "$outfile"
        echo ""
    done
done

echo "=== Extended GPU sweep complete ==="
echo "Results: $OUTDIR"
