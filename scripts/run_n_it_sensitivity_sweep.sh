#!/usr/bin/env bash
# n_it sensitivity sweep: fix n_candidates in {100,500,1000,5000,10000} and
# sweep --n-it in {1,5,10,25,50,100}.  Three conflict densities = 90 configs.
# Purpose: quantify eviction loop over-execution and find the optimal per-regime
# n_it value that minimises GPU time without correctness loss (hash_match=true).
# Requires a CUDA-capable node.
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
TRACCC_BIN="${TRACCC_BIN:-$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda}"

if [[ ! -x "$TRACCC_BIN" ]]; then
    echo "traccc_benchmark_resolver_cuda not found at $TRACCC_BIN"
    exit 1
fi

THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_n_it_sweep}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$RUN_ID}"
mkdir -p "$OUTDIR"

echo "=== n_it sensitivity sweep ==="
echo "Binary:  $TRACCC_BIN"
echo "Output:  $OUTDIR"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1 | tee "$OUTDIR/gpu_info.txt" || true
echo ""

# Warm-up
"$TRACCC_BIN" --synthetic --n-candidates=1000 --conflict-density=med \
    --repeats=1 --warmup=1 > /dev/null 2>&1 || true

for n in 100 500 1000 5000 10000; do
    for density in low med high; do
        for n_it in 1 5 10 25 50 100; do
            outfile="$OUTDIR/n${n}_${density}_nit${n_it}.txt"
            printf "n=%-6d density=%-4s n_it=%-3d -> %s\n" \
                "$n" "$density" "$n_it" "$(basename $outfile)"
            "$TRACCC_BIN" --synthetic \
                --n-candidates="$n" \
                --conflict-density="$density" \
                --n-it="$n_it" \
                --repeats=10 --warmup=3 \
                2>&1 | tee "$outfile"
            echo ""
        done
    done
done

echo "=== n_it sensitivity sweep complete ==="
echo "Results: $OUTDIR"
