#!/usr/bin/env bash
# GPU resolver profiling script — runs with nsys (Nsight Systems) and optionally ncu
# (Nsight Compute) to produce full GPU timeline + kernel-level metrics.
#
# Usage:
#   ./run_resolver_profile_nsys.sh [--nsys-only] [--ncu-only] [--n-candidates=N]
#                                   [--conflict-density=low|med|high]
#
# Outputs are written to $OUTDIR (default: $THESIS_RESULTS_ROOT/<timestamp>_profile_nsys/).
#   <n>_<density>.nsys-rep  — Nsight Systems report  (open with nsys-ui or nsys stats)
#   <n>_<density>.ncu-rep   — Nsight Compute report   (open with ncu-ui or ncu --import)
#   <n>_<density>_profile.txt — benchmark --profile text output (profile_*_ms fields)
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
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_profile_nsys}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$RUN_ID}"
mkdir -p "$OUTDIR"

N_CANDIDATES=10000
CONFLICT_DENSITY="med"
RUN_NSYS=true
RUN_NCU=true

for arg in "$@"; do
    case "$arg" in
        --nsys-only)         RUN_NCU=false ;;
        --ncu-only)          RUN_NSYS=false ;;
        --n-candidates=*)    N_CANDIDATES="${arg#*=}" ;;
        --conflict-density=*) CONFLICT_DENSITY="${arg#*=}" ;;
    esac
done

BENCH_ARGS="--synthetic --n-candidates=$N_CANDIDATES --conflict-density=$CONFLICT_DENSITY --repeats=5 --warmup=2 --profile"
BASE="$OUTDIR/n${N_CANDIDATES}_${CONFLICT_DENSITY}"

echo "=== GPU Resolver profiling ==="
echo "Host:    $(hostname)"
echo "Binary:  $TRACCC_BIN"
echo "Config:  n_candidates=$N_CANDIDATES  conflict_density=$CONFLICT_DENSITY"
echo "Output:  $OUTDIR"
echo ""

nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null | head -1 | tee "$OUTDIR/gpu_info.txt" || true
echo ""

# Warm-up the GPU before any profiled run
"$TRACCC_BIN" --synthetic --n-candidates=1000 --conflict-density=med \
    --repeats=1 --warmup=1 > /dev/null 2>&1 || true

# ------------------------------------------------------------------
# Benchmark --profile: emit profile_*_ms text output
# ------------------------------------------------------------------
echo "--- benchmark --profile run ---"
"$TRACCC_BIN" $BENCH_ARGS 2>&1 | tee "${BASE}_profile.txt"
echo ""

# ------------------------------------------------------------------
# nsys: full CUDA timeline with NVTX phase ranges
# ------------------------------------------------------------------
if $RUN_NSYS; then
    if ! command -v nsys &>/dev/null; then
        echo "nsys not found in PATH — skipping Nsight Systems profiling."
        echo "Add /path/to/nsight-systems/bin to PATH, or set: export PATH=\$CUDA_HOME/../nsight-systems-*/bin:\$PATH"
    else
        echo "--- nsys profile ---"
        nsys profile \
            --trace=cuda,nvtx \
            --gpu-metrics-device=all \
            --output="${BASE}" \
            --force-overwrite=true \
            --stats=true \
            "$TRACCC_BIN" $BENCH_ARGS \
            2>&1 | tee "${BASE}_nsys_stats.txt"
        echo ""
        echo "nsys report: ${BASE}.nsys-rep"
        echo "View with:   nsys-ui ${BASE}.nsys-rep"
        echo "  or stats:  nsys stats --report gputrace ${BASE}.nsys-rep"
        echo ""
    fi
fi

# ------------------------------------------------------------------
# ncu: per-kernel metrics for the eviction-loop graph kernels
# ------------------------------------------------------------------
if $RUN_NCU; then
    if ! command -v ncu &>/dev/null; then
        echo "ncu not found in PATH — skipping Nsight Compute profiling."
        echo "Add /path/to/nsight-compute/bin to PATH."
    else
        echo "--- ncu profile ---"
        # Use --kernel-name-base=function to match by function name.
        # Target the main eviction kernels by regex; --launch-count limits overhead.
        ncu \
            --target-processes all \
            --kernel-name "remove_tracks|update_status|rearrange_tracks|fill_vectors|fill_tracks_per_measurement|count_shared_measurements|fill_track_candidates" \
            --set full \
            --launch-count 1 \
            --export "${BASE}_ncu" \
            --force-overwrite \
            "$TRACCC_BIN" $BENCH_ARGS \
            2>&1 | tee "${BASE}_ncu_summary.txt"
        echo ""
        echo "ncu report: ${BASE}_ncu.ncu-rep"
        echo "View with:  ncu-ui ${BASE}_ncu.ncu-rep"
        echo "  or print: ncu --import ${BASE}_ncu.ncu-rep --print-summary"
        echo ""
    fi
fi

echo "=== Profiling complete ==="
echo "Results in: $OUTDIR"
