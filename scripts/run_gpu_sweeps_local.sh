#!/usr/bin/env bash
# Run all GPU benchmark sweeps sequentially on the current GPU node.
# Sequential execution is required for accurate GPU benchmarking — parallel
# CUDA processes share SM resources and produce unreliable timings.
#
# Usage (on wn-lot-001 after: spack env activate traccc):
#   nohup bash scripts/run_gpu_sweeps_local.sh > /tmp/gpu_sweep.log 2>&1 &
#   tail -f /tmp/gpu_sweep.log          # monitor progress
#
# Flags:
#   --extended-only   skip n_it sweep and ODD benchmark
#   --n-it-only       skip extended sweep and ODD benchmark
#   --odd-only        skip extended and n_it sweeps
#
# Results land in:
#   $THESIS_RESULTS_ROOT/<RUN_ID>_extended_cuda/
#   $THESIS_RESULTS_ROOT/<RUN_ID>_n_it_sweep/
#   $THESIS_RESULTS_ROOT/<RUN_ID>_odd_muon_cuda/
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda"
if [[ ! -x "$TRACCC_BIN" ]]; then
    echo "ERROR: GPU benchmark binary not found at $TRACCC_BIN"; exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

RUN_EXTENDED=true
RUN_N_IT=true
RUN_ODD=true

for arg in "$@"; do
    case "$arg" in
        --extended-only) RUN_N_IT=false; RUN_ODD=false ;;
        --n-it-only)     RUN_EXTENDED=false; RUN_ODD=false ;;
        --odd-only)      RUN_EXTENDED=false; RUN_N_IT=false ;;
    esac
done

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

echo "=== Local GPU sweep runner ==="
echo "RUN_ID:    $RUN_ID"
echo "Binary:    $TRACCC_BIN"
echo "GPU:       $GPU_NAME"
echo "Started:   $(date)"
echo ""

TOTAL=0
DONE=0
if $RUN_EXTENDED; then TOTAL=$(( TOTAL + 30 )); fi
if $RUN_N_IT;     then TOTAL=$(( TOTAL + 90 )); fi
if $RUN_ODD; then
    N_ODD=$(ls "$SCRIPT_DIR/data/odd_muon_dumps/"*/event_*.json 2>/dev/null | wc -l)
    TOTAL=$(( TOTAL + N_ODD ))
fi

progress() {
    DONE=$(( DONE + 1 ))
    echo "  [${DONE}/${TOTAL}] $*"
}

# -----------------------------------------------------------------------
# Extended sweep — 10 n_candidates × 3 densities = 30 configs
# -----------------------------------------------------------------------
if $RUN_EXTENDED; then
    OUTDIR="$THESIS_RESULTS_ROOT/${RUN_ID}_extended_cuda"
    mkdir -p "$OUTDIR"
    echo "--- Extended sweep (30 configs) -> $OUTDIR"
    echo "    $(date)"

    for n in 100 500 1000 2000 3000 5000 7500 10000 20000 50000; do
        for density in low med high; do
            outfile="$OUTDIR/n${n}_${density}.txt"
            progress "n=$n density=$density"
            "$TRACCC_BIN" --synthetic \
                --n-candidates="$n" \
                --conflict-density="$density" \
                --repeats=10 --warmup=3 --profile \
                > "$outfile" 2>&1 \
                || echo "  FAILED (exit $?) — n=$n density=$density (logged to $outfile)"
        done
    done
    echo "  extended sweep done — $(ls "$OUTDIR"/*.txt | wc -l) files"
    echo ""
fi

# -----------------------------------------------------------------------
# n_it sensitivity sweep — 5 n × 3 densities × 6 n_it values = 90 configs
# -----------------------------------------------------------------------
if $RUN_N_IT; then
    OUTDIR="$THESIS_RESULTS_ROOT/${RUN_ID}_n_it_sweep"
    mkdir -p "$OUTDIR"
    echo "--- n_it sensitivity sweep (90 configs) -> $OUTDIR"
    echo "    $(date)"

    for n in 100 500 1000 5000 10000; do
        for density in low med high; do
            for n_it in 1 5 10 25 50 100; do
                outfile="$OUTDIR/n${n}_${density}_nit${n_it}.txt"
                progress "n=$n density=$density n_it=$n_it"
                "$TRACCC_BIN" --synthetic \
                    --n-candidates="$n" \
                    --conflict-density="$density" \
                    --n-it="$n_it" \
                    --repeats=10 --warmup=3 \
                    > "$outfile" 2>&1 \
                    || echo "  FAILED (exit $?) — n=$n density=$density n_it=$n_it"
            done
        done
    done
    echo "  n_it sweep done — $(ls "$OUTDIR"/*.txt | wc -l) files"
    echo ""
fi

# -----------------------------------------------------------------------
# ODD muon GPU benchmark — one result file per event dump
# -----------------------------------------------------------------------
if $RUN_ODD; then
    # Find the most recently modified subdirectory that actually contains JSON dumps.
    LATEST_DUMPS=$(for d in "$SCRIPT_DIR/data/odd_muon_dumps"/*/; do
        ls "$d"event_*.json 2>/dev/null | grep -q . && echo "$d"
    done | xargs -I{} stat --format="%Y {}" {} 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
    if [[ -z "$LATEST_DUMPS" ]]; then
        echo "No ODD dumps found in data/odd_muon_dumps/ — skipping."
    else
        OUTDIR="$THESIS_RESULTS_ROOT/${RUN_ID}_odd_muon_cuda"
        mkdir -p "$OUTDIR"
        echo "--- ODD GPU benchmark -> $OUTDIR"
        echo "    Dumps: $LATEST_DUMPS"
        echo "    $(date)"

        for dump in "$LATEST_DUMPS"event_*.json; do
            [[ -f "$dump" ]] || continue
            base="$(basename "$dump" .json)"
            outfile="$OUTDIR/${base}_cuda.txt"
            progress "$base"
            "$TRACCC_BIN" --input-dump="$dump" \
                --repeats=10 --warmup=3 --profile \
                > "$outfile" 2>&1 \
                || echo "  FAILED (exit $?) — $base"
        done
        echo "  ODD benchmark done — $(ls "$OUTDIR"/*.txt | wc -l) files"
        echo ""
    fi
fi

echo "=== All GPU sweeps complete ==="
echo "Finished:  $(date)"
echo "Results:   $THESIS_RESULTS_ROOT/"
ls -lhd "$THESIS_RESULTS_ROOT/${RUN_ID}_"* 2>/dev/null || true
