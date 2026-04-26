#!/usr/bin/env bash
# Submit the full extended sweep (CPU + GPU) as parallel Condor array jobs.
# Each backend submits 30 jobs (10 n_candidates × 3 densities) running simultaneously.
#
# Usage:
#   cd condor/extended_sweep
#   ./submit_extended_sweep.sh              # both CPU + GPU
#   ./submit_extended_sweep.sh --cpu-only   # CPU only
#   ./submit_extended_sweep.sh --gpu-only   # GPU only
#
# Results land in:
#   $THESIS_RESULTS_ROOT/<RUN_ID>_extended/        (CPU)
#   $THESIS_RESULTS_ROOT/<RUN_ID>_extended_cuda/   (GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THESIS_REPO="${THESIS_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

CPU_OUTDIR="$THESIS_RESULTS_ROOT/${RUN_ID}_extended"
GPU_OUTDIR="$THESIS_RESULTS_ROOT/${RUN_ID}_extended_cuda"

SUBMIT_CPU=true
SUBMIT_GPU=true
for arg in "$@"; do
    case "$arg" in
        --cpu-only) SUBMIT_GPU=false ;;
        --gpu-only) SUBMIT_CPU=false ;;
    esac
done

mkdir -p "$SCRIPT_DIR/logs"

echo "=== Extended sweep submission ==="
echo "RUN_ID:   $RUN_ID"
if $SUBMIT_CPU; then echo "CPU out:  $CPU_OUTDIR"; fi
if $SUBMIT_GPU; then echo "GPU out:  $GPU_OUTDIR"; fi
echo ""

if $SUBMIT_CPU; then
    mkdir -p "$CPU_OUTDIR"
    echo "Submitting CPU sweep (30 jobs)..."
    condor_submit -a "OUTDIR=$CPU_OUTDIR" "$SCRIPT_DIR/sweep_cpu.submit"
    echo ""
fi

if $SUBMIT_GPU; then
    mkdir -p "$GPU_OUTDIR"
    echo "Submitting GPU sweep (30 jobs)..."
    condor_submit -a "OUTDIR=$GPU_OUTDIR" "$SCRIPT_DIR/sweep_cuda.submit"
    echo ""
fi

echo "=== Jobs submitted ==="
echo "Monitor with: condor_q"
echo "When done, collect results from the output directories above."
