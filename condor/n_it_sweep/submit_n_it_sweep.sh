#!/usr/bin/env bash
# Submit the n_it sensitivity sweep (90 GPU jobs) as a Condor array job.
#
# Usage:
#   cd condor/n_it_sweep
#   ./submit_n_it_sweep.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
THESIS_REPO="${THESIS_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${RUN_ID}_n_it_sweep}"

mkdir -p "$OUTDIR"
mkdir -p "$SCRIPT_DIR/logs"

echo "=== n_it sensitivity sweep submission ==="
echo "RUN_ID:  $RUN_ID"
echo "Output:  $OUTDIR"
echo "Jobs:    90 (5 n × 3 density × 6 n_it)"
echo ""

condor_submit -a "OUTDIR=$OUTDIR" "$SCRIPT_DIR/sweep_n_it.submit"

echo ""
echo "Monitor with: condor_q"
