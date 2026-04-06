#!/usr/bin/env bash
# Submit GPU benchmarks for all event dump files in a directory.
# Generates dump_list.txt (one path per line) and submits the Condor array job.
#
# Usage:
#   cd condor/odd_benchmark
#   ./submit_odd_gpu_benchmark.sh --dumps-dir /path/to/dumps --outdir /path/to/results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DUMPS_DIR=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dumps-dir) DUMPS_DIR="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$DUMPS_DIR" || -z "$OUTDIR" ]]; then
    echo "Usage: $0 --dumps-dir DIR --outdir DIR"
    exit 1
fi

if [[ ! -d "$DUMPS_DIR" ]]; then
    echo "Dumps directory not found: $DUMPS_DIR"
    exit 1
fi

mkdir -p "$OUTDIR"
mkdir -p "$SCRIPT_DIR/logs"

DUMP_LIST="$SCRIPT_DIR/dump_list.txt"
ls -1 "$DUMPS_DIR"/event_*.json > "$DUMP_LIST"
N_JOBS=$(wc -l < "$DUMP_LIST")

if [[ "$N_JOBS" -eq 0 ]]; then
    echo "No event_*.json files found in $DUMPS_DIR"
    exit 1
fi

echo "=== ODD GPU benchmark submission ==="
echo "Dumps:    $DUMPS_DIR ($N_JOBS files)"
echo "Output:   $OUTDIR"
echo "Job list: $DUMP_LIST"
echo ""

condor_submit -a "OUTDIR=$OUTDIR" "$SCRIPT_DIR/sweep_gpu_dumps.submit"

echo ""
echo "Submitted $N_JOBS GPU jobs."
echo "Monitor with: condor_q"
