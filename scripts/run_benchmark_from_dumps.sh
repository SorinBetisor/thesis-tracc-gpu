#!/usr/bin/env bash
# Run CPU (and optionally GPU) resolver benchmark on a directory of JSON event dumps.
# Produces one result file per event per backend.
# GPU mode requires a CUDA-capable node (use --gpu flag explicitly to enable).
#
# Usage:
#   ./run_benchmark_from_dumps.sh --dumps-dir DIR [--cpu] [--gpu] [--outdir DIR]
#
# Defaults:
#   --cpu   always on
#   --gpu   off (must pass --gpu explicitly; requires CUDA node)
#   --outdir <dumps-dir>/../benchmark_<timestamp>
#
# Environment overrides:
#   TRACCC_SRC   path to traccc build root (default /data/alice/sbetisor/traccc)
set -euo pipefail

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
CPU_BIN="$TRACCC_SRC/build/bin/traccc_benchmark_resolver"
GPU_BIN="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"

DUMPS_DIR=""
OUTDIR=""
RUN_CPU=true
RUN_GPU=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dumps-dir) DUMPS_DIR="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --cpu) RUN_CPU=true; shift ;;
        --gpu) RUN_GPU=true; shift ;;
        --no-cpu) RUN_CPU=false; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$DUMPS_DIR" ]]; then
    echo "Usage: $0 --dumps-dir DIR [--gpu] [--outdir DIR]"
    exit 1
fi

if [[ ! -d "$DUMPS_DIR" ]]; then
    echo "Dumps directory not found: $DUMPS_DIR"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$(dirname "$DUMPS_DIR")/benchmark_${TIMESTAMP}}"
mkdir -p "$OUTDIR"

if $RUN_CPU && [[ ! -x "$CPU_BIN" ]]; then
    echo "CPU benchmark binary not found: $CPU_BIN"; exit 1
fi
if $RUN_GPU; then
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    if [[ ! -x "$GPU_BIN" ]]; then
        echo "GPU benchmark binary not found: $GPU_BIN"; exit 1
    fi
fi

echo "=== Benchmark from real physics event dumps ==="
echo "Dumps:   $DUMPS_DIR"
echo "Output:  $OUTDIR"
$RUN_CPU && echo "CPU:     enabled"
$RUN_GPU && echo "GPU:     enabled ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'no GPU info'))"
echo ""

SUMMARY="$OUTDIR/summary.txt"
if $RUN_CPU && $RUN_GPU; then
    echo "event dump_file n_candidates cpu_time_ms gpu_resolver_ms cpu_hash gpu_hash hash_match" > "$SUMMARY"
elif $RUN_CPU; then
    echo "event dump_file n_candidates cpu_time_ms cpu_hash" > "$SUMMARY"
else
    echo "event dump_file n_candidates gpu_resolver_ms cpu_hash gpu_hash hash_match" > "$SUMMARY"
fi

for DUMP in "$DUMPS_DIR"/event_*.json; do
    [[ -f "$DUMP" ]] || continue
    BASENAME="$(basename "$DUMP" .json)"

    CPU_TIME="n/a"; GPU_TIME="n/a"
    CPU_HASH="n/a"; GPU_HASH="n/a"; HASH_MATCH="n/a"
    N_CANDS="n/a"

    if $RUN_CPU; then
        CPU_OUT="$OUTDIR/${BASENAME}_cpu.txt"
        "$CPU_BIN" --input-dump="$DUMP" --repeats=10 --warmup=3 --profile \
            2>&1 | tee "$CPU_OUT" > /dev/null
        CPU_TIME=$(grep '^time_ms_mean' "$CPU_OUT" | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        CPU_HASH=$(grep '^output_hash' "$CPU_OUT" | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        N_CANDS=$(grep '^n_candidates' "$CPU_OUT" | head -1 | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
    fi

    if $RUN_GPU; then
        GPU_OUT="$OUTDIR/${BASENAME}_cuda.txt"
        "$GPU_BIN" --input-dump="$DUMP" --repeats=10 --warmup=3 --profile \
            2>&1 | tee "$GPU_OUT" > /dev/null
        GPU_TIME=$(grep '^time_resolver_ms_mean' "$GPU_OUT" | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        GPU_HASH=$(grep '^gpu_hash' "$GPU_OUT" | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        HASH_MATCH=$(grep '^hash_match' "$GPU_OUT" | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        if [[ "$N_CANDS" == "n/a" ]]; then
            N_CANDS=$(grep '^n_candidates' "$GPU_OUT" | head -1 | awk -F= '{print $2}' | tr -d ' ' || echo "n/a")
        fi
    fi

    printf "%-30s  n=%-6s  cpu=%-10s  gpu=%s\n" "$BASENAME" "$N_CANDS" "$CPU_TIME" "$GPU_TIME"
    echo "$BASENAME $DUMP $N_CANDS $CPU_TIME $GPU_TIME $CPU_HASH $GPU_HASH $HASH_MATCH" >> "$SUMMARY"
done

echo ""
echo "=== Benchmark from dumps complete ==="
echo "Summary: $SUMMARY"
