#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/data/alice/sbetisor/spack/install/linux-zen/intel-oneapi-compilers-2024.2.0-2paotxrdgntn64w5uupe7z4b2imrx3ad/compiler/2024.2/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

CPU_BIN="${CPU_BIN:-/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver}"
GPU_BIN="${GPU_BIN:-/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda}"
DUMPS_DIR="${DUMPS_DIR:-/user/sbetisor/thesis/sorin-thesis-work/data/odd_muon_dumps/20260406}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$(date +%Y%m%d_%H%M%S)_odd_muon_graph_reuse}"
GPU_EXTRA_ARGS="${GPU_EXTRA_ARGS:---reuse-eviction-graph}"

mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.txt"
LOG="$OUTDIR/run.log"

echo "event dump_file n_candidates cpu_time_ms gpu_time_ms cpu_hash gpu_hash hash_match" > "$SUMMARY"

{
    echo "=== ODD dump benchmark with graph reuse ==="
    echo "Dumps:  $DUMPS_DIR"
    echo "Output: $OUTDIR"
    echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unavailable)"
    echo
} | tee "$LOG"

for dump in "$DUMPS_DIR"/event_*.json; do
    [[ -f "$dump" ]] || continue

    base="$(basename "$dump" .json)"
    cpu_out="$OUTDIR/${base}_cpu.txt"
    gpu_out="$OUTDIR/${base}_cuda.txt"

    "$CPU_BIN" --input-dump="$dump" --repeats=10 --warmup=3 --profile \
        > "$cpu_out" 2>&1
    "$GPU_BIN" --input-dump="$dump" --repeats=10 --warmup=3 --profile \
        $GPU_EXTRA_ARGS > "$gpu_out" 2>&1

    cpu_time="$(sed -n 's/^time_ms_mean=\([^ ]*\).*/\1/p' "$cpu_out")"
    gpu_time="$(sed -n 's/^time_ms_mean=\([^ ]*\).*/\1/p' "$gpu_out")"
    n_cands="$(sed -n 's/^n_candidates=\([^ ]*\).*/\1/p' "$gpu_out" | head -1)"
    cpu_hash="$(sed -n 's/^output_hash=//p' "$cpu_out")"
    gpu_hash="$(sed -n 's/^gpu_hash=//p' "$gpu_out")"
    match="$(sed -n 's/^hash_match=//p' "$gpu_out")"

    printf "%-12s n=%-4s cpu=%-8s gpu=%-8s match=%s\n" \
        "$base" "$n_cands" "$cpu_time" "$gpu_time" "$match" | tee -a "$LOG"

    echo "$base $dump $n_cands $cpu_time $gpu_time $cpu_hash $gpu_hash $match" \
        >> "$SUMMARY"
done

echo | tee -a "$LOG"
echo "Summary: $SUMMARY" | tee -a "$LOG"
echo "$OUTDIR"
