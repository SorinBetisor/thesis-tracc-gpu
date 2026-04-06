#!/usr/bin/env bash
# Run all GPU benchmark sweeps locally on the current node (wn-lot-001).
# Uses bounded parallelism: at most MAX_PARALLEL jobs at once.
# Intended as the fallback when Condor containers lack GLIBC_2.35.
#
# Usage (run on wn-lot-001 after: spack env activate traccc):
#   bash scripts/run_gpu_sweeps_local.sh [--extended-only] [--n-it-only] [--odd-only]
#   bash scripts/run_gpu_sweeps_local.sh              # all three sweeps
#
# Results land in:
#   results/<RUN_ID>_extended_cuda/
#   results/<RUN_ID>_n_it_sweep/
#   results/<RUN_ID>_odd_muon_cuda/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

TRACCC_BIN="$SCRIPT_DIR/../../../data-work/traccc/build/bin/traccc_benchmark_resolver_cuda"
if [[ ! -x "$TRACCC_BIN" ]]; then
    TRACCC_BIN="/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda"
fi
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
        --n-it-only) RUN_EXTENDED=false; RUN_ODD=false ;;
        --odd-only) RUN_EXTENDED=false; RUN_N_IT=false ;;
    esac
done

echo "=== Local GPU sweep runner ==="
echo "RUN_ID:        $RUN_ID"
echo "Binary:        $TRACCC_BIN"
echo "MAX_PARALLEL:  $MAX_PARALLEL"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -i nvidia | head -1 || echo 'unknown')"
echo ""

pids=()
wait_for_slot() {
    while [[ ${#pids[@]} -ge $MAX_PARALLEL ]]; do
        local new_pids=()
        for p in "${pids[@]}"; do
            if kill -0 "$p" 2>/dev/null; then new_pids+=("$p"); fi
        done
        pids=("${new_pids[@]}")
        [[ ${#pids[@]} -ge $MAX_PARALLEL ]] && sleep 2
    done
}

run_config() {
    local outfile="$1"; shift
    mkdir -p "$(dirname "$outfile")"
    "$TRACCC_BIN" "$@" --repeats=10 --warmup=3 --profile > "$outfile" 2>&1
}

if $RUN_EXTENDED; then
    OUTDIR="$SCRIPT_DIR/results/${RUN_ID}_extended_cuda"
    mkdir -p "$OUTDIR"
    echo "--- Extended sweep -> $OUTDIR"
    for n in 100 500 1000 2000 3000 5000 7500 10000 20000 50000; do
        for density in low med high; do
            wait_for_slot
            outfile="$OUTDIR/n${n}_${density}.txt"
            run_config "$outfile" --synthetic \
                --n-candidates="$n" --conflict-density="$density" &
            pids+=($!)
            echo "  launched n=$n density=$density (pid $!)"
        done
    done
    echo "  waiting for extended sweep to finish..."
    wait "${pids[@]}" 2>/dev/null || true
    pids=()
    echo "  extended sweep done: $(ls "$OUTDIR"/*.txt 2>/dev/null | wc -l) files"
    echo ""
fi

if $RUN_N_IT; then
    OUTDIR="$SCRIPT_DIR/results/${RUN_ID}_n_it_sweep"
    mkdir -p "$OUTDIR"
    echo "--- n_it sweep -> $OUTDIR"
    for n in 100 500 1000 5000 10000; do
        for density in low med high; do
            for n_it in 1 5 10 25 50 100; do
                wait_for_slot
                outfile="$OUTDIR/n${n}_${density}_nit${n_it}.txt"
                run_config "$outfile" --synthetic \
                    --n-candidates="$n" --conflict-density="$density" --n-it="$n_it" &
                pids+=($!)
            done
        done
    done
    echo "  waiting for n_it sweep to finish..."
    wait "${pids[@]}" 2>/dev/null || true
    pids=()
    echo "  n_it sweep done: $(ls "$OUTDIR"/*.txt 2>/dev/null | wc -l) files"
    echo ""
fi

if $RUN_ODD; then
    LATEST_DUMPS=$(ls -dt "$SCRIPT_DIR/data/odd_muon_dumps/"*/ 2>/dev/null | head -1)
    if [[ -z "$LATEST_DUMPS" ]]; then
        echo "No ODD dumps found in data/odd_muon_dumps/; skipping ODD sweep."
    else
        OUTDIR="$SCRIPT_DIR/results/${RUN_ID}_odd_muon_cuda"
        mkdir -p "$OUTDIR"
        echo "--- ODD GPU benchmark -> $OUTDIR"
        echo "    Dumps: $LATEST_DUMPS"
        for dump in "$LATEST_DUMPS"event_*.json; do
            [[ -f "$dump" ]] || continue
            base="$(basename "$dump" .json)"
            wait_for_slot
            outfile="$OUTDIR/${base}_cuda.txt"
            "$TRACCC_BIN" --input-dump="$dump" --repeats=10 --warmup=3 --profile \
                > "$outfile" 2>&1 &
            pids+=($!)
            echo "  launched $base (pid $!)"
        done
        echo "  waiting for ODD benchmark to finish..."
        wait "${pids[@]}" 2>/dev/null || true
        pids=()
        echo "  ODD benchmark done: $(ls "$OUTDIR"/*.txt 2>/dev/null | wc -l) files"
        echo ""
    fi
fi

echo "=== All GPU sweeps complete ==="
echo "Results under: $SCRIPT_DIR/results/"
