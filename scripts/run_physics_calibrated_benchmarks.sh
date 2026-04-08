#!/bin/bash
# Run traccc CPU + GPU benchmarks at physics-calibrated n values.
# n values correspond to mean CKF track counts from the Fatras pileup sweep:
#   mu=0->56, mu=20->154, mu=50->307, mu=100->602,
#   mu=140->821, mu=200->1167, mu=300->1770
#
# Results: thesis/results/<RUN_ID>_physics_calibrated/
# Run: bash scripts/run_physics_calibrated_benchmarks.sh

set -o pipefail

# Set library path directly — avoids slow spack env activate network fetch
INTEL_LIB=/data/alice/sbetisor/spack/install/linux-zen/intel-oneapi-compilers-2024.2.0-2paotxrdgntn64w5uupe7z4b2imrx3ad/compiler/2024.2/lib
export LD_LIBRARY_PATH=$INTEL_LIB:$LD_LIBRARY_PATH

CPU_BIN=/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver
GPU_BIN=/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_ROOT="$SCRIPT_DIR/../results"
RUN_ID="$(date +%Y%m%d_%H%M%S)_physics_calibrated"
OUTDIR="$RESULTS_ROOT/$RUN_ID"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/run.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== Physics-calibrated CPU+GPU benchmark ==="
echo "Run ID: $RUN_ID"
echo "Output: $OUTDIR"
echo "Date: $(date)"
echo ""

# Physics-calibrated n values (mean CKF candidates from Fatras pileup sweep)
declare -A N_TO_PU
N_TO_PU[56]="mu0"
N_TO_PU[154]="mu20"
N_TO_PU[307]="mu50"
N_TO_PU[602]="mu100"
N_TO_PU[821]="mu140"
N_TO_PU[1167]="mu200"
N_TO_PU[1770]="mu300"

CONFLICT_DENSITIES="low med"
REPEATS=20
WARMUP=5

echo "=== GPU device ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no GPU visible)"
echo ""

for N in 56 154 307 602 821 1167 1770; do
    PU_LABEL="${N_TO_PU[$N]}"
    for DENSITY in $CONFLICT_DENSITIES; do
        TAG="n${N}_${DENSITY}_${PU_LABEL}"
        echo "--- CPU: n=$N density=$DENSITY ($PU_LABEL) ---"
        $CPU_BIN \
            --synthetic \
            --n-candidates=$N \
            --conflict-density=$DENSITY \
            --backend=cpu \
            --repeats=$REPEATS \
            --warmup=$WARMUP \
            > "$OUTDIR/cpu_${TAG}.txt" 2>&1
        grep "time_ms_mean\|n_candidates\|events_per_sec" "$OUTDIR/cpu_${TAG}.txt"

        echo "--- GPU: n=$N density=$DENSITY ($PU_LABEL) ---"
        $GPU_BIN \
            --synthetic \
            --n-candidates=$N \
            --conflict-density=$DENSITY \
            --repeats=$REPEATS \
            --warmup=$WARMUP \
            > "$OUTDIR/gpu_${TAG}.txt" 2>&1
        grep "time_ms_mean\|n_candidates\|events_per_sec\|hash_match" "$OUTDIR/gpu_${TAG}.txt"
        echo ""
    done
done

echo ""
echo "=== DONE: $RUN_ID ==="
echo ""

echo "=== Summary table (low conflict density) ==="
echo "n_candidates | pileup_equiv | cpu_mean_ms | gpu_mean_ms | cpu/gpu_ratio | hash_match"
echo "-------------|--------------|-------------|-------------|----------------|----------"
for N in 56 154 307 602 821 1167 1770; do
    PU_LABEL="${N_TO_PU[$N]}"
    CPU_FILE="$OUTDIR/cpu_n${N}_low_${PU_LABEL}.txt"
    GPU_FILE="$OUTDIR/gpu_n${N}_low_${PU_LABEL}.txt"
    CPU_T=$(grep "time_ms_mean" "$CPU_FILE" 2>/dev/null | awk -F= '{print $2}')
    GPU_T=$(grep "time_ms_mean" "$GPU_FILE" 2>/dev/null | awk -F= '{print $2}')
    MATCH=$(grep "hash_match" "$GPU_FILE" 2>/dev/null | awk -F= '{print $2}')
    RATIO=$(echo "$CPU_T $GPU_T" | awk '{if($2>0) printf "%.2f", $1/$2; else print "N/A"}')
    printf "%12s | %12s | %11s | %11s | %14s | %s\n" \
        "$N" "$PU_LABEL" "$CPU_T" "$GPU_T" "$RATIO" "$MATCH"
done
