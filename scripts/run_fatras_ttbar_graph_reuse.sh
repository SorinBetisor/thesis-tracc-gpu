#!/usr/bin/env bash
# Convert real Fatras ttbar CKF CSVs into ambiguity dumps and benchmark them
# with GPU graph reuse on/off.
#
# Example:
#   ./run_fatras_ttbar_graph_reuse.sh --pileups "300 400 500 600" --events 5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THESIS_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data/alice/sbetisor/traccc/data/odd}"
PILEUP_LEVELS="${PILEUP_LEVELS:-300 400 500 600}"
N_EVENTS="${N_EVENTS:-5}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-2}"
PVAL_SOURCE="${PVAL_SOURCE:-approx-chi2}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pileups) PILEUP_LEVELS="$2"; shift 2 ;;
        --events) N_EVENTS="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --pval-source) PVAL_SOURCE="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)_fatras_real_graph_reuse"
OUTDIR="$THESIS_REPO/results/$RUN_ID"
mkdir -p "$OUTDIR"

CONVERTER="$SCRIPT_DIR/convert_fatras_ckf_csv_to_dumps.py"
BENCH="$SCRIPT_DIR/run_benchmark_from_dumps.sh"
SUMMARY_TABLE="$OUTDIR/summary_table.txt"

echo "pileup mean_n_candidates cpu_mean_ms gpu_no_reuse_ms gpu_reuse_ms reuse_speedup hash_all_match" \
    > "$SUMMARY_TABLE"

echo "=== Real Fatras ttbar graph reuse benchmark ==="
echo "Pileups: $PILEUP_LEVELS"
echo "Events:  $N_EVENTS"
echo "Repeats: $REPEATS"
echo "Warmup:  $WARMUP"
echo "Output:  $OUTDIR"
echo ""

for PU in $PILEUP_LEVELS; do
    DATASET="$DATA_ROOT/fatras_ttbar_mu${PU}"
    DUMPS_DIR="$OUTDIR/dumps_mu${PU}"
    CONTROL_OUT="$OUTDIR/mu${PU}_no_reuse"
    REUSE_OUT="$OUTDIR/mu${PU}_reuse"

    if [[ ! -d "$DATASET" ]]; then
        echo "Missing dataset: $DATASET"
        exit 1
    fi

    echo "--- Convert mu=$PU ---"
    python3 "$CONVERTER" "$DATASET" \
        --outdir "$DUMPS_DIR" \
        --events "$N_EVENTS" \
        --pval-source "$PVAL_SOURCE"

    echo "--- Benchmark mu=$PU control ---"
    REPEATS="$REPEATS" WARMUP="$WARMUP" \
        "$BENCH" --dumps-dir "$DUMPS_DIR" --cpu --gpu --outdir "$CONTROL_OUT"

    echo "--- Benchmark mu=$PU reuse ---"
    REPEATS="$REPEATS" WARMUP="$WARMUP" \
        GPU_EXTRA_ARGS="--reuse-eviction-graph" \
        "$BENCH" --dumps-dir "$DUMPS_DIR" --cpu --gpu --outdir "$REUSE_OUT"

    read -r MEAN_N CPU_MEAN GPU_CONTROL_MEAN MATCH_OK <<<"$(
        awk '
            BEGIN { ok = 1 }
            NR==1 {next}
            {n+=$3; cpu+=$4; gpu+=$5; ok=ok && ($8=="true"); c++}
            END {
                if (c == 0) exit 1;
                printf "%.3f %.6f %.6f %d", n/c, cpu/c, gpu/c, ok
            }
        ' "$CONTROL_OUT/summary.txt"
    )"

    GPU_REUSE_MEAN="$(
        awk '
            BEGIN { ok = 1 }
            NR==1 {next}
            {gpu+=$5; ok=ok && ($8=="true"); c++}
            END {
                if (c == 0) exit 1;
                printf "%.6f", gpu/c
            }
        ' "$REUSE_OUT/summary.txt"
    )"

    REUSE_SPEEDUP="$(
        awk -v a="$GPU_CONTROL_MEAN" -v b="$GPU_REUSE_MEAN" '
            BEGIN {
                if (b > 0) printf "%.3f", a / b;
                else print "nan";
            }
        '
    )"

    REUSE_MATCH_OK="$(
        awk '
            BEGIN { ok = 1 }
            NR==1 {next}
            {ok=ok && ($8=="true")}
            END {print ok ? "true" : "false"}
        ' "$REUSE_OUT/summary.txt"
    )"

    HASH_ALL_MATCH="false"
    if [[ "$MATCH_OK" == "1" && "$REUSE_MATCH_OK" == "true" ]]; then
        HASH_ALL_MATCH="true"
    fi

    printf "%s %s %s %s %s %s %s\n" \
        "$PU" "$MEAN_N" "$CPU_MEAN" "$GPU_CONTROL_MEAN" "$GPU_REUSE_MEAN" \
        "$REUSE_SPEEDUP" "$HASH_ALL_MATCH" >> "$SUMMARY_TABLE"
done

echo ""
echo "=== Summary ==="
column -t "$SUMMARY_TABLE"
echo ""
echo "Saved: $SUMMARY_TABLE"
