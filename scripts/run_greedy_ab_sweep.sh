#!/usr/bin/env bash
# Greedy-only A/B sweep: compare two saved libtraccc_cuda.so builds.
#
# Usage:
#   LIB_A=<path.so> LIB_B=<path.so> LABEL_A=gb0 LABEL_B=gb1 \
#       ./run_greedy_ab_sweep.sh
#
# For each (library, event), runs:
#   - baseline greedy only (no --conflict-graph)
# with REPEATS timed iterations and DET_RUNS determinism checks.

set -euo pipefail

SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
unset CUDA_VISIBLE_DEVICES

BUILD_DIR="/data/alice/sbetisor/traccc-jp/build"
BIN="$BUILD_DIR/bin/traccc_benchmark_resolver_cuda"
LIB_LIVE="$BUILD_DIR/lib64/libtraccc_cuda.so.1.1.0"

LIB_A="${LIB_A:-$BUILD_DIR/lib64/libtraccc_cuda.so.tuned}"
LIB_B="${LIB_B:-$BUILD_DIR/lib64/libtraccc_cuda.so.gb1}"
LABEL_A="${LABEL_A:-gb0}"
LABEL_B="${LABEL_B:-gb1}"

RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_greedy_ab_${LABEL_A}_vs_${LABEL_B}"
mkdir -p "$OUT/raw"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

echo "=== Greedy A/B: ${LABEL_A} vs ${LABEL_B} ==="
echo "Binary : $BIN"
echo "Lib A  : $LIB_A  ($(md5sum "$LIB_A" | awk '{print $1}'))"
echo "Lib B  : $LIB_B  ($(md5sum "$LIB_B" | awk '{print $1}'))"
echo "Output : $OUT"
echo "Repeats/warmup/det : $REPEATS / $WARMUP / $DET_RUNS"
echo
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || true
echo

SUMMARY="$OUT/summary.tsv"
printf 'label\tcorpus\tevent\tn_candidates\tn_selected\ttime_ms_mean\ttime_ms_std\ttime_ms_median\ttime_ms_p95\thash_match\tdet_pass\tdet_fail\n' \
    > "$SUMMARY"

# Swap in the given library, run the binary, swap back.
run_with_lib() {
    local lib="$1"; local label="$2"
    local corpus="$3"; local event_label="$4"
    shift 4
    local extra_args=("$@")

    local raw="$OUT/raw/${label}__${corpus}__${event_label}.txt"

    cp "$lib" "$LIB_LIVE"
    set +e
    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" "${extra_args[@]}" \
        > "$raw" 2>&1
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        printf '%s\t%s\t%s\tERR\tERR\t-1\t-1\t-1\t-1\tfalse\t0\t0\n' \
            "$label" "$corpus" "$event_label" >> "$SUMMARY"
        echo "  FAIL  ${label}/${corpus}/${event_label}  (rc=$rc)"
        return
    fi

    local n_cand n_sel mean std med p95 hm det_pass det_fail
    n_cand=$(grep -oE '^baseline_n_candidates=[0-9]+'   "$raw" | head -1 | cut -d= -f2)
    n_sel=$(grep -oE '^baseline_n_selected=[0-9]+'      "$raw" | head -1 | cut -d= -f2)
    local tl; tl=$(grep -E '^baseline_time_ms_mean=' "$raw" | head -1)
    mean=$(echo "$tl" | grep -oE 'time_ms_mean=[0-9.eE+-]+'   | cut -d= -f2)
    std=$(echo  "$tl" | grep -oE 'time_ms_std=[0-9.eE+-]+'    | cut -d= -f2)
    med=$(echo  "$tl" | grep -oE 'time_ms_median=[0-9.eE+-]+' | cut -d= -f2)
    p95=$(echo  "$tl" | grep -oE 'time_ms_p95=[0-9.eE+-]+'    | cut -d= -f2)
    hm=$(grep -oE 'baseline_hash_match=(true|false)' "$raw" | head -1 | cut -d= -f2)
    det_pass=$(grep -oE 'det_baseline_pass=[0-9]+' "$raw" | head -1 | grep -oE '[0-9]+$' || echo "0")
    det_fail=$(grep -oE 'det_baseline_fail=[0-9]+' "$raw" | head -1 | grep -oE '[0-9]+$' || echo "0")
    [[ -z "$det_pass" ]] && det_pass=0
    [[ -z "$det_fail" ]] && det_fail=0

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$label" "$corpus" "$event_label" \
        "${n_cand:-NA}" "${n_sel:-NA}" \
        "${mean:-NA}" "${std:-NA}" "${med:-NA}" "${p95:-NA}" \
        "${hm:-NA}" "$det_pass" "$det_fail" \
        >> "$SUMMARY"

    printf '  %-5s  %-45s  mean=%-8s  hash=%s  det=%s/%s\n' \
        "$label" "${corpus}/${event_label}" \
        "${mean:-NA}ms" "${hm:-?}" "$det_pass" "$(( det_pass + det_fail ))"
}

sweep_all() {
    local lib="$1"; local label="$2"

    echo
    echo "--- ${label}: Fatras ---"
    for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
        [[ -d "$d" ]] || continue
        local corpus; corpus="$(basename "$d")"
        for ev in "$d"/event_*.json; do
            [[ -e "$ev" ]] || continue
            local evl; evl="$(basename "$ev" .json)"
            run_with_lib "$lib" "$label" "$corpus" "$evl" --input-dump="$ev"
        done
    done

    echo
    echo "--- ${label}: ODD ---"
    local ODD_DIR="$RAW_ROOT/odd_dumps"
    if [[ -d "$ODD_DIR" ]]; then
        for d in "$ODD_DIR"/geant4_*; do
            [[ -d "$d" ]] || continue
            local corpus; corpus="$(basename "$d")"
            for ev in "$d"/event_*.json; do
                [[ -e "$ev" ]] || continue
                local evl; evl="$(basename "$ev" .json)"
                run_with_lib "$lib" "$label" "$corpus" "$evl" --input-dump="$ev"
            done
        done
    fi

    echo
    echo "--- ${label}: Synthetic ---"
    for density in $SYNTH_DENSITIES; do
        for n in $SYNTH_SIZES; do
            run_with_lib "$lib" "$label" "synthetic_${density}" "n${n}" \
                --synthetic --n-candidates="$n" --conflict-density="$density"
        done
    done
}

# Save the current live library so we can restore it at the end.
ORIG_LIB="$BUILD_DIR/lib64/libtraccc_cuda.so.orig_backup"
cp "$LIB_LIVE" "$ORIG_LIB"

sweep_all "$LIB_A" "$LABEL_A"
sweep_all "$LIB_B" "$LABEL_B"

# Restore original
cp "$ORIG_LIB" "$LIB_LIVE"
rm -f "$ORIG_LIB"

echo
echo "=== A/B sweep complete. Summary: $SUMMARY ==="

# Roll-up mean per (label, corpus)
awk -F'\t' '
NR>1 && $6 != "NA" && $6 != "-1" && $6 != "ERR" {
    key=$1 "\t" $2; sum[key]+=$6; n[key]++
}
END {
    for (k in sum) printf "%s\t%.3f\n", k, sum[k]/n[k]
}' "$SUMMARY" | sort > "$OUT/mean_by_corpus.tsv"

echo
echo "Mean time_ms per (label, corpus):"
column -t -s $'\t' "$OUT/mean_by_corpus.tsv"

# Speedup table: gb0 vs gb1 for Fatras pile-up
echo
echo "Speedup (gb0_mean / gb1_mean per corpus, Fatras only):"
awk -F'\t' '
NR>1 && $1 == "'"$LABEL_A"'" && $2 ~ /fatras/ && $6 != "NA" {
    a[$2] += $6; na[$2]++
}
NR>1 && $1 == "'"$LABEL_B"'" && $2 ~ /fatras/ && $6 != "NA" {
    b[$2] += $6; nb[$2]++
}
END {
    for (c in a) {
        if (nb[c] > 0) {
            ma = a[c]/na[c]; mb = b[c]/nb[c]
            delta = (ma - mb) / ma * 100
            printf "%s\t%.3f\t%.3f\t%+.1f%%\n", c, ma, mb, delta
        }
    }
}' "$SUMMARY" | sort | column -t -s $'\t'
