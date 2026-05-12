#!/usr/bin/env bash
# Greedy-only per-tier benchmark sweep.
#
# Usage:
#   TIER=gb0  ./run_greedy_tier_sweep.sh
#   TIER=gb1  ./run_greedy_tier_sweep.sh
#   TIER=gb2  ./run_greedy_tier_sweep.sh
#
# Each invocation runs the greedy-only baseline on all corpora using the
# per-tier shared library in build/lib64/tier_<TIER>/.
# LD_LIBRARY_PATH is set so the binary loads that tier's kernels without
# touching any file in the live build directory.
#
# Outputs land in:
#   $RESULTS_ROOT/<TS>_greedy_<TIER>/
#     summary.tsv
#     raw/<corpus>__<event>.txt

set -euo pipefail

SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
unset CUDA_VISIBLE_DEVICES

BUILD_DIR="${BUILD_DIR:-/data/alice/sbetisor/traccc-jp/build}"
BIN="$BUILD_DIR/bin/traccc_benchmark_resolver_cuda"

TIER="${TIER:-gb2}"
TIER_LIB_DIR="$BUILD_DIR/lib64/tier_${TIER}"

if [[ ! -d "$TIER_LIB_DIR" ]]; then
    echo "ERROR: tier lib dir not found: $TIER_LIB_DIR" >&2
    exit 1
fi

# Put the tier-specific lib ahead of the standard build lib64.
export LD_LIBRARY_PATH="${TIER_LIB_DIR}:${BUILD_DIR}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_greedy_${TIER}"
mkdir -p "$OUT/raw"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

echo "=== Greedy tuning sweep — Tier ${TIER} ==="
echo "Binary      : $BIN"
echo "Tier lib dir: $TIER_LIB_DIR"
echo "Lib MD5     : $(md5sum "$TIER_LIB_DIR/libtraccc_cuda.so.1.1.0" | awk '{print $1}')"
echo "Output dir  : $OUT"
echo "Repeats/warmup/det : $REPEATS / $WARMUP / $DET_RUNS"
echo
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || true
echo

SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tn_selected\ttime_ms_mean\ttime_ms_std\ttime_ms_median\ttime_ms_p95\thash_match\tdet_pass\tdet_fail\n' \
    > "$SUMMARY"

run_one() {
    local corpus="$1"; local event_label="$2"; shift 2
    local extra_args=("$@")
    local raw="$OUT/raw/${corpus}__${event_label}.txt"

    set +e
    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" "${extra_args[@]}" \
        > "$raw" 2>&1
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        printf '%s\t%s\tERR\tERR\t-1\t-1\t-1\t-1\tfalse\t0\t0\n' \
            "$corpus" "$event_label" >> "$SUMMARY"
        echo "  FAIL  ${corpus}/${event_label}  (rc=$rc)"
        return
    fi

    # Parse fields — use || true on every grep so set -e never fires on no-match.
    local n_cand n_sel tl mean std med p95 hm det_line det_pass det_fail
    n_cand=$(grep -oE 'n_candidates=[0-9]+'      "$raw" | head -1 | cut -d= -f2 || true)
    n_sel=$(grep -oE  'baseline_n_selected=[0-9]+' "$raw" | head -1 | cut -d= -f2 || true)
    tl=$(grep -E 'baseline_time_ms_mean=' "$raw" | head -1 || true)
    mean=$(echo "$tl" | grep -oE 'time_ms_mean=[0-9.eE+-]+'   | cut -d= -f2 || true)
    std=$(echo  "$tl" | grep -oE 'time_ms_std=[0-9.eE+-]+'    | cut -d= -f2 || true)
    med=$(echo  "$tl" | grep -oE 'time_ms_median=[0-9.eE+-]+' | cut -d= -f2 || true)
    p95=$(echo  "$tl" | grep -oE 'time_ms_p95=[0-9.eE+-]+'    | cut -d= -f2 || true)
    hm=$(grep -oE 'baseline_hash_match=(true|false)' "$raw" | head -1 | cut -d= -f2 || true)
    # det_pass and det_fail appear on the same line: "det_baseline_pass=N det_baseline_fail=M"
    det_line=$(grep -oE 'det_baseline_pass=[0-9]+' "$raw" | head -1 || true)
    det_pass=$(echo "$det_line" | grep -oE '[0-9]+$' || true)
    det_fail=$(grep -oE 'det_baseline_fail=[0-9]+' "$raw" | head -1 | grep -oE '[0-9]+$' || true)
    [[ -z "$det_pass" ]] && det_pass=0
    [[ -z "$det_fail" ]] && det_fail=0

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" \
        "${n_cand:-NA}" "${n_sel:-NA}" \
        "${mean:-NA}" "${std:-NA}" "${med:-NA}" "${p95:-NA}" \
        "${hm:-NA}" "$det_pass" "$det_fail" >> "$SUMMARY"

    printf '  OK  %-45s  mean=%-8s  hash=%s  det=%s/%s\n' \
        "${corpus}/${event_label}" "${mean:-NA}ms" "${hm:-?}" \
        "$det_pass" "$(( det_pass + det_fail ))"
}

echo "--- Fatras pile-up sweep ---"
for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
    [[ -d "$d" ]] || continue
    corpus="$(basename "$d")"
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        run_one "$corpus" "$(basename "$ev" .json)" --input-dump="$ev"
    done
done

echo
echo "--- ODD / Geant4 sweep ---"
ODD_DIR="$RAW_ROOT/odd_dumps"
if [[ -d "$ODD_DIR" ]]; then
    for d in "$ODD_DIR"/geant4_*; do
        [[ -d "$d" ]] || continue
        corpus="$(basename "$d")"
        for ev in "$d"/event_*.json; do
            [[ -e "$ev" ]] || continue
            run_one "$corpus" "$(basename "$ev" .json)" --input-dump="$ev"
        done
    done
else
    echo "  (no odd_dumps dir, skipping)"
fi

echo
echo "--- Synthetic sweep ---"
for density in $SYNTH_DENSITIES; do
    for n in $SYNTH_SIZES; do
        run_one "synthetic_${density}" "n${n}" \
            --synthetic --n-candidates="$n" --conflict-density="$density"
    done
done

echo
echo "=== Tier ${TIER} sweep complete. Summary: $SUMMARY ==="

awk -F'\t' '
NR>1 && $5 != "NA" && $5 != "-1" && $5 != "ERR" {
    sum[$1] += $5; n[$1]++
}
END {
    for (k in sum) printf "%s\t%.3f\n", k, sum[k]/n[k]
}' "$SUMMARY" | sort > "$OUT/mean_by_corpus.tsv"

echo
echo "Mean time_ms per corpus (Tier ${TIER}):"
column -t -s $'\t' "$OUT/mean_by_corpus.tsv"
