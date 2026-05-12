#!/usr/bin/env bash
# Greedy-only hardware tuning sweep.
#
# Runs a single benchmark binary on ALL corpora (Fatras, ODD/Geant4,
# synthetic) using ONLY the baseline greedy backend.  Designed to be
# re-run after each greedy tuning tier so deltas are reproducible.
#
# Usage:
#   ./run_greedy_tuning_sweep.sh                    # full sweep with defaults
#   TIER=gb1 ./run_greedy_tuning_sweep.sh           # label the tier
#   BIN=/path/to/binary TIER=gb0 ./run_greedy_tuning_sweep.sh
#
# Outputs (all under RESULTS_ROOT/<TS>_greedy_tuning_<TIER>/):
#   summary.tsv          one row per (corpus, event)
#   raw/<corpus>__<event>.txt   full stdout per run
#   mean_by_corpus.tsv   rolled-up means for quick eyeballing

set -euo pipefail

# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------
SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
unset CUDA_VISIBLE_DEVICES

# ---------------------------------------------------------------------------
# Binary  (default: the current tuned build)
# ---------------------------------------------------------------------------
BIN="${BIN:-/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda.tuned}"
TIER="${TIER:-gb0}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found or not executable: $BIN" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_greedy_tuning_${TIER}"
mkdir -p "$OUT/raw"

# ---------------------------------------------------------------------------
# Benchmark knobs
# ---------------------------------------------------------------------------
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"

# Synthetic sizes to sweep (n_candidates)
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
# Synthetic conflict densities
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=== Greedy-only hardware tuning sweep (Tier ${TIER}) ==="
echo "Binary     : $BIN"
echo "Binary MD5 : $(md5sum "$BIN" | awk '{print $1}')"
echo "Output dir : $OUT"
echo "Repeats/warmup/det : $REPEATS / $WARMUP / $DET_RUNS"
echo
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"
echo

# ---------------------------------------------------------------------------
# TSV header
# ---------------------------------------------------------------------------
SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tn_selected\ttime_ms_mean\ttime_ms_std\ttime_ms_median\ttime_ms_p95\tlatency_ms\tevents_per_sec\thash_match\tdup_post\tdet_pass\tdet_fail\n' \
    > "$SUMMARY"

# ---------------------------------------------------------------------------
# Per-run helper
# ---------------------------------------------------------------------------
run_one() {
    local corpus="$1"
    local event_label="$2"
    local extra_args=("${@:3}")

    local raw="$OUT/raw/${corpus}__${event_label}.txt"

    set +e
    "$BIN" \
        --repeats="$REPEATS" \
        --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        "${extra_args[@]}" \
        > "$raw" 2>&1
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        printf '%s\t%s\tERR\tERR\t-1\t-1\t-1\t-1\t-1\t-1\tfalse\t-1\t0\t0\n' \
            "$corpus" "$event_label" >> "$SUMMARY"
        echo "  FAIL  $corpus/$event_label  (rc=$rc)"
        return
    fi

    # Parse the baseline_ prefixed output lines
    local n_cand n_sel mean std med p95 latency eps hm dup det_pass det_fail
    n_cand=$(grep -oE '^baseline_n_candidates=[0-9]+'      "$raw" | head -1 | cut -d= -f2)
    n_sel=$(grep -oE '^baseline_n_selected=[0-9]+'         "$raw" | head -1 | cut -d= -f2)
    local timing_line
    timing_line=$(grep -E '^baseline_time_ms_mean=' "$raw" | head -1)
    mean=$(echo "$timing_line" | grep -oE 'time_ms_mean=[0-9.eE+-]+'   | cut -d= -f2)
    std=$(echo  "$timing_line" | grep -oE 'time_ms_std=[0-9.eE+-]+'    | cut -d= -f2)
    med=$(echo  "$timing_line" | grep -oE 'time_ms_median=[0-9.eE+-]+' | cut -d= -f2)
    p95=$(echo  "$timing_line" | grep -oE 'time_ms_p95=[0-9.eE+-]+'    | cut -d= -f2)
    latency=$(grep -oE '^baseline_latency_ms_per_event=[0-9.eE+-]+'    "$raw" | head -1 | cut -d= -f2)
    eps=$(grep -oE '^baseline_single_event_equiv_events_per_sec=[0-9.eE+-]+' "$raw" | head -1 | cut -d= -f2)
    hm=$(grep -oE '^baseline_hash_match=(true|false)'                  "$raw" | head -1 | cut -d= -f2)
    dup=$(grep -oE '^baseline_duplicate_rate_post=[0-9.eE+-]+'         "$raw" | head -1 | cut -d= -f2)
    det_pass=$(grep -oE '^det_baseline_pass=[0-9]+'                    "$raw" | head -1 | cut -d= -f2)
    det_fail=$(grep -oE '^det_baseline_fail=[0-9]+'                    "$raw" | head -1 | cut -d= -f2)

    [[ -z "$latency"  ]] && latency="${mean:-NA}"
    [[ -z "$eps"      ]] && eps="NA"
    [[ -z "$det_pass" ]] && det_pass=0
    [[ -z "$det_fail" ]] && det_fail=0

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" \
        "${n_cand:-NA}" "${n_sel:-NA}" \
        "${mean:-NA}" "${std:-NA}" "${med:-NA}" "${p95:-NA}" \
        "${latency:-NA}" "${eps:-NA}" \
        "${hm:-NA}" "${dup:-NA}" \
        "$det_pass" "$det_fail" \
        >> "$SUMMARY"

    printf '  OK  %-45s  mean=%-8s  hash=%s  det=%s/%s\n' \
        "${corpus}/${event_label}" "${mean:-NA}ms" "${hm:-?}" \
        "$det_pass" "$(( det_pass + det_fail ))"
}

# ---------------------------------------------------------------------------
# 1. Fatras pile-up sweep
# ---------------------------------------------------------------------------
echo "--- Fatras pile-up sweep ---"
for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
    [[ -d "$d" ]] || continue
    corpus="$(basename "$d")"
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        event_label="$(basename "$ev" .json)"
        run_one "$corpus" "$event_label" --input-dump="$ev"
    done
done

# ---------------------------------------------------------------------------
# 2. ODD / Geant4 datasets
# ---------------------------------------------------------------------------
echo
echo "--- ODD / Geant4 dataset sweep ---"
ODD_DIR="$RAW_ROOT/odd_dumps"
if [[ -d "$ODD_DIR" ]]; then
    for d in "$ODD_DIR"/geant4_*; do
        [[ -d "$d" ]] || continue
        corpus="$(basename "$d")"
        for ev in "$d"/event_*.json; do
            [[ -e "$ev" ]] || continue
            event_label="$(basename "$ev" .json)"
            run_one "$corpus" "$event_label" --input-dump="$ev"
        done
    done
else
    echo "  (no odd_dumps dir found, skipping)"
fi

# ---------------------------------------------------------------------------
# 3. Synthetic sweep
# ---------------------------------------------------------------------------
echo
echo "--- Synthetic data sweep ---"
for density in $SYNTH_DENSITIES; do
    for n in $SYNTH_SIZES; do
        corpus="synthetic_${density}"
        event_label="n${n}"
        run_one "$corpus" "$event_label" \
            --synthetic --n-candidates="$n" --conflict-density="$density"
    done
done

# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------
echo
echo "Sweep complete. Summary TSV: $SUMMARY"

awk -F'\t' '
NR>1 && $5 != "NA" && $5 != "-1" && $5 != "ERR" {
    sum[$1] += $5; n[$1]++
}
END {
    for (k in sum) printf "%s\t%.3f\n", k, sum[k]/n[k]
}' "$SUMMARY" | sort > "$OUT/mean_by_corpus.tsv"

echo
echo "Mean time_ms per corpus (greedy, Tier ${TIER}):"
column -t -s $'\t' "$OUT/mean_by_corpus.tsv"
echo
echo "All raw outputs: $OUT/raw/"
