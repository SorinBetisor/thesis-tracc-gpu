#!/usr/bin/env bash
# Tier A hardware-tuning A/B benchmark.
#
# Runs both the untuned (thesis-novelty-conflict-graph) and tuned
# (thesis-novelty-hardware-tuning) binaries on the same Fatras dumps,
# in three configurations each:
#   - baseline   (no --conflict-graph flag)
#   - graph_jp   (--conflict-graph=jp)
#   - graph_mis  (--conflict-graph=mis)
#
# Per-event stdout is captured to a raw/ folder; a TSV summary is parsed at
# the end. Validity gate: every (binary, backend, event) row must have
# hash_match=true and 5/5 determinism (or it is flagged in the TSV).
#
# Must run on a CUDA-capable node (e.g. wn-lot-001 interactive, or via
# HTCondor). See REPORT.md for the LD_PRELOAD pattern.

set -euo pipefail

# ----------------------------------------------------------------------------
# Toolchain + binaries
# ----------------------------------------------------------------------------
SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"

# Some interactive sessions on Stoomboot inherit CUDA_VISIBLE_DEVICES=""
# (empty string), which the runtime interprets as "no GPUs visible" and
# fails with cudaErrorInvalidDevice. Drop it explicitly so the runtime
# enumerates all attached GPUs.
unset CUDA_VISIBLE_DEVICES

UNTUNED_BIN="${UNTUNED_BIN:-/data/alice/sbetisor/traccc/build/bin/traccc_benchmark_resolver_cuda}"
TUNED_BIN="${TUNED_BIN:-/data/alice/sbetisor/traccc-jp/build/bin/traccc_benchmark_resolver_cuda.tuned}"

for B in "$UNTUNED_BIN" "$TUNED_BIN"; do
    if [[ ! -x "$B" ]]; then
        echo "ERROR: binary not found or not executable: $B" >&2
        exit 1
    fi
done

# ----------------------------------------------------------------------------
# Inputs and output
# ----------------------------------------------------------------------------
RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_tier_a_tuning"
mkdir -p "$OUT/raw"

# Default corpus: full Fatras pile-up sweep. Override with FATRAS_DIRS to
# subset (e.g. FATRAS_DIRS="$RAW_ROOT/fatras_csv_dumps/fatras_ttbar_mu300").
# Globs are expanded here at parse time so the for loop sees real dirs.
if [[ -z "${FATRAS_DIRS+x}" ]]; then
    FATRAS_DIRS=($RAW_ROOT/fatras_csv_dumps/fatras_ttbar_mu*)
else
    # Allow the user to pass a single literal path or a space-separated list.
    read -r -a FATRAS_DIRS <<< "$FATRAS_DIRS"
fi
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"

echo "=== Tier A hardware-tuning A/B sweep ==="
echo "Untuned binary : $UNTUNED_BIN  ($(md5sum "$UNTUNED_BIN" | awk '{print $1}'))"
echo "Tuned binary   : $TUNED_BIN    ($(md5sum "$TUNED_BIN"   | awk '{print $1}'))"
echo "Output dir     : $OUT"
echo "Repeats/warmup : $REPEATS / $WARMUP   Det runs: $DET_RUNS"
echo "Corpora        : ${FATRAS_DIRS[*]}"
echo
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv 2>/dev/null || echo "(nvidia-smi unavailable)"
echo

# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
SUMMARY="$OUT/summary.tsv"
{
    printf 'binary\tbackend\tcorpus\tevent\tn_selected\ttime_ms_mean\tlatency_ms_per_event\tsingle_event_equiv_events_per_sec\ttime_ms_median\ttime_ms_p95\thash_match\toverlap_vs_cpu\tselected_jaccard\tcpu_only_selected\tgpu_only_selected\tn_selected_delta\tdup_post\tdet_pass\tdet_fail\n'
} > "$SUMMARY"

run_event () {
    local binary_label="$1"; local binary="$2"
    local backend_label="$3"; local backend_flag="$4"
    local input="$5"
    local corpus_name; corpus_name="$(basename "$(dirname "$input")")"
    local event_name; event_name="$(basename "$input" .json)"
    local raw="$OUT/raw/${binary_label}__${backend_label}__${corpus_name}__${event_name}.txt"

    set +e
    "$binary" \
        --input-dump="$input" \
        --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        $backend_flag \
        > "$raw" 2>&1
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]]; then
        printf '%s\t%s\t%s\t%s\tERR\t-1\t-1\t-1\t-1\t-1\tfalse\t-1\t-1\t-1\t-1\t0\t-1\t0\t0\n' \
            "$binary_label" "$backend_label" "$corpus_name" "$event_name" \
            >> "$SUMMARY"
        echo "  FAIL  $binary_label/$backend_label  $corpus_name/$event_name (rc=$rc)"
        return
    fi

    # Parse the dumped block for the relevant backend prefix. Both binaries
    # emit "<prefix>label=...", "<prefix>time_ms_mean=...", etc.
    local prefix
    case "$backend_label" in
        baseline)  prefix='baseline_' ;;
        graph_jp)  prefix='graph_jp_' ;;
        graph_mis) prefix='graph_mis_' ;;
    esac

    # The per-backend timing line is one line:
    #   <prefix>time_ms_mean=X time_ms_std=Y time_ms_median=Z time_ms_p95=W
    # so we anchor on the prefix-bearing token then extract each field.
    local n_sel mean latency eps med p95 hm overlap jaccard cpu_only gpu_only delta dup timing_line
    n_sel=$(grep -oE "^${prefix}n_selected=[0-9]+" "$raw" | head -1 | cut -d= -f2)
    timing_line=$(grep -E "^${prefix}time_ms_mean=" "$raw" | head -1)
    mean=$(echo "$timing_line" | grep -oE "time_ms_mean=[0-9.eE+-]+"   | cut -d= -f2)
    med=$( echo "$timing_line" | grep -oE "time_ms_median=[0-9.eE+-]+" | cut -d= -f2)
    p95=$( echo "$timing_line" | grep -oE "time_ms_p95=[0-9.eE+-]+"    | cut -d= -f2)
    latency=$(grep -oE "^${prefix}latency_ms_per_event=[0-9.eE+-]+" "$raw" | head -1 | cut -d= -f2)
    eps=$(grep -oE "^${prefix}single_event_equiv_events_per_sec=[0-9.eE+-]+" "$raw" | head -1 | cut -d= -f2)
    hm=$(   grep -oE "^${prefix}hash_match=(true|false)"             "$raw" | head -1 | cut -d= -f2)
    overlap=$(grep -oE "^${prefix}track_overlap_vs_cpu=[0-9.eE+-]+"  "$raw" | head -1 | cut -d= -f2)
    jaccard=$(grep -oE "^${prefix}selected_jaccard=[0-9.eE+-]+"       "$raw" | head -1 | cut -d= -f2)
    cpu_only=$(grep -oE "^${prefix}cpu_only_selected_count=[0-9]+"    "$raw" | head -1 | cut -d= -f2)
    gpu_only=$(grep -oE "^${prefix}gpu_only_selected_count=[0-9]+"    "$raw" | head -1 | cut -d= -f2)
    delta=$(grep -oE "^${prefix}n_selected_delta=-?[0-9]+"            "$raw" | head -1 | cut -d= -f2)
    dup=$(  grep -oE "^${prefix}duplicate_rate_post=[0-9.eE+-]+"     "$raw" | head -1 | cut -d= -f2)
    [[ -z "$latency" ]] && latency="$mean"
    [[ -z "$eps" && -n "$mean" ]] && eps=$(awk -v t="$mean" 'BEGIN { if (t > 0) printf "%.8g", 1000.0/t; else print "NA" }')
    [[ -z "$jaccard" ]] && jaccard="NA"
    [[ -z "$cpu_only" ]] && cpu_only="NA"
    [[ -z "$gpu_only" ]] && gpu_only="NA"
    [[ -z "$delta" ]] && delta="NA"

    # Determinism block format (one line per backend):
    #   det_<label>_pass=N det_<label>_fail=M
    local det_pass det_fail
    det_pass=$(grep -oE "det_${backend_label}_pass=[0-9]+" "$raw" | head -1 | cut -d= -f2)
    det_fail=$(grep -oE "det_${backend_label}_fail=[0-9]+" "$raw" | head -1 | cut -d= -f2)
    [[ -z "$det_pass" ]] && det_pass=0
    [[ -z "$det_fail" ]] && det_fail=0

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$binary_label" "$backend_label" "$corpus_name" "$event_name" \
        "${n_sel:-NA}" "${mean:-NA}" "${latency:-NA}" "${eps:-NA}" \
        "${med:-NA}" "${p95:-NA}" \
        "${hm:-NA}" "${overlap:-NA}" "${jaccard:-NA}" \
        "${cpu_only:-NA}" "${gpu_only:-NA}" "${delta:-NA}" "${dup:-NA}" \
        "$det_pass" "$det_fail" \
        >> "$SUMMARY"
}

declare -A backend_flags=(
    [baseline]=""
    [graph_jp]="--conflict-graph=jp"
    [graph_mis]="--conflict-graph=mis"
)

n_total=0; n_done=0
for d in "${FATRAS_DIRS[@]}"; do
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        n_total=$(( n_total + 6 ))  # 2 binaries x 3 backends
    done
done
echo "Total configs to run: $n_total"

for d in "${FATRAS_DIRS[@]}"; do
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        for backend in baseline graph_jp graph_mis; do
            for binary_pair in "untuned:$UNTUNED_BIN" "tuned:$TUNED_BIN"; do
                bin_label="${binary_pair%%:*}"
                bin_path="${binary_pair#*:}"
                run_event "$bin_label" "$bin_path" \
                    "$backend" "${backend_flags[$backend]}" "$ev"
                n_done=$(( n_done + 1 ))
                printf '\r  progress: %d / %d' "$n_done" "$n_total"
            done
        done
    done
done
echo

echo
echo "Sweep complete. Summary TSV: $SUMMARY"
echo "Per-event stdout dumps:    $OUT/raw/"
echo
echo "Quick speedup roll-up (mean per (binary, backend, corpus)):"
awk -F'\t' 'NR>1 && $6 != "NA" && $6 != "-1" {
    key=$1"\t"$2"\t"$3
    sum[key]+=$6; n[key]++
}
END {
    for (k in sum) printf "%s\t%.3f\n", k, sum[k]/n[k]
}' "$SUMMARY" | sort > "$OUT/mean_by_corpus.tsv"
column -t -s $'\t' "$OUT/mean_by_corpus.tsv"
