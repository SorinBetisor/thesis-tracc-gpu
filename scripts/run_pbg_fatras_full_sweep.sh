#!/usr/bin/env bash
# Full PBG sweep across all Fatras pileup levels (mu0 → mu600)
#
# Runs three backends per event:
#   baseline  : default greedy (no --parallel-batch)
#   pbg_w8192 : --parallel-batch --parallel-batch-window=8192  (primary)
#
# Outputs:
#   $RESULTS_ROOT/<TS>_pbg_fatras_full/
#     summary.tsv             — per-event: n_cand, baseline_ms, pbg_ms, speedup, hash_match
#     mean_by_corpus.tsv      — per-corpus averages
#     raw_baseline/           — per-event stdout, baseline
#     raw_pbg/                — per-event stdout, PBG W=8192
#     batch_sizes/            — per-event CSV from --log-batch-sizes
#     run_metadata.txt
#
# Env overrides:
#   REPEATS       (default 10)
#   WARMUP        (default 3)
#   DET_RUNS      (default 5)
#   PBG_WINDOW    (default 8192)
#   TRACCC_BUILD  (default /data/alice/sbetisor/traccc/build)

set -euo pipefail

SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
TRACCC_BUILD="${TRACCC_BUILD:-/data/alice/sbetisor/traccc/build}"
export LD_LIBRARY_PATH="${TRACCC_BUILD}/lib64:$SPACK_VIEW/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"

BIN="$TRACCC_BUILD/bin/traccc_benchmark_resolver_cuda"
RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_pbg_fatras_full"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"
PBG_WINDOW="${PBG_WINDOW:-8192}"

mkdir -p "$OUT/raw_baseline" "$OUT/raw_pbg" "$OUT/batch_sizes"

[[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN" >&2; exit 1; }

if ! "$BIN" --help 2>&1 | grep -q "parallel-batch"; then
    echo "ERROR: --parallel-batch flag not found in binary — wrong build?" >&2; exit 1
fi

echo "=== PBG Fatras Full Sweep (mu0 → mu600) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
echo "Binary     : $BIN"
echo "Output     : $OUT"
echo "Repeats    : $REPEATS  Warmup: $WARMUP  DetRuns: $DET_RUNS  PBG_WINDOW: $PBG_WINDOW"
echo ""

# ── TSV header ──────────────────────────────────────────────────────────────
SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tn_selected\tbaseline_ms\tbaseline_std\tpbg_ms\tpbg_std\tspeedup_pct\thash_baseline\thash_pbg\tn_outer_iter\tavg_batch\tmax_batch\n' \
    > "$SUMMARY"

# ── Parse helper ────────────────────────────────────────────────────────────
# Outputs: n_cand n_sel mean std hm n_outer avg_batch max_batch
parse_output() {
    local f="$1"
    local n_cand n_sel mean std hm n_outer avg_batch max_batch
    n_cand=$(grep -oE   'n_candidates=[0-9]+'               "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    n_sel=$(grep -oE    'baseline_n_selected=[0-9]+'         "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    [[ "$n_sel" == "NA" ]] && n_sel=$(grep -oE 'n_selected=[0-9]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    mean=$(grep -E      'baseline_time_ms_mean='             "$f" 2>/dev/null \
           | grep -oE 'time_ms_mean=[0-9.eE+\-]+' | cut -d= -f2 || echo NA)
    [[ "$mean" == "NA" ]] && mean=$(grep -oE 'time_ms_mean=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    std=$(grep -E       'baseline_time_ms_std='              "$f" 2>/dev/null \
          | grep -oE 'time_ms_std=[0-9.eE+\-]+'  | cut -d= -f2 || echo NA)
    [[ "$std" == "NA" ]] && std=$(grep -oE 'time_ms_std=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    hm=$(grep -oE       'baseline_hash_match=(true|false)'   "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    [[ "$hm" == "NA" ]] && hm=$(grep -oE 'hash_match=(true|false)' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    # PBG-specific fields (present in pbg output files only)
    n_outer=$(grep -oE  'pbg_n_outer_iterations=[0-9]+'      "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    avg_batch=$(grep -oE 'pbg_avg_batch_size=[0-9.]+'        "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    max_batch=$(grep -oE 'pbg_max_batch_size=[0-9]+'         "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    echo "$n_cand $n_sel $mean $std $hm $n_outer $avg_batch $max_batch"
}

# ── run_one ──────────────────────────────────────────────────────────────────
run_one() {
    local corpus="$1" event_label="$2" ev_path="$3"

    local base_raw="$OUT/raw_baseline/${corpus}__${event_label}.txt"
    local pbg_raw="$OUT/raw_pbg/${corpus}__${event_label}.txt"
    local batch_csv="$OUT/batch_sizes/${corpus}__${event_label}.csv"

    set +e
    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        --input-dump="$ev_path" \
        > "$base_raw" 2>&1
    local rc_base=$?

    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        --parallel-batch --parallel-batch-window="$PBG_WINDOW" \
        --log-batch-sizes="$batch_csv" \
        --input-dump="$ev_path" \
        > "$pbg_raw" 2>&1
    local rc_pbg=$?
    set -e

    local n_cand n_sel base_mean base_std base_hm
    local pbg_mean pbg_std pbg_hm n_outer avg_batch max_batch speedup

    if [[ $rc_base -ne 0 ]]; then
        n_cand=ERR; n_sel=ERR; base_mean=ERR; base_std=ERR; base_hm=ERR
    else
        read -r n_cand n_sel base_mean base_std base_hm _no _ab _mb \
            < <(parse_output "$base_raw")
    fi

    if [[ $rc_pbg -ne 0 ]]; then
        pbg_mean=ERR; pbg_std=ERR; pbg_hm=ERR; n_outer=ERR; avg_batch=ERR; max_batch=ERR
    else
        read -r _nc _ns pbg_mean pbg_std pbg_hm n_outer avg_batch max_batch \
            < <(parse_output "$pbg_raw")
    fi

    speedup=$(awk -v a="$base_mean" -v b="$pbg_mean" \
        'BEGIN{if(a+0>0&&b+0>0)printf "%.2f",(a-b)/a*100;else print "NA"}' \
        2>/dev/null || echo NA)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" \
        "$n_cand" "$n_sel" \
        "$base_mean" "$base_std" \
        "$pbg_mean" "$pbg_std" \
        "$speedup" \
        "$base_hm" "$pbg_hm" \
        "$n_outer" "$avg_batch" "$max_batch" \
        >> "$SUMMARY"

    printf '  %-50s  n_cand=%-5s  base=%-9s  pbg=%-9s  Δ=%s%%\n' \
        "${corpus}/${event_label}" \
        "${n_cand:-?}" "${base_mean:-ERR}" "${pbg_mean:-ERR}" "${speedup:-NA}"
}

# ── Main: all Fatras pile-up levels ──────────────────────────────────────────
for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
    [[ -d "$d" ]] || continue
    corpus="$(basename "$d")"
    echo "--- $corpus ---"
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        run_one "$corpus" "$(basename "$ev" .json)" "$ev"
    done
done

# ── Per-corpus means ──────────────────────────────────────────────────────────
MEAN_TSV="$OUT/mean_by_corpus.tsv"
printf 'corpus\tn_events\tn_candidates_mean\tbaseline_mean_ms\tpbg_mean_ms\tmean_speedup_pct\tany_hash_fail\n' \
    > "$MEAN_TSV"

awk -F'\t' '
NR > 1 && $5 != "ERR" && $5 != "NA" && $7 != "ERR" && $7 != "NA" {
    base[$1] += $5; pbg[$1] += $7; spd[$1] += $9; nc[$1] += $3; n[$1]++;
    if ($10 != "true" || $11 != "true") fail[$1] = 1
}
END {
    for (k in n) {
        printf "%s\t%d\t%.0f\t%.3f\t%.3f\t%.2f\t%s\n",
            k, n[k], nc[k]/n[k], base[k]/n[k], pbg[k]/n[k], spd[k]/n[k],
            (k in fail) ? "YES" : "no"
    }
}' "$SUMMARY" | sort >> "$MEAN_TSV"

echo ""
echo "=== Per-corpus means ==="
column -t -s $'\t' "$MEAN_TSV"

# ── Metadata ──────────────────────────────────────────────────────────────────
COMMIT=$(cd /data/alice/sbetisor/traccc && git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(cd /data/alice/sbetisor/traccc && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
cat > "$OUT/run_metadata.txt" << EOF
run_id=${TS}_pbg_fatras_full
binary=$BIN
branch=$BRANCH
commit=$COMMIT
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
repeats=$REPEATS
warmup=$WARMUP
det_runs=$DET_RUNS
pbg_window=$PBG_WINDOW
corpora=$(ls "$RAW_ROOT/fatras_csv_dumps/" 2>/dev/null | tr '\n' ' ')
note=full_fatras_sweep_mu0_through_mu600
EOF

echo ""
echo "=== PBG Fatras full sweep complete ==="
echo "Summary  : $SUMMARY"
echo "Means    : $MEAN_TSV"
echo "Output   : $OUT"
