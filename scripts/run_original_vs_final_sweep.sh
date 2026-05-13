#!/usr/bin/env bash
# Compare original (n_it=100 fixed, no graph reuse) vs final (adaptive n_it + graph reuse)
#
# The "original" column reproduces the April 1 behaviour: fixed n_it=100, no adaptive
# formula, no graph reuse. Run with --n-it=100 which explicitly disables adaptive.
#
# The "final" column reuses the raw_reuse/ output from a prior graph reuse sweep
# (which ran with adaptive n_it default + --reuse-eviction-graph). If PRIOR_REUSE_DIR
# is not set, it regenerates the final column too.
#
# Outputs:
#   $RESULTS_ROOT/<TS>_original_vs_final/
#     comparison.tsv        — original_ms, final_ms, speedup_pct, hash_ok
#     mean_by_corpus.tsv    — per-corpus averages
#     raw_original/         — per-event stdout, n_it=100 fixed
#     raw_final/            — per-event stdout, adaptive + reuse  (may be symlinked)
#     run_metadata.txt
#
# Env overrides:
#   PRIOR_REUSE_DIR   path to an existing graph_reuse_full_sweep output dir
#                     whose raw_reuse/ subtree is used as the "final" column.
#                     If empty, re-runs the final column here.
#   REPEATS           (default 10)
#   WARMUP            (default 3)
#   DET_RUNS          (default 5)
#   SYNTH_SIZES       (default "500 1000 2000 5000 10000 20000 50000")
#   SYNTH_DENSITIES   (default "low med high")
#   TRACCC_BUILD      (default /data/alice/sbetisor/traccc/build)

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
OUT="$RESULTS_ROOT/${TS}_original_vs_final"
PRIOR_REUSE_DIR="${PRIOR_REUSE_DIR:-/user/sbetisor/data-work/results/20260513_005418_graph_reuse_full_sweep}"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

mkdir -p "$OUT/raw_original" "$OUT/raw_final"

[[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN" >&2; exit 1; }

echo "=== Original (n_it=100 fixed) vs Final (adaptive n_it + graph reuse) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
echo "Binary   : $BIN"
echo "Output   : $OUT"
echo "Repeats  : $REPEATS  Warmup: $WARMUP  DetRuns: $DET_RUNS"
if [[ -d "$PRIOR_REUSE_DIR/raw_reuse" ]]; then
    echo "Final col: reusing $PRIOR_REUSE_DIR/raw_reuse/"
    REUSE_SRC="$PRIOR_REUSE_DIR/raw_reuse"
else
    echo "Final col: generating fresh (no PRIOR_REUSE_DIR found)"
    REUSE_SRC=""
fi
echo ""

# ── TSV header ──────────────────────────────────────────────────────────────
CMP="$OUT/comparison.tsv"
printf 'corpus\tevent\tn_candidates\tn_selected\torig_ms\torig_std\tfinal_ms\tfinal_std\tspeedup_pct\thash_orig\thash_final\tdet_orig\tdet_final\n' \
    > "$CMP"

# ── Parse helper ────────────────────────────────────────────────────────────
parse_output() {
    local f="$1"
    local n_cand n_sel mean std hm det
    n_cand=$(grep -oE 'n_candidates=[0-9]+'             "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    n_sel=$(grep -oE  'baseline_n_selected=[0-9]+'       "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    [[ "$n_sel" == "NA" ]] && \
        n_sel=$(grep -oE 'n_selected=[0-9]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    mean=$(grep -E    'baseline_time_ms_mean='           "$f" 2>/dev/null \
           | grep -oE 'time_ms_mean=[0-9.eE+\-]+' | cut -d= -f2 || echo NA)
    [[ "$mean" == "NA" ]] && \
        mean=$(grep -oE 'time_ms_mean=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    std=$(grep -E     'baseline_time_ms_std='            "$f" 2>/dev/null \
          | grep -oE 'time_ms_std=[0-9.eE+\-]+'  | cut -d= -f2 || echo NA)
    [[ "$std" == "NA" ]] && \
        std=$(grep -oE 'time_ms_std=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    hm=$(grep -oE     'baseline_hash_match=(true|false)' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    [[ "$hm" == "NA" ]] && \
        hm=$(grep -oE 'hash_match=(true|false)' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    det=$(grep -oE    'det_baseline_pass=[0-9]+' "$f" 2>/dev/null | head -1 \
          | grep -oE '[0-9]+$' || echo 0)
    echo "$n_cand $n_sel $mean $std $hm $det"
}

# ── run_one: run original only (or both if no prior reuse dir) ───────────────
run_one() {
    local corpus="$1" event_label="$2"; shift 2
    local extra_args=("$@")

    local orig_raw="$OUT/raw_original/${corpus}__${event_label}.txt"
    local final_raw

    # Original: n_it=100 fixed, no adaptive, no graph reuse
    set +e
    "$BIN" --n-it=100 --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        "${extra_args[@]}" \
        > "$orig_raw" 2>&1
    local rc_orig=$?
    set -e

    # Final: from prior sweep if available, else regenerate
    if [[ -n "$REUSE_SRC" ]]; then
        final_raw="$REUSE_SRC/${corpus}__${event_label}.txt"
        if [[ ! -f "$final_raw" ]]; then
            final_raw="$OUT/raw_final/${corpus}__${event_label}.txt"
            set +e
            "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
                --determinism-runs="$DET_RUNS" \
                --reuse-eviction-graph \
                "${extra_args[@]}" \
                > "$final_raw" 2>&1
            set -e
        else
            # Symlink into raw_final for auditability
            ln -sf "$final_raw" "$OUT/raw_final/${corpus}__${event_label}.txt" 2>/dev/null || true
        fi
    else
        final_raw="$OUT/raw_final/${corpus}__${event_label}.txt"
        set +e
        "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
            --determinism-runs="$DET_RUNS" \
            --reuse-eviction-graph \
            "${extra_args[@]}" \
            > "$final_raw" 2>&1
        set -e
    fi

    local n_cand n_sel orig_mean orig_std orig_hm orig_det
    local final_mean final_std final_hm final_det speedup

    if [[ $rc_orig -ne 0 ]]; then
        n_cand=ERR; n_sel=ERR; orig_mean=ERR; orig_std=ERR; orig_hm=ERR; orig_det=0
    else
        read -r n_cand n_sel orig_mean orig_std orig_hm orig_det \
            < <(parse_output "$orig_raw")
    fi

    if [[ ! -f "$final_raw" ]] || [[ ! -s "$final_raw" ]]; then
        final_mean=ERR; final_std=ERR; final_hm=ERR; final_det=0
    else
        read -r _nc _ns final_mean final_std final_hm final_det \
            < <(parse_output "$final_raw")
    fi

    speedup=$(awk -v a="$orig_mean" -v b="$final_mean" \
        'BEGIN{if(a+0>0&&b+0>0)printf "%.2f",(a-b)/a*100;else print "NA"}' \
        2>/dev/null || echo NA)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" \
        "$n_cand" "$n_sel" \
        "$orig_mean" "$orig_std" \
        "$final_mean" "$final_std" \
        "$speedup" \
        "$orig_hm" "$final_hm" \
        "$orig_det" "$final_det" \
        >> "$CMP"

    printf '  %-55s  orig=%-9s  final=%-9s  Δ=%s%%\n' \
        "${corpus}/${event_label}" \
        "${orig_mean:-ERR}" "${final_mean:-ERR}" "${speedup:-NA}"
}

# ── Fatras sweep ─────────────────────────────────────────────────────────────
echo "--- Fatras pile-up sweep ---"
for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
    [[ -d "$d" ]] || continue
    corpus="$(basename "$d")"
    echo "  corpus: $corpus"
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        run_one "$corpus" "$(basename "$ev" .json)" --input-dump="$ev"
    done
done

# ── ODD sweep ────────────────────────────────────────────────────────────────
echo ""
echo "--- ODD / Geant4 muon sweep ---"
ODD_DIR="$RAW_ROOT/odd_dumps"
if [[ -d "$ODD_DIR" ]]; then
    for d in "$ODD_DIR"/geant4_*; do
        [[ -d "$d" ]] || continue
        corpus="$(basename "$d")"
        echo "  corpus: $corpus"
        for ev in "$d"/event_*.json; do
            [[ -e "$ev" ]] || continue
            run_one "$corpus" "$(basename "$ev" .json)" --input-dump="$ev"
        done
    done
else
    echo "  (no $ODD_DIR — skipping)"
fi

# ── Synthetic sweep ───────────────────────────────────────────────────────────
echo ""
echo "--- Synthetic sweep ---"
for density in $SYNTH_DENSITIES; do
    for n in $SYNTH_SIZES; do
        run_one "synthetic_${density}" "n${n}" \
            --synthetic --n-candidates="$n" --conflict-density="$density"
    done
done

# ── Per-corpus means ──────────────────────────────────────────────────────────
MEAN_TSV="$OUT/mean_by_corpus.tsv"
printf 'corpus\tn_events\torig_mean_ms\tfinal_mean_ms\tmean_speedup_pct\tany_hash_fail\n' \
    > "$MEAN_TSV"

awk -F'\t' '
NR > 1 && $5 != "ERR" && $5 != "NA" && $7 != "ERR" && $7 != "NA" {
    orig[$1] += $5; fin[$1] += $7; spd[$1] += $9; n[$1]++;
    if ($10 != "true" || $11 != "true") fail[$1] = 1
}
END {
    for (k in n) {
        printf "%s\t%d\t%.3f\t%.3f\t%.2f\t%s\n",
            k, n[k], orig[k]/n[k], fin[k]/n[k], spd[k]/n[k],
            (k in fail) ? "YES" : "no"
    }
}' "$CMP" | sort >> "$MEAN_TSV"

echo ""
echo "=== Per-corpus means ==="
column -t -s $'\t' "$MEAN_TSV"

# ── Metadata ──────────────────────────────────────────────────────────────────
COMMIT=$(cd /data/alice/sbetisor/traccc && git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(cd /data/alice/sbetisor/traccc && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)

cat > "$OUT/run_metadata.txt" << EOF
run_id=${TS}_original_vs_final
binary=$BIN
branch=$BRANCH
commit=$COMMIT
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
original_config=n_it=100_fixed_no_adaptive_no_graph_reuse
final_config=adaptive_n_it_plus_graph_reuse
prior_reuse_dir=${PRIOR_REUSE_DIR:-none}
repeats=$REPEATS
warmup=$WARMUP
det_runs=$DET_RUNS
synth_sizes=$SYNTH_SIZES
synth_densities=$SYNTH_DENSITIES
EOF

echo ""
echo "=== Sweep complete ==="
echo "Comparison : $CMP"
echo "Means      : $MEAN_TSV"
echo "Metadata   : $OUT/run_metadata.txt"
echo "Output     : $OUT"
