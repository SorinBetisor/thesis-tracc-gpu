#!/usr/bin/env bash
# Full CUDA graph reuse sweep — baseline GPU vs --reuse-eviction-graph
#
# Uses the binary at TRACCC_BUILD/bin/traccc_benchmark_resolver_cuda which
# must already be compiled with the graph-reuse feature (set_reuse_eviction_graph).
# No branch checkout is performed; SKIP_BUILD is effectively always 1.
#
# The graph-reuse feature was forward-ported onto the thesis-novelty-conflict-graph
# codebase. Benchmarks run in pure greedy mode (no --conflict-graph, no --parallel-batch)
# so the comparison is: baseline greedy CUDA graph vs graph-reuse (one-time capture,
# subsequent iterations use cudaGraphExecKernelNodeSetParams).
#
# Outputs:
#   $RESULTS_ROOT/<TS>_graph_reuse_full_sweep/
#     summary.tsv          — one row per (corpus, event, backend)
#     mean_by_corpus.tsv   — corpus-level averages
#     raw_baseline/        — per-event txt files, baseline
#     raw_reuse/           — per-event txt files, reuse
#     run_metadata.txt     — provenance
#
# Usage:
#   ./run_graph_reuse_full_sweep.sh
#
# Env overrides:
#   REPEATS           (default 10)
#   WARMUP            (default 3)
#   DET_RUNS          (default 5)
#   SYNTH_SIZES       (default "500 1000 2000 5000 10000 20000 50000")
#   SYNTH_DENSITIES   (default "low med high")
#   TRACCC_BUILD      (default /data/alice/sbetisor/traccc/build)
#   RAW_ROOT          (default /user/sbetisor/data-work/data)
#   RESULTS_ROOT      (default /user/sbetisor/data-work/results)

set -euo pipefail

# ── Environment ────────────────────────────────────────────────────────────────
SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
export LD_LIBRARY_PATH="${TRACCC_BUILD:-/data/alice/sbetisor/traccc/build}/lib64:$SPACK_VIEW/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_VISIBLE_DEVICES

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"

# ── Paths ──────────────────────────────────────────────────────────────────────
TRACCC_BUILD="${TRACCC_BUILD:-/data/alice/sbetisor/traccc/build}"
BIN="$TRACCC_BUILD/bin/traccc_benchmark_resolver_cuda"

RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_graph_reuse_full_sweep"
mkdir -p "$OUT/raw_baseline" "$OUT/raw_reuse"

# ── Parameters ─────────────────────────────────────────────────────────────────
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-5}"
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

# ── Sanity check ──────────────────────────────────────────────────────────────
if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found or not executable: $BIN" >&2
    exit 1
fi

# Verify graph reuse flag is present
if ! "$BIN" --help 2>&1 | grep -q "reuse-eviction-graph"; then
    echo "ERROR: --reuse-eviction-graph flag not found in binary help output" >&2
    echo "       Please rebuild the binary with the graph-reuse feature." >&2
    exit 1
fi

echo "=== Graph Reuse Full Sweep ==="
nvidia-smi --query-gpu=name,driver_version,compute_cap \
    --format=csv,noheader 2>/dev/null || true
echo "Binary      : $BIN"
echo "Repeats     : $REPEATS   Warmup: $WARMUP   DetRuns: $DET_RUNS"
echo "Output      : $OUT"
echo "Fatras data : $RAW_ROOT/fatras_csv_dumps/"
echo ""

# ── Summary TSV header ────────────────────────────────────────────────────────
SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tn_selected\tbaseline_ms\tbaseline_std\treuse_ms\treuse_std\tspeedup_pct\thash_match_baseline\thash_match_reuse\tdet_pass_baseline\tdet_pass_reuse\n' \
    > "$SUMMARY"

# ── Helper: parse one benchmark output file ───────────────────────────────────
# Outputs: n_cand n_sel mean std hm det_pass
parse_output() {
    local f="$1"
    local n_cand n_sel mean std hm det_pass
    n_cand=$(grep -oE 'n_candidates=[0-9]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    n_sel=$(grep -oE  'baseline_n_selected=[0-9]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    mean=$(grep -E    'baseline_time_ms_mean=' "$f" 2>/dev/null \
           | grep -oE 'time_ms_mean=[0-9.eE+\-]+' | cut -d= -f2 || echo NA)
    std=$(grep -E     'baseline_time_ms_std='  "$f" 2>/dev/null \
          | grep -oE 'time_ms_std=[0-9.eE+\-]+'  | cut -d= -f2 || echo NA)
    hm=$(grep -oE     'baseline_hash_match=(true|false)' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    det_pass=$(grep -oE 'det_baseline_pass=[0-9]+' "$f" 2>/dev/null | head -1 \
               | grep -oE '[0-9]+$' || echo 0)
    echo "$n_cand $n_sel $mean $std $hm $det_pass"
}

# ── run_one: run both backends and append one row to summary.tsv ──────────────
run_one() {
    local corpus="$1" event_label="$2"; shift 2
    local extra_args=("$@")

    local base_raw="$OUT/raw_baseline/${corpus}__${event_label}.txt"
    local reuse_raw="$OUT/raw_reuse/${corpus}__${event_label}.txt"

    set +e
    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        "${extra_args[@]}" \
        > "$base_raw" 2>&1
    local rc_base=$?

    "$BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        --reuse-eviction-graph \
        "${extra_args[@]}" \
        > "$reuse_raw" 2>&1
    local rc_reuse=$?
    set -e

    local n_cand n_sel base_mean base_std base_hm base_det
    local reuse_mean reuse_std reuse_hm reuse_det speedup

    if [[ $rc_base -ne 0 ]]; then
        n_cand=ERR; n_sel=ERR; base_mean=ERR; base_std=ERR; base_hm=ERR; base_det=0
    else
        read -r n_cand n_sel base_mean base_std base_hm base_det \
            < <(parse_output "$base_raw")
    fi

    if [[ $rc_reuse -ne 0 ]]; then
        reuse_mean=ERR; reuse_std=ERR; reuse_hm=ERR; reuse_det=0
    else
        read -r _nc _ns reuse_mean reuse_std reuse_hm reuse_det \
            < <(parse_output "$reuse_raw")
    fi

    speedup=$(awk -v a="$base_mean" -v b="$reuse_mean" \
        'BEGIN{if(a+0>0&&b+0>0)printf "%.2f",(a-b)/a*100;else print "NA"}' 2>/dev/null \
        || echo NA)

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" \
        "$n_cand" "$n_sel" \
        "$base_mean" "$base_std" \
        "$reuse_mean" "$reuse_std" \
        "$speedup" \
        "$base_hm" "$reuse_hm" \
        "$base_det" "$reuse_det" \
        >> "$SUMMARY"

    printf '  %-55s  base=%-9s  reuse=%-9s  Δ=%s%%\n' \
        "${corpus}/${event_label}" \
        "${base_mean:-ERR}" "${reuse_mean:-ERR}" "${speedup:-NA}"
}

# ── 1. Fatras pile-up sweep ───────────────────────────────────────────────────
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

# ── 2. ODD muon sweep (if available) ─────────────────────────────────────────
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

# ── 3. Synthetic sweep ────────────────────────────────────────────────────────
echo ""
echo "--- Synthetic sweep ---"
for density in $SYNTH_DENSITIES; do
    for n in $SYNTH_SIZES; do
        run_one "synthetic_${density}" "n${n}" \
            --synthetic --n-candidates="$n" --conflict-density="$density"
    done
done

# ── Per-corpus mean aggregation ───────────────────────────────────────────────
MEAN_TSV="$OUT/mean_by_corpus.tsv"
printf 'corpus\tn_events\tbaseline_mean_ms\treuse_mean_ms\tmean_speedup_pct\tany_hash_fail\n' \
    > "$MEAN_TSV"

awk -F'\t' '
NR > 1 && $5 != "ERR" && $5 != "NA" {
    base[$1] += $5; reuse[$1] += $7; spd[$1] += $9; n[$1]++;
    if ($10 != "true" || $11 != "true") fail[$1] = 1
}
END {
    for (k in n) {
        printf "%s\t%d\t%.3f\t%.3f\t%.2f\t%s\n",
            k, n[k], base[k]/n[k], reuse[k]/n[k], spd[k]/n[k],
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
run_id=${TS}_graph_reuse_full_sweep
binary=$BIN
branch=$BRANCH
commit=$COMMIT
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
repeats=$REPEATS
warmup=$WARMUP
det_runs=$DET_RUNS
synth_sizes=$SYNTH_SIZES
synth_densities=$SYNTH_DENSITIES
feature=graph_reuse_forward_ported_from_thesis-novelty-graph-reuse
note=greedy_only_path_no_conflict_graph_no_pbg
fatras_corpora=$(ls "$RAW_ROOT/fatras_csv_dumps/" 2>/dev/null | tr '\n' ' ')
EOF

echo ""
echo "=== Graph reuse full sweep complete ==="
echo "Summary  : $SUMMARY"
echo "Means    : $MEAN_TSV"
echo "Metadata : $OUT/run_metadata.txt"
echo "Output   : $OUT"
