#!/usr/bin/env bash
# Fatras end-to-end throughput sweep: CPU vs GPU baseline vs JP
#
# Resolver-only latency is what prior tables used (time_ms_mean).
# This sweep also reports end-to-end per-event cost:
#   e2e_ms = time_h2d_ms + resolver_ms + time_d2h_ms
#   e2e_events_per_sec = 1000 / e2e_ms
#
# CPU has no H2D/D2H; cpu_e2e_ms = cpu resolver time only.
#
# One CUDA invocation per event runs baseline + JP (--conflict-graph=jp).
#
# Outputs:
#   $RESULTS_ROOT/<TS>_fatras_throughput/
#     summary.tsv
#     mean_by_corpus.tsv
#     raw_cpu/
#     raw_gpu/
#     run_metadata.txt

set -euo pipefail

SPACK_VIEW="/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/view"
export LD_PRELOAD="${LD_PRELOAD:-$SPACK_VIEW/lib64/libstdc++.so.6}"
TRACCC_BUILD="${TRACCC_BUILD:-/data/alice/sbetisor/traccc/build}"
export LD_LIBRARY_PATH="${TRACCC_BUILD}/lib64:$SPACK_VIEW/lib64:${LD_LIBRARY_PATH:-}"
unset CUDA_VISIBLE_DEVICES
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"

CUDA_BIN="$TRACCC_BUILD/bin/traccc_benchmark_resolver_cuda"
CPU_BIN="$TRACCC_BUILD/bin/traccc_benchmark_resolver"
RAW_ROOT="${RAW_ROOT:-/user/sbetisor/data-work/data}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_fatras_throughput"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-3}"

mkdir -p "$OUT/raw_cpu" "$OUT/raw_gpu"

[[ -x "$CUDA_BIN" ]] || { echo "ERROR: $CUDA_BIN not found" >&2; exit 1; }
[[ -x "$CPU_BIN" ]]  || { echo "ERROR: $CPU_BIN not found" >&2; exit 1; }

echo "=== Fatras throughput sweep (resolver-only + end-to-end) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
echo "CUDA bin : $CUDA_BIN"
echo "CPU bin  : $CPU_BIN"
echo "Output   : $OUT"
echo "Repeats  : $REPEATS  Warmup: $WARMUP"
echo ""

SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tcpu_ms\tcpu_e2e_eps\th2d_ms\tbase_res_ms\tbase_d2h_ms\tbase_e2e_ms\tbase_e2e_eps\tjp_res_ms\tjp_d2h_ms\tjp_e2e_ms\tjp_e2e_eps\tjp_hash_match\n' \
    > "$SUMMARY"

# Parse float field: prefix_field=value
getf() {
    local f="$1" prefix="$2"
    grep -E "${prefix}${prefix:+_}${f}=" "$f" 2>/dev/null | head -1 | sed 's/.*=//' || true
}

parse_cpu() {
    local f="$1"
    local mean
    mean=$(grep -oE 'time_ms_mean=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    echo "$mean"
}

parse_gpu() {
    local f="$1"
    local h2d base_res base_d2h jp_res jp_d2h hm
    h2d=$(grep -oE 'time_h2d_ms=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    base_res=$(grep -oE 'baseline_time_ms_mean=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    base_d2h=$(grep -oE 'baseline_time_d2h_ms=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    jp_res=$(grep -oE 'graph_jp_time_ms_mean=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    jp_d2h=$(grep -oE 'graph_jp_time_d2h_ms=[0-9.eE+\-]+' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    hm=$(grep -oE 'graph_jp_hash_match=(true|false)' "$f" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)
    echo "$h2d $base_res $base_d2h $jp_res $jp_d2h $hm"
}

run_one() {
    local corpus="$1" event_label="$2" ev_path="$3"
    local cpu_raw="$OUT/raw_cpu/${corpus}__${event_label}.txt"
    local gpu_raw="$OUT/raw_gpu/${corpus}__${event_label}.txt"

    set +e
    "$CPU_BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --input-dump="$ev_path" > "$cpu_raw" 2>&1
    local rc_cpu=$?

    "$CUDA_BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        --conflict-graph=jp \
        --input-dump="$ev_path" > "$gpu_raw" 2>&1
    local rc_gpu=$?
    set -e

    local n_cand cpu_ms cpu_e2e_eps
    local h2d base_res base_d2h jp_res jp_d2h hm
    local base_e2e jp_e2e base_e2e_eps jp_e2e_eps

    n_cand=$(grep -oE 'n_candidates=[0-9]+' "$gpu_raw" 2>/dev/null | head -1 | cut -d= -f2 || echo NA)

    if [[ $rc_cpu -ne 0 ]]; then cpu_ms=ERR; cpu_e2e_eps=NA
    else
        cpu_ms=$(parse_cpu "$cpu_raw")
        cpu_e2e_eps=$(awk -v t="$cpu_ms" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
    fi

    if [[ $rc_gpu -ne 0 ]]; then
        h2d=ERR; base_res=ERR; base_d2h=ERR; jp_res=ERR; jp_d2h=ERR; hm=ERR
        base_e2e=ERR; jp_e2e=ERR; base_e2e_eps=NA; jp_e2e_eps=NA
    else
        read -r h2d base_res base_d2h jp_res jp_d2h hm < <(parse_gpu "$gpu_raw")
        base_e2e=$(awk -v a="$h2d" -v b="$base_res" -v c="$base_d2h" \
            'BEGIN{if(a+0>=0&&b+0>0&&c+0>=0)printf "%.4f",a+b+c;else print "NA"}')
        jp_e2e=$(awk -v a="$h2d" -v b="$jp_res" -v c="$jp_d2h" \
            'BEGIN{if(a+0>=0&&b+0>0&&c+0>=0)printf "%.4f",a+b+c;else print "NA"}')
        base_e2e_eps=$(awk -v t="$base_e2e" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
        jp_e2e_eps=$(awk -v t="$jp_e2e" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" "$n_cand" \
        "$cpu_ms" "$cpu_e2e_eps" \
        "$h2d" "$base_res" "$base_d2h" "$base_e2e" "$base_e2e_eps" \
        "$jp_res" "$jp_d2h" "$jp_e2e" "$jp_e2e_eps" "$hm" \
        >> "$SUMMARY"

    printf '  %-45s  cpu=%-8s  base_e2e=%-8s (%.1f ev/s)  jp_e2e=%-8s (%.1f ev/s)\n' \
        "${corpus}/${event_label}" \
        "${cpu_ms:-ERR}" "${base_e2e:-ERR}" "${base_e2e_eps:-0}" \
        "${jp_e2e:-ERR}" "${jp_e2e_eps:-0}"
}

for d in "$RAW_ROOT"/fatras_csv_dumps/fatras_ttbar_mu*; do
    [[ -d "$d" ]] || continue
    corpus="$(basename "$d")"
    echo "--- $corpus ---"
    for ev in "$d"/event_*.json; do
        [[ -e "$ev" ]] || continue
        run_one "$corpus" "$(basename "$ev" .json)" "$ev"
    done
done

MEAN_TSV="$OUT/mean_by_corpus.tsv"
printf 'corpus\tn_events\tn_cand_mean\tcpu_e2e_eps\tbase_res_eps\tjp_res_eps\tbase_e2e_eps\tjp_e2e_eps\te2e_jp_vs_base\tres_jp_vs_base\thash_fail\n' \
    > "$MEAN_TSV"

awk -F'\t' '
NR > 1 && $4 != "ERR" && $7 != "ERR" && $11 != "ERR" {
    nc[$1]+=$3; n[$1]++;
    cpu_eps[$1]+=$5;
    base_res_eps[$1]+=(1000/$7);
    jp_res_eps[$1]+=(1000/$11);
    base_e2e_eps[$1]+=$10;
    jp_e2e_eps[$1]+=$14;
    if ($15 != "true") fail[$1]=1
}
END {
    for (k in n) {
        bres = base_res_eps[k]/n[k]; jres = jp_res_eps[k]/n[k];
        be2e = base_e2e_eps[k]/n[k]; je2e = jp_e2e_eps[k]/n[k];
        printf "%s\t%d\t%.0f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.2f\t%.2f\t%s\n",
            k, n[k], nc[k]/n[k], cpu_eps[k]/n[k], bres, jres, be2e, je2e,
            be2e/je2e, bres/jres, (k in fail) ? "YES" : "no"
    }
}' "$SUMMARY" | sort >> "$MEAN_TSV"

echo ""
echo "=== Per-corpus means ==="
column -t -s $'\t' "$MEAN_TSV"

COMMIT=$(cd /data/alice/sbetisor/traccc && git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(cd /data/alice/sbetisor/traccc && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
cat > "$OUT/run_metadata.txt" << EOF
run_id=${TS}_fatras_throughput
cuda_binary=$CUDA_BIN
cpu_binary=$CPU_BIN
branch=$BRANCH
commit=$COMMIT
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
repeats=$REPEATS
warmup=$WARMUP
det_runs=$DET_RUNS
e2e_formula=h2d_ms + resolver_ms_mean + d2h_ms
note=h2d measured once per event before warmup; not included in resolver repeats
EOF

echo ""
echo "=== Done ==="
echo "Summary : $SUMMARY"
echo "Means   : $MEAN_TSV"
echo "Output  : $OUT"
