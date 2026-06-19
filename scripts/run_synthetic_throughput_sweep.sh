#!/usr/bin/env bash
# Synthetic end-to-end throughput sweep: CPU vs GPU baseline vs JP
#
# Same metrics as run_fatras_throughput_sweep.sh but uses --synthetic inputs.
#
# Outputs:
#   $RESULTS_ROOT/<TS>_synthetic_throughput/
#     summary.tsv
#     pileup_aggregate.tsv
#     mean_by_corpus.tsv
#     raw_cpu/
#     raw_gpu/
#     run_metadata.txt
#
# Env overrides:
#   SYNTH_SIZES       (default "500 1000 2000 5000 10000 20000 50000")
#   SYNTH_DENSITIES   (default "low med high")
#   REPEATS WARMUP DET_RUNS TRACCC_BUILD RESULTS_ROOT

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
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
OUT="$RESULTS_ROOT/${TS}_synthetic_throughput"

REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DET_RUNS="${DET_RUNS:-3}"
SYNTH_SIZES="${SYNTH_SIZES:-500 1000 2000 5000 10000 20000 50000}"
SYNTH_DENSITIES="${SYNTH_DENSITIES:-low med high}"

mkdir -p "$OUT/raw_cpu" "$OUT/raw_gpu"

[[ -x "$CUDA_BIN" ]] || { echo "ERROR: $CUDA_BIN not found" >&2; exit 1; }
[[ -x "$CPU_BIN" ]]  || { echo "ERROR: $CPU_BIN not found" >&2; exit 1; }

echo "=== Synthetic throughput sweep (resolver-only + end-to-end) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
echo "CUDA bin : $CUDA_BIN"
echo "CPU bin  : $CPU_BIN"
echo "Output   : $OUT"
echo "Sizes    : $SYNTH_SIZES"
echo "Densities: $SYNTH_DENSITIES"
echo "Repeats  : $REPEATS  Warmup: $WARMUP"
echo ""

SUMMARY="$OUT/summary.tsv"
printf 'corpus\tevent\tn_candidates\tcpu_ms\tcpu_e2e_eps\th2d_ms\tbase_res_ms\tbase_d2h_ms\tbase_e2e_ms\tbase_e2e_eps\tjp_res_ms\tjp_d2h_ms\tjp_e2e_ms\tjp_e2e_eps\tjp_hash_match\tstatus\n' \
    > "$SUMMARY"

parse_cpu() {
    grep -oE 'time_ms_mean=[0-9.eE+\-]+' "$1" 2>/dev/null | head -1 | cut -d= -f2 || echo NA
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
    local corpus="$1" event_label="$2" density="$3" n="$4"
    local cpu_raw="$OUT/raw_cpu/${corpus}__${event_label}.txt"
    local gpu_raw="$OUT/raw_gpu/${corpus}__${event_label}.txt"
    local synth_args=(--synthetic --n-candidates="$n" --conflict-density="$density")
    local status=ok

    set +e
    "$CPU_BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        "${synth_args[@]}" > "$cpu_raw" 2>&1
    local rc_cpu=$?

    "$CUDA_BIN" --repeats="$REPEATS" --warmup="$WARMUP" \
        --determinism-runs="$DET_RUNS" \
        --conflict-graph=jp \
        "${synth_args[@]}" > "$gpu_raw" 2>&1
    local rc_gpu=$?
    set -e

    local n_cand cpu_ms cpu_e2e_eps
    local h2d base_res base_d2h jp_res jp_d2h hm
    local base_e2e jp_e2e base_e2e_eps jp_e2e_eps

    n_cand=$(grep -oE 'n_candidates=[0-9]+' "$gpu_raw" 2>/dev/null | head -1 | cut -d= -f2 || echo "$n")

    if [[ $rc_cpu -ne 0 ]]; then cpu_ms=ERR; cpu_e2e_eps=NA; status=cpu_fail
    else
        cpu_ms=$(parse_cpu "$cpu_raw")
        cpu_e2e_eps=$(awk -v t="$cpu_ms" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
    fi

    if [[ $rc_gpu -ne 0 ]]; then
        h2d=ERR; base_res=ERR; base_d2h=ERR; jp_res=ERR; jp_d2h=ERR; hm=ERR
        base_e2e=ERR; jp_e2e=ERR; base_e2e_eps=NA; jp_e2e_eps=NA
        status=gpu_fail
    else
        read -r h2d base_res base_d2h jp_res jp_d2h hm < <(parse_gpu "$gpu_raw")
        base_e2e=$(awk -v a="$h2d" -v b="$base_res" -v c="$base_d2h" \
            'BEGIN{if(a+0>=0&&b+0>0&&c+0>=0)printf "%.4f",a+b+c;else print "NA"}')
        jp_e2e=$(awk -v a="$h2d" -v b="$jp_res" -v c="$jp_d2h" \
            'BEGIN{if(a+0>=0&&b+0>0&&c+0>=0)printf "%.4f",a+b+c;else print "NA"}')
        base_e2e_eps=$(awk -v t="$base_e2e" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
        jp_e2e_eps=$(awk -v t="$jp_e2e" 'BEGIN{if(t+0>0)printf "%.2f",1000/t;else print "NA"}')
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$corpus" "$event_label" "$n_cand" \
        "$cpu_ms" "$cpu_e2e_eps" \
        "$h2d" "$base_res" "$base_d2h" "$base_e2e" "$base_e2e_eps" \
        "$jp_res" "$jp_d2h" "$jp_e2e" "$jp_e2e_eps" "$hm" \
        "$status" \
        >> "$SUMMARY"

    printf '  %-40s  cpu=%-9s  base_e2e=%-9s  jp_e2e=%-9s  %s\n' \
        "${corpus}/${event_label}" \
        "${cpu_ms:-ERR}" "${base_e2e:-ERR}" "${jp_e2e:-ERR}" "$status"
}

for density in $SYNTH_DENSITIES; do
    echo "--- synthetic_${density} ---"
    for n in $SYNTH_SIZES; do
        run_one "synthetic_${density}" "n${n}" "$density" "$n"
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

AGG="$OUT/config_aggregate.tsv"
python3 - "$SUMMARY" "$AGG" << 'PY'
import csv, statistics as stats, sys
from pathlib import Path

src, out = map(Path, sys.argv[1:3])
rows = list(csv.DictReader(src.open(), delimiter='\t'))
fields = [
    'corpus','event','density','n_candidates',
    'cpu_ms_mean','h2d_ms_mean',
    'gpu_res_ms_mean','gpu_d2h_ms_mean','gpu_e2e_ms_mean','gpu_e2e_eps_mean',
    'jp_res_ms_mean','jp_d2h_ms_mean','jp_e2e_ms_mean','jp_e2e_eps_mean',
    'res_jp_vs_gpu','e2e_jp_vs_gpu',
    'jp_hash_match','status'
]
out_rows = []
for r in rows:
    if r['base_res_ms'] == 'ERR' or r['jp_res_ms'] == 'ERR':
        out_rows.append({
            'corpus': r['corpus'], 'event': r['event'],
            'density': r['corpus'].replace('synthetic_', ''),
            'n_candidates': r['n_candidates'],
            'cpu_ms_mean': r['cpu_ms'], 'h2d_ms_mean': r['h2d_ms'],
            'gpu_res_ms_mean': 'ERR', 'gpu_d2h_ms_mean': 'ERR',
            'gpu_e2e_ms_mean': 'ERR', 'gpu_e2e_eps_mean': 'ERR',
            'jp_res_ms_mean': 'ERR', 'jp_d2h_ms_mean': 'ERR',
            'jp_e2e_ms_mean': 'ERR', 'jp_e2e_eps_mean': 'ERR',
            'res_jp_vs_gpu': 'ERR', 'e2e_jp_vs_gpu': 'ERR',
            'jp_hash_match': r.get('jp_hash_match', 'ERR'),
            'status': r.get('status', 'fail'),
        })
        continue
    br = float(r['base_res_ms']); jr = float(r['jp_res_ms'])
    be = float(r['base_e2e_ms']); je = float(r['jp_e2e_ms'])
    out_rows.append({
        'corpus': r['corpus'], 'event': r['event'],
        'density': r['corpus'].replace('synthetic_', ''),
        'n_candidates': r['n_candidates'],
        'cpu_ms_mean': r['cpu_ms'], 'h2d_ms_mean': r['h2d_ms'],
        'gpu_res_ms_mean': r['base_res_ms'], 'gpu_d2h_ms_mean': r['base_d2h_ms'],
        'gpu_e2e_ms_mean': r['base_e2e_ms'], 'gpu_e2e_eps_mean': f"{1000/be:.1f}",
        'jp_res_ms_mean': r['jp_res_ms'], 'jp_d2h_ms_mean': r['jp_d2h_ms'],
        'jp_e2e_ms_mean': r['jp_e2e_ms'], 'jp_e2e_eps_mean': f"{1000/je:.1f}",
        'res_jp_vs_gpu': f"{br/jr:.2f}", 'e2e_jp_vs_gpu': f"{be/je:.2f}",
        'jp_hash_match': r.get('jp_hash_match', 'NA'),
        'status': r.get('status', 'ok'),
    })

def sort_key(row):
    order = {'low': 0, 'med': 1, 'high': 2}
    n = int(row['event'].replace('n', '')) if row['event'].startswith('n') else 0
    return (order.get(row['density'], 9), n)

out_rows.sort(key=sort_key)
with out.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
    w.writeheader()
    w.writerows(out_rows)
PY

echo ""
echo "=== Per-density means ==="
column -t -s $'\t' "$MEAN_TSV"
echo ""
echo "=== Per-config table ==="
column -t -s $'\t' "$AGG"

COMMIT=$(cd /data/alice/sbetisor/traccc && git rev-parse HEAD 2>/dev/null || echo unknown)
BRANCH=$(cd /data/alice/sbetisor/traccc && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
cat > "$OUT/run_metadata.txt" << EOF
run_id=${TS}_synthetic_throughput
cuda_binary=$CUDA_BIN
cpu_binary=$CPU_BIN
branch=$BRANCH
commit=$COMMIT
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
repeats=$REPEATS
warmup=$WARMUP
det_runs=$DET_RUNS
synth_sizes=$SYNTH_SIZES
synth_densities=$SYNTH_DENSITIES
e2e_formula=h2d_ms + resolver_ms_mean + d2h_ms
note=synthetic physics-calibrated generator seed=42
EOF

echo ""
echo "=== Done ==="
echo "Summary   : $SUMMARY"
echo "Aggregate : $AGG"
echo "Output    : $OUT"
