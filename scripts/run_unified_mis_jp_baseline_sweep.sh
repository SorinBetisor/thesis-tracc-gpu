#!/usr/bin/env bash
# Unified sweep: CPU greedy vs GPU baseline greedy vs GPU MIS vs GPU JP (Tier 2c).
# Same corpora shape as 20260426 unified sweep; uses --conflict-graph=both (not --enable-jp).
# ODD: ODD_DUMPS_ROOT default flat tree data-work/data/odd_dumps/geant4_*/event_*.json
#   (optional: ODD_D1_DUMPS, ODD_D5_NESTED_ROOT for legacy layouts).
# Writes per-event raw logs + wide summary.tsv under $OUT (default: data-work/results/<ts>_unified_mis_jp_baseline).
set -euo pipefail

LIBSTDCXX="${LIBSTDCXX:-}"
CPU_BIN="${CPU_BIN:-/user/sbetisor/data-work/traccc/build/bin/traccc_benchmark_resolver}"
GPU_BIN="${GPU_BIN:-/user/sbetisor/data-work/traccc/build/bin/traccc_benchmark_resolver_cuda}"
RESULTS_ROOT="${RESULTS_ROOT:-/user/sbetisor/data-work/results}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)_unified_mis_jp_baseline}"
OUT="${OUT:-$RESULTS_ROOT/$TS}"
RAW="$OUT/raw"
SUM="$OUT/summary.tsv"
LOG="$OUT/run.log"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-3}"
DETRUNS="${DETRUNS:-5}"

mkdir -p "$RAW"

cpu_wrap() {
  if [[ -n "$LIBSTDCXX" && -f "$LIBSTDCXX" ]]; then
    LD_PRELOAD="$LIBSTDCXX" "$@"
  else
    "$@"
  fi
}

gpu_wrap() {
  if [[ -n "$LIBSTDCXX" && -f "$LIBSTDCXX" ]]; then
    LD_PRELOAD="$LIBSTDCXX" "$@"
  else
    "$@"
  fi
}

echo "CPU_BIN=$CPU_BIN GPU_BIN=$GPU_BIN OUT=$OUT" | tee "$LOG"

cat >"$SUM" <<'HDR'
family	corpus	event	n_cand	n_meas	cpu_n_sel	cpu_time_mean_ms	cpu_time_med_ms	cpu_time_p95_ms	cpu_eps_mean	cpu_cand_per_s_mean	cpu_cand_per_s_med	base_n_sel	base_time_mean_ms	base_time_med_ms	base_time_p95_ms	base_eps_mean	base_cand_per_s_mean	base_cand_per_s_med	base_hash	base_ov	base_dup	mis_n_sel	mis_time_mean_ms	mis_time_med_ms	mis_time_p95_ms	mis_eps_mean	mis_cand_per_s_mean	mis_cand_per_s_med	mis_hash	mis_ov	mis_dup	mis_outer	mis_avg_v	mis_avg_e	jp_n_sel	jp_time_mean_ms	jp_time_med_ms	jp_time_p95_ms	jp_eps_mean	jp_cand_per_s_mean	jp_cand_per_s_med	jp_hash	jp_ov	jp_dup	jp_outer	jp_avg_v	jp_avg_e	det_base_p	det_base_f	det_mis_p	det_mis_f	det_jp_p	det_jp_f	status
HDR

extract() {
  grep -oE "(^|[[:space:]])$2=[^[:space:]]+" "$1" 2>/dev/null | head -1 | tr -d ' ' | sed "s/^$2=//"
  return 0
}

cand_rate() {
  python3 -c "import sys; n=float(sys.argv[1]); t=float(sys.argv[2]); print(f'{n*1000/t:.12g}' if t>0 else '?')" "$1" "$2" 2>/dev/null || echo "?"
}

run_event() {
  local family="$1" corpus="$2" event="$3" cpu_args="$4" gpu_args="$5"
  local stem="${family}__${corpus}__${event}"
  local cpu_file="$RAW/${stem}.cpu.txt"
  local gpu_file="$RAW/${stem}.gpu.txt"
  local row_ok="ok"

  echo "[run] $stem" | tee -a "$LOG"

  if ! cpu_wrap "$CPU_BIN" $cpu_args --backend=cpu --repeats=$REPEATS --warmup=$WARMUP >"$cpu_file" 2>&1; then
    row_ok="cpu_fail"
  fi
  if ! gpu_wrap "$GPU_BIN" $gpu_args --conflict-graph=both \
    --repeats=$REPEATS --warmup=$WARMUP --determinism-runs=$DETRUNS >"$gpu_file" 2>&1; then
    row_ok="${row_ok},gpu_fail"
  fi

  local n_cand n_meas
  n_cand=$(extract "$gpu_file" "n_candidates")
  [[ -z "$n_cand" ]] && n_cand=$(extract "$cpu_file" "n_candidates")
  n_meas=$(extract "$gpu_file" "n_meas")

  local cpu_sel cpu_tm cpu_med cpu_p95 cpu_eps cpu_cs cpu_cs_med
  cpu_sel=$(extract "$cpu_file" "n_selected")
  cpu_tm=$(extract "$cpu_file" "time_ms_mean")
  cpu_med=$(extract "$cpu_file" "time_ms_median")
  cpu_p95=$(extract "$cpu_file" "time_ms_p95")
  cpu_eps=$(extract "$cpu_file" "events_per_sec")
  cpu_cs=$(cand_rate "${n_cand:-0}" "${cpu_tm:-nan}")
  cpu_cs_med=$(cand_rate "${n_cand:-0}" "${cpu_med:-nan}")

  local bsel btm bmed bp95 beps bcs bcs_med bh bov bdup
  bsel=$(extract "$gpu_file" "baseline_n_selected")
  btm=$(extract "$gpu_file" "baseline_time_ms_mean")
  bmed=$(extract "$gpu_file" "baseline_time_ms_median")
  bp95=$(extract "$gpu_file" "baseline_time_ms_p95")
  beps=$(extract "$gpu_file" "baseline_events_per_sec")
  bcs=$(cand_rate "${n_cand:-0}" "${btm:-nan}")
  bcs_med=$(cand_rate "${n_cand:-0}" "${bmed:-nan}")
  bh=$(extract "$gpu_file" "baseline_hash_match")
  bov=$(extract "$gpu_file" "baseline_track_overlap_vs_cpu")
  bdup=$(extract "$gpu_file" "baseline_duplicate_rate_post")

  local msel mtm mmed mp95 meps mcs mcs_med mh mov mdup mou mv me
  msel=$(extract "$gpu_file" "graph_mis_n_selected")
  mtm=$(extract "$gpu_file" "graph_mis_time_ms_mean")
  mmed=$(extract "$gpu_file" "graph_mis_time_ms_median")
  mp95=$(extract "$gpu_file" "graph_mis_time_ms_p95")
  meps=$(extract "$gpu_file" "graph_mis_events_per_sec")
  mcs=$(cand_rate "${n_cand:-0}" "${mtm:-nan}")
  mcs_med=$(cand_rate "${n_cand:-0}" "${mmed:-nan}")
  mh=$(extract "$gpu_file" "graph_mis_hash_match")
  mov=$(extract "$gpu_file" "graph_mis_track_overlap_vs_cpu")
  mdup=$(extract "$gpu_file" "graph_mis_duplicate_rate_post")
  mou=$(extract "$gpu_file" "graph_mis_n_outer_iterations")
  mv=$(extract "$gpu_file" "graph_mis_avg_vertices")
  me=$(extract "$gpu_file" "graph_mis_avg_edges")

  local jsel jtm jmed jp95 jeps jcs jcs_med jh jov jdup jou jv je
  jsel=$(extract "$gpu_file" "graph_jp_n_selected")
  jtm=$(extract "$gpu_file" "graph_jp_time_ms_mean")
  jmed=$(extract "$gpu_file" "graph_jp_time_ms_median")
  jp95=$(extract "$gpu_file" "graph_jp_time_ms_p95")
  jeps=$(extract "$gpu_file" "graph_jp_events_per_sec")
  jcs=$(cand_rate "${n_cand:-0}" "${jtm:-nan}")
  jcs_med=$(cand_rate "${n_cand:-0}" "${jmed:-nan}")
  jh=$(extract "$gpu_file" "graph_jp_hash_match")
  jov=$(extract "$gpu_file" "graph_jp_track_overlap_vs_cpu")
  jdup=$(extract "$gpu_file" "graph_jp_duplicate_rate_post")
  jou=$(extract "$gpu_file" "graph_jp_n_outer_iterations")
  jv=$(extract "$gpu_file" "graph_jp_avg_vertices")
  je=$(extract "$gpu_file" "graph_jp_avg_edges")

  local dbp dbf dmp dmf djp djf
  dbp=$(extract "$gpu_file" "det_baseline_pass")
  dbf=$(extract "$gpu_file" "det_baseline_fail")
  dmp=$(extract "$gpu_file" "det_graph_mis_pass")
  dmf=$(extract "$gpu_file" "det_graph_mis_fail")
  djp=$(extract "$gpu_file" "det_graph_jp_pass")
  djf=$(extract "$gpu_file" "det_graph_jp_fail")

  declare -a _col=(
    "$family" "$corpus" "$event" "${n_cand:-?}" "${n_meas:-?}"
    "${cpu_sel:-?}" "${cpu_tm:-?}" "${cpu_med:-?}" "${cpu_p95:-?}" "${cpu_eps:-?}" "${cpu_cs:-?}" "${cpu_cs_med:-?}"
    "${bsel:-?}" "${btm:-?}" "${bmed:-?}" "${bp95:-?}" "${beps:-?}" "${bcs:-?}" "${bcs_med:-?}" "${bh:-?}" "${bov:-?}" "${bdup:-?}"
    "${msel:-?}" "${mtm:-?}" "${mmed:-?}" "${mp95:-?}" "${meps:-?}" "${mcs:-?}" "${mcs_med:-?}" "${mh:-?}" "${mov:-?}" "${mdup:-?}" "${mou:-?}" "${mv:-?}" "${me:-?}"
    "${jsel:-?}" "${jtm:-?}" "${jmed:-?}" "${jp95:-?}" "${jeps:-?}" "${jcs:-?}" "${jcs_med:-?}" "${jh:-?}" "${jov:-?}" "${jdup:-?}" "${jou:-?}" "${jv:-?}" "${je:-?}"
    "${dbp:-?}" "${dbf:-?}" "${dmp:-?}" "${dmf:-?}" "${djp:-?}" "${djf:-?}" "$row_ok"
  )
  printf "%s" "${_col[0]}" >>"$SUM"
  local _ci
  for ((_ci = 1; _ci < ${#_col[@]}; _ci++)); do printf "\t%s" "${_col[$_ci]}" >>"$SUM"; done
  printf "\n" >>"$SUM"
}

echo "=== Synthetic ===" | tee -a "$LOG"
for n in 500 1000 2000 5000 10000; do
  for d in low med high; do
    args="--synthetic --n-candidates=$n --conflict-density=$d"
    run_event "synthetic" "n${n}_${d}" "single" "$args" "$args"
  done
done

echo "=== FATRAS ===" | tee -a "$LOG"
FAT="${FATRAS_ROOT:-/user/sbetisor/data-work/data/fatras_csv_dumps}"
for cdir in "$FAT"/fatras_ttbar_mu*; do
  [[ -d "$cdir" ]] || continue
  corpus=$(basename "$cdir")
  for dump in "$cdir"/event_*.json; do
    [[ -f "$dump" ]] || continue
    ev=$(basename "$dump" .json)
    args="--input-dump=$dump"
    run_event "fatras" "$corpus" "$ev" "$args" "$args"
  done
done

ODD_FLAT="${ODD_DUMPS_ROOT:-/user/sbetisor/data-work/data/odd_dumps}"
if [[ -d "$ODD_FLAT" ]]; then
  echo "=== ODD geant4 (flat: odd_dumps/geant4_*/event_*.json) ===" | tee -a "$LOG"
  for cdir in "$ODD_FLAT"/geant4_*; do
    [[ -d "$cdir" ]] || continue
    corpus=$(basename "$cdir")
    for dump in "$cdir"/event_*.json; do
      [[ -f "$dump" ]] || continue
      ev=$(basename "$dump" .json)
      args="--input-dump=$dump"
      run_event "odd" "$corpus" "$ev" "$args" "$args"
    done
  done
fi

D1="${ODD_D1_DUMPS:-}"
if [[ -n "$D1" && -d "$D1" ]]; then
  echo "=== ODD D1 (legacy path; set ODD_D1_DUMPS) ===" | tee -a "$LOG"
  for dump in "$D1"/event_*.json; do
    [[ -f "$dump" ]] || continue
    ev=$(basename "$dump" .json)
    args="--input-dump=$dump"
    run_event "odd" "geant4_10muon_10GeV" "$ev" "$args" "$args"
  done
fi

D5_NESTED="${ODD_D5_NESTED_ROOT:-/user/sbetisor/thesis/sorin-thesis-work/results/20260426_175604_phase_d5_odd_corpus}"
if [[ -d "$D5_NESTED" ]]; then
  echo "=== ODD D5 nested (geant4_*/dumps/) ===" | tee -a "$LOG"
  for cdir in "$D5_NESTED"/geant4_*/dumps; do
    [[ -d "$cdir" ]] || continue
    corpus=$(basename "$(dirname "$cdir")")
    for dump in "$cdir"/event_*.json; do
      [[ -f "$dump" ]] || continue
      ev=$(basename "$dump" .json)
      args="--input-dump=$dump"
      run_event "odd" "$corpus" "$ev" "$args" "$args"
    done
  done
fi

echo "done" | tee -a "$LOG"
wc -l "$SUM" | tee -a "$LOG"
