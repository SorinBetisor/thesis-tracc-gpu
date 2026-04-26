#!/usr/bin/env bash
# Phase D7: Cross-detector validation. Beyond ODD, traccc ships three
# alternative geometries we can simulate locally without external data:
#
#   - telescope        : simple stack of planes (already covered by D3 at 200p)
#   - toy_detector     : cylindrical multi-layer toy detector
#   - wire_chamber     : drift-chamber-style geometry, very different topology
#
# This script produces a small corpus per geometry (multiple particle-density
# points) so the resolver is exercised on independent geometric layouts.
# Useful for the CERN review answer: "is this an ODD-specific result?"
#
# Usage:
#   ./scripts/phase_d7_alt_geom_corpus.sh
#
# Environment:
#   N_EVENTS         events per (geometry, density) cell  (default 5)
#   GEOMETRIES       space-separated list (default "telescope toy_detector wire_chamber")
#   DENSITIES        space-separated particle counts (default "50 200 500 1000")
#   FORCE            1 = re-simulate even if outputs exist
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
N_EVENTS="${N_EVENTS:-5}"
FORCE="${FORCE:-0}"
GEOMETRIES="${GEOMETRIES:-telescope toy_detector wire_chamber}"
DENSITIES="${DENSITIES:-50 200 500 1000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_REPO/results/${TIMESTAMP}_phase_d7_alt_geom}"
mkdir -p "$OUTDIR"

GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"

echo "=== Phase D7: alternative-geometry corpus ==="
echo "Geometries: $GEOMETRIES"
echo "Densities:  $DENSITIES"
echo "N events:   $N_EVENTS per cell"
echo "Output:     $OUTDIR"
echo ""

CORPUS_OUT="$OUTDIR/corpus_summary.tsv"
echo -e "geometry\tn_particles\tn_events\tdump_dir\tsweep_file" > "$CORPUS_OUT"

declare -A SIM_BIN_FOR
SIM_BIN_FOR[telescope]="$TRACCC_SRC/build/bin/traccc_simulate_telescope"
SIM_BIN_FOR[toy_detector]="$TRACCC_SRC/build/bin/traccc_simulate_toy_detector"
SIM_BIN_FOR[wire_chamber]="$TRACCC_SRC/build/bin/traccc_simulate_wire_chamber"

simulate_cell() {
  local geom="$1" n_part="$2"
  local sim_bin="${SIM_BIN_FOR[$geom]:-}"
  local sim_dir="$OUTDIR/$geom/n${n_part}/sim/detray_simulation/${geom}/n_particles_${n_part}"
  mkdir -p "$(dirname "$sim_dir")"

  if [[ -z "$sim_bin" || ! -x "$sim_bin" ]]; then
    echo "  WARNING: simulator not found: $sim_bin (skipping $geom)"
    return 1
  fi

  if compgen -G "$sim_dir/event*-particles_initial.csv" > /dev/null && [[ "$FORCE" -ne 1 ]]; then
    echo "  [skip] simulation already present at $sim_dir"
    echo "$sim_dir"; return 0
  fi
  mkdir -p "$sim_dir"

  echo "  --- simulating $geom n=$n_part ---"
  "$sim_bin" \
    --gen-vertex-xyz-mm=0:0:0 \
    --gen-vertex-xyz-std-mm=0:0:0 \
    --gen-mom-gev=10:10 \
    --gen-phi-degree=0:360 \
    --gen-eta=-3:3 \
    --gen-events="$N_EVENTS" \
    --gen-nparticles="$n_part" \
    --output-directory="$sim_dir/" \
    > "$OUTDIR/$geom/n${n_part}/sim.log" 2>&1
  echo "$sim_dir"
}

dump_cell() {
  local geom="$1" n_part="$2" sim_dir="$3"
  local cell="$OUTDIR/$geom/n${n_part}"
  local dump_dir="$cell/dumps"
  local log_dir="$cell/logs"
  mkdir -p "$dump_dir" "$log_dir"

  for (( i=0; i<N_EVENTS; i++ )); do
    local dump_path="$dump_dir/event_$(printf '%03d' $i).json"
    if [[ -s "$dump_path" && "$FORCE" -ne 1 ]]; then continue; fi
    (
      export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
      "$SEQ_BIN" \
        --input-directory="$sim_dir/" \
        --input-skip="$i" \
        --input-events=1 \
        --dump-ambiguity-input="$dump_path" \
        > "$log_dir/event_$(printf '%03d' $i)_seq.log" 2>&1
    ) || { echo "  WARNING: $geom n=$n_part event $i failed"; rm -f "$dump_path"; continue; }
    local n_cands
    n_cands=$(python3 -c "
import json
try:
    d=json.load(open('$dump_path'))
    print(len(d['tracks']) if isinstance(d, dict) else len(d))
except Exception: print('?')
" 2>/dev/null || echo "?")
    printf "  %s n=%d event %3d  n_candidates=%s\n" "$geom" "$n_part" "$i" "$n_cands"
  done
  echo "$dump_dir"
}

sweep_cell() {
  local geom="$1" n_part="$2" dump_dir="$3"
  local cell="$OUTDIR/$geom/n${n_part}"
  local sweep="$cell/sweep.txt"
  local per_event_dir="$cell/per_event"
  mkdir -p "$per_event_dir"

  if [[ -s "$sweep" && "$FORCE" -ne 1 ]]; then echo "$sweep"; return; fi

  if [[ ! -x "$GPU_BENCH" ]]; then
    echo "  GPU bench missing; skipping sweep for $geom n=$n_part"
    return 1
  fi

  {
    echo "# Phase D7 sweep: $geom n=$n_part"
    echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "# date: $(date)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
      | head -1 | { echo -n "# gpu: "; cat; echo; } || true
  } > "$sweep"

  for dump in "$dump_dir"/event_*.json; do
    [[ -f "$dump" ]] || continue
    local ev; ev=$(basename "$dump" .json)
    local per_event_file="$per_event_dir/${ev}.txt"
    "$GPU_BENCH" \
      --input-dump="$dump" \
      --repeats=10 --warmup=3 \
      --parallel-batch --parallel-batch-window=8192 \
      --conflict-graph=both \
      --determinism-runs=5 \
      2>&1 | tee "$per_event_file"
    { echo ""; echo "=== $ev ==="; cat "$per_event_file"; } >> "$sweep"
  done
  echo "$sweep"
}

for geom in $GEOMETRIES; do
  for n in $DENSITIES; do
    echo "--- cell: $geom n=$n ---"
    sim_dir=$(simulate_cell "$geom" "$n" | tail -1) || continue
    [[ -d "$sim_dir" ]] || continue
    dump_dir=$(dump_cell "$geom" "$n" "$sim_dir" | tail -1)
    if ls "$dump_dir"/event_*.json 2>/dev/null | head -1 >/dev/null; then
      sweep_file=$(sweep_cell "$geom" "$n" "$dump_dir" | tail -1 || echo "")
      echo -e "$geom\t$n\t$N_EVENTS\t$dump_dir\t$sweep_file" >> "$CORPUS_OUT"
    fi
  done
done

echo ""
echo "=== Phase D7 done ==="
echo "Corpus index: $CORPUS_OUT"
echo "Aggregate with: ./scripts/phase_d6_aggregate.sh --root=$OUTDIR"
