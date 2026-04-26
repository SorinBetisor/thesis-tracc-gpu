#!/usr/bin/env bash
# Phase E3: End-to-end full-chain events/s measurement.
# Runs traccc_seq_example (full CKF + ambiguity resolution) with baseline
# and JP resolvers at one Fatras pile-up point and reports events/s.
#
# This converts the resolver-only speedup into the number reviewers care about:
# the full-pipeline throughput impact.
#
# Usage:
#   ./scripts/phase_e3_full_chain.sh [--mu=400]
# Environment:
#   TRACCC_SRC, THESIS_REPO
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
MU="${MU:-400}"
N_EVENTS="${N_EVENTS:-20}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${TIMESTAMP}_phase_e3_fullchain}"
mkdir -p "$OUTDIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mu=*) MU="${1#*=}"; shift ;;
    --n-events=*) N_EVENTS="${1#*=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"
GPU_SEQ="${TRACCC_SRC}/build/bin/traccc_seq_example_cuda"

if [[ ! -x "$SEQ_BIN" ]] && [[ ! -x "$GPU_SEQ" ]]; then
  echo "ERROR: no traccc_seq_example binary found."
  exit 1
fi

FATRAS_DIR="$TRACCC_SRC/data/odd/fatras_ttbar_mu${MU}"
if [[ ! -d "$FATRAS_DIR" ]]; then
  echo "ERROR: Fatras mu=$MU data not found at $FATRAS_DIR"
  exit 1
fi

echo "=== Phase E3: Full-chain events/s measurement ==="
echo "Fatras mu=$MU, $N_EVENTS events"
echo "Output: $OUTDIR"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

WARMUP=3
REPEATS=5

run_chain() {
  local label="$1"
  local bin="$2"
  shift 2
  local extra_flags=("$@")

  local outfile="$OUTDIR/${label}.txt"
  echo "--- $label ---"

  # Time N_EVENTS events REPEATS times and compute mean.
  local times=()
  for (( r=0; r<WARMUP+REPEATS; r++ )); do
    local t0 t1
    t0=$(date +%s%3N)
    (
      export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
      "$bin" \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --material-file=geometries/odd/odd-detray_material_detray.json \
        --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory="odd/fatras_ttbar_mu${MU}/" \
        --input-events="$N_EVENTS" \
        "${extra_flags[@]}" \
        > "$outfile.run${r}" 2>&1
    )
    t1=$(date +%s%3N)
    local dt=$(( t1 - t0 ))
    [[ $r -ge $WARMUP ]] && times+=("$dt")
    printf "  run %d: %d ms (%.1f events/s)\n" \
      "$r" "$dt" "$(python3 -c "print(${N_EVENTS}*1000/${dt})")"
  done

  # Mean over timed repeats.
  local sum=0
  for t in "${times[@]}"; do sum=$(( sum + t )); done
  local mean_ms=$(( sum / REPEATS ))
  local evts_per_s
  evts_per_s=$(python3 -c "print(f'{${N_EVENTS}*1000/${mean_ms}:.2f}')")
  echo "  MEAN: ${mean_ms} ms -> ${evts_per_s} events/s" | tee -a "$outfile"
  echo "label=$label mean_chain_ms=${mean_ms} events_per_s=${evts_per_s} n_events=${N_EVENTS}" \
    >> "$OUTDIR/summary.txt"
  echo ""
}

echo "# Phase E3 full-chain summary" > "$OUTDIR/summary.txt"
echo "# mu=$MU n_events=$N_EVENTS" >> "$OUTDIR/summary.txt"
echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)" \
  >> "$OUTDIR/summary.txt"
echo "" >> "$OUTDIR/summary.txt"

# CPU baseline (sequential, default resolver).
if [[ -x "$SEQ_BIN" ]]; then
  run_chain "cpu_baseline" "$SEQ_BIN"
fi

# GPU baseline (default resolver, if GPU binary exists).
if [[ -x "$GPU_SEQ" ]]; then
  run_chain "gpu_baseline" "$GPU_SEQ"
  # GPU + JP conflict-graph resolver (if binary supports --conflict-graph flag).
  run_chain "gpu_jp" "$GPU_SEQ" "--ambig-mode=conflict_graph_jp" || \
    echo "NOTE: --ambig-mode flag not supported by this binary; JP full-chain pending upstream integration."
fi

echo "=== Phase E3 complete ==="
echo "Summary: $OUTDIR/summary.txt"
cat "$OUTDIR/summary.txt"
