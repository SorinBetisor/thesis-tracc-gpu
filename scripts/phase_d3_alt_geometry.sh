#!/usr/bin/env bash
# Phase D3: Generate one alternative-geometry corpus (telescope detector) via
# traccc_simulate_telescope, dump ambiguity-resolver inputs, and benchmark.
#
# The telescope geometry is fully local (no external data fetch) and runs on
# any CPU node. Provides a non-ODD geometry data point that kills the
# "ODD-specific" reviewer objection.
#
# Usage:
#   ./scripts/phase_d3_alt_geometry.sh
# Environment:
#   TRACCC_SRC, THESIS_REPO, N_EVENTS, N_PARTICLES
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
N_EVENTS="${N_EVENTS:-10}"
N_PARTICLES="${N_PARTICLES:-200}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_REPO/results/${TIMESTAMP}_phase_d3_telescope}"
SIM_DIR="$OUTDIR/detray_simulation/telescope_detector/n_particles_${N_PARTICLES}"
DUMP_DIR="$OUTDIR/telescope_dumps"
mkdir -p "$SIM_DIR" "$DUMP_DIR"

SIM_BIN="$TRACCC_SRC/build/bin/traccc_simulate_telescope"
SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"
GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"

for b in "$SIM_BIN" "$SEQ_BIN"; do
  if [[ ! -x "$b" ]]; then
    echo "ERROR: binary not found: $b — rebuild traccc with this geometry enabled"
    exit 1
  fi
done

echo "=== Phase D3: telescope geometry simulation + benchmark ==="
echo "N events:    $N_EVENTS"
echo "N particles: $N_PARTICLES"
echo "Output:      $OUTDIR"
echo ""

# Step 1: simulate telescope events.
echo "--- Step 1: simulate ---"
"$SIM_BIN" \
  --gen-vertex-xyz-mm=0:0:0 \
  --gen-vertex-xyz-std-mm=0:0:0 \
  --gen-mom-gev=10:10 \
  --gen-phi-degree=0:0 \
  --gen-events="$N_EVENTS" \
  --gen-nparticles="$N_PARTICLES" \
  --output-directory="$SIM_DIR/" \
  --gen-eta=1:3

echo "Simulation done."

# Step 2: dump ambiguity-resolver inputs.
echo "--- Step 2: dump ---"
SUMMARY="$DUMP_DIR/dump_summary.txt"
echo "event_idx n_candidates" > "$SUMMARY"

for (( i=0; i<N_EVENTS; i++ )); do
  DUMP_PATH="$DUMP_DIR/event_$(printf '%03d' $i).json"
  (
    export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
    "$SEQ_BIN" \
      --input-directory="$SIM_DIR/" \
      --input-skip="$i" \
      --input-events=1 \
      --dump-ambiguity-input="$DUMP_PATH" \
      > "$DUMP_DIR/event_$(printf '%03d' $i)_seq.log" 2>&1
  ) || { echo "WARNING: event $i seq failed (rc=$?)"; continue; }

  N_CANDS=$(python3 -c "
import json
with open('$DUMP_PATH') as f: d=json.load(f)
print(len(d['tracks']) if isinstance(d, dict) else len(d))
" 2>/dev/null || echo "?")
  echo "$i $N_CANDS" >> "$SUMMARY"
  printf "  event %3d  n_candidates=%s\n" "$i" "$N_CANDS"
done

# Step 3: GPU benchmark sweep if GPU bench is available.
SWEEP_OUT="$OUTDIR/telescope_sweep.txt"
if [[ -x "$GPU_BENCH" ]]; then
  echo "--- Step 3: GPU benchmark ---"
  echo "# Phase D3 telescope sweep" > "$SWEEP_OUT"
  echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)" >> "$SWEEP_OUT"
  echo "# date: $(date)" >> "$SWEEP_OUT"

  for dump in "$DUMP_DIR"/event_*.json; do
    ev=$(basename "$dump" .json)
    echo "" >> "$SWEEP_OUT"
    echo "=== $ev ===" >> "$SWEEP_OUT"
    "$GPU_BENCH" \
      --input-dump="$dump" \
      --repeats=10 --warmup=3 \
      --parallel-batch --parallel-batch-window=8192 \
      --conflict-graph=both \
      --determinism-runs=3 \
      2>&1 | tee -a "$SWEEP_OUT"
  done
  echo "GPU sweep written to $SWEEP_OUT"
else
  echo "GPU bench not available (run Step 3 on a GPU node separately):"
  echo "  OUTDIR=$OUTDIR ./scripts/phase_d4_resweep.sh --dump-dir=$DUMP_DIR"
fi

echo ""
echo "=== Phase D3 complete: $OUTDIR ==="
