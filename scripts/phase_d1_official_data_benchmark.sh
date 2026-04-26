#!/usr/bin/env bash
# Phase D1: Fetch official traccc-data-v10 and run all four resolver backends
# on the ODD geant4_10muon_10GeV corpus.
#
# Requires: GPU node (wn-lot-001), CUDA build of traccc on the
# thesis-novelty-conflict-graph branch.
#
# Usage (interactive GPU node):
#   ./scripts/phase_d1_official_data_benchmark.sh
#
# Environment overrides:
#   TRACCC_SRC    path to traccc checkout (default /data/alice/sbetisor/traccc)
#   THESIS_REPO   path to thesis-work repo (auto-detected)
#   OUTDIR              output directory (default under THESIS_RESULTS_ROOT)
#   THESIS_RESULTS_ROOT output root (default $HOME/data-work/results)
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

# traccc_seq_example has a stale RPATH that resolves libstdc++ to the
# spack view's older lib (3.4.30); force-load the lib64/3.4.32 variant.
SPACK_VIEW=/data/alice/sbetisor/spack/var/spack/environments/traccc/.spack-env/._view/3une3gz6c3dhaus2ggcji3vznrc4qrll
if [[ -f "$SPACK_VIEW/lib64/libstdc++.so.6.0.32" ]]; then
  export LD_PRELOAD="$SPACK_VIEW/lib64/libstdc++.so.6.0.32"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.5}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_phase_d1}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$RUN_ID}"
mkdir -p "$OUTDIR"

GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"

for b in "$GPU_BENCH" "$SEQ_BIN"; do
  if [[ ! -x "$b" ]]; then
    echo "ERROR: binary not found: $b"
    echo "Rebuild on a GPU node with TRACCC_BUILD_CUDA=ON"
    exit 1
  fi
done

# Step 1: fetch official data if not present.
ODD_10GEV="$TRACCC_SRC/data/odd/geant4_10muon_10GeV"
if [[ ! -d "$ODD_10GEV" ]]; then
  echo "=== Fetching traccc-data-v10 ==="
  (cd "$TRACCC_SRC" && bash data/traccc_data_get_files.sh)
fi

if [[ ! -d "$ODD_10GEV" ]]; then
  echo "ERROR: ODD 10muon 10GeV data still not found after fetch."
  exit 1
fi

N_EVENTS=$(ls "$ODD_10GEV"/event*.csv 2>/dev/null | wc -l || echo 0)
if [[ "$N_EVENTS" -eq 0 ]]; then
  # Count via directory structure
  N_EVENTS=$(ls "$ODD_10GEV"/ | grep -c '^event' || echo 10)
fi
echo "ODD 10muon 10GeV: $N_EVENTS events in $ODD_10GEV"

# Step 2: dump all events as ambiguity-resolver input JSON.
DUMP_DIR="$OUTDIR/odd_10gev_dumps"
mkdir -p "$DUMP_DIR"
SUMMARY="$DUMP_DIR/dump_summary.txt"
echo "event_idx n_candidates" > "$SUMMARY"

echo "=== Dumping ODD 10muon 10GeV events ==="
for (( i=0; i<N_EVENTS; i++ )); do
  DUMP_PATH="$DUMP_DIR/event_$(printf '%03d' $i).json"
  MAX_TRIES=5; RC=1
  for (( attempt=1; attempt<=MAX_TRIES; attempt++ )); do
    if (
      export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
      "$SEQ_BIN" \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --material-file=geometries/odd/odd-detray_material_detray.json \
        --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=odd/geant4_10muon_10GeV/ \
        --input-skip="$i" \
        --input-events=1 \
        --dump-ambiguity-input="$DUMP_PATH" \
        > "$DUMP_DIR/event_$(printf '%03d' $i)_seq.log" 2>&1
    ); then RC=0; break
    else
      RC=$?; printf "  [event %d attempt %d/%d failed, retrying]\n" "$i" "$attempt" "$MAX_TRIES"
      rm -f "$DUMP_PATH"
    fi
  done
  [[ $RC -ne 0 ]] && { echo "ERROR: event $i failed"; exit 1; }

  N_CANDS=$(python3 -c "
import json
with open('$DUMP_PATH') as f: d=json.load(f)
print(len(d['tracks']) if isinstance(d, dict) else len(d))
" 2>/dev/null || echo "?")
  echo "$i $N_CANDS" >> "$SUMMARY"
  printf "  event %3d  n_candidates=%s\n" "$i" "$N_CANDS"
done

# Step 3: run all four backends on every dump.
SWEEP_OUT="$OUTDIR/odd_10gev_sweep.txt"
echo "=== Running four-backend sweep on ODD 10muon 10GeV ==="
echo "# Phase D1 sweep: ODD geant4_10muon_10GeV" > "$SWEEP_OUT"
echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)" >> "$SWEEP_OUT"
echo "# date: $(date)" >> "$SWEEP_OUT"

for dump in "$DUMP_DIR"/event_*.json; do
  ev=$(basename "$dump" .json)
  echo "--- $ev ---"
  echo "" >> "$SWEEP_OUT"
  echo "=== $ev ===" >> "$SWEEP_OUT"
  "$GPU_BENCH" \
    --input-dump="$dump" \
    --repeats=10 --warmup=3 \
    --parallel-batch --parallel-batch-window=8192 \
    --conflict-graph=both \
    --determinism-runs=3 \
    2>&1 | tee -a "$SWEEP_OUT"
  echo ""
done

echo ""
echo "=== Phase D1 complete ==="
echo "Dumps:  $DUMP_DIR"
echo "Sweep:  $SWEEP_OUT"
