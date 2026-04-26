#!/usr/bin/env bash
# Phase D2: Extend Fatras dump corpus to mu=200 and >=10 events per pile-up
# point for mu in {200, 300, 400, 500, 600}.
#
# Prerequisites:
#   - Fatras CSV files already generated under
#     /data/alice/sbetisor/traccc/data/odd/fatras_ttbar_mu{PU}/
#     for each PU in the MU_LIST below.
#   - CPU build of traccc_seq_example.
#
# For each pile-up point the script:
#   1. Dumps N_PER_MU events as ambiguity-resolver JSON.
#   2. Writes a dump_summary.txt with n_candidates per event.
#
# Usage:
#   ./scripts/phase_d2_fatras_expand.sh
# Environment:
#   TRACCC_SRC, THESIS_REPO, N_PER_MU, MU_LIST
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
N_PER_MU="${N_PER_MU:-10}"
MU_LIST="${MU_LIST:-200 300 400 500 600}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${TIMESTAMP}_phase_d2_fatras}"
mkdir -p "$OUTDIR"

SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"
if [[ ! -x "$SEQ_BIN" ]]; then
  echo "ERROR: traccc_seq_example not found at $SEQ_BIN"
  exit 1
fi

echo "=== Phase D2: Fatras ttbar dump expansion ==="
echo "N per mu-point: $N_PER_MU"
echo "Pileup list:    $MU_LIST"
echo "Output root:    $OUTDIR"
echo ""

for MU in $MU_LIST; do
  FATRAS_CSV="$TRACCC_SRC/data/odd/fatras_ttbar_mu${MU}"
  if [[ ! -d "$FATRAS_CSV" ]]; then
    echo "WARNING: Fatras CSV for mu=$MU not found at $FATRAS_CSV — skipping."
    continue
  fi

  # Count available events in the CSV directory.
  N_AVAIL=$(find "$FATRAS_CSV" -name "event*.json" -o -name "particles*.csv" 2>/dev/null |
            grep -c 'particles' || true)
  # Fall back: count any event-indexed directory or file.
  if [[ "$N_AVAIL" -eq 0 ]]; then
    N_AVAIL=$(ls "$FATRAS_CSV" 2>/dev/null | grep -cE '^event[0-9]' || echo 20)
  fi
  DUMP_COUNT=$(( N_AVAIL < N_PER_MU ? N_AVAIL : N_PER_MU ))
  echo "mu=$MU: found ~$N_AVAIL events, will dump $DUMP_COUNT"

  DUMP_DIR="$OUTDIR/dumps_mu${MU}"
  mkdir -p "$DUMP_DIR"
  SUMMARY="$DUMP_DIR/dump_summary.txt"
  echo "event_idx n_candidates" > "$SUMMARY"

  for (( i=0; i<DUMP_COUNT; i++ )); do
    DUMP_PATH="$DUMP_DIR/event_$(printf '%09d' $i).json"
    MAX_TRIES=5; RC=1
    for (( attempt=1; attempt<=MAX_TRIES; attempt++ )); do
      if (
        export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
        "$SEQ_BIN" \
          --detector-file=geometries/odd/odd-detray_geometry_detray.json \
          --material-file=geometries/odd/odd-detray_material_detray.json \
          --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
          --digitization-file=geometries/odd/odd-digi-geometric-config.json \
          --input-directory="odd/fatras_ttbar_mu${MU}/" \
          --input-skip="$i" \
          --input-events=1 \
          --dump-ambiguity-input="$DUMP_PATH" \
          > "$DUMP_DIR/event_$(printf '%09d' $i)_seq.log" 2>&1
      ); then RC=0; break
      else
        RC=$?
        printf "  [mu=%s event %d attempt %d/%d failed, retrying]\n" \
               "$MU" "$i" "$attempt" "$MAX_TRIES"
        rm -f "$DUMP_PATH"
      fi
    done
    [[ $RC -ne 0 ]] && { echo "ERROR: mu=$MU event $i failed"; exit 1; }

    N_CANDS=$(python3 -c "
import json
with open('$DUMP_PATH') as f: d=json.load(f)
print(len(d['tracks']) if isinstance(d, dict) else len(d))
" 2>/dev/null || echo "?")
    echo "$i $N_CANDS" >> "$SUMMARY"
    printf "  mu=%s event %3d  n_candidates=%s\n" "$MU" "$i" "$N_CANDS"
  done

  echo "  mu=$MU dump complete: $DUMP_DIR"
  echo ""
done

echo "=== Phase D2 complete: $OUTDIR ==="
