#!/usr/bin/env bash
# Dump pre-resolver state from ODD geant4_10muon_1GeV events.
# Produces one JSON dump per event (event_000.json ... event_N-1.json) by
# running traccc_seq_example with --skip=i --input-events=1 per event.
# Run on a CPU interactive node (e.g. stbc-i2). No GPU required.
#
# Usage:
#   ./dump_odd_events.sh [--n-events N] [--skip-start S] [--outdir DIR]
#
# Defaults:
#   --n-events 20
#   --skip-start 0
#   --outdir $THESIS_REPO/data/odd_muon_dumps/<timestamp>
#
# Environment overrides:
#   TRACCC_SRC   path to traccc source/build root (default /data/alice/sbetisor/traccc)
#   THESIS_REPO  path to thesis repo (auto-detected from script location)
set -euo pipefail

. /data/alice/sbetisor/spack/share/spack/setup-env.sh
spack env activate traccc

TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

N_EVENTS=20
SKIP_START=0
OUTDIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-events) N_EVENTS="$2"; shift 2 ;;
        --skip-start) SKIP_START="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_REPO/data/odd_muon_dumps/${TIMESTAMP}}"
mkdir -p "$OUTDIR"

SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"
DATA_DIR="$TRACCC_SRC/data"

if [[ ! -x "$SEQ_BIN" ]]; then
    echo "traccc_seq_example not found at $SEQ_BIN"
    exit 1
fi

# Verify input data exists
MUON_DIR="$DATA_DIR/odd/geant4_10muon_1GeV"
if [[ ! -d "$MUON_DIR" ]]; then
    echo "ODD muon data not found at $MUON_DIR"
    echo "Run data/traccc_data_get_files.sh from traccc root to fetch it."
    exit 1
fi

echo "=== Dumping ODD muon ambiguity inputs ==="
echo "Dataset:  $MUON_DIR"
echo "Events:   ${SKIP_START}..$(( SKIP_START + N_EVENTS - 1 ))"
echo "Output:   $OUTDIR"
echo ""

SUMMARY_FILE="$OUTDIR/dump_summary.txt"
echo "event_idx n_candidates" > "$SUMMARY_FILE"

export TRACCC_TEST_DATA_DIR="$DATA_DIR"

for (( i=0; i<N_EVENTS; i++ )); do
    EVENT_IDX=$(( SKIP_START + i ))
    DUMP_PATH="$OUTDIR/event_$(printf '%03d' $EVENT_IDX).json"
    LOG_PATH="$OUTDIR/event_$(printf '%03d' $EVENT_IDX)_seq.log"

    printf "Event %3d -> %s\n" "$EVENT_IDX" "$(basename $DUMP_PATH)"

    # Note: --use-acts-geom-source is intentionally omitted; it causes a
    # segfault in Acts::from_json for this geometry + Spack Acts combination.
    "$SEQ_BIN" \
        --detector-file=geometries/odd/odd-detray_geometry_detray.json \
        --material-file=geometries/odd/odd-detray_material_detray.json \
        --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
        --digitization-file=geometries/odd/odd-digi-geometric-config.json \
        --input-directory=odd/geant4_10muon_1GeV/ \
        --skip="$EVENT_IDX" \
        --input-events=1 \
        --dump-ambiguity-input="$DUMP_PATH" \
        2>&1 | tee "$LOG_PATH" > /dev/null

    # Extract n_candidates from the dump JSON (top-level "n_tracks" or array length)
    N_CANDS="unknown"
    if command -v python3 &>/dev/null; then
        N_CANDS=$(python3 -c "
import json, sys
with open('$DUMP_PATH') as f:
    d = json.load(f)
# dump format: list of tracks or dict with 'tracks' key
if isinstance(d, list):
    print(len(d))
elif isinstance(d, dict) and 'tracks' in d:
    print(len(d['tracks']))
else:
    print('?')
" 2>/dev/null || echo "?")
    fi

    echo "$EVENT_IDX $N_CANDS" >> "$SUMMARY_FILE"
    printf "  -> n_candidates: %s\n" "$N_CANDS"
done

echo ""
echo "=== Dump complete ==="
echo "Summary: $SUMMARY_FILE"
echo "$(wc -l < "$SUMMARY_FILE") events recorded (including header)"
