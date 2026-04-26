#!/usr/bin/env bash
# Generate ODD Fatras ttbar datasets at configurable pileup levels.
#
# Example:
#   ./generate_fatras_high_pileup.sh --pileups "400 500 600" --events 10
#
# Output:
#   /data/alice/sbetisor/traccc/data/odd/fatras_ttbar_mu{PU}/
set -euo pipefail

LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt}"
ACTS_SRC="${ACTS_SRC:-/data/alice/sbetisor/acts}"
INSTALL="${INSTALL:-/data/alice/sbetisor/thesis/acts-python-install}"
ODD_BUILD="${ODD_BUILD:-/data/alice/sbetisor/thesis/acts-python-build/_deps/odd-build}"
ODD_SRC="${ODD_SRC:-/data/alice/sbetisor/thesis/acts-python-build/_deps/odd-src}"
DATAROOT="${DATAROOT:-/data/alice/sbetisor/traccc/data/odd}"

PILEUP_LEVELS="${PILEUP_LEVELS:-400 500 600}"
N_EVENTS="${N_EVENTS:-10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pileups) PILEUP_LEVELS="$2"; shift 2 ;;
        --events) N_EVENTS="$2"; shift 2 ;;
        --force) SKIP_EXISTING=0; shift ;;
        --out-root) DATAROOT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)_fatras_high_pileup"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
LOG_DIR="$THESIS_RESULTS_ROOT/$RUN_ID"
MASTER_LOG="$LOG_DIR/generation.log"
mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

set +u
source "$LCG_VIEW/setup.sh"
set -u

export PYTHONPATH="$INSTALL:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$INSTALL/lib64:$ODD_BUILD/factory:${LD_LIBRARY_PATH:-}"
export ODD_PATH="$ODD_SRC"
export LD_PRELOAD="$INSTALL/lib64/libActsCore.so:$INSTALL/lib64/libActsPluginDD4hep.so:$INSTALL/lib64/libActsPluginTGeo.so:$INSTALL/lib64/libActsPluginRoot.so"

log "=== High-pileup Fatras ttbar generation ==="
log "Pileups: $PILEUP_LEVELS"
log "Events:  $N_EVENTS"
log "Output:  $DATAROOT"
log "Log dir: $LOG_DIR"
log ""

python3 -c "
import acts, acts.examples, acts.examples.dd4hep
from acts.examples.odd import getOpenDataDetector
getOpenDataDetector()
print('ACTS import OK:', acts.__version__)
" 2>&1 | tee -a "$MASTER_LOG"

TOTAL_START="$(date +%s)"

for PU in $PILEUP_LEVELS; do
    OUTDIR="$DATAROOT/fatras_ttbar_mu${PU}"
    GENLOG="$OUTDIR/generation.log"

    if [[ "$SKIP_EXISTING" == "1" && -f "$GENLOG" ]]; then
        DONE_EVENTS="$(find "$OUTDIR" -maxdepth 1 -name 'event*-measurements.csv' | wc -l)"
        if [[ "$DONE_EVENTS" -ge "$N_EVENTS" ]]; then
            log "SKIP mu=$PU: already has $DONE_EVENTS events"
            continue
        fi
    fi

    mkdir -p "$OUTDIR"
    PU_START="$(date +%s)"
    log "--- mu=$PU -> $OUTDIR ---"

    python3 "$ACTS_SRC/Examples/Scripts/Python/full_chain_odd.py" \
        --ttbar \
        --ttbar-pu "$PU" \
        --events "$N_EVENTS" \
        --output "$OUTDIR" \
        --skip 0 \
        --output-csv \
        2>&1 | tee "$GENLOG"

    RC="${PIPESTATUS[0]}"
    PU_END="$(date +%s)"
    DONE_EVENTS="$(find "$OUTDIR" -maxdepth 1 -name 'event*-measurements.csv' | wc -l)"

    if [[ "$RC" -eq 0 ]]; then
        log "DONE mu=$PU: $DONE_EVENTS events in $((PU_END - PU_START))s"
    else
        log "FAIL mu=$PU: rc=$RC after $((PU_END - PU_START))s"
        exit "$RC"
    fi
done

TOTAL_END="$(date +%s)"
log ""
log "Finished in $((TOTAL_END - TOTAL_START))s"
