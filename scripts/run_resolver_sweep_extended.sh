#!/usr/bin/env bash
# Extended CPU resolver benchmark sweep — 10 n_candidates points × 3 densities = 30 configs.
# Covers n=100 (real muon event scale) through n=50000 to produce the full crossover curve.
# Outputs profile_*_ms fields when PROFILE=1.
set -euo pipefail

TRACCC_BIN="${TRACCC_BIN:-}"
if [[ -z "$TRACCC_BIN" ]]; then
    TRACCC_SRC="${TRACCC_SRC:-/data/alice/sbetisor/traccc}"
    TRACCC_BIN="$TRACCC_SRC/build/bin/traccc_benchmark_resolver"
fi
if [[ ! -x "$TRACCC_BIN" ]]; then
    echo "traccc_benchmark_resolver not found at $TRACCC_BIN"
    echo "Build traccc and set TRACCC_BIN or TRACCC_SRC"
    exit 1
fi

THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
THESIS_RESULTS_ROOT="${THESIS_RESULTS_ROOT:-$HOME/data-work/results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_extended}"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/$RUN_ID}"
mkdir -p "$OUTDIR"

PROFILE="${PROFILE:-1}"
PROFILE_FLAG=""
[[ "$PROFILE" == "1" ]] && PROFILE_FLAG="--profile"

echo "=== Extended CPU resolver sweep (10×3: n_candidates × conflict_density) ==="
echo "Binary:  $TRACCC_BIN"
echo "Output:  $OUTDIR"
echo "Profile: ${PROFILE_FLAG:-off}"
echo ""

# n=100 is near the real physics scale (geant4_10muon_1GeV: ~87 candidates/event)
# n=50000 captures the high-pileup regime (comparable to ttbar_mu200)
for n in 100 500 1000 2000 3000 5000 7500 10000 20000 50000; do
    for density in low med high; do
        outfile="$OUTDIR/n${n}_${density}.txt"
        echo "n_candidates=$n conflict_density=$density -> $outfile"
        "$TRACCC_BIN" --synthetic --n-candidates="$n" \
            --conflict-density="$density" --repeats=10 --warmup=3 \
            ${PROFILE_FLAG} \
            2>&1 | tee "$outfile"
        echo ""
    done
done

echo "=== Extended sweep complete ==="
echo "Results: $OUTDIR"
