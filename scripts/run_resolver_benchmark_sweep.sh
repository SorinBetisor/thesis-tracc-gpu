#!/usr/bin/env bash
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
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTDIR="${OUTDIR:-$THESIS_REPO/results/$RUN_ID}"
mkdir -p "$OUTDIR"

# Set PROFILE=1 to append per-phase timing to each result file.
PROFILE="${PROFILE:-0}"
PROFILE_FLAG=""
if [[ "$PROFILE" == "1" ]]; then
  PROFILE_FLAG="--profile"
fi

echo "=== Resolver benchmark sweep (3x3: n_candidates x conflict_density) ==="
echo "Output:  $OUTDIR"
echo "Profile: ${PROFILE_FLAG:-off}"
echo ""

for n in 1000 5000 10000; do
  for density in low med high; do
    outfile="$OUTDIR/n${n}_${density}.txt"
    echo "Running n_candidates=$n conflict_density=$density -> $outfile"
    "$TRACCC_BIN" --synthetic --n-candidates="$n" \
      --conflict-density="$density" --repeats=10 --warmup=3 \
      ${PROFILE_FLAG} \
      2>&1 | tee "$outfile"
    echo ""
  done
done

echo "=== Sweep complete ==="
