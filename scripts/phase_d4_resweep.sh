#!/usr/bin/env bash
# Phase D4: Run the full four-backend sweep (baseline + PBG + MIS + JP) on a
# directory of pre-dumped ambiguity-resolver JSON inputs. Includes
# determinism validation (--determinism-runs=5) and produces per-event JSON
# quality reports suitable for the validity-contract check.
#
# Usage:
#   ./scripts/phase_d4_resweep.sh --dump-dir=<path> [--outdir=<path>]
#
# The script can be run on any directory of event_*.json dump files,
# including those from Phase D1 (ODD 10GeV), D2 (Fatras expanded), and
# D3 (telescope).
#
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

DUMP_DIR=""
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dump-dir=*) DUMP_DIR="${1#*=}"; shift ;;
    --outdir=*)   OUTDIR="${1#*=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$DUMP_DIR" ]]; then
  echo "Usage: phase_d4_resweep.sh --dump-dir=<path> [--outdir=<path>]"
  exit 1
fi

OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${TIMESTAMP}_phase_d4_sweep}"
mkdir -p "$OUTDIR"

GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
if [[ ! -x "$GPU_BENCH" ]]; then
  echo "ERROR: traccc_benchmark_resolver_cuda not found at $GPU_BENCH"
  echo "Run on a GPU node with TRACCC_BUILD_CUDA=ON build."
  exit 1
fi

SWEEP_OUT="$OUTDIR/sweep.txt"
PER_EVENT_DIR="$OUTDIR/per_event"
mkdir -p "$PER_EVENT_DIR"

echo "=== Phase D4: Full four-backend sweep ==="
echo "Dump dir: $DUMP_DIR"
echo "Output:   $OUTDIR"
echo ""

echo "# Phase D4 sweep" > "$SWEEP_OUT"
echo "# dump_dir: $DUMP_DIR" >> "$SWEEP_OUT"
echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)" >> "$SWEEP_OUT"
echo "# date: $(date)" >> "$SWEEP_OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
  | head -1 | { echo -n "# gpu: "; cat; echo; } >> "$SWEEP_OUT" || true

DUMP_COUNT=0
for dump in "$DUMP_DIR"/event_*.json; do
  [[ -f "$dump" ]] || continue
  ev=$(basename "$dump" .json)
  DUMP_COUNT=$(( DUMP_COUNT + 1 ))

  echo "--- $ev ---"
  PER_EVENT_FILE="$PER_EVENT_DIR/${ev}.txt"

  "$GPU_BENCH" \
    --input-dump="$dump" \
    --repeats=10 --warmup=3 \
    --parallel-batch --parallel-batch-window=8192 \
    --conflict-graph=both \
    --determinism-runs=5 \
    2>&1 | tee "$PER_EVENT_FILE"

  # Append to the combined sweep file with event label.
  { echo ""; echo "=== $ev ==="; cat "$PER_EVENT_FILE"; } >> "$SWEEP_OUT"
  echo ""
done

echo "=== Phase D4 summary ==="
echo "Events processed: $DUMP_COUNT"
echo "Sweep file:       $SWEEP_OUT"
echo "Per-event files:  $PER_EVENT_DIR/"

# Quick quality summary extracted from the sweep.
python3 - << 'PYEOF' "$SWEEP_OUT"
import sys, re, collections

fname = sys.argv[1]
events = []
cur = {}

with open(fname) as f:
    for line in f:
        line = line.strip()
        if line.startswith("=== event"):
            if cur:
                events.append(cur)
            cur = {"event": line.strip("= ")}
        for key in ["baseline_time_ms_mean", "pbg_time_ms_mean",
                    "graph_mis_time_ms_mean", "graph_jp_time_ms_mean",
                    "graph_jp_hash_match", "graph_mis_hash_match",
                    "graph_jp_track_overlap_vs_cpu",
                    "determinism_all_pass"]:
            m = re.search(rf"{key}=(\S+)", line)
            if m:
                cur[key] = m.group(1)
if cur:
    events.append(cur)

if not events:
    print("No events found in sweep file.")
    sys.exit(0)

def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

cols = ["baseline_time_ms_mean", "pbg_time_ms_mean",
        "graph_mis_time_ms_mean", "graph_jp_time_ms_mean"]
print(f"\n{'Event':<30} {'baseline':>10} {'pbg':>10} {'mis':>10} {'jp':>10} {'jp_hash':>8} {'det_pass':>9}")
print("-" * 100)
for e in events:
    row = [e.get("event","?")[:30]]
    for c in cols:
        v = safe_float(e.get(c))
        row.append(f"{v:10.2f}" if v is not None else f"{'?':>10}")
    row.append(f"{e.get('graph_jp_hash_match','?'):>8}")
    row.append(f"{e.get('determinism_all_pass','?'):>9}")
    print("".join(row))

# Speedup summary.
jp_speedups = []
for e in events:
    b = safe_float(e.get("baseline_time_ms_mean"))
    j = safe_float(e.get("graph_jp_time_ms_mean"))
    if b and j and j > 0:
        jp_speedups.append(b / j)
if jp_speedups:
    print(f"\nJP speedup vs baseline: min={min(jp_speedups):.2f}x mean={sum(jp_speedups)/len(jp_speedups):.2f}x max={max(jp_speedups):.2f}x")

n_jp_match = sum(1 for e in events if e.get("graph_jp_hash_match") == "true")
print(f"JP selection-identical to CPU: {n_jp_match}/{len(events)}")
det_pass = sum(1 for e in events if e.get("determinism_all_pass") == "true")
print(f"Determinism pass: {det_pass}/{len(events)}")
PYEOF

echo ""
echo "=== Phase D4 complete ==="
