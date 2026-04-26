#!/usr/bin/env bash
# Phase E1: nsight-compute (ncu) kernel-level profiling for the conflict-graph
# kernels: build_conflict_coo, graph_mis_propose, graph_mis_finalize,
# apply_graph_removals.
#
# Reports: SM occupancy, memory throughput, L2 hit rate, warp efficiency.
# Run on wn-lot-001 (Quadro GV100) with CUDA 12.x.
#
# Usage:
#   ./scripts/phase_e1_ncu_profiling.sh --input-dump=<path>
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
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${TIMESTAMP}_phase_e1_ncu}"
mkdir -p "$OUTDIR"

INPUT_DUMP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dump=*) INPUT_DUMP="${1#*=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$INPUT_DUMP" ]]; then
  # Use any available Fatras mu=600 dump as a representative high-n input.
  INPUT_DUMP=$(find "$THESIS_RESULTS_ROOT" -path "*/dumps_mu600/event_*.json" \
    | sort | tail -1)
  if [[ -z "$INPUT_DUMP" ]]; then
    echo "ERROR: no input dump found. Pass --input-dump=<path> or run Phase D first."
    exit 1
  fi
  echo "Auto-selected input: $INPUT_DUMP"
fi

GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
NCU=$(command -v ncu 2>/dev/null || echo "")

if [[ -z "$NCU" ]]; then
  echo "ERROR: ncu (nsight-compute) not found. Install CUDA Toolkit >= 11."
  exit 1
fi

echo "=== Phase E1: ncu kernel profiling ==="
echo "Input:  $INPUT_DUMP"
echo "Output: $OUTDIR"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

KERNELS=(
  "traccc::cuda::kernels::build_conflict_coo"
  "traccc::cuda::kernels::graph_mis_propose"
  "traccc::cuda::kernels::graph_mis_finalize"
  "traccc::cuda::kernels::apply_graph_removals"
)

for kernel in "${KERNELS[@]}"; do
  safe_name="${kernel//::/_}"
  NCU_OUT="$OUTDIR/${safe_name}.ncu-rep"
  TXT_OUT="$OUTDIR/${safe_name}.txt"

  echo "Profiling: $kernel"
  "$NCU" \
    --set full \
    --kernel-name "${kernel}" \
    --launch-count 1 \
    --export "$NCU_OUT" \
    "$GPU_BENCH" \
      --input-dump="$INPUT_DUMP" \
      --repeats=1 --warmup=0 \
      --conflict-graph=both \
      2>/dev/null | tail -20 | tee "$TXT_OUT" || true

  # Also export a text summary from the .ncu-rep.
  if [[ -f "${NCU_OUT}.ncu-rep" ]]; then
    "$NCU" --import "${NCU_OUT}.ncu-rep" \
      --print-summary per-gpu 2>/dev/null \
      >> "$TXT_OUT" || true
  fi
  echo ""
done

# Phase E2 bonus: edge-count histogram from the existing sweep output.
# Look for graph_size log files.
echo "=== Bonus: edge-count histogram from Phase D sweep (if available) ==="
SWEEP_FILE=$(find "$THESIS_RESULTS_ROOT" -name "*.jp.csv" | sort | tail -1)
if [[ -n "$SWEEP_FILE" ]]; then
  python3 - "$SWEEP_FILE" << 'PYEOF'
import sys, csv, collections

fname = sys.argv[1]
edges = []
with open(fname) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            edges.append(int(row.get("n_edges", 0)))
        except ValueError:
            pass

if not edges:
    print("No edge data found.")
    sys.exit(0)

buckets = [0, 100, 500, 1000, 5000, 10000, 50000, 100000, float("inf")]
hist = collections.defaultdict(int)
for e in edges:
    for i in range(len(buckets) - 1):
        if buckets[i] <= e < buckets[i + 1]:
            hist[f"{buckets[i]}-{buckets[i+1]}"] += 1
            break

print(f"Edge-count distribution across {len(edges)} outer iterations:")
for k in sorted(hist):
    print(f"  {k:>20}: {hist[k]}")
print(f"  mean={sum(edges)/len(edges):.0f}  max={max(edges)}")
PYEOF
else
  echo "(No .jp.csv graph-size log found from Phase D sweep.)"
fi

echo ""
echo "=== Phase E1 complete: $OUTDIR ==="
