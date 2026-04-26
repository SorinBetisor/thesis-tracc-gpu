#!/usr/bin/env bash
# Phase D5: Validate the conflict-graph resolver on the *full* on-disk ODD
# corpus shipped with traccc-data-v10. Three orthogonal axes:
#
#   axis 1: geant4 1muon  GeV ladder { 1, 5, 10, 50, 100 }
#           -- low-density, single-particle events. Sanity ceiling for
#              "no-conflict regime" behavior; CPU greedy and GPU graph
#              must agree exactly here.
#   axis 2: geant4 10muon GeV ladder { 1, 5, 10, 50, 100 }
#           -- multi-muon momentum ladder; 10 GeV already covered by D1.
#   axis 3: fatras ttbar pileup ladder { 0, 20, 50, 100, 140 }
#           -- low/medium pileup; fills the gap below D2 (200..600).
#
# Each (sample, n_events) cell:
#   1. dump ambiguity inputs via traccc_seq_example (ODD geometry)
#   2. run all four backends (baseline, PBG, MIS, JP) with --determinism-runs=5
#   3. emit a per-cell sweep TXT + per-event TXT files
#
# Designed to be idempotent: existing dumps and sweeps are skipped unless
# FORCE=1 is set. Suitable for both interactive GPU node and HTCondor.
#
# Usage (interactive GPU node):
#   ./scripts/phase_d5_odd_corpus.sh
#
# Subset selection (run only one axis or one cell):
#   AXES="muon10 ttbar" ./scripts/phase_d5_odd_corpus.sh
#   CELLS="geant4_10muon_50GeV fatras_ttbar_mu100" ./scripts/phase_d5_odd_corpus.sh
#
# Environment overrides:
#   TRACCC_SRC      path to traccc checkout
#   THESIS_REPO     path to this thesis-work repo
#   OUTDIR          parent output dir (default results/<ts>_phase_d5_odd_corpus)
#   N_EVENTS        events per cell (default 10)
#   FORCE           1 = re-dump and re-sweep even if outputs exist
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
N_EVENTS="${N_EVENTS:-10}"
FORCE="${FORCE:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-$THESIS_RESULTS_ROOT/${TIMESTAMP}_phase_d5_odd_corpus}"
mkdir -p "$OUTDIR"

GPU_BENCH="$TRACCC_SRC/build/bin/traccc_benchmark_resolver_cuda"
SEQ_BIN="$TRACCC_SRC/build/bin/traccc_seq_example"

for b in "$GPU_BENCH" "$SEQ_BIN"; do
  if [[ ! -x "$b" ]]; then
    echo "ERROR: binary not found: $b"
    echo "Rebuild traccc on a GPU node with TRACCC_BUILD_CUDA=ON"
    exit 1
  fi
done

# Cell catalogue: SAMPLE_DIR  AXIS  LABEL.
# All cells use the canonical ODD geometry/digi configs.
declare -a CELLS_ALL=(
  # axis 1 — single-muon GeV ladder
  "odd/geant4_1muon_1GeV     muon1   geant4_1muon_1GeV"
  "odd/geant4_1muon_5GeV     muon1   geant4_1muon_5GeV"
  "odd/geant4_1muon_10GeV    muon1   geant4_1muon_10GeV"
  "odd/geant4_1muon_50GeV    muon1   geant4_1muon_50GeV"
  "odd/geant4_1muon_100GeV   muon1   geant4_1muon_100GeV"
  # axis 2 — multi-muon GeV ladder
  "odd/geant4_10muon_1GeV    muon10  geant4_10muon_1GeV"
  "odd/geant4_10muon_5GeV    muon10  geant4_10muon_5GeV"
  "odd/geant4_10muon_50GeV   muon10  geant4_10muon_50GeV"
  "odd/geant4_10muon_100GeV  muon10  geant4_10muon_100GeV"
  # axis 3 — low/medium ttbar pileup ladder
  "odd/fatras_ttbar_mu0      ttbar   fatras_ttbar_mu0"
  "odd/fatras_ttbar_mu20     ttbar   fatras_ttbar_mu20"
  "odd/fatras_ttbar_mu50     ttbar   fatras_ttbar_mu50"
  "odd/fatras_ttbar_mu100    ttbar   fatras_ttbar_mu100"
  "odd/fatras_ttbar_mu140    ttbar   fatras_ttbar_mu140"
)

# Filter by AXES / CELLS env overrides.
AXES_FILTER="${AXES:-}"
CELLS_FILTER="${CELLS:-}"

declare -a CELLS=()
for line in "${CELLS_ALL[@]}"; do
  read -r sample axis label <<<"$line"
  if [[ -n "$AXES_FILTER"  && " $AXES_FILTER "  != *" $axis "* ]]; then continue; fi
  if [[ -n "$CELLS_FILTER" && " $CELLS_FILTER " != *" $label "* ]]; then continue; fi
  CELLS+=("$line")
done

if [[ ${#CELLS[@]} -eq 0 ]]; then
  echo "No cells selected (AXES='$AXES_FILTER' CELLS='$CELLS_FILTER')"
  exit 1
fi

# Fetch official data if any sample dir is missing.
NEED_FETCH=0
for line in "${CELLS[@]}"; do
  read -r sample axis label <<<"$line"
  if [[ ! -d "$TRACCC_SRC/data/$sample" ]]; then NEED_FETCH=1; fi
done
if [[ "$NEED_FETCH" -eq 1 ]]; then
  echo "=== Fetching traccc-data-v10 ==="
  (cd "$TRACCC_SRC" && bash data/traccc_data_get_files.sh)
fi

CORPUS_OUT="$OUTDIR/corpus_summary.tsv"
echo -e "axis\tlabel\tn_events\tdump_dir\tsweep_file" > "$CORPUS_OUT"

dump_one_cell() {
  local sample="$1" label="$2"
  local cell_dir="$OUTDIR/$label"
  local dump_dir="$cell_dir/dumps"
  local log_dir="$cell_dir/logs"
  mkdir -p "$dump_dir" "$log_dir"

  for (( i=0; i<N_EVENTS; i++ )); do
    local dump_path="$dump_dir/event_$(printf '%03d' $i).json"
    if [[ -s "$dump_path" && "$FORCE" -ne 1 ]]; then
      printf "  [skip] event %d already dumped\n" "$i"
      continue
    fi
    local rc=1
    for attempt in 1 2 3; do
      if (
        export TRACCC_TEST_DATA_DIR="$TRACCC_SRC/data"
        "$SEQ_BIN" \
          --detector-file=geometries/odd/odd-detray_geometry_detray.json \
          --material-file=geometries/odd/odd-detray_material_detray.json \
          --grid-file=geometries/odd/odd-detray_surface_grids_detray.json \
          --digitization-file=geometries/odd/odd-digi-geometric-config.json \
          --input-directory="$sample/" \
          --input-skip="$i" \
          --input-events=1 \
          --dump-ambiguity-input="$dump_path" \
          > "$log_dir/event_$(printf '%03d' $i)_seq.log" 2>&1
      ); then rc=0; break
      else rc=$?; printf "  [event %d attempt %d failed]\n" "$i" "$attempt"; rm -f "$dump_path"
      fi
    done
    if [[ $rc -ne 0 ]]; then
      echo "  WARNING: event $i never dumped — moving on"
      continue
    fi
    local n_cands
    n_cands=$(python3 -c "
import json,sys
try:
    d=json.load(open('$dump_path'))
    print(len(d['tracks']) if isinstance(d, dict) else len(d))
except Exception as e:
    print('?')
" 2>/dev/null || echo "?")
    printf "  event %3d  n_candidates=%s\n" "$i" "$n_cands"
  done
  echo "$dump_dir"
}

sweep_one_cell() {
  local label="$1" dump_dir="$2"
  local cell_dir="$OUTDIR/$label"
  local sweep="$cell_dir/sweep.txt"
  local per_event_dir="$cell_dir/per_event"
  mkdir -p "$per_event_dir"

  if [[ -s "$sweep" && "$FORCE" -ne 1 ]]; then
    echo "  [skip] sweep already exists at $sweep"
    echo "$sweep"; return
  fi

  {
    echo "# Phase D5 sweep: $label"
    echo "# branch: $(cd $TRACCC_SRC && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "# date: $(date)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
      | head -1 | { echo -n "# gpu: "; cat; echo; } || true
  } > "$sweep"

  for dump in "$dump_dir"/event_*.json; do
    [[ -f "$dump" ]] || continue
    local ev; ev=$(basename "$dump" .json)
    local per_event_file="$per_event_dir/${ev}.txt"
    "$GPU_BENCH" \
      --input-dump="$dump" \
      --repeats=10 --warmup=3 \
      --parallel-batch --parallel-batch-window=8192 \
      --conflict-graph=both \
      --determinism-runs=5 \
      2>&1 | tee "$per_event_file"
    { echo ""; echo "=== $ev ==="; cat "$per_event_file"; } >> "$sweep"
  done
  echo "$sweep"
}

echo "=== Phase D5: ODD corpus, ${#CELLS[@]} cell(s), $N_EVENTS events each ==="
echo "Output: $OUTDIR"

for line in "${CELLS[@]}"; do
  read -r sample axis label <<<"$line"
  echo ""
  echo "============================================================"
  echo "Cell: $label  (axis: $axis)"
  echo "============================================================"
  if [[ ! -d "$TRACCC_SRC/data/$sample" ]]; then
    echo "  WARNING: $TRACCC_SRC/data/$sample missing — skipping"
    continue
  fi
  dump_dir=$(dump_one_cell "$sample" "$label" | tail -1)
  if ls "$dump_dir"/event_*.json 2>/dev/null | head -1 >/dev/null; then
    sweep_file=$(sweep_one_cell "$label" "$dump_dir" | tail -1)
    echo -e "$axis\t$label\t$N_EVENTS\t$dump_dir\t$sweep_file" >> "$CORPUS_OUT"
  else
    echo "  no dumps for $label, skipping sweep"
  fi
done

echo ""
echo "=== Phase D5 done ==="
echo "Corpus index: $CORPUS_OUT"
echo "To aggregate across phases run:"
echo "  ./scripts/phase_d6_aggregate.sh --root=$THESIS_RESULTS_ROOT"
