#!/usr/bin/env bash
# Phase D6: Aggregate every Phase D sweep file under a results-root into a
# single corpus-wide validation table.
#
# Walks the results tree, collects every `sweep.txt` / `*sweep*.txt` produced
# by phases D1–D5 (and Phase E1 if present), parses the per-event metrics, and
# emits:
#   - <outdir>/aggregate.csv       (one row per (cell, event))
#   - <outdir>/aggregate_by_cell.tsv  (one row per cell, mean over events)
#   - <outdir>/aggregate_summary.md   (markdown table for the CERN note)
#
# Usage:
#   ./scripts/phase_d6_aggregate.sh                 # walks default results dir
#   ./scripts/phase_d6_aggregate.sh --root=<dir>    # walks any results dir
#   ./scripts/phase_d6_aggregate.sh --outdir=<dir>  # explicit output location
set -euo pipefail

THESIS_REPO="${THESIS_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="$THESIS_REPO/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root=*)   ROOT="${1#*=}"; shift ;;
    --outdir=*) OUTDIR="${1#*=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUTDIR="${OUTDIR:-$THESIS_REPO/results/${TIMESTAMP}_phase_d6_aggregate}"
mkdir -p "$OUTDIR"

echo "=== Phase D6: aggregate sweeps under $ROOT ==="
echo "Output: $OUTDIR"

python3 - "$ROOT" "$OUTDIR" << 'PYEOF'
import os, re, sys, csv, json, statistics
from pathlib import Path

root = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

# Numeric metrics emitted by traccc_benchmark_resolver_cuda we care about.
NUMERIC = [
    "n_candidates",
    "baseline_time_ms_mean", "baseline_time_ms_median",
    "pbg_time_ms_mean",      "pbg_time_ms_median",
    "graph_mis_time_ms_mean", "graph_mis_time_ms_median",
    "graph_jp_time_ms_mean",  "graph_jp_time_ms_median",
    "n_selected_baseline",   "n_selected_pbg",
    "n_selected_graph_mis",  "n_selected_graph_jp",
    "duplicate_rate_post_baseline", "duplicate_rate_post_pbg",
    "duplicate_rate_post_graph_mis", "duplicate_rate_post_graph_jp",
    "graph_jp_track_overlap_vs_cpu", "graph_mis_track_overlap_vs_cpu",
    "graph_jp_n_outer_iterations", "graph_mis_n_outer_iterations",
    "graph_max_edges",
    "selection_efficiency_baseline", "selection_efficiency_graph_jp",
    "fake_rate_baseline", "fake_rate_graph_jp",
]
BOOL = [
    "graph_jp_hash_match", "graph_mis_hash_match",
    "pbg_hash_match", "determinism_all_pass",
]
ALL_KEYS = NUMERIC + BOOL


def parse_sweep(path: Path):
    """Yield (event_label, metrics_dict) tuples from a sweep TXT."""
    cur_event, cur_metrics = None, {}
    with path.open() as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()
            if stripped.startswith("=== ") and stripped.endswith(" ==="):
                if cur_event is not None and cur_metrics:
                    yield cur_event, cur_metrics
                cur_event = stripped.strip("= ").strip()
                cur_metrics = {}
                continue
            for k in ALL_KEYS:
                m = re.search(rf"\b{k}=(\S+)", line)
                if m:
                    cur_metrics[k] = m.group(1)
    if cur_event is not None and cur_metrics:
        yield cur_event, cur_metrics


def cell_label_from_path(p: Path):
    parts = p.parts
    # Common patterns we lay down:
    #   results/<ts>_phase_d1/odd_10gev_sweep.txt
    #   results/<ts>_phase_d2_*/...
    #   results/<ts>_phase_d3_telescope/telescope_sweep.txt
    #   results/<ts>_phase_d4_*/sweep.txt
    #   results/<ts>_phase_d5_odd_corpus/<label>/sweep.txt
    name = p.name
    parent = p.parent.name
    grandparent = p.parent.parent.name if len(parts) >= 2 else ""
    if "phase_d5" in grandparent:
        return ("d5", parent)
    if "phase_d3" in parent or "telescope" in name:
        return ("d3", "telescope_10GeV")
    if "phase_d2" in parent or "fatras_expand" in parent:
        return ("d2", parent.replace("_phase_d2_fatras_expand", ""))
    if "phase_d1" in parent or "odd_10gev_sweep" in name:
        return ("d1", "geant4_10muon_10GeV")
    if "phase_d4" in parent:
        return ("d4", parent.replace("_phase_d4_sweep", ""))
    return (parent[:8] if parent else "?", name.replace(".txt",""))


sweep_files = []
for sweep in root.rglob("*.txt"):
    # only files that look like a four-backend sweep
    head = ""
    try:
        with sweep.open() as f:
            head = f.read(4096)
    except Exception:
        continue
    if "graph_jp_time_ms_mean" not in head and "baseline_time_ms_mean" not in head:
        continue
    sweep_files.append(sweep)

print(f"Found {len(sweep_files)} sweep file(s) under {root}")

rows_per_event = []
for sweep in sweep_files:
    phase, cell = cell_label_from_path(sweep)
    for event, metrics in parse_sweep(sweep):
        row = {"phase": phase, "cell": cell, "sweep_file": str(sweep), "event": event}
        for k in ALL_KEYS:
            row[k] = metrics.get(k, "")
        rows_per_event.append(row)

print(f"Parsed {len(rows_per_event)} per-event row(s)")

csv_path = outdir / "aggregate.csv"
with csv_path.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["phase","cell","event","sweep_file"] + ALL_KEYS)
    w.writeheader()
    for r in rows_per_event:
        w.writerow(r)
print(f"Wrote {csv_path}")


def to_float(v):
    try: return float(v)
    except Exception: return None

def is_true(v):
    return str(v).lower() == "true"

# Per-cell rollups.
cells = {}
for r in rows_per_event:
    key = (r["phase"], r["cell"])
    cells.setdefault(key, []).append(r)

cell_rows = []
for (phase, cell), events in sorted(cells.items()):
    n = len(events)
    def col(name):
        vals = [to_float(e[name]) for e in events if to_float(e[name]) is not None]
        return vals
    base = col("baseline_time_ms_mean")
    jp   = col("graph_jp_time_ms_mean")
    mis  = col("graph_mis_time_ms_mean")
    pbg  = col("pbg_time_ms_mean")
    n_cand = col("n_candidates")
    speedups_jp = []
    speedups_mis = []
    for e in events:
        b = to_float(e["baseline_time_ms_mean"])
        j = to_float(e["graph_jp_time_ms_mean"])
        m = to_float(e["graph_mis_time_ms_mean"])
        if b and j and j > 0: speedups_jp.append(b/j)
        if b and m and m > 0: speedups_mis.append(b/m)
    jp_match  = sum(1 for e in events if is_true(e["graph_jp_hash_match"]))
    mis_match = sum(1 for e in events if is_true(e["graph_mis_hash_match"]))
    det_pass  = sum(1 for e in events if is_true(e["determinism_all_pass"]))

    def stat(vals):
        if not vals: return ""
        return f"{statistics.mean(vals):.3f}"

    cell_rows.append({
        "phase": phase, "cell": cell, "n_events": n,
        "n_cand_mean": stat(n_cand),
        "baseline_ms_mean": stat(base),
        "pbg_ms_mean": stat(pbg),
        "mis_ms_mean": stat(mis),
        "jp_ms_mean":  stat(jp),
        "jp_speedup_mean":  stat(speedups_jp),
        "mis_speedup_mean": stat(speedups_mis),
        "jp_selection_identical":  f"{jp_match}/{n}",
        "mis_selection_identical": f"{mis_match}/{n}",
        "determinism_pass":        f"{det_pass}/{n}",
    })

cell_tsv = outdir / "aggregate_by_cell.tsv"
with cell_tsv.open("w") as f:
    headers = ["phase","cell","n_events","n_cand_mean",
               "baseline_ms_mean","pbg_ms_mean","mis_ms_mean","jp_ms_mean",
               "jp_speedup_mean","mis_speedup_mean",
               "jp_selection_identical","mis_selection_identical",
               "determinism_pass"]
    f.write("\t".join(headers) + "\n")
    for r in cell_rows:
        f.write("\t".join(str(r[h]) for h in headers) + "\n")
print(f"Wrote {cell_tsv}")

# Markdown summary suitable for the CERN review note.
md = outdir / "aggregate_summary.md"
with md.open("w") as f:
    f.write("# Cross-corpus validation summary\n\n")
    f.write(f"Aggregated from {len(rows_per_event)} per-event rows across {len(cells)} cell(s).\n\n")
    f.write("| phase | cell | n_ev | n_cand | base ms | pbg ms | mis ms | jp ms | jp×base | mis×base | jp ident | mis ident | det |\n")
    f.write("|------:|:-----|----:|------:|--------:|-------:|-------:|------:|--------:|---------:|---------:|----------:|----:|\n")
    for r in cell_rows:
        f.write("| " + " | ".join(str(r[h]) for h in
            ["phase","cell","n_events","n_cand_mean",
             "baseline_ms_mean","pbg_ms_mean","mis_ms_mean","jp_ms_mean",
             "jp_speedup_mean","mis_speedup_mean",
             "jp_selection_identical","mis_selection_identical",
             "determinism_pass"]) + " |\n")
    # Overall.
    all_jp_speed = []
    all_mis_speed = []
    n_jp_match = n_mis_match = n_det = total = 0
    for r in rows_per_event:
        b = to_float(r["baseline_time_ms_mean"])
        j = to_float(r["graph_jp_time_ms_mean"])
        m = to_float(r["graph_mis_time_ms_mean"])
        if b and j and j > 0: all_jp_speed.append(b/j)
        if b and m and m > 0: all_mis_speed.append(b/m)
        n_jp_match  += int(is_true(r["graph_jp_hash_match"]))
        n_mis_match += int(is_true(r["graph_mis_hash_match"]))
        n_det       += int(is_true(r["determinism_all_pass"]))
        total += 1

    if total:
        f.write("\n## Overall\n\n")
        if all_jp_speed:
            f.write(f"- JP speedup vs CPU baseline: min={min(all_jp_speed):.2f}x, mean={statistics.mean(all_jp_speed):.2f}x, max={max(all_jp_speed):.2f}x ({len(all_jp_speed)} events)\n")
        if all_mis_speed:
            f.write(f"- MIS speedup vs CPU baseline: min={min(all_mis_speed):.2f}x, mean={statistics.mean(all_mis_speed):.2f}x, max={max(all_mis_speed):.2f}x ({len(all_mis_speed)} events)\n")
        f.write(f"- JP selection-identical to CPU: {n_jp_match}/{total}\n")
        f.write(f"- MIS selection-identical to CPU: {n_mis_match}/{total}\n")
        f.write(f"- Determinism pass: {n_det}/{total}\n")

print(f"Wrote {md}")
PYEOF

echo ""
echo "=== Phase D6 done ==="
echo "Outputs in: $OUTDIR"
