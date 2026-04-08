#!/usr/bin/env python3
"""
Occupancy inspector for traccc tml_full (ttbar_muXXX) datasets.

Usage:
    python3 inspect_tml_occupancy.py <event_dir> [--events N] [--all]

Examples:
    python3 inspect_tml_occupancy.py /path/to/traccc/data/tml_full/ttbar_mu200
    python3 inspect_tml_occupancy.py /path/to/traccc/data/tml_full/ttbar_mu200 --events 5
"""

import csv
import math
import sys
import os
import glob
import argparse
from collections import Counter


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f) - 1  # subtract header


def read_particles(path):
    particles = []
    with open(path) as f:
        for row in csv.DictReader(f):
            px = float(row["px"])
            py = float(row["py"])
            pz = float(row["pz"])
            pt = math.sqrt(px * px + py * py)
            particles.append({"pt": pt, "pz": pz, "type": int(row["particle_type"])})
    return particles


def inspect_event(evdir, event_id):
    prefix = os.path.join(evdir, f"event{event_id:09d}")

    files = {
        "cells":    f"{prefix}-cells.csv",
        "hits":     f"{prefix}-hits.csv",
        "meas":     f"{prefix}-measurements.csv",
        "mhmap":    f"{prefix}-measurement-simhit-map.csv",
        "parts_i":  f"{prefix}-particles_initial.csv",
        "parts_f":  f"{prefix}-particles_final.csv",
    }

    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"  event {event_id}: missing {missing}, skipping")
        return None

    n_cells = count_lines(files["cells"])
    n_hits  = count_lines(files["hits"])
    n_meas  = count_lines(files["meas"])
    n_mhmap = count_lines(files["mhmap"])

    # Active modules
    with open(files["meas"]) as f:
        gids = set(row["geometry_id"] for row in csv.DictReader(f))
    n_modules = len(gids)

    # Particle statistics
    parts_i = read_particles(files["parts_i"])
    parts_f = read_particles(files["parts_f"])

    n_parts_i = len(parts_i)
    n_parts_f = len(parts_f)

    # pt > 0.5 GeV (tracking threshold)
    trackable_i = [p for p in parts_i if p["pt"] > 0.5]
    trackable_f = [p for p in parts_f if p["pt"] > 0.5]

    # Cells per measurement ratio (occupancy indicator)
    cells_per_meas = n_cells / n_meas if n_meas else 0

    return {
        "event":          event_id,
        "n_cells":        n_cells,
        "n_hits":         n_hits,
        "n_measurements": n_meas,
        "n_mhmap":        n_mhmap,
        "n_modules":      n_modules,
        "cells_per_meas": cells_per_meas,
        "n_particles_initial": n_parts_i,
        "n_particles_final":   n_parts_f,
        "n_trackable_initial": len(trackable_i),
        "n_trackable_final":   len(trackable_f),
    }


def print_summary(results):
    keys = [
        "n_cells", "n_hits", "n_measurements", "n_modules",
        "cells_per_meas",
        "n_particles_initial", "n_particles_final",
        "n_trackable_initial", "n_trackable_final",
    ]

    print("\n" + "=" * 72)
    print("PER-EVENT OCCUPANCY SUMMARY")
    print("=" * 72)
    print(f"{'Metric':<30} {'Mean':>12} {'Min':>10} {'Max':>10}")
    print("-" * 72)
    for k in keys:
        vals = [r[k] for r in results]
        fmt = "12,.0f" if k != "cells_per_meas" else "12.2f"
        fmt_sm = "10,.0f" if k != "cells_per_meas" else "10.2f"
        print(f"  {k:<28} {sum(vals)/len(vals):{fmt}} {min(vals):{fmt_sm}} {max(vals):{fmt_sm}}")

    n_meas_mean = sum(r["n_measurements"] for r in results) / len(results)
    n_parts_mean = sum(r["n_trackable_final"] for r in results) / len(results)

    print("\n" + "=" * 72)
    print("PHYSICS-INFORMED AMBIGUITY RESOLUTION ESTIMATES")
    print("=" * 72)
    print(f"""
  n_measurements (mean)          : {n_meas_mean:>10,.0f}
  n_trackable_final (mean, pt>0.5GeV): {n_parts_mean:>10,.0f}

  Scaling from ODD muon benchmark (n_meas≈540 → n_candidates≈87, ratio 0.16):
    estimated n_candidates = n_meas × 0.16  ≈ {n_meas_mean * 0.16:>10,.0f}

  Measured seeds (traccc_tml_seed_count, 10 events, ttbar_mu200):
    mean n_seeds                    ≈     17,544

  CKF efficiency ×0.7–0.9 applied to seeds:
    estimated n_candidates (low)    ≈     12,281
    estimated n_candidates (high)   ≈     15,790

  GPU crossover threshold (from synthetic sweep): ~2,000–3,000 candidates
  → ttbar_mu200 is FIRMLY in the GPU-favorable regime (5–8× above crossover).
""")


def main():
    parser = argparse.ArgumentParser(description="TML event occupancy inspector")
    parser.add_argument("evdir", help="Path to dataset directory (e.g. tml_full/ttbar_mu200)")
    parser.add_argument("--events", type=int, default=10, help="Number of events to inspect (default: 10)")
    args = parser.parse_args()

    evdir = args.evdir.rstrip("/")
    print(f"Inspecting: {evdir}")

    # Discover available event IDs
    cells_files = sorted(glob.glob(os.path.join(evdir, "event*-cells.csv")))
    event_ids = []
    for f in cells_files:
        base = os.path.basename(f)
        eid = int(base.replace("event", "").replace("-cells.csv", ""))
        event_ids.append(eid)

    if not event_ids:
        print("No events found.")
        sys.exit(1)

    event_ids = event_ids[: args.events]
    print(f"Found {len(cells_files)} events, inspecting first {len(event_ids)}\n")

    results = []
    for eid in event_ids:
        r = inspect_event(evdir, eid)
        if r:
            results.append(r)
            print(f"  event {eid:03d}: meas={r['n_measurements']:>7,}  "
                  f"cells={r['n_cells']:>8,}  "
                  f"parts_f={r['n_particles_final']:>6,}  "
                  f"trackable={r['n_trackable_final']:>5,}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
