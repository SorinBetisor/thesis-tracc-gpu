#!/usr/bin/env python3
"""Convert FATRAS / CKF csv outputs to the traccc ambiguity-dump JSON format.

The traccc seq pipeline starts from cells.csv and runs the full chain
(clusterization -> measurements -> seeds -> CKF -> ambiguity), but the
FATRAS samples shipped under traccc/data/odd/fatras_ttbar_mu* contain
empty cells.csv files: the pre-resolution track candidates are stored
directly in event*-tracks_ckf.csv, with the per-track measurement list
in the Measurements_ID column. This script reads those CSVs and emits
the same JSON layout that traccc::io::write_ambiguity_input writes /
read_ambiguity_input expects.

Output schema (matches traccc/io/src/ambiguity_io.cpp):

    {
      "config":       {min_meas_per_track, max_iterations, max_shared_meas},
      "measurements": [{"identifier": <int>}, ...],
      "tracks":       [{"pval": <float>, "measurement_ids": [<int>, ...]}, ...]
    }

The pval column is derived from chi2 / ndf via a monotonic transform
(exp(-0.5 * chi2/ndf)) bounded in (0, 1]. The ambiguity resolver uses
pval only for ranking candidate tracks (higher = better), so any
strictly decreasing function of chi2/ndf yields the same selection;
matching the absolute distribution shape of the upstream chi2 survival
function would require scipy.stats.chi2.sf and isn't needed for
selection-identical behaviour.

Usage:
    fatras_csv_to_ambig_dump.py <event_dir> <event_idx> <out.json>
        [--min-meas-per-track 3] [--max-iterations 4294967295] [--max-shared-meas 1]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

MEAS_LIST_RE = re.compile(r"\[([^\]]*)\]")


def parse_meas_id_list(field: str) -> list[int]:
    field = field.strip().strip('"')
    m = MEAS_LIST_RE.search(field)
    if not m:
        return []
    body = m.group(1).strip()
    if not body:
        return []
    parts = [p for p in (x.strip() for x in body.split(",")) if p]
    return [int(p) for p in parts]


def load_track_rows(tracks_csv: Path) -> list[dict]:
    import csv
    out: list[dict] = []
    with tracks_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row)
    return out


def load_measurement_ids(meas_csv: Path) -> list[int]:
    import csv
    ids: list[int] = []
    with meas_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(int(row["measurement_id"]))
    return ids


def chi2_to_pval(chi2: float, ndf: float) -> float:
    """Monotonically decreasing transform of chi2/ndf into (0, 1].

    The resolver ranks candidates by pval, so a strictly decreasing
    function of chi2/ndf preserves the selection order produced by the
    canonical chi2 survival function. Using a closed-form transform
    avoids a scipy dependency in environments where the local numpy is
    too old for scipy's _typing imports.
    """
    if ndf <= 0:
        return 0.0
    chi2_per_ndf = max(0.0, chi2) / max(1.0, ndf)
    p = math.exp(-0.5 * chi2_per_ndf)
    if p <= 0.0:
        return 1e-12
    if p >= 1.0:
        return 1.0 - 1e-12
    return p


def event_files(event_dir: Path, event_idx: int) -> tuple[Path, Path]:
    stem = f"event{event_idx:09d}"
    tracks = event_dir / f"{stem}-tracks_ckf.csv"
    meas = event_dir / f"{stem}-measurements.csv"
    if not tracks.exists():
        raise FileNotFoundError(f"missing {tracks}")
    if not meas.exists():
        raise FileNotFoundError(f"missing {meas}")
    return tracks, meas


def build_dump(
    event_dir: Path,
    event_idx: int,
    min_meas_per_track: int,
    max_iterations: int,
    max_shared_meas: int,
) -> dict:
    tracks_csv, meas_csv = event_files(event_dir, event_idx)

    rows = load_track_rows(tracks_csv)
    all_meas_ids = load_measurement_ids(meas_csv)
    meas_id_set = set(all_meas_ids)

    track_records: list[dict] = []
    referenced_ids: set[int] = set()
    n_skipped_empty = 0
    n_skipped_unknown = 0

    for row in rows:
        try:
            chi2 = float(row["chi2"])
            ndf_raw = float(row["ndf"])
        except (KeyError, ValueError):
            continue
        ndf = max(1.0, ndf_raw)
        pval = chi2_to_pval(chi2, ndf)

        meas_ids = parse_meas_id_list(row.get("Measurements_ID", "[]"))
        clean_ids: list[int] = []
        for mid in meas_ids:
            if mid in meas_id_set:
                clean_ids.append(mid)
            else:
                n_skipped_unknown += 1
        if not clean_ids:
            n_skipped_empty += 1
            continue
        track_records.append({"pval": pval, "measurement_ids": clean_ids})
        referenced_ids.update(clean_ids)

    sorted_ids = sorted(referenced_ids)

    dump = {
        "config": {
            "min_meas_per_track": int(min_meas_per_track),
            "max_iterations": int(max_iterations),
            "max_shared_meas": int(max_shared_meas),
        },
        "measurements": [{"identifier": int(i)} for i in sorted_ids],
        "tracks": track_records,
    }
    dump["__meta__"] = {
        "source": "fatras_csv_to_ambig_dump.py",
        "tracks_csv": str(tracks_csv),
        "measurements_csv": str(meas_csv),
        "n_track_rows": len(rows),
        "n_track_records": len(track_records),
        "n_measurements_referenced": len(sorted_ids),
        "n_measurements_total_in_csv": len(all_meas_ids),
        "n_skipped_empty_meas_list": n_skipped_empty,
        "n_skipped_unknown_meas_id": n_skipped_unknown,
    }
    return dump


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("event_dir", type=Path)
    p.add_argument("event_idx", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--min-meas-per-track", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=4294967295)
    p.add_argument("--max-shared-meas", type=int, default=1)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    dump = build_dump(
        args.event_dir,
        args.event_idx,
        min_meas_per_track=args.min_meas_per_track,
        max_iterations=args.max_iterations,
        max_shared_meas=args.max_shared_meas,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(dump, f)
    if not args.quiet:
        meta = dump["__meta__"]
        print(
            f"{args.out}  tracks={meta['n_track_records']}  "
            f"meas_referenced={meta['n_measurements_referenced']}  "
            f"skipped_empty={meta['n_skipped_empty_meas_list']}  "
            f"skipped_unknown={meta['n_skipped_unknown_meas_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
