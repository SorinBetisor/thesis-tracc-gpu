#!/usr/bin/env python3
"""
Convert ODD Fatras ttbar CKF CSV outputs into ambiguity-resolution JSON dumps.

The generated Fatras directories already contain `event*-tracks_ckf.csv`, which
captures the pre-ambiguity track candidates and their measurement membership.
This script converts those files into the compact JSON format consumed by:

  traccc_benchmark_resolver
  traccc_benchmark_resolver_cuda

The original CKF CSV does not store the resolver p-value explicitly, so the
default mode reconstructs a close proxy from `(chi2, ndf)` using the
Wilson-Hilferty approximation to the chi-square survival function.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


EVENT_RE = re.compile(r"event(\d+)-tracks_ckf\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Fatras CKF CSV outputs into ambiguity-input JSON dumps"
    )
    parser.add_argument("dataset_dir", help="Path to fatras_ttbar_muXXX directory")
    parser.add_argument(
        "--outdir",
        help="Output directory for event_*.json dumps "
        "(default: <dataset_dir>/ambiguity_dumps_<timestamp>)",
    )
    parser.add_argument(
        "--events",
        type=int,
        default=0,
        help="Number of events to convert (0 = all discovered events)",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=0,
        help="Number of discovered events to skip before converting",
    )
    parser.add_argument(
        "--pval-source",
        choices=("approx-chi2", "truth-prob"),
        default="approx-chi2",
        help="How to populate track p-values in the dump",
    )
    return parser.parse_args()


def wh_chi2_survival(chi2: float, ndf: float) -> float:
    """Approximate chi-square survival function using Wilson-Hilferty."""
    if ndf <= 0.0:
        return 0.0
    if chi2 <= 0.0:
        return 1.0
    k = float(ndf)
    x = float(chi2)
    z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(
        2.0 / (9.0 * k)
    )
    sf = 0.5 * math.erfc(z / math.sqrt(2.0))
    return min(1.0, max(0.0, sf))


def parse_measurement_ids(raw: str) -> list[int]:
    text = raw.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [int(tok) for tok in text.split(",") if tok.strip()]


def choose_pval(row: dict[str, str], source: str) -> float:
    if source == "truth-prob":
        return float(row["truthMatchProbability"])
    chi2 = float(row["chi2"])
    ndf = float(row["ndf"])
    return wh_chi2_survival(chi2, ndf)


def convert_event(src: Path, dst: Path, pval_source: str) -> tuple[int, int]:
    tracks: list[dict[str, object]] = []
    unique_meas: set[int] = set()

    with src.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            meas_ids = parse_measurement_ids(row["Measurements_ID"])
            if not meas_ids:
                continue
            unique_meas.update(meas_ids)
            tracks.append(
                {
                    "pval": choose_pval(row, pval_source),
                    "measurement_ids": meas_ids,
                }
            )

    payload = {
        "config": {
            "min_meas_per_track": 3,
            "max_iterations": 0xFFFFFFFF,
            "max_shared_meas": 1,
        },
        "measurements": [{"identifier": mid} for mid in sorted(unique_meas)],
        "tracks": tracks,
    }

    dst.write_text(json.dumps(payload, indent=2))
    return len(tracks), len(unique_meas)


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir}")

    track_files: list[tuple[int, Path]] = []
    for path in sorted(dataset_dir.glob("event*-tracks_ckf.csv")):
        match = EVENT_RE.search(path.name)
        if match is None:
            continue
        track_files.append((int(match.group(1)), path))

    if not track_files:
        raise SystemExit(f"No event*-tracks_ckf.csv files found in {dataset_dir}")

    selected = track_files[args.skip_start :]
    if args.events > 0:
        selected = selected[: args.events]

    if not selected:
        raise SystemExit("No events selected after applying skip/events filters")

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        outdir = dataset_dir / "ambiguity_dumps"
    outdir.mkdir(parents=True, exist_ok=True)

    summary_path = outdir / "dump_summary.txt"
    with summary_path.open("w") as summary:
        summary.write("event_idx n_tracks n_unique_measurements pval_source source_csv\n")
        for event_idx, src in selected:
            dst = outdir / f"event_{event_idx:09d}.json"
            n_tracks, n_meas = convert_event(src, dst, args.pval_source)
            summary.write(
                f"{event_idx} {n_tracks} {n_meas} {args.pval_source} {src}\n"
            )
            print(
                f"event {event_idx:09d} -> {dst.name} | "
                f"tracks={n_tracks} unique_meas={n_meas}"
            )

    print(f"\nSummary: {summary_path}")
    print(f"Output:  {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
