#!/usr/bin/env python3
"""Aggregate per-event summary.tsv into per-corpus latency & throughput stats.

Reads a unified-sweep summary.tsv (columns: family, corpus, event, cpu_time_ms,
base_time_ms, jp_time_ms, …) and prints TSV to stdout:

  family, corpus, n_events,
  mean_cpu_ms, median_cpu_ms, p95_cpu_ms,
  mean_base_ms, median_base_ms, p95_base_ms,
  mean_jp_ms, median_jp_ms, p95_jp_ms,
  jp_hash_match_events (count true), effective_events_per_sec_jp_mean

Effective single-event throughput uses 1000/mean_ms (same convention as the
benchmark harness).

Example:
  python3 aggregate_unified_sweep_summary.py \\
    /path/to/summary.tsv --fatras-only > rollup.tsv
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys


def load(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text().splitlines()
    if not lines:
        return [], []
    h = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) != len(h):
            continue
        rows.append(dict(zip(h, p)))
    return h, rows


def p95_val(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = int(round(0.95 * (len(s) - 1)))
    return s[min(i, len(s) - 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary_tsv", type=pathlib.Path)
    ap.add_argument(
        "--fatras-only",
        action="store_true",
        help="Only rows where family == fatras",
    )
    args = ap.parse_args()

    _, rows = load(args.summary_tsv)
    from collections import defaultdict

    bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if args.fatras_only and r.get("family") != "fatras":
            continue
        fam = r.get("family", "")
        corp = r.get("corpus", "")
        bucket[(fam, corp)].append(r)

    header = (
        "family\tcorpus\tn_events\t"
        "mean_cpu_ms\tmedian_cpu_ms\tp95_cpu_ms\t"
        "mean_base_ms\tmedian_base_ms\tp95_base_ms\t"
        "mean_jp_ms\tmedian_jp_ms\tp95_jp_ms\t"
        "jp_hash_match_true\teps_jp_mean\n"
    )
    sys.stdout.write(header)

    for (fam, corp) in sorted(bucket.keys()):
        br = bucket[(fam, corp)]
        n = len(br)

        def fcol(name: str) -> list[float]:
            out = []
            for x in br:
                try:
                    out.append(float(x[name]))
                except (KeyError, ValueError):
                    pass
            return out

        cpus = fcol("cpu_time_ms")
        bases = fcol("base_time_ms")
        jps = fcol("jp_time_ms")

        def stats(vals: list[float]) -> tuple[float, float, float]:
            if not vals:
                return (float("nan"), float("nan"), float("nan"))
            m = statistics.mean(vals)
            med = statistics.median(vals)
            p9 = p95_val(vals)
            return (m, med, p9)

        mc, medc, p95c = stats(cpus)
        mb, medb, p95b = stats(bases)
        mj, medj, p95j = stats(jps)

        hm = sum(1 for x in br if x.get("jp_hash_match") == "true")
        eps = 1000.0 / mj if mj and mj > 0 else float("nan")

        sys.stdout.write(
            f"{fam}\t{corp}\t{n}\t"
            f"{mc:.6g}\t{medc:.6g}\t{p95c:.6g}\t"
            f"{mb:.6g}\t{medb:.6g}\t{p95b:.6g}\t"
            f"{mj:.6g}\t{medj:.6g}\t{p95j:.6g}\t"
            f"{hm}\t{eps:.6g}\n"
        )


if __name__ == "__main__":
    main()
