#!/usr/bin/env python3
"""Roll up run_unified_mis_jp_baseline_sweep.sh summary.tsv by (family, corpus).

Latency: median of per-event mean times (matches prior reports).
Throughput: median events/s and median candidates/s from per-event *_eps_mean /
  *_cand_per_s_mean (frozen-input single-stream convention).

Prints wide TSV to stdout.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
from collections import defaultdict


def load(path: pathlib.Path):
    lines = path.read_text().splitlines()
    if not lines:
        return [], []
    h = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) == len(h):
            rows.append(dict(zip(h, p)))
    return h, rows


def row_ok_status(r):
    return r.get("status", "").strip() == "ok"


def fvals(rows, col):
    out = []
    for r in rows:
        if not row_ok_status(r):
            continue
        try:
            v = float(r[col])
            if v == v:
                out.append(v)
        except (KeyError, ValueError):
            pass
    return out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def p95(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = int(round(0.95 * (len(s) - 1)))
    return s[min(i, len(s) - 1)]


def count_match(rows, col, want="true"):
    return sum(1 for r in rows if r.get(col) == want and row_ok_status(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary_tsv", type=pathlib.Path)
    args = ap.parse_args()

    _, rows = load(args.summary_tsv)
    bucket: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        bucket[(r.get("family", ""), r.get("corpus", ""))].append(r)

    hdr = (
        "family\tcorpus\tn_events\t"
        "med_cpu_ms\tmed_base_ms\tmed_mis_ms\tmed_jp_ms\t"
        "p95_cpu_ms\tp95_base_ms\tp95_mis_ms\tp95_jp_ms\t"
        "med_cpu_eps\tmed_base_eps\tmed_mis_eps\tmed_jp_eps\t"
        "med_cpu_cand_s\tmed_base_cand_s\tmed_mis_cand_s\tmed_jp_cand_s\t"
        "med_n_cand\t"
        "n_base_hash_match\tn_mis_hash_match\tn_jp_hash_match\t"
        "n_mis_ov1\tn_jp_ov1\n"
    )
    sys.stdout.write(hdr)

    for (fam, corp) in sorted(bucket.keys()):
        br = bucket[(fam, corp)]
        nc = fvals(br, "n_cand")

        def colmed(c):
            return med(fvals(br, c))

        def colp95(c):
            return p95(fvals(br, c))

        n_ok = sum(1 for r in br if row_ok_status(r))
        sys.stdout.write(
            f"{fam}\t{corp}\t{n_ok}\t"
            f"{colmed('cpu_time_mean_ms'):.6g}\t{colmed('base_time_mean_ms'):.6g}\t"
            f"{colmed('mis_time_mean_ms'):.6g}\t{colmed('jp_time_mean_ms'):.6g}\t"
            f"{colp95('cpu_time_mean_ms'):.6g}\t{colp95('base_time_mean_ms'):.6g}\t"
            f"{colp95('mis_time_mean_ms'):.6g}\t{colp95('jp_time_mean_ms'):.6g}\t"
            f"{colmed('cpu_eps_mean'):.6g}\t{colmed('base_eps_mean'):.6g}\t"
            f"{colmed('mis_eps_mean'):.6g}\t{colmed('jp_eps_mean'):.6g}\t"
            f"{colmed('cpu_cand_per_s_mean'):.6g}\t{colmed('base_cand_per_s_mean'):.6g}\t"
            f"{colmed('mis_cand_per_s_mean'):.6g}\t{colmed('jp_cand_per_s_mean'):.6g}\t"
            f"{med(nc) if nc else float('nan'):.6g}\t"
            f"{count_match(br, 'base_hash')}\t{count_match(br, 'mis_hash')}\t"
            f"{count_match(br, 'jp_hash')}\t"
            f"{sum(1 for r in br if row_ok_status(r) and r.get('mis_ov','')=='1')}\t"
            f"{sum(1 for r in br if row_ok_status(r) and r.get('jp_ov','')=='1')}\n"
        )


if __name__ == "__main__":
    main()
