#!/usr/bin/env python3
"""
compare_greedy_tiers.py  —  post-process greedy-tuning tier sweep summaries.

Usage:
    python3 compare_greedy_tiers.py \
        --gb0 /path/to/gb0/summary.tsv \
        --gb1 /path/to/gb1/summary.tsv \
        --gb2 /path/to/gb2/summary.tsv \
        [--out /path/to/output_dir]

Outputs:
    comparison_table.tsv   per-corpus mean±std time for each tier
    speedup_table.md       markdown speedup table (gb0 baseline)
    validity_report.md     hash_match / det_fail anomalies
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict


def load_tsv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def mean_std(values):
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    s = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    return m, s


def parse_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def parse_int(s):
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def aggregate(rows):
    """Return dict: corpus -> {mean, std, n, n_hash_fail, n_det_fail}"""
    by_corpus = defaultdict(list)
    hash_fails = defaultdict(int)
    det_fails = defaultdict(int)
    for row in rows:
        corpus = row["corpus"]
        t = parse_float(row.get("time_ms_mean"))
        if t is None or t < 0:
            continue
        by_corpus[corpus].append(t)
        if row.get("hash_match", "true").strip() == "false":
            hash_fails[corpus] += 1
        det_fails[corpus] += parse_int(row.get("det_fail", "0"))
    result = {}
    for corpus, vals in by_corpus.items():
        m, s = mean_std(vals)
        result[corpus] = {
            "mean": m,
            "std": s,
            "n": len(vals),
            "n_hash_fail": hash_fails[corpus],
            "n_det_fail": det_fails[corpus],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb0", required=True)
    ap.add_argument("--gb1", required=True)
    ap.add_argument("--gb2", required=True)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    tiers = {
        "GB-0 (baseline)": load_tsv(args.gb0),
        "GB-1 (prefix-scan)": load_tsv(args.gb1),
        "GB-2 (warp-sort)": load_tsv(args.gb2),
    }
    agg = {label: aggregate(rows) for label, rows in tiers.items()}

    os.makedirs(args.out, exist_ok=True)

    # All corpora across all tiers
    all_corpora = sorted(
        set(c for a in agg.values() for c in a.keys())
    )

    # ---- comparison_table.tsv ----
    tsv_path = os.path.join(args.out, "comparison_table.tsv")
    tier_labels = list(tiers.keys())
    with open(tsv_path, "w") as f:
        header = ["corpus"] + [
            f"{l}_mean_ms\t{l}_std_ms\t{l}_n" for l in tier_labels
        ]
        f.write("\t".join(header) + "\n")
        for corpus in all_corpora:
            row = [corpus]
            for label in tier_labels:
                d = agg[label].get(corpus)
                if d:
                    row += [f"{d['mean']:.4f}", f"{d['std']:.4f}", str(d["n"])]
                else:
                    row += ["NA", "NA", "0"]
            f.write("\t".join(row) + "\n")
    print(f"Written: {tsv_path}")

    # ---- speedup_table.md ----
    md_path = os.path.join(args.out, "speedup_table.md")
    base_label = tier_labels[0]
    with open(md_path, "w") as f:
        f.write("# Greedy Tuning Tier Speedup Table\n\n")
        f.write(
            "Speedup = GB-0 mean / tier mean. "
            "Positive = faster than baseline.\n\n"
        )

        # Fatras pile-up table
        fatras = [c for c in all_corpora if "fatras_ttbar_mu" in c]
        if fatras:
            f.write("## Fatras pile-up sweep\n\n")
            cols = ["Corpus", "GB-0 (ms)"] + [
                f"{l} (ms)" for l in tier_labels[1:]
            ] + [f"Δ% {l}" for l in tier_labels[1:]]
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
            for corpus in sorted(fatras, key=lambda c: int(c.split("mu")[1])):
                base = agg[base_label].get(corpus)
                if not base:
                    continue
                b = base["mean"]
                row_vals = [corpus, f"{b:.3f}"]
                deltas = []
                for label in tier_labels[1:]:
                    d = agg[label].get(corpus)
                    if d and d["mean"] > 0:
                        ms = d["mean"]
                        delta = (b - ms) / b * 100
                        row_vals.append(f"{ms:.3f}")
                        deltas.append(f"{delta:+.1f}%")
                    else:
                        row_vals.append("NA")
                        deltas.append("NA")
                f.write("| " + " | ".join(row_vals + deltas) + " |\n")
            f.write("\n")

        # ODD table
        odd = [c for c in all_corpora if "geant4" in c]
        if odd:
            f.write("## ODD / Geant4 datasets\n\n")
            f.write("| Corpus | GB-0 (ms) | GB-1 (ms) | GB-2 (ms) | Δ% GB-1 | Δ% GB-2 |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for corpus in sorted(odd):
                base = agg[base_label].get(corpus)
                if not base:
                    continue
                b = base["mean"]
                gb1 = agg.get("GB-1 (prefix-scan)", {}).get(corpus)
                gb2 = agg.get("GB-2 (warp-sort)", {}).get(corpus)
                gb1_s = f"{gb1['mean']:.3f}" if gb1 else "NA"
                gb2_s = f"{gb2['mean']:.3f}" if gb2 else "NA"
                d1 = f"{(b - gb1['mean'])/b*100:+.1f}%" if gb1 else "NA"
                d2 = f"{(b - gb2['mean'])/b*100:+.1f}%" if gb2 else "NA"
                f.write(f"| {corpus} | {b:.3f} | {gb1_s} | {gb2_s} | {d1} | {d2} |\n")
            f.write("\n")

        # Synthetic table
        synth = [c for c in all_corpora if "synthetic" in c]
        if synth:
            f.write("## Synthetic sweep\n\n")
            f.write("| Corpus | GB-0 (ms) | GB-1 (ms) | GB-2 (ms) | Δ% GB-1 | Δ% GB-2 |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for corpus in sorted(synth):
                base = agg[base_label].get(corpus)
                if not base:
                    continue
                b = base["mean"]
                gb1 = agg.get("GB-1 (prefix-scan)", {}).get(corpus)
                gb2 = agg.get("GB-2 (warp-sort)", {}).get(corpus)
                gb1_s = f"{gb1['mean']:.3f}" if gb1 else "NA"
                gb2_s = f"{gb2['mean']:.3f}" if gb2 else "NA"
                d1 = f"{(b - gb1['mean'])/b*100:+.1f}%" if gb1 else "NA"
                d2 = f"{(b - gb2['mean'])/b*100:+.1f}%" if gb2 else "NA"
                f.write(f"| {corpus} | {b:.3f} | {gb1_s} | {gb2_s} | {d1} | {d2} |\n")
            f.write("\n")

    print(f"Written: {md_path}")

    # ---- validity_report.md ----
    val_path = os.path.join(args.out, "validity_report.md")
    with open(val_path, "w") as f:
        f.write("# Validity Report — Greedy Tuning Tiers\n\n")
        f.write(
            "Lists hash_match failures and determinism failures that are "
            "**new** relative to GB-0 (baseline), per tier.\n\n"
        )
        base_fails = {
            c: (agg[base_label][c]["n_hash_fail"], agg[base_label][c]["n_det_fail"])
            for c in agg[base_label]
        }
        any_new = False
        for label in tier_labels[1:]:
            f.write(f"## {label}\n\n")
            new_rows = []
            for corpus in all_corpora:
                d = agg[label].get(corpus)
                if not d:
                    continue
                bf_hash, bf_det = base_fails.get(corpus, (0, 0))
                if d["n_hash_fail"] > bf_hash:
                    new_rows.append(
                        f"| {corpus} | hash_match=false | "
                        f"+{d['n_hash_fail'] - bf_hash} new failures |"
                    )
                    any_new = True
                if d["n_det_fail"] > bf_det:
                    new_rows.append(
                        f"| {corpus} | det_fail | "
                        f"+{d['n_det_fail'] - bf_det} new failures |"
                    )
                    any_new = True
            if new_rows:
                f.write("| corpus | failure type | count |\n")
                f.write("|---|---|---|\n")
                f.write("\n".join(new_rows) + "\n\n")
            else:
                f.write("✓ No new failures vs GB-0 baseline.\n\n")
        if not any_new:
            f.write("## Summary\n\nAll tuning tiers pass validity gate.\n")
    print(f"Written: {val_path}")

    # ---- console summary ----
    print("\n=== Quick Fatras speedup roll-up ===")
    fatras = sorted(
        [c for c in all_corpora if "fatras_ttbar_mu" in c],
        key=lambda c: int(c.split("mu")[1]),
    )
    print(f"{'Corpus':<25} {'GB-0':>8} {'GB-1':>8} {'Δ%GB-1':>8} {'GB-2':>8} {'Δ%GB-2':>8}")
    for corpus in fatras:
        base = agg[base_label].get(corpus)
        if not base:
            continue
        b = base["mean"]
        gb1 = agg.get("GB-1 (prefix-scan)", {}).get(corpus)
        gb2 = agg.get("GB-2 (warp-sort)", {}).get(corpus)
        gb1_s = f"{gb1['mean']:8.3f}" if gb1 else "      NA"
        gb2_s = f"{gb2['mean']:8.3f}" if gb2 else "      NA"
        d1 = f"{(b-gb1['mean'])/b*100:+8.1f}%" if gb1 else "      NA"
        d2 = f"{(b-gb2['mean'])/b*100:+8.1f}%" if gb2 else "      NA"
        print(f"{corpus:<25} {b:8.3f} {gb1_s} {d1} {gb2_s} {d2}")


if __name__ == "__main__":
    main()
