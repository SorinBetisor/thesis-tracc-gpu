#!/usr/bin/env python3
"""Summarise a Tier A tuning sweep into per-corpus mean times and speedups.

Reads the summary.tsv produced by run_tier_a_tuning_compare.sh and emits:
  - <out_root>/per_corpus_mean.tsv    : mean(time_ms_mean) per (binary, backend, corpus)
  - <out_root>/speedup_table.md       : markdown table of tuned/untuned speedups
  - <out_root>/validity_report.md     : flags any backend where hash_match=false
                                        or determinism failed in either binary
"""

import argparse
import collections
import pathlib
import statistics
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("summary_tsv", type=pathlib.Path,
                   help="Path to summary.tsv produced by the sweep.")
    p.add_argument("--out", type=pathlib.Path, default=None,
                   help="Output directory (defaults to alongside summary_tsv).")
    return p.parse_args()


def load_rows(path):
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    return rows


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def main():
    args = parse_args()
    out_dir = args.out or args.summary_tsv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.summary_tsv)
    if not rows:
        print(f"ERROR: no usable rows in {args.summary_tsv}", file=sys.stderr)
        sys.exit(1)

    # Aggregate mean(time_ms_mean) per (binary, backend, corpus).
    bucket = collections.defaultdict(list)
    validity = collections.defaultdict(list)
    for r in rows:
        binary = r["binary"]
        backend = r["backend"]
        corpus = r["corpus"]
        t = to_float(r.get("time_ms_mean"))
        hm = r.get("hash_match", "")
        df = to_float(r.get("det_fail")) or 0.0
        if t is not None and t > 0:
            bucket[(binary, backend, corpus)].append(t)
        if hm != "true" or df > 0:
            validity[(binary, backend, corpus)].append(
                f"event={r['event']} hash_match={hm} det_fail={int(df)}")

    means = {k: statistics.mean(v) for k, v in bucket.items() if v}

    # Per-corpus mean TSV.
    per_corpus_path = out_dir / "per_corpus_mean.tsv"
    with per_corpus_path.open("w") as f:
        f.write("binary\tbackend\tcorpus\tn_events\tmean_time_ms\tmedian_event_ms\tmin_event_ms\tmax_event_ms\n")
        for k in sorted(bucket):
            v = bucket[k]
            f.write(f"{k[0]}\t{k[1]}\t{k[2]}\t{len(v)}\t{statistics.mean(v):.4f}\t"
                    f"{statistics.median(v):.4f}\t{min(v):.4f}\t{max(v):.4f}\n")

    # Speedup table.
    backends = sorted({k[1] for k in means.keys()})
    corpora = sorted({k[2] for k in means.keys()})
    md_lines = ["# Tier A hardware-tuning speedup", ""]
    md_lines.append("Speedup = untuned mean / tuned mean. Higher is better.")
    md_lines.append("")
    for backend in backends:
        md_lines.append(f"## Backend: `{backend}`")
        md_lines.append("")
        md_lines.append("| Corpus | Untuned (ms) | Tuned (ms) | Speedup |")
        md_lines.append("|---|---:|---:|---:|")
        for corpus in corpora:
            t_un = means.get(("untuned", backend, corpus))
            t_tu = means.get(("tuned",   backend, corpus))
            if t_un is None or t_tu is None or t_tu <= 0:
                md_lines.append(f"| {corpus} | { '—' if t_un is None else f'{t_un:.3f}' }"
                                f" | { '—' if t_tu is None else f'{t_tu:.3f}' } | — |")
            else:
                md_lines.append(f"| {corpus} | {t_un:.3f} | {t_tu:.3f} | {t_un/t_tu:.2f}× |")
        md_lines.append("")
    speedup_path = out_dir / "speedup_table.md"
    speedup_path.write_text("\n".join(md_lines))

    # Validity report.
    val_lines = ["# Tier A validity report", ""]
    if not validity:
        val_lines.append("All (binary, backend, corpus, event) rows passed the validity gate"
                         " (`hash_match=true`, `det_fail=0`).")
    else:
        val_lines.append("The following (binary, backend, corpus) tuples had at least one"
                         " offending event. Each row lists the failing events.")
        val_lines.append("")
        for k in sorted(validity):
            val_lines.append(f"- **{k[0]} / {k[1]} / {k[2]}** ({len(validity[k])} events):")
            for ev in validity[k][:8]:
                val_lines.append(f"    - {ev}")
            if len(validity[k]) > 8:
                val_lines.append(f"    - ... ({len(validity[k]) - 8} more)")
    validity_path = out_dir / "validity_report.md"
    validity_path.write_text("\n".join(val_lines))

    print(f"Wrote: {per_corpus_path}")
    print(f"Wrote: {speedup_path}")
    print(f"Wrote: {validity_path}")


if __name__ == "__main__":
    main()
