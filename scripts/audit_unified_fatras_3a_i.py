#!/usr/bin/env python3
"""Regenerate / verify Sec. 3a-i Fatras table from canonical sweep outputs.

Canonical sources (April 2026 unified three-backend sweep):
  - per_corpus_aggregate.json  (preferred — pre-aggregated means)
  - summary.tsv                (optional cross-check — recomputes means per row)

Usage:
  ./audit_unified_fatras_3a_i.py \\
    /path/to/per_corpus_aggregate.json

  ./audit_unified_fatras_3a_i.py \\
    /path/to/per_corpus_aggregate.json \\
    --summary-tsv /path/to/summary.tsv

Exits 1 if JSON vs TSV means differ by more than --tol-ms (when TSV given).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("aggregate_json", type=pathlib.Path)
    p.add_argument("--summary-tsv", type=pathlib.Path, default=None)
    p.add_argument("--tol-ms", type=float, default=0.05)
    return p.parse_args()


def mu_from_corpus(name: str) -> int | None:
    m = re.match(r"fatras_ttbar_mu(\d+)$", name)
    return int(m.group(1)) if m else None


def load_json_rows(path: pathlib.Path):
    data = json.loads(path.read_text())
    rows = {}
    for r in data:
        if r.get("family") != "fatras":
            continue
        corp = r.get("corpus", "")
        mu = mu_from_corpus(corp)
        if mu is None:
            continue
        rows[mu] = r
    return rows


def load_tsv_means(path: pathlib.Path) -> dict[int, dict]:
    """Returns mu -> {cpu_ms, base_ms, jp_ms} from raw per-event rows."""
    lines = path.read_text().splitlines()
    if not lines:
        return {}
    header = lines[0].split("\t")
    need = {"corpus", "cpu_time_ms", "base_time_ms", "jp_time_ms"}
    idx = {c: header.index(c) for c in need if c in header}
    if len(idx) != len(need):
        print("ERROR: summary.tsv missing required columns", file=sys.stderr)
        sys.exit(2)

    buckets: dict[str, list[tuple[float, float, float]]] = {}
    for line in lines[1:]:
        p = line.split("\t")
        if len(p) <= max(idx.values()):
            continue
        corp = p[idx["corpus"]]
        mu = mu_from_corpus(corp)
        if mu is None:
            continue
        try:
            cpu = float(p[idx["cpu_time_ms"]])
            base = float(p[idx["base_time_ms"]])
            jp = float(p[idx["jp_time_ms"]])
        except ValueError:
            continue
        buckets.setdefault(corp, []).append((cpu, base, jp))

    out: dict[int, dict] = {}
    for corp, triples in buckets.items():
        mu = mu_from_corpus(corp)
        if mu is None:
            continue
        cpus = [t[0] for t in triples]
        bases = [t[1] for t in triples]
        jps = [t[2] for t in triples]
        out[mu] = {
            "cpu_ms": statistics.mean(cpus),
            "base_ms": statistics.mean(bases),
            "jp_ms": statistics.mean(jps),
            "n": len(triples),
        }
    return out


def fmt_round(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def fmt_int(x: float) -> str:
    return str(int(round(x)))


def main():
    args = parse_args()
    jrows = load_json_rows(args.aggregate_json)
    if not jrows:
        print("ERROR: no fatras_ttbar_mu* rows in JSON", file=sys.stderr)
        sys.exit(1)

    tsv_m = load_tsv_means(args.summary_tsv) if args.summary_tsv else {}

    fail = False
    print("# Markdown rows (paste into conflict_graph_results_mis_jp.md §3a-i)\n")
    print(
        "| μ | n cand | CPU (ms) | baseline (ms) | **JP (ms)** | "
        "CPU / JP | baseline / JP | JP hash |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    total_fatras_events = sum(jrows[k]["n"] for k in jrows)
    total_jp_match = sum(jrows[k]["jp_match"] for k in jrows)
    mu_keys = sorted(jrows.keys())

    for mu in mu_keys:
        r = jrows[mu]
        cpu = r["cpu_ms"]
        base = r["base_ms"]
        jp = r["jp_ms"]
        cand = r["cand_mean"]
        n_ev = int(r["n"])
        hm = int(r["jp_match"])
        cpu_jp = cpu / jp if jp else float("nan")
        base_jp = base / jp if jp else float("nan")

        if mu in tsv_m:
            t = tsv_m[mu]
            for label, a, b in (
                ("cpu_ms", cpu, t["cpu_ms"]),
                ("base_ms", base, t["base_ms"]),
                ("jp_ms", jp, t["jp_ms"]),
            ):
                if abs(a - b) > args.tol_ms:
                    print(
                        f"# WARN μ={mu} {label} JSON={a:g} TSV-mean={b:g} "
                        f"delta={abs(a-b):g}",
                        file=sys.stderr,
                    )
                    fail = True

        print(
            f"| {mu} | {fmt_int(cand)} | {fmt_round(cpu, 2)} | {fmt_round(base, 2)} | "
            f"**{fmt_round(jp, 2)}** | "
            f"{fmt_round(cpu_jp, 2)}× | {fmt_round(base_jp, 2)}× | "
            f"{hm}/{n_ev} |"
        )

    print("\n# Totals (hash audit)\n")
    print(f"# sum_jp_match_events={total_jp_match} / sum_events={total_fatras_events}")
    print(f"# fraction_match={total_jp_match / total_fatras_events:.4f}")

    if args.summary_tsv and fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
