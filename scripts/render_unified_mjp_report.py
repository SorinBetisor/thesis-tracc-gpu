#!/usr/bin/env python3
"""Regenerate corpus_rollup.tsv, fatras_median_by_mu.tsv, and RESULTS.md for a sweep OUT dir."""

from __future__ import annotations

import argparse
import collections
import pathlib
import statistics
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent


def load_tsv(path: pathlib.Path) -> tuple[list[str], list[dict]]:
    lines = path.read_text().splitlines()
    if not lines:
        return [], []
    h = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        p = ln.split("\t")
        if len(p) == len(h):
            rows.append(dict(zip(h, p)))
    return h, rows


def row_ok(r: dict) -> bool:
    return r.get("status", "").strip() == "ok"


def pct(a: int, b: int) -> str:
    if b <= 0:
        return "—"
    return f"{100.0 * a / b:.1f}%"


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def fatras_by_mu(rows: list[dict]) -> list[dict]:
    mu_b: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if not row_ok(r):
            continue
        corp = r.get("corpus", "")
        if not corp.startswith("fatras_ttbar_mu"):
            continue
        mu = int(corp.split("mu")[-1])
        mu_b[mu].append(r)
    out = []
    for mu in sorted(mu_b.keys()):
        br = mu_b[mu]
        nc = [float(x["n_cand"]) for x in br]

        def g(k):
            return [float(x[k]) for x in br]

        mc, mb, mm, mj = med(g("cpu_time_mean_ms")), med(g("base_time_mean_ms")), med(
            g("mis_time_mean_ms")
        ), med(g("jp_time_mean_ms"))
        ncm = med(nc)

        def cs(t):
            return ncm * 1000 / t if t and t > 0 else float("nan")

        out.append(
            {
                "mu": mu,
                "n": len(br),
                "n_cand_med": ncm,
                "med_cpu": mc,
                "med_base": mb,
                "med_mis": mm,
                "med_jp": mj,
                "cands_base": cs(mb),
                "cands_jp": cs(mj),
            }
        )
    return out


def fmt_float(x: float, nd: int = 3) -> str:
    if x != x:
        return "—"
    return f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    w = [len(h) for h in headers]
    for row in rows:
        for i, c in enumerate(row):
            w[i] = max(w[i], len(c))

    def pad(s: str, i: int) -> str:
        return s.ljust(w[i])

    sep = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    top = "| " + " | ".join(pad(headers[i], i) for i in range(len(headers))) + " |"
    lines = [top, sep]
    for row in rows:
        lines.append("| " + " | ".join(pad(row[i], i) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", type=pathlib.Path, help="Sweep output directory")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    summary_path = out / "summary.tsv"
    if not summary_path.is_file():
        print("ERROR: missing", summary_path, file=sys.stderr)
        sys.exit(1)

    rollup_script = SCRIPT_DIR / "aggregate_mis_jp_sweep_rollup.py"
    rollup_path = out / "corpus_rollup.tsv"
    subprocess.run(
        [sys.executable, str(rollup_script), str(summary_path)],
        stdout=rollup_path.open("w"),
        check=True,
    )

    _, rows = load_tsv(summary_path)
    fm = fatras_by_mu(rows)
    fatras_path = out / "fatras_median_by_mu.tsv"
    fl = [
        "mu\tn\tn_cand_med\tmed_cpu_ms\tmed_base_ms\tmed_mis_ms\tmed_jp_ms\tcand_s_base\tcand_s_jp\n"
    ]
    for r in fm:
        fl.append(
            f'{r["mu"]}\t{r["n"]}\t{r["n_cand_med"]:.6g}\t{r["med_cpu"]:.6g}\t{r["med_base"]:.6g}\t'
            f'{r["med_mis"]:.6g}\t{r["med_jp"]:.6g}\t{r["cands_base"]:.6g}\t{r["cands_jp"]:.6g}\n'
        )
    fatras_path.write_text("".join(fl))

    by_fam = collections.Counter(r.get("family", "") for r in rows if row_ok(r))
    n_all = sum(1 for r in rows if row_ok(r))
    bh = sum(1 for r in rows if row_ok(r) and r.get("base_hash") == "true")
    mh = sum(1 for r in rows if row_ok(r) and r.get("mis_hash") == "true")
    jh = sum(1 for r in rows if row_ok(r) and r.get("jp_hash") == "true")

    _, roll_rows = load_tsv(rollup_path)
    syn = [x for x in roll_rows if x["family"] == "synthetic"]
    syn.sort(key=lambda x: x["corpus"])
    syn_md = md_table(
        [
            "Corpus",
            "n",
            "med CPU ms",
            "med base",
            "med MIS",
            "med JP",
            "med n_cand",
            "hash B/M/J",
        ],
        [
            [
                x["corpus"],
                x["n_events"],
                fmt_float(float(x["med_cpu_ms"]), 2),
                fmt_float(float(x["med_base_ms"]), 2),
                fmt_float(float(x["med_mis_ms"]), 2),
                fmt_float(float(x["med_jp_ms"]), 2),
                fmt_float(float(x["med_n_cand"]), 0),
                f'{x["n_base_hash_match"]}/{x["n_events"]},{x["n_mis_hash_match"]}/{x["n_events"]},{x["n_jp_hash_match"]}/{x["n_events"]}',
            ]
            for x in syn
        ],
    )

    odd = [x for x in roll_rows if x["family"] == "odd"]
    odd.sort(key=lambda x: x["corpus"])
    odd_md = md_table(
        ["Corpus", "n", "med CPU ms", "med base", "med MIS", "med JP", "hash B/M/J"],
        [
            [
                x["corpus"],
                x["n_events"],
                fmt_float(float(x["med_cpu_ms"]), 3),
                fmt_float(float(x["med_base_ms"]), 3),
                fmt_float(float(x["med_mis_ms"]), 3),
                fmt_float(float(x["med_jp_ms"]), 3),
                f'{x["n_base_hash_match"]}/{x["n_events"]},{x["n_mis_hash_match"]}/{x["n_events"]},{x["n_jp_hash_match"]}/{x["n_events"]}',
            ]
            for x in odd
        ],
    )

    fat_md = md_table(
        [
            "μ",
            "n",
            "n_cand†",
            "CPU ms",
            "base ms",
            "MIS ms",
            "JP ms",
            "cand/s base",
            "cand/s JP",
        ],
        [
            [
                str(r["mu"]),
                str(r["n"]),
                fmt_float(r["n_cand_med"], 0),
                fmt_float(r["med_cpu"], 2),
                fmt_float(r["med_base"], 2),
                fmt_float(r["med_mis"], 2),
                fmt_float(r["med_jp"], 2),
                fmt_float(r["cands_base"], 0),
                fmt_float(r["cands_jp"], 0),
            ]
            for r in fm
        ],
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    inv = md_table(
        ["Family", "Events", "Source"],
        [
            ["Synthetic", str(by_fam.get("synthetic", 0)), "5×n × 3 densities"],
            ["FATRAS", str(by_fam.get("fatras", 0)), "`data-work/data/fatras_csv_dumps/`"],
            ["ODD Geant4", str(by_fam.get("odd", 0)), "`data-work/data/odd_dumps/geant4_*/`"],
        ],
    )

    md_path = out / "RESULTS.md"
    md_path.write_text(
        f"""# Unified resolver benchmark — CPU / GPU baseline / MIS / JP

Generated: {now}

**Directory:** `{out}`

## Data inventory

{inv}

**Total events (status=ok):** {n_all}

**Hash vs CPU:** baseline {bh}/{n_all} ({pct(bh, n_all)}), MIS {mh}/{n_all} ({pct(mh, n_all)}), JP {jh}/{n_all} ({pct(jh, n_all)})

Artifacts: `summary.tsv`, `corpus_rollup.tsv`, `fatras_median_by_mu.tsv`, `raw/*.cpu.txt`, `raw/*.gpu.txt`

## Synthetic corpora

{syn_md}

## ODD Geant4 corpora

{odd_md}

## FATRAS — median by pile-up μ

† Median `n_candidates` in the μ bin.

{fat_md}

## Throughput

Per-event `events_per_sec` and candidates/s are **single-stream**, frozen-input metrics (inverse latency). Use them to compare backends on the **same** event, not as full-GPU sustained throughput.

## Protocol

- CPU: `traccc_benchmark_resolver` `--backend=cpu` `--repeats=10` `--warmup=3`
- GPU: `traccc_benchmark_resolver_cuda` `--conflict-graph=both` `--repeats=10` `--warmup=3` `--determinism-runs=5`

Driver: `run_unified_mis_jp_baseline_sweep.sh`
"""
    )
    print("Wrote", rollup_path, fatras_path, md_path)


if __name__ == "__main__":
    main()
