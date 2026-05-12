#!/usr/bin/env python3
"""Run traccc_benchmark_resolver_cuda --profile --profile-kernels across
synthetic (all conflict densities × n_candidates), Fatras (all μ in dumps),
and ODD (each geant4 corpus, one event). Writes Markdown tables + machine TSV.

Example:
  python3 profile_resolver_sweep_to_md.py \\
    --bin /user/sbetisor/data-work/traccc/build/bin/traccc_benchmark_resolver_cuda \\
    --out-md /user/sbetisor/data-work/results/20260511_profile_matrix/profile_report.md \\
    --out-tsv /user/sbetisor/data-work/results/20260511_profile_matrix/profile_matrix.tsv
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROFILE_KEYS = (
    "profile_filter_setup_ms",
    "profile_unique_meas_ms",
    "profile_inverted_index_ms",
    "profile_shared_count_ms",
    "profile_initial_sort_ms",
    "profile_eviction_loop_ms",
    "profile_output_copy_ms",
    "profile_eviction_graph_launches",
    "profile_unique_meas_count",
    "profile_greedy_remove_tracks_ms",
    "profile_greedy_sort_updated_tracks_ms",
    "profile_greedy_fill_inverted_ids_ms",
    "profile_greedy_block_inclusive_scan_ms",
    "profile_greedy_scan_block_offsets_ms",
    "profile_greedy_add_block_offset_ms",
    "profile_greedy_rearrange_tracks_ms",
    "profile_greedy_update_status_ms",
)

SUMMARY_KEYS = (
    "n_candidates",
    "baseline_time_ms_mean",
    "baseline_hash_match",
    "baseline_latency_ms_per_event",
    "time_h2d_ms",
    "peak_memory_mb",
)


def parse_kv_block(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line or line.startswith("WARNING"):
            continue
        for tok in line.split():
            if "=" not in tok:
                continue
            k, _, v = tok.partition("=")
            k = k.strip()
            v = v.strip()
            if k and k not in out:
                out[k] = v
    return out


def run_case(
    bin_path: Path,
    args: list[str],
) -> tuple[int, str]:
    r = subprocess.run(
        [str(bin_path)] + args,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = r.stdout
    if r.stderr:
        out += "\n" + r.stderr
    return r.returncode, out


def find_dump(preferred: Path, fallback: Path) -> Path | None:
    if preferred.is_file():
        return preferred
    if fallback.is_file():
        return fallback
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bin", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--out-tsv", type=Path, required=True)
    ap.add_argument(
        "--fatras-root",
        type=Path,
        default=Path("/user/sbetisor/data-work/data/fatras_csv_dumps"),
    )
    ap.add_argument(
        "--odd-root",
        type=Path,
        default=Path("/user/sbetisor/data-work/data/odd_dumps"),
    )
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument(
        "--event",
        default="event_002",
        help="JSON stem for dump corpora (fallback event_000)",
    )
    args = ap.parse_args()

    bin_path = args.bin.expanduser().resolve()
    if not bin_path.is_file():
        print(f"ERROR: binary not found: {bin_path}", file=sys.stderr)
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict[str, str]] = []
    errors: list[str] = []

    def add_row(family: str, label: str, extra: dict[str, str]) -> None:
        row = {"family": family, "label": label, **extra}
        rows.append(row)

    base_flags = [
        f"--repeats={args.repeats}",
        f"--warmup={args.warmup}",
        "--profile",
        "--profile-kernels",
    ]

    n_cands = [500, 1000, 2000, 5000, 10000]
    densities = ["low", "med", "high"]

    for n in n_cands:
        for d in densities:
            label = f"synthetic_n{n}_{d}"
            argv = [
                "--synthetic",
                f"--n-candidates={n}",
                f"--conflict-density={d}",
                *base_flags,
            ]
            code, text = run_case(bin_path, argv)
            kv = parse_kv_block(text)
            if code != 0:
                errors.append(f"{label}: exit {code}")
            rec = {k: kv.get(k, "") for k in SUMMARY_KEYS + PROFILE_KEYS}
            add_row("synthetic", label, rec)
            print(f"OK {label}", flush=True)

    fatras_dirs: list[tuple[int, Path]] = []
    for p in sorted(args.fatras_root.glob("fatras_ttbar_mu*")):
        m = re.match(r"fatras_ttbar_mu(\d+)$", p.name)
        if m:
            fatras_dirs.append((int(m.group(1)), p))
    fatras_dirs.sort(key=lambda x: x[0])
    for mu, d in fatras_dirs:
        preferred = d / f"{args.event}.json"
        fallback = d / "event_000.json"
        dump = find_dump(preferred, fallback)
        if dump is None:
            errors.append(f"fatras_mu{mu}: no dump json")
            continue
        label = f"fatras_mu{mu}_{dump.stem}"
        argv = [f"--input-dump={dump}", *base_flags]
        code, text = run_case(bin_path, argv)
        kv = parse_kv_block(text)
        if code != 0:
            errors.append(f"{label}: exit {code}")
        rec = {k: kv.get(k, "") for k in SUMMARY_KEYS + PROFILE_KEYS}
        add_row("fatras", label, rec)
        print(f"OK {label}", flush=True)

    odd_dirs = sorted(
        [p for p in args.odd_root.iterdir() if p.is_dir() and p.name.startswith("geant4_")]
    )
    for d in odd_dirs:
        preferred = d / f"{args.event}.json"
        fallback = d / "event_000.json"
        dump = find_dump(preferred, fallback)
        if dump is None:
            errors.append(f"odd {d.name}: no dump")
            continue
        label = f"odd_{d.name}_{dump.stem}"
        argv = [f"--input-dump={dump}", *base_flags]
        code, text = run_case(bin_path, argv)
        kv = parse_kv_block(text)
        if code != 0:
            errors.append(f"{label}: exit {code}")
        rec = {k: kv.get(k, "") for k in SUMMARY_KEYS + PROFILE_KEYS}
        add_row("odd", label, rec)
        print(f"OK {label}", flush=True)

    columns = ["family", "label"] + list(SUMMARY_KEYS) + list(PROFILE_KEYS)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_tsv.parent.mkdir(parents=True, exist_ok=True)

    with args.out_tsv.open("w") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            f.write("\t".join(r.get(c, "") for c in columns) + "\n")

    def md_table(title: str, pred) -> str:
        sub = [r for r in rows if pred(r)]
        if not sub:
            return f"### {title}\n\n_(no rows)_\n\n"
        head = "| " + " | ".join(
            ["label", "n_cand", "base_ms", "evict_ms", "rm_trk", "rearr", "graph_launches"]
        )
        sep = "| " + " | ".join(["---"] * 7)
        lines = [head, sep]
        for r in sub:
            lines.append(
                "| "
                + " | ".join(
                    [
                        r["label"].replace("|", "\\|"),
                        r.get("n_candidates", ""),
                        r.get("baseline_time_ms_mean", ""),
                        r.get("profile_eviction_loop_ms", ""),
                        r.get("profile_greedy_remove_tracks_ms", ""),
                        r.get("profile_greedy_rearrange_tracks_ms", ""),
                        r.get("profile_eviction_graph_launches", ""),
                    ]
                )
                + " |"
            )
        return f"### {title}\n\n" + "\n".join(lines) + "\n\n"

    md = []
    md.append(f"# GPU resolver profiling matrix\n\n")
    md.append(f"Generated `{ts}` (UTC). Binary `{bin_path}`.\n\n")
    md.append(
        "Flags: `--profile --profile-kernels` (eager per-kernel cumulative ms; "
        "timed path uses CUDA graphs). "
        f"Repeats={args.repeats}, warmup={args.warmup}. "
        f"Dumps use `{args.event}.json` when present, else `event_000.json`.\n\n"
    )
    md.append(f"Machine-readable TSV: `{args.out_tsv}`\n\n")
    md.append(
        "Columns in TSV suitable for stacked bars: `profile_greedy_*_ms` plus "
        "`profile_eviction_loop_ms` (phase envelope) and `baseline_time_ms_mean` (timed mean).\n\n"
    )

    md.append("---\n\n")
    md.append(md_table("Synthetic (all n_candidates × conflict density)", lambda r: r["family"] == "synthetic"))
    md.append(md_table("Fatras (one event per μ)", lambda r: r["family"] == "fatras"))
    md.append(md_table("ODD Geant4 corpora (one event per corpus)", lambda r: r["family"] == "odd"))

    md.append("### Greedy kernel columns (full names)\n\n")
    md.append("| Column | Meaning |\n|---|---|\n")
    for k in PROFILE_KEYS:
        if k.startswith("profile_greedy"):
            md.append(f"| `{k}` | cumulative eager-path GPU ms |\n")
    md.append("\n")

    if errors:
        md.append("### Errors\n\n")
        for e in errors:
            md.append(f"- {e}\n")
        md.append("\n")

    args.out_md.write_text("".join(md))
    print(f"Wrote {args.out_md}", file=sys.stderr)
    print(f"Wrote {args.out_tsv}", file=sys.stderr)
    if errors:
        print(f"WARNING: {len(errors)} cases reported errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
