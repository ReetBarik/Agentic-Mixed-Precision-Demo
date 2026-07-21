#!/usr/bin/env python3
"""slice_gen_failures.py — WAVE3 characterization helper (read-only).

Slices a strategy run's iterations.jsonl + per-iter build logs into the residual
`llm_gen_failed` cluster, grouped by the *actual* compiler error signature (not
the misleading `llm_capacity` dispatch tag).

Usage:
    python3 slice_gen_failures.py --run-dir runs/qcdloop/strategy/<run_id> [--csv out.csv]

No writes outside the given --csv. No agents/ or tests/ touched.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# First `error:` line in a gcc build log, normalized to a coarse signature.
ERR_RE = re.compile(r"error:\s*(.*)")


def error_signature(build_log: Path) -> tuple[str, str]:
    """Return (raw_first_error, coarse_signature) for a build log."""
    if not build_log.exists():
        return ("<no build log>", "no_log")
    raw = ""
    for line in build_log.read_text(errors="replace").splitlines():
        m = ERR_RE.search(line)
        if m:
            raw = m.group(1).strip()
            break
    if not raw:
        return ("<no error line>", "no_error_line")
    # Coarse buckets — let the data define them.
    if "redefinition of" in raw and "Constants<" in raw:
        return (raw, "redefinition_Constants_specialization")
    if "redefinition" in raw:
        return (raw, "redefinition_other")
    if raw.startswith("duplicate"):
        return (raw, "duplicate_qualifier")
    if "#error" in raw or "R4" in raw:
        return (raw, "R4_constant_derivation_escape")
    if "cannot convert" in raw or "no matching function" in raw:
        return (raw, "missing_overload_or_convert")
    if "ambiguous" in raw:
        return (raw, "overload_ambiguous")
    return (raw, "other:" + raw[:40])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--csv")
    args = ap.parse_args()

    run = Path(args.run_dir)
    logs = run / "logs"
    rows = []
    with (run / "iterations.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("patcher_status") != "llm_gen_failed":
                continue
            t = d["target"]
            raw, sig = error_signature(logs / f"iter_{d['iter_id']}_build.log")
            rows.append({
                "iter": d["iter_id"],
                "kind": d["kind"],
                "phase": d["phase"],
                "file": t["file"],
                "line": t.get("line_start"),
                "sig": sig,
                "raw": raw,
            })

    rows.sort(key=lambda r: r["iter"])
    print(f"total llm_gen_failed: {len(rows)}\n")

    by_sig = Counter(r["sig"] for r in rows)
    print("=== clusters by coarse error signature ===")
    for sig, n in by_sig.most_common():
        print(f"{n:4d}  {sig}")

    print("\n=== by (kind x signature) ===")
    kx = Counter((r["kind"], r["sig"]) for r in rows)
    for (k, s), n in kx.most_common():
        print(f"{n:4d}  {k:16s} {s}")

    # distinct (file,line,kind) regions vs total attempts (retries inflate)
    distinct = {(r["file"], r["line"], r["kind"]) for r in rows}
    print(f"\ndistinct (file,line,target) regions: {len(distinct)}  "
          f"(vs {len(rows)} attempts)")

    # per-file-per-type: is the FIRST specialization in a TU ok and later ones colliding?
    print("\n=== distinct failed regions per (file, target-type) ===")
    byft = defaultdict(list)
    for r in distinct:
        byft[(r[0], r[2])].append(r[1])
    for (fn, kind), lines in sorted(byft.items()):
        print(f"{fn:16s} {kind:16s} {len(lines):2d} lines: {sorted(lines)}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
