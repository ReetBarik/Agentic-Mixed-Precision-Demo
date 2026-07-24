#!/usr/bin/env python3
"""Backfill ``value_range_ok_for_float`` onto cascade-chain records in a report.

Region records already carry the WI1 float-range flag, but cascade-chain records
predate it (``_extract_cascade_chains`` unions source spans, not value ranges), so
the Strategy float-rung prune fails open on chains and warns.  Rather than pay a
full re-characterization (report_5k.json is ~850 MB / a 5k whole-app tracked run),
derive the chain flag from the report's own already-classified region records —
identical to what ``finalize_report`` now stamps for fresh reports.

This is a pure, idempotent post-process: for each chain, the flag is the AND over
its contributor lines' region ``value_range_ok_for_float`` (missing line → True,
fail-open, unchanged behavior).  Re-running on an already-backfilled report is a
no-op modulo recompute.

Usage:
    python -m agents.shared.backfill_chain_range_flag REPORT.json [--in-place]
    python -m agents.shared.backfill_chain_range_flag REPORT.json -o OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.shared.stability_reducer import chain_range_ok_for_float


def backfill(report: dict) -> dict[str, int]:
    """Stamp the chain flag in-place; return per-integral counts of chains set."""
    counts: dict[str, int] = {}
    for name, idata in report.get("integrals", {}).items():
        regions = idata.get("regions", {})
        chains = idata.get("cascade_chains", [])
        n = 0
        for ch in chains:
            ch["value_range_ok_for_float"] = chain_range_ok_for_float(ch, regions)
            n += 1
        counts[name] = n
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("-o", "--out", default=None,
                    help="Output path (default: alongside input as *.backfilled.json)")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the input report.")
    args = ap.parse_args(argv)

    src = Path(args.report)
    report = json.loads(src.read_text())
    counts = backfill(report)

    total_chains = sum(counts.values())
    unsafe = 0
    for idata in report.get("integrals", {}).values():
        for ch in idata.get("cascade_chains", []):
            if not ch.get("value_range_ok_for_float", True):
                unsafe += 1
    print(f"backfilled {total_chains} chains across {len(counts)} integrals; "
          f"{unsafe} flagged range-UNSAFE for float", file=sys.stderr)

    if args.in_place:
        dst = src
    elif args.out:
        dst = Path(args.out)
    else:
        dst = src.with_suffix(".backfilled.json")
    dst.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {dst}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
