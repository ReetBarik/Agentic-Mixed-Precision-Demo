#!/usr/bin/env python3
"""Phase 2b analysis — the 2a-verdict vs 2b-delta comparison table.

Reads the Phase-2b scorer manifest (``manifest_scorer_<I>.jsonl``) and the Phase-2a
per-integral manifest (``manifest_<I>.json``, whose ``decisions`` carry the old
per-intent verdict), joins them on ``(region_id, rung)``, and prints the artifact
that proves the reframe: regions the whole-app Validator stamped ``insufficient_fix``
now show a concrete, un-buried delta.

Usage:
    python runs/qcdloop/analyze_2b_manifest.py \
        --scorer runs/qcdloop/per_integral_out_b1_2b/B1/manifest_scorer_B1.jsonl \
        --manifest-2a runs/qcdloop/per_integral_out_b1_fanout/B1/manifest_B1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.validator import scorer as sc  # noqa: E402


def _rung(kind: str) -> str:
    return sc.rung_from_kind(kind or "")


def _region_id(d: dict) -> str:
    ls = d.get("line")
    le = d.get("line_end", ls)
    return sc.canonical_region_id(d.get("file"), ls, le)


def load_2a(path: Path) -> dict:
    """``{(region_id, rung): decision}`` from a 2a per-integral manifest."""
    m = json.loads(Path(path).read_text())
    out = {}
    for d in m.get("decisions", []):
        out[(_region_id(d), _rung(d.get("kind")))] = d
    return out, m


def fmt_delta(v) -> str:
    if v is None:
        return "     —    "
    return f"{v:.3e}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scorer", required=True, help="manifest_scorer_<I>.jsonl (2b)")
    ap.add_argument("--manifest-2a", required=True, help="manifest_<I>.json (2a)")
    ap.add_argument("--md", action="store_true", help="emit a Markdown table")
    args = ap.parse_args(argv)

    rows = sc.read_rows(args.scorer)
    two_a, m2a = load_2a(args.manifest_2a)

    # collapse fan-out over-generation to the best cell per (region_id, rung)
    best = sc.collapse_min_delta(rows)
    keys = sorted(best.keys())

    measured = sum(1 for r in rows if r["status"] == sc.STATUS_MEASURED)
    print(f"scorer cells: {len(rows)} rows, {len(keys)} distinct (region_id,rung); "
          f"{measured} measured")
    print(f"2a: status={m2a.get('status')} counts={m2a.get('counts')} "
          f"failure_modes={m2a.get('failure_modes')}\n")

    if args.md:
        print("| region_id | rung | 2a verdict | 2a reason | 2b status | "
              "2b delta_eff | baseline_delta_eff | inert? |")
        print("|---|---|---|---|---|---|---|---|")
    else:
        print(f"{'region_id':<16} {'rung':<7} {'2a verdict':<10} "
              f"{'2a reason':<17} {'2b status':<15} {'delta_eff':<11} "
              f"{'base_delta':<11} {'inert?':<6}")
        print("-" * 100)

    for key in keys:
        region_id, rung = key
        row = best[key]
        d2a = two_a.get(key, {})
        v2a = d2a.get("verdict", "—")
        r2a = d2a.get("verdict_reason") or (d2a.get("patcher_status") or "—")
        st = row["status"]
        de = fmt_delta(row.get("delta_effective"))
        bde = fmt_delta(row.get("baseline_delta_effective"))
        d_eff = row.get("delta_effective")
        b_eff = row.get("baseline_delta_effective")
        inert = "—"
        if d_eff is not None and b_eff is not None:
            inert = "yes" if d_eff == b_eff else "no"
        if args.md:
            print(f"| `{region_id}` | {rung} | {v2a} | {r2a} | {st} | "
                  f"{de.strip()} | {bde.strip()} | {inert} |")
        else:
            print(f"{region_id:<16} {rung:<7} {v2a:<10} {r2a:<17} {st:<15} "
                  f"{de:<11} {bde:<11} {inert:<6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
