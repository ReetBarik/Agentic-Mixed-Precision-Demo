"""Backfill ``predicted_rel_err_if_ff`` into stability reports that predate it.

The reducer's ``predicted_rel_err_if_ff`` signal (``U_FF * max_sensitivity``) was
added after ``report_1k.json`` / ``report_100k.json`` were frozen, so those
reports carry only ``predicted_rel_err_if_float`` (``U_FLOAT * max_sensitivity``).
Strategy's speedup gate now keys off the ff prediction (see
:func:`agents.strategy.ranking.build_speedup_queue`); on a report lacking it the
loader falls back to the float value — a conservative upper bound that admits
*fewer* ff speedups than the true ff prediction would.  This utility rewrites the
ff signal onto such a report **without re-characterizing**, so a frozen report
gets the tighter, correct ff admissions.

Derivation (per region / variable / cascade chain that has a float prediction
but no ff prediction):

    predicted_rel_err_if_ff = U_FF * max_sensitivity                 (exact)

when ``max_sensitivity`` is recorded, else the algebraically-equivalent

    predicted_rel_err_if_ff = predicted_rel_err_if_float * (U_FF / U_FLOAT)

since both are ``U_x * max_sensitivity`` over the same sensitivity.  The two
agree to floating-point rounding; the sensitivity form is preferred when present.

Usage::

    python -m agents.shared.backfill_ff_prediction runs/qcdloop/report_1k.json
    python -m agents.shared.backfill_ff_prediction --dry-run report_100k.json
    python -m agents.shared.backfill_ff_prediction a.json b.json      # in place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agents.shared.stability_reducer import U_FF, U_FLOAT

_FF_FROM_FLOAT = U_FF / U_FLOAT   # 2**-46 / 2**-24 == 2**-22


def _ff_value(entry: dict) -> float | None:
    """The ff prediction to write for one region/variable/chain, or None if the
    entry carries no float prediction to derive from (nothing to backfill)."""
    sens = entry.get("max_sensitivity")
    if sens is not None:
        return U_FF * float(sens)
    pf = entry.get("predicted_rel_err_if_float")
    if pf is not None:
        return float(pf) * _FF_FROM_FLOAT
    return None


def _backfill_entry(entry: dict) -> bool:
    """Add ``predicted_rel_err_if_ff`` to one dict if missing + derivable.

    Returns True if it wrote the field.  Idempotent: a present ff field is left
    untouched.
    """
    if not isinstance(entry, dict) or "predicted_rel_err_if_ff" in entry:
        return False
    val = _ff_value(entry)
    if val is None:
        return False
    entry["predicted_rel_err_if_ff"] = val
    return True


def backfill_report(report: dict) -> int:
    """Backfill every region / variable / cascade chain in a report in place.

    Returns the number of entries updated.
    """
    updated = 0
    for idata in report.get("integrals", {}).values():
        for reg in idata.get("regions", {}).values():
            updated += _backfill_entry(reg)
        for var in idata.get("variables", {}).values():
            updated += _backfill_entry(var)
        # cascade chains may be a list (finalized) or a dict keyed by chain_id
        chains = idata.get("cascade_chains", [])
        chain_iter = chains.values() if isinstance(chains, dict) else chains
        for chain in chain_iter:
            updated += _backfill_entry(chain)
    return updated


def backfill_file(path: str | Path, *, dry_run: bool = False) -> int:
    """Load, backfill, and (unless ``dry_run``) rewrite a report file.

    Returns the number of entries updated.
    """
    path = Path(path)
    report = json.loads(path.read_text())
    updated = backfill_report(report)
    if updated and not dry_run:
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    return updated


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", help="stability_report JSON file(s)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts without rewriting the files")
    args = ap.parse_args(argv)

    total = 0
    for r in args.reports:
        n = backfill_file(r, dry_run=args.dry_run)
        total += n
        verb = "would update" if args.dry_run else ("updated" if n else "no change")
        print(f"{r}: {verb} {n} entr{'y' if n == 1 else 'ies'}", flush=True)
    print(f"total: {total} entries {'(dry run)' if args.dry_run else 'written'}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
