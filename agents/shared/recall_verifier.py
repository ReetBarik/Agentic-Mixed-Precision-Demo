"""Recall verifier — end-to-end signal usefulness check.

For each validation fixture, recall = fraction of ``symbolic_hints[*].location``
that is covered by some ``sensitivity_profile.top_hotspots[*].location``,
grouped by hint ``severity``.

Pass criteria: ≥0.80 recall on ``high`` severity,
≥0.50 on ``medium``, precision unbounded (false positives acceptable).  These
are surfaced as a non-blocking ``pass`` status — a threshold miss never makes
the script exit non-zero.  Only structural errors (missing files, malformed
JSON) cause a non-zero exit.

Location matcher (v1)
---------------------
Symbolic-hint locations and profile-hotspot locations use *different* formats:

  hint     ``cancellation_check:2-4``     (func:line-range)
  hint     ``file:log_sum_exp_naive:11``  (literal-"file":func:line-range)
  hint     ``Lnrat:complex_overload``     (kernel:logical-overload, no line)
  hotspot  ``src/micro_driver.cpp:exp:28``                (relpath:func:line)
  hotspot  ``/abs/path/cancellation.cpp:cancellation_check:9``

So an exact/substring match alone is too strict.  A hint is considered matched
against a hotspot location if EITHER:

  (a) one string is a substring of the other (after stripping the literal
      ``file:`` prefix from hints), OR
  (b) any C-identifier token of length ≥ 3 extracted from the hint location
      (excluding the literal ``file``) appears as a substring of the hotspot
      location.

Rule (b) is what lets ``cancellation_check:2-4`` match
``…/cancellation.cpp:cancellation_check:9`` (shared function-name token) — since
the call-site ``TRACKED_HERE`` forwarding pass, hotspots carry the *kernel*
function name, so any shared identifier token ≥ 3 chars matches.  When a kernel
calls ``std::`` math that cannot be forwarded, ops stay shim-attributed and the
resulting misses are reported in ``findings`` rather than papered over.
Precision is unbounded by spec, so leniency here is acceptable.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Pass thresholds. `low` is reported but ungated.
THRESHOLDS = {"high": 0.80, "medium": 0.50}
SEVERITIES = ("high", "medium", "low")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STOP_TOKENS = {"file"}  # literal placeholders that aren't real identifiers


def _hint_tokens(hint_loc: str) -> list[str]:
    """Identifier tokens (len ≥ 3, minus literal placeholders) from a hint loc."""
    return [
        t for t in _IDENT_RE.findall(hint_loc)
        if len(t) >= 3 and t.lower() not in _STOP_TOKENS
    ]


def _strip_file_prefix(loc: str) -> str:
    return loc[5:] if loc.startswith("file:") else loc


def match_hint(hint_loc: str, hotspot_locs: list[str]) -> str | None:
    """Return the first hotspot location matching ``hint_loc``, else None."""
    if not hint_loc:
        return None
    stripped = _strip_file_prefix(hint_loc)
    tokens = _hint_tokens(hint_loc)
    for hloc in hotspot_locs:
        if not hloc:
            continue
        # (a) substring either direction
        if stripped in hloc or hloc in stripped:
            return hloc
        # (b) shared identifier token
        if any(tok in hloc for tok in tokens):
            return hloc
    return None


def verify_fixture(run_dir: Path) -> dict:
    """Compute per-severity recall for one ``runs/<kernel>/`` directory."""
    run_dir = Path(run_dir)
    fixture = run_dir.name

    profile_path = run_dir / "sensitivity_profile.json"
    hints_path = run_dir / "symbolic_hints.json"

    if not profile_path.exists():
        raise FileNotFoundError(f"missing {profile_path}")
    if not hints_path.exists():
        raise FileNotFoundError(f"missing {hints_path}")

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    hints = json.loads(hints_path.read_text(encoding="utf-8"))

    hotspot_locs = [
        h.get("location", "") for h in profile.get("top_hotspots", [])
        if h.get("location", "")
    ]

    # Empty annotations: nothing to recall against.
    if not hints:
        return {"fixture": fixture, "no_annotations": True}

    buckets: dict[str, dict] = {s: {"matched": 0, "total": 0} for s in SEVERITIES}
    findings: list[str] = []

    for hint in hints:
        sev = hint.get("severity", "")
        loc = hint.get("location", "")
        bucket = buckets.setdefault(sev, {"matched": 0, "total": 0})
        bucket["total"] += 1
        hit = match_hint(loc, hotspot_locs)
        if hit is not None:
            bucket["matched"] += 1
        else:
            findings.append(
                f"unmatched {sev or '?'} hint location {loc!r} "
                f"(idiom={hint.get('idiom', '?')!r}); "
                f"top_hotspots locations={hotspot_locs or '[]'}"
            )

    result: dict = {"fixture": fixture}
    for sev in SEVERITIES:
        b = buckets.get(sev, {"matched": 0, "total": 0})
        recall = (b["matched"] / b["total"]) if b["total"] else None
        result[sev] = {"matched": b["matched"], "total": b["total"], "recall": recall}

    # Any non-standard severities (shouldn't happen, but don't silently drop).
    for sev, b in buckets.items():
        if sev not in SEVERITIES and b["total"]:
            recall = b["matched"] / b["total"]
            result[sev] = {"matched": b["matched"], "total": b["total"], "recall": recall}

    result["pass"] = _passes(result)
    if findings:
        result["findings"] = findings
    return result


def _passes(result: dict) -> bool:
    """Threshold gate: a severity with no hints (recall None) passes vacuously."""
    for sev, threshold in THRESHOLDS.items():
        recall = result.get(sev, {}).get("recall")
        if recall is not None and recall < threshold:
            return False
    return True


def verify_all(runs_root: Path, fixtures: list[str] | None = None) -> dict:
    runs_root = Path(runs_root)
    if fixtures is None:
        fixtures = sorted(
            p.name for p in runs_root.iterdir()
            if p.is_dir() and (p / "sensitivity_profile.json").exists()
        )

    per_fixture = []
    errors = []
    for name in fixtures:
        try:
            per_fixture.append(verify_fixture(runs_root / name))
        except Exception as exc:  # noqa: BLE001
            errors.append({"fixture": name, "error": str(exc)})

    gated = [f for f in per_fixture if not f.get("no_annotations")]
    overall_pass = all(f.get("pass", False) for f in gated) if gated else None

    return {
        "schema_version": SCHEMA_VERSION,
        "thresholds": THRESHOLDS,
        "overall_pass": overall_pass,
        "fixtures": per_fixture,
        "errors": errors,
    }


def _fmt_recall(v) -> str:
    if v is None:
        return "  n/a"
    return f"{v:5.0%}"


def print_report(summary: dict) -> None:
    print("Recall verifier — symbolic_hints vs top_hotspots")
    print(f"  thresholds: high ≥ {THRESHOLDS['high']:.0%}, "
          f"medium ≥ {THRESHOLDS['medium']:.0%}; precision unbounded\n")
    header = f"  {'fixture':<16} {'high':>10} {'medium':>10} {'low':>10}   status"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f in summary["fixtures"]:
        if f.get("no_annotations"):
            print(f"  {f['fixture']:<16} {'—':>10} {'—':>10} {'—':>10}   "
                  "no annotations")
            continue
        def cell(sev):
            b = f.get(sev, {})
            return f"{b.get('matched', 0)}/{b.get('total', 0)} {_fmt_recall(b.get('recall'))}"
        status = "PASS" if f.get("pass") else "FAIL"
        print(f"  {f['fixture']:<16} {cell('high'):>10} {cell('medium'):>10} "
              f"{cell('low'):>10}   {status}")
    if summary["errors"]:
        print("\n  errors:")
        for e in summary["errors"]:
            print(f"    {e['fixture']}: {e['error']}")
    # Surface findings (format mismatches) below the table.
    any_findings = False
    for f in summary["fixtures"]:
        for finding in f.get("findings", []):
            if not any_findings:
                print("\n  findings:")
                any_findings = True
            print(f"    [{f['fixture']}] {finding}")
    op = summary["overall_pass"]
    label = "PASS" if op else ("FAIL" if op is False else "n/a (no gated fixtures)")
    print(f"\n  overall: {label}")


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs_root", nargs="?", default="runs",
        help="directory containing runs/<kernel>/ subdirs (default: runs)",
    )
    parser.add_argument(
        "--fixtures", nargs="*", default=None,
        help="explicit fixture names; default = all run dirs with a profile",
    )
    parser.add_argument(
        "--out", default=None,
        help="write the JSON summary here (default: <runs_root>/recall_summary.json)",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print only; do not write summary",
    )
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    summary = verify_all(runs_root, args.fixtures)
    print_report(summary)

    if not args.no_write:
        out = Path(args.out) if args.out else runs_root / "recall_summary.json"
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n  wrote {out}")

    # Never fail on threshold miss; only on structural errors.
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
