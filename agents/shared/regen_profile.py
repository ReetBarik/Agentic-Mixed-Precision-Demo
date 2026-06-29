"""Re-parse a characterizer run's ``journal.jsonl`` into ``sensitivity_profile.json``.

This is the deterministic post-processing half of the characterizer (no LLM, no
build).  Rebuild + rerun the micro-driver first so the journal reflects current
source (including ``TRACKED_HERE`` attribution), then call this to refresh the
profile.

It reuses :func:`agents.characterizer.log_parser.parse` so the output is exactly
what the live pipeline would emit — same op aggregation, same ``per_line``
rollup, and the same ``work_dir``-relative location keys (commit ``918738e``).

The kernel name and per-run sample count are recovered from the run directory
itself (the existing profile + the driver source) so callers need only point at
``runs/<kernel>/``.

Usage:
    python -m agents.shared.regen_profile runs/lnrat runs/cln ...
    python -m agents.shared.regen_profile runs/*        # all fixtures
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from pathlib import Path

from agents.characterizer import log_parser

_SAMPLE_RE = re.compile(r"sample_count\s*=\s*(\d+)")


def _profile_to_dict(p) -> dict:
    """Serialize a SensitivityProfile, converting provenance sets to sorted lists.

    Mirrors ``agents.characterizer.agent._profile_to_dict`` so re-parsed
    profiles are byte-compatible with pipeline-emitted ones.  ``sorted`` (not
    ``list``) keeps the JSON stable run-over-run — set iteration order is
    hash-randomized across processes.
    """
    d = dataclasses.asdict(p)
    for rec in d.get("per_op", []):
        rec["provenance_union"] = sorted(rec.get("provenance_union", []))
    for rec in d.get("per_line", {}).values():
        rec["provenance_union"] = sorted(rec.get("provenance_union", []))
    for rec in d.get("top_hotspots", []):
        rec["provenance_union"] = sorted(rec.get("provenance_union", []))
    return d


def _recover_kernel_name(run_dir: Path) -> str:
    """Read the existing profile's ``kernel`` field; fall back to the dir name."""
    prof = run_dir / "sensitivity_profile.json"
    if prof.exists():
        try:
            return json.loads(prof.read_text(encoding="utf-8")).get(
                "kernel", run_dir.name
            )
        except (json.JSONDecodeError, OSError):
            pass
    return run_dir.name


def _recover_sample_count(run_dir: Path) -> int | None:
    """Grep the driver source for ``sample_count = N`` (the kernel-invocation
    count).  Returns None if not found, in which case the parser falls back to
    the raw JSONL record count."""
    driver = run_dir / "src" / "micro_driver.cpp"
    if driver.exists():
        m = _SAMPLE_RE.search(driver.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))
    return None


def regen(run_dir: Path, flag_threshold: float = 1e8, top_n: int = 10) -> Path:
    """Re-parse ``run_dir/journal.jsonl`` → ``run_dir/sensitivity_profile.json``.

    ``work_dir`` is the run directory itself, so any ``TRACKED_HERE`` location
    that points inside the run dir (driver shims) is relativized; locations
    pointing at fixture kernels outside the run dir are left absolute (the
    parser's documented fallback).
    """
    run_dir = Path(run_dir)
    journal = run_dir / "journal.jsonl"
    if not journal.exists():
        raise FileNotFoundError(
            f"{journal} not found — rebuild + rerun the micro-driver first"
        )

    profile = log_parser.parse(
        journal_path=journal,
        kernel_name=_recover_kernel_name(run_dir),
        flag_threshold=flag_threshold,
        top_n=top_n,
        sample_count=_recover_sample_count(run_dir),
        work_dir=run_dir,
    )

    out = run_dir / "sensitivity_profile.json"
    out.write_text(json.dumps(_profile_to_dict(profile), indent=2), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for arg in argv:
        run_dir = Path(arg)
        if not (run_dir / "journal.jsonl").exists():
            print(f"[regen] skip {run_dir} (no journal.jsonl)", file=sys.stderr)
            continue
        try:
            out = regen(run_dir)
            print(f"[regen] wrote {out}")
        except Exception as exc:  # noqa: BLE001 — report and continue
            print(f"[regen] FAILED {run_dir}: {exc}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
