"""Pure Python JSONL → SensitivityProfile.

Tracked writes one JSON object per line.  Expected fields per record:
  op       : str   — operation name ("add", "sub", "mul", "opaque", ...)
  at       : str   — "file:fn:line" source location (may be absent → "")
  cond     : float — condition number for this sample
  rel_err  : float — relative forward error for this sample
  prov_vars: list[str] — source-variable roots (v0.3+; older journals used a
             single flat ``prov``, still accepted as a fallback)

RETIRED (IMPROVEMENT_PLAN 5.A.1): the legacy ``per_variable`` rollup.  It read
only the pre-v0.3 ``prov`` key, so on every v0.3+ journal it was silently
empty — and the reducer's ``variables{}`` map (per-source sensitivity from the
forward-cone pass) is the signal to consume instead.  The profile field
remains (always ``{}``) for serialized-profile compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.characterizer.profile import OpRecord, SensitivityProfile


def parse(
    journal_path: Path,
    kernel_name: str,
    flag_threshold: float = 1e8,
    top_n: int = 10,
    sample_count: int | None = None,
    work_dir: Path | None = None,
) -> SensitivityProfile:
    """Parse a Tracked JSONL journal into a SensitivityProfile.

    Parameters
    ----------
    work_dir:
        If provided, source-location keys (``file:fn:line``) whose file part
        is inside ``work_dir`` are normalized to paths relative to it.
        This stabilizes ``per_line`` keys across machines/clones so that
        downstream agents can match locations across runs.  Paths that lie
        outside ``work_dir`` (or that can't be resolved) are left unchanged.
    """

    raw_records = _load_jsonl(journal_path)
    work_dir_resolved = work_dir.resolve() if work_dir is not None else None

    # Aggregate by (op, location)
    aggregated: dict[tuple[str, str], _Agg] = {}
    for rec in raw_records:
        op = rec.get("op", "unknown")
        loc = rec.get("at", rec.get("loc", rec.get("location", "")))
        loc = _normalize_loc(loc, work_dir_resolved)
        cond = float(rec.get("cond", 0.0))
        rel_err = float(rec.get("rel_err", 0.0))
        # v0.3 split provenance into prov_vars/prov_consts; older journals
        # used a single flat prov.  Source roots feed provenance_union.
        if "prov_vars" in rec:
            prov = set(rec.get("prov_vars") or [])
        else:
            prov = set(rec.get("prov", rec.get("provenance", [])) or [])

        key = (op, loc)
        if key not in aggregated:
            aggregated[key] = _Agg(op=op, location=loc)
        aggregated[key].update(cond, rel_err, prov)

    # Build OpRecord list
    op_records: list[OpRecord] = []
    for agg in aggregated.values():
        op_records.append(
            OpRecord(
                op=agg.op,
                location=agg.location,
                max_cond=agg.max_cond,
                max_rel_err=agg.max_rel_err,
                sample_count=agg.sample_count,
                provenance_union=agg.provenance_union,
                flagged=agg.max_cond > flag_threshold,
            )
        )

    # Sort by max_cond descending
    op_records.sort(key=lambda r: r.max_cond, reverse=True)

    # per_line rollup: for each source location keep the worst record
    per_line: dict[str, OpRecord] = {}
    for rec in op_records:
        loc = rec.location
        if not loc:
            continue
        if loc not in per_line or rec.max_cond > per_line[loc].max_cond:
            per_line[loc] = rec

    # per_variable rollup: RETIRED (5.A.1) — always empty.  The reducer's
    # variables{} map (forward-cone per-source sensitivity) supersedes it.
    per_variable: dict[str, float] = {}

    total = len(op_records)
    opaque_count = sum(1 for r in op_records if r.op == "opaque")
    opaque_coverage = opaque_count / total if total > 0 else 0.0

    notes: list[str] = []
    if opaque_coverage > 0.5:
        notes.append(
            f"kernel is heavily opaque ({opaque_coverage:.0%}); "
            "consider expanding Tracked op coverage or switching opaque→interop "
            "for major framework calls."
        )

    # sample_count is kernel invocations; fall back to total JSONL records if unknown
    samples_run = sample_count if sample_count is not None else len(raw_records)

    return SensitivityProfile(
        kernel=kernel_name,
        samples_run=samples_run,
        per_op=op_records,
        per_line=per_line,
        per_variable=per_variable,
        top_hotspots=op_records[:top_n],
        opaque_coverage=opaque_coverage,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Agg:
    def __init__(self, op: str, location: str):
        self.op = op
        self.location = location
        self.max_cond = 0.0
        self.max_rel_err = 0.0
        self.sample_count = 0
        self.provenance_union: set[str] = set()

    def update(self, cond: float, rel_err: float, prov: set[str]) -> None:
        self.max_cond = max(self.max_cond, cond)
        self.max_rel_err = max(self.max_rel_err, rel_err)
        self.sample_count += 1
        self.provenance_union |= prov


def _normalize_loc(loc: str, work_dir: Path | None) -> str:
    """Make the file part of a ``file:fn:line`` location relative to work_dir.

    Falls back to the original string if normalization is impossible (path
    outside ``work_dir``, file doesn't exist, malformed location, etc.).
    """
    if not loc or work_dir is None:
        return loc
    file_part, sep, rest = loc.partition(":")
    if not sep or not file_part:
        return loc
    try:
        rel = Path(file_part).resolve().relative_to(work_dir)
    except (ValueError, OSError):
        return loc
    return f"{rel.as_posix()}:{rest}"


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {lineno} of {path}: {exc}") from exc
    return records
