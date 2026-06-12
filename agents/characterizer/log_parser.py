"""Pure Python JSONL → SensitivityProfile.

Tracked writes one JSON object per line.  Expected fields per record:
  op       : str   — operation name ("add", "sub", "mul", "opaque", ...)
  loc      : str   — "file:fn:line" source location (may be absent → "")
  cond     : float — condition number for this sample
  rel_err  : float — relative forward error for this sample
  prov     : list[str] — tracked variable names involved
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
) -> SensitivityProfile:
    """Parse a Tracked JSONL journal into a SensitivityProfile."""

    raw_records = _load_jsonl(journal_path)

    # Aggregate by (op, location)
    aggregated: dict[tuple[str, str], _Agg] = {}
    for rec in raw_records:
        op = rec.get("op", "unknown")
        loc = rec.get("loc", rec.get("location", ""))
        cond = float(rec.get("cond", 0.0))
        rel_err = float(rec.get("rel_err", 0.0))
        prov = set(rec.get("prov", rec.get("provenance", [])))

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

    # per_variable rollup: variable → max cond it appeared in
    per_variable: dict[str, float] = {}
    for rec in op_records:
        for var in rec.provenance_union:
            per_variable[var] = max(per_variable.get(var, 0.0), rec.max_cond)

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

    samples_run = sum(r.sample_count for r in op_records)

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
