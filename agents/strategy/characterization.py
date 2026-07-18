"""Load the fixed characterization report into ranked-ready region records.

The report is the characterizer's ``stability_report`` JSON (schema_version 1):

    {"integrals": {"B12": {"class_counts": {...},
                           "regions": {"B2m.h:355": {<region fields>}, "": {...}}}},
     "kind": "stability_report", "samples_seen": {...}, "schema_version": 1}

Each localizable region key is ``"file:line"`` (single line).  The ``""`` key is
the whole-integral rollup / non-localizable bucket — it carries no line range so
Strategy cannot emit a region-scoped intent for it; those are skipped and
counted (``non_localizable_skipped``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agents.strategy.models import RegionTarget


# Signal-class severity — merging a source line across integrals keeps the
# most severe class (a line that is local_cancellation in ANY integral is a
# local_cancellation target).
_SEVERITY = {
    "stable": 0, "log_near_root": 1, "cancellation_cascade": 2,
    "local_cancellation": 3,
}


@dataclass
class ChainRecord:
    """A localized ``cancellation_cascade`` spanning multiple source lines.

    The characterizer emits one ``cascade_chain`` per victim (see
    stability_reducer): a list of contributing sub-region lines (multi-file
    allowed) that together accumulate the cancellation.  A chain is promoted as a
    unit; the resulting precision floor is distributed across every line in
    ``lines`` via ``required_by`` bookkeeping (design "Chain promotion
    semantics").  ``chain_id`` is stable across runs.
    """

    integral: str
    chain_id: str
    lines: list[RegionTarget]
    signal_class: str
    max_cond: float
    max_rel_err: float
    predicted_rel_err_if_float: float
    op_count: int
    n: int
    variables: list[str] = field(default_factory=list)

    def walk_record(self) -> "RegionRecord":
        """A single-target ``RegionRecord`` the retry walk drives on.

        The walk/Patcher intent is region-shaped (single span); the chain's first
        sub-region is the representative target.  The promoted precision is
        distributed to ALL ``lines`` afterward — the representative is a driver
        for the walk, not the assignment scope.  (Real multi-line chain intents
        for Patcher are deferred; see HANDOFF.md.)
        """
        rep = self.lines[0]
        return RegionRecord(
            integral=self.integral, target=rep, signal_class=self.signal_class,
            max_cond=self.max_cond, max_rel_err=self.max_rel_err,
            predicted_rel_err_if_float=self.predicted_rel_err_if_float,
            op_count=self.op_count, n=self.n, integrals=[self.integral])


@dataclass
class RegionRecord:
    """One localizable characterization *code region* + the fields ranking reads.

    A source line in a shared header is compiled into many integrals, so the
    characterization emits one entry per (integral, file, line).  Since a line
    can only be promoted once, records sharing ``(file, line_start, line_end)``
    are merged (see :func:`load_regions`) with worst-case signals.  ``integral``
    is the representative (highest-cond) contributor; ``integrals`` lists them all.
    """

    integral: str
    target: RegionTarget
    signal_class: str
    max_cond: float
    max_rel_err: float
    predicted_rel_err_if_float: float
    op_count: int
    n: int
    integrals: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int]:
        return self.target.key


def _parse_location(rkey: str) -> tuple[str, int] | None:
    """``"B2m.h:355"`` -> ``("B2m.h", 355)``; None if not a localizable key."""
    if not rkey or ":" not in rkey:
        return None
    file, _, line = rkey.rpartition(":")
    if not file or not line.isdigit():
        return None
    return file, int(line)


def load_regions(report_path: str | Path, *, merge: bool = True,
                 ) -> tuple[list[RegionRecord], dict]:
    """Parse the report into ``RegionRecord``s plus a small meta summary.

    Returns ``(records, meta)`` where meta carries ``non_localizable_skipped``,
    ``raw_regions`` (pre-merge count) and ``schema_version``.  ``variables`` is
    the region-local *reads* set (``region_local_vars``, falling back to
    ``prov_vars`` for older reports) — the tight in-scope set ff_integrator /
    dd_integrator consume, not the full transitive provenance union.

    When ``merge`` (default), per-integral entries sharing ``(file, line)`` are
    collapsed into one code-region target with worst-case signals, because a
    source line can only be promoted once (Q1 region contract has no integral).
    """
    data = json.loads(Path(report_path).read_text())
    raw: list[tuple[str, dict, str, int]] = []   # (integral, region, file, line)
    skipped = 0

    for integral, idata in data.get("integrals", {}).items():
        for rkey, region in idata.get("regions", {}).items():
            loc = _parse_location(rkey)
            if loc is None or region.get("non_localizable"):
                skipped += 1
                continue
            raw.append((integral, region, loc[0], loc[1]))

    records = _merge_by_line(raw) if merge else [_one_record(*r) for r in raw]

    meta = {
        "schema_version": data.get("schema_version"),
        "kind": data.get("kind"),
        "non_localizable_skipped": skipped,
        "raw_regions": len(raw),
        "n_regions": len(records),
        "merged": merge,
    }
    return records, meta


def load_chains(report_path: str | Path) -> tuple[list[ChainRecord], dict]:
    """Parse the report's ``cascade_chain`` records into ``ChainRecord``s.

    These are the concrete population of correctness tier 2 (cancellation
    cascade): the localized replacement for the non-localizable cascade regions
    that ``load_regions`` skips.  Chains are NOT merged (each victim is its own
    chain); Strategy resolves per-line overlap via ``required_by`` bookkeeping.
    Returns ``(chains, meta)`` with ``meta['n_chains']``.
    """
    data = json.loads(Path(report_path).read_text())
    chains: list[ChainRecord] = []
    for integral, idata in data.get("integrals", {}).items():
        for chain in idata.get("cascade_chains", []) or []:
            lines = []
            for span in chain.get("chain", []) or []:
                lines.append(RegionTarget(
                    file=span["file"],
                    line_start=int(span["line_start"]),
                    line_end=int(span["line_end"]),
                    variables=list(chain.get("region_local_vars", []) or [])))
            if not lines:
                continue
            ops = chain.get("ops", {}) or {}
            chains.append(ChainRecord(
                integral=integral,
                chain_id=chain["chain_id"],
                lines=lines,
                signal_class=chain.get("signal_class", "cancellation_cascade"),
                max_cond=float(chain.get("max_cond", 0.0) or 0.0),
                max_rel_err=float(chain.get("max_rel_err", 0.0) or 0.0),
                predicted_rel_err_if_float=float(
                    chain.get("predicted_rel_err_if_float", 0.0) or 0.0),
                op_count=int(sum(ops.values())),
                n=int(chain.get("n", 0) or 0),
                variables=list(chain.get("region_local_vars", []) or [])))
    return chains, {"n_chains": len(chains)}


def _region_vars(region: dict) -> list[str]:
    """The region-local *reads* for a single-span region (Q1 variable set).

    Prefer ``region_local_vars`` — the tight set of source vars used as direct
    leaf operands at the line — over ``prov_vars`` (the full transitive
    provenance union).  ff_integrator / dd_integrator want the region-local set,
    not the DAG closure (see HANDOFF.md: consumer migration).  Falls back to
    ``prov_vars`` for reports predating the ``region_local_vars`` field.
    """
    if "region_local_vars" in region:
        return list(region.get("region_local_vars") or [])
    return list(region.get("prov_vars", []) or [])


def _one_record(integral: str, region: dict, file: str, line: int) -> RegionRecord:
    ops = region.get("ops", {}) or {}
    return RegionRecord(
        integral=integral,
        target=RegionTarget(file=file, line_start=line, line_end=line,
                            variables=_region_vars(region)),
        signal_class=region.get("signal_class", "stable"),
        max_cond=float(region.get("max_cond", 0.0) or 0.0),
        max_rel_err=float(region.get("max_rel_err", 0.0) or 0.0),
        predicted_rel_err_if_float=float(region.get("predicted_rel_err_if_float", 0.0) or 0.0),
        op_count=int(sum(ops.values())),
        n=int(region.get("n", 0) or 0),
        integrals=[integral],
    )


def _merge_by_line(raw: list[tuple[str, dict, str, int]]) -> list[RegionRecord]:
    """Collapse entries sharing (file, line) into one worst-case code region."""
    by_key: dict[tuple[str, int, int], RegionRecord] = {}
    order: list[tuple[str, int, int]] = []
    for integral, region, file, line in raw:
        rec = _one_record(integral, region, file, line)
        key = rec.key
        if key not in by_key:
            by_key[key] = rec
            order.append(key)
            continue
        cur = by_key[key]
        # representative integral = highest-conditioning contributor
        if rec.max_cond > cur.max_cond:
            cur.integral = rec.integral
        cur.integrals.append(integral)
        cur.max_cond = max(cur.max_cond, rec.max_cond)
        cur.max_rel_err = max(cur.max_rel_err, rec.max_rel_err)
        # worst-case float safety: unsafe if unsafe in ANY integral
        cur.predicted_rel_err_if_float = max(
            cur.predicted_rel_err_if_float, rec.predicted_rel_err_if_float)
        cur.op_count = max(cur.op_count, rec.op_count)
        cur.n = max(cur.n, rec.n)
        if _SEVERITY.get(rec.signal_class, 0) > _SEVERITY.get(cur.signal_class, 0):
            cur.signal_class = rec.signal_class
        # union variables preserving order
        seen = set(cur.target.variables)
        for v in rec.target.variables:
            if v not in seen:
                cur.target.variables.append(v)
                seen.add(v)
    return [by_key[k] for k in order]
