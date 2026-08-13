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
import sys
from dataclasses import dataclass, field
from pathlib import Path

from agents.shared.bound_decomposition import (
    chain_row, chain_tightness, chain_predicted_lift)
from agents.strategy.models import RegionTarget


# Signal-class severity — merging a source line across integrals keeps the
# most severe class (a line that is local_cancellation in ANY integral is a
# local_cancellation target).
_SEVERITY = {
    "stable": 0, "log_near_root": 1, "cancellation_cascade": 2,
    "local_cancellation": 3,
}

# One-time fail-open warning when a report predates the
# ``value_range_ok_for_float`` field (Wave-3 WI1).  A missing field defaults to
# ``True`` (do not gate) so an older report never *silently* disables the
# float-rung range guard — it warns once so the report gets regenerated.
_warned_missing_range_flag = False


def _range_ok_for_float(region: dict) -> bool:
    """Read ``value_range_ok_for_float`` with fail-open default ``True``.

    Reports predating the field (Wave-3 WI1) carry no range signal; defaulting to
    ``True`` means the range guard does not fire — the historical behavior.
    Warns once so the omission is visible and future reports get regenerated.

    SCOPE (wider than the name).  The reducer computes this as
    ``abs_val_min >= FLT_MIN_NORMAL and abs_val_max <= FLT_MAX``, so it measures
    whether the region's values fit **FP32's exponent range** — it is not specific
    to the ``float`` rung.  Every fp32-family rung inherits that range regardless of
    how many FP32 words it stacks (``float`` 1x, ``ff`` 2x, ``qf`` 4x), so this one
    flag governs all three; see ``models.FP32_FAMILY``.  The field name is retained
    for report-schema compatibility with existing characterization runs.

    Note also what it does NOT measure: it bounds the region's own |value|, so it
    catches the value itself overflowing or falling below FP32's smallest normal.
    It does not model **low-limb underflow** — a value comfortably inside FP32 range
    whose 3rd/4th limbs nonetheless fall under FLT_MIN_NORMAL and flush to zero,
    costing precision (not range) on the multi-word rungs.  That failure mode is
    real (the qf constant tables show it in the ~1e-35 tail) but unmodelled here.
    """
    global _warned_missing_range_flag
    if "value_range_ok_for_float" not in region:
        if not _warned_missing_range_flag:
            _warned_missing_range_flag = True
            print("[strategy] report lacks `value_range_ok_for_float`; float-rung "
                  "range guard fails open (default true) — regenerate the report "
                  "to enable the WI1 prune", file=sys.stderr)
        return True
    return bool(region.get("value_range_ok_for_float"))


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
    predicted_rel_err_if_ff: float
    op_count: int
    n: int
    variables: list[str] = field(default_factory=list)
    value_range_ok_for_float: bool = True
    ops: dict[str, int] = field(default_factory=dict)
    # Phase 2f: first-order bound decomposition (agents.shared.bound_decomposition).
    # ``max_sensitivity = cond * amp`` is precision-invariant; ``tightness`` =
    # predicted_if_double/measured (COMPUTED band gates chain-dd enqueue);
    # ``predicted_lift`` = digits recovered double->dd (ranks the chain-dd tier).
    max_sensitivity: float = 0.0
    tightness: float | None = None
    predicted_lift: float = 0.0

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
            predicted_rel_err_if_ff=self.predicted_rel_err_if_ff,
            op_count=self.op_count, n=self.n, integrals=[self.integral],
            value_range_ok_for_float=self.value_range_ok_for_float,
            ops=dict(self.ops))


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
    predicted_rel_err_if_ff: float
    op_count: int
    n: int
    integrals: list[str] = field(default_factory=list)
    # Wave-3 WI1: FP32 exponent-range safety flag (fail-open true).  A region
    # flagged false MUST NOT be walked to ANY fp32-family rung — float, ff or qf,
    # which all share FP32's exponent range — in either walk direction; the error
    # model (`predicted_rel_err_if_*`) is blind to over/underflow.  (It was
    # originally read as float-only, which let a rejected float fall back to the
    # equally range-limited ff.)  See _range_ok_for_float for the exact measurement.
    value_range_ok_for_float: bool = True
    # Wave-3 WI3: per-op dynamic mix (op_kind -> count).  ``op_count`` is its
    # sum; the mix drives the flop-weighted speedup ordering (div/log ≫ add).
    ops: dict[str, int] = field(default_factory=dict)

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
            ops = _op_mix(chain)
            pred_float = float(chain.get("predicted_rel_err_if_float", 0.0) or 0.0)
            # Phase 2f: decompose the raw chain via the shared bound arithmetic so
            # tightness / predicted-lift match the analysis script exactly.
            row = chain_row(chain)
            chains.append(ChainRecord(
                integral=integral,
                chain_id=chain["chain_id"],
                lines=lines,
                signal_class=chain.get("signal_class", "cancellation_cascade"),
                max_cond=float(chain.get("max_cond", 0.0) or 0.0),
                max_rel_err=float(chain.get("max_rel_err", 0.0) or 0.0),
                predicted_rel_err_if_float=pred_float,
                predicted_rel_err_if_ff=_pred_ff(chain, pred_float),
                op_count=int(sum(ops.values())),
                n=int(chain.get("n", 0) or 0),
                variables=list(chain.get("region_local_vars", []) or []),
                value_range_ok_for_float=_range_ok_for_float(chain),
                ops=ops,
                max_sensitivity=float(chain.get("max_sensitivity", 0.0) or 0.0),
                tightness=chain_tightness(row),
                predicted_lift=chain_predicted_lift(row)))
    return chains, {"n_chains": len(chains)}


def _pred_ff(entry: dict, pred_float: float) -> float:
    """Read ``predicted_rel_err_if_ff`` from a region/chain dict.

    Reports predating the reducer's ff signal (report_1k/100k) carry only
    ``predicted_rel_err_if_float``.  Fall back to the float prediction — a
    conservative upper bound (ff is never *worse* than float), so a stale report
    never admits an ff speedup it can't actually make.  Run the backfill utility
    (``agents/shared/backfill_ff_prediction.py``) to compute the true, tighter ff
    value on such reports.
    """
    val = entry.get("predicted_rel_err_if_ff")
    return float(val) if val is not None else pred_float


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


def _op_mix(region: dict) -> dict[str, int]:
    """Per-op dynamic counts (op_kind -> int), sanitized (Wave-3 WI3)."""
    ops = region.get("ops", {}) or {}
    return {str(k): int(v) for k, v in ops.items()}


def _one_record(integral: str, region: dict, file: str, line: int) -> RegionRecord:
    ops = _op_mix(region)
    pred_float = float(region.get("predicted_rel_err_if_float", 0.0) or 0.0)
    return RegionRecord(
        integral=integral,
        target=RegionTarget(file=file, line_start=line, line_end=line,
                            variables=_region_vars(region)),
        signal_class=region.get("signal_class", "stable"),
        max_cond=float(region.get("max_cond", 0.0) or 0.0),
        max_rel_err=float(region.get("max_rel_err", 0.0) or 0.0),
        predicted_rel_err_if_float=pred_float,
        predicted_rel_err_if_ff=_pred_ff(region, pred_float),
        op_count=int(sum(ops.values())),
        n=int(region.get("n", 0) or 0),
        integrals=[integral],
        value_range_ok_for_float=_range_ok_for_float(region),
        ops=ops,
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
        # worst-case float/ff safety: unsafe if unsafe in ANY integral
        cur.predicted_rel_err_if_float = max(
            cur.predicted_rel_err_if_float, rec.predicted_rel_err_if_float)
        cur.predicted_rel_err_if_ff = max(
            cur.predicted_rel_err_if_ff, rec.predicted_rel_err_if_ff)
        cur.op_count = max(cur.op_count, rec.op_count)
        cur.n = max(cur.n, rec.n)
        # WI1: float range-unsafe if unsafe in ANY integral (worst-case safety).
        cur.value_range_ok_for_float = (
            cur.value_range_ok_for_float and rec.value_range_ok_for_float)
        # WI3: worst-case per-op mix (element-wise max across integrals).
        for op, cnt in rec.ops.items():
            cur.ops[op] = max(cur.ops.get(op, 0), cnt)
        if _SEVERITY.get(rec.signal_class, 0) > _SEVERITY.get(cur.signal_class, 0):
            cur.signal_class = rec.signal_class
        # union variables preserving order
        seen = set(cur.target.variables)
        for v in rec.target.variables:
            if v not in seen:
                cur.target.variables.append(v)
                seen.add(v)
    return [by_key[k] for k in order]
