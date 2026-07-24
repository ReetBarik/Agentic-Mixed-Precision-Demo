"""Reconstruct a Patcher ``RemediationIntent`` from a solver Candidate (Phase 2e).

The solver rides on the *existing* float/ff/dd rungs the fan-out already measured
(no new patch shapes — PLAN 2e §Do NOT), so a candidate maps back to exactly the
intent Strategy would have emitted for that ``(region, rung)`` cell: a
``double-to-<rung>`` transition, ``via`` from the measurement, with the region's
source-local variables looked up from the (filtered) characterization report.
"""

from __future__ import annotations

from agents.solver.queue import Candidate
from agents.strategy.models import RegionTarget, RemediationIntent


def parse_region_id(region_id: str) -> tuple[str, int, int]:
    """``"file:line"`` or ``"file:start-end"`` -> ``(file, line_start, line_end)``.

    The file part may itself contain ':' on exotic paths, so split on the LAST
    colon (region ids are ``<path>:<line-spec>``).
    """
    path, _, line_spec = region_id.rpartition(":")
    if not path:
        raise ValueError(f"malformed region_id {region_id!r} (no ':')")
    if "-" in line_spec:
        a, _, b = line_spec.partition("-")
        return path, int(a), int(b)
    ln = int(line_spec)
    return path, ln, ln


def region_variables(report_regions: dict, region_id: str) -> list[str]:
    """The region-local source reads for ``region_id`` from a report's regions map.

    Prefers ``region_local_vars`` (tight leaf-operand set) over ``prov_vars``
    (transitive provenance), matching Strategy ``_region_vars``.  Empty when the
    region is absent (e.g. a chain representative) — the fan-out derives reads
    from source anyway (Phase 2c), so variables are advisory here.
    """
    reg = (report_regions or {}).get(region_id) or {}
    if "region_local_vars" in reg:
        return list(reg.get("region_local_vars") or [])
    return list(reg.get("prov_vars") or [])


def intent_from_candidate(cand: Candidate, *, variables: list[str],
                          rationale_id: str) -> RemediationIntent:
    """Build the ``RemediationIntent`` the Patcher applies for this candidate.

    ``current_precision`` is always ``"double"``: under first-accept-per-region
    layering a region is patched at most once, from the double baseline, so the
    candidate's ``kind`` (``double-to-<rung>``) is consistent with the tree state.
    """
    file, lo, hi = parse_region_id(cand.region_id)
    target = RegionTarget(file=file, line_start=lo, line_end=hi,
                          variables=list(variables))
    return RemediationIntent(
        target=target,
        kind=cand.kind,
        intent=cand.intent,
        current_precision="double",
        rationale_id=rationale_id,
        via=cand.via,
    )
