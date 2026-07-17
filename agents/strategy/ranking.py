"""Ranking function — two class-driven queues (design: "Ranking function").

Correctness queue (fixed 4-tier order, drained first):

  1. All ``local_cancellation`` regions (cond > 1e15 by construction).
  2. ``cancellation_cascade`` regions with ``max_rel_err > 10^-tolerance``.
  3. ``log_near_root`` regions with ``max_rel_err > 10^-tolerance``.
  4. ``stable`` regions that surprisingly show ``max_rel_err > 10^-tolerance``.

Speedup queue: ``stable`` regions whose ``predicted_rel_err_if_float`` is at or
below the tolerance bar (float alone already meets tolerance), ranked by
``op_count`` descending — biggest hardware win first.

Intra-tier order (design leaves it open): ``max_cond`` desc then ``location``
for a deterministic, worst-first walk.  The downstream-leverage tiebreaker
(walk the prov-var DAG) is **deferred** — not implemented here.
"""

from __future__ import annotations

from agents.strategy.characterization import RegionRecord
from agents.strategy.models import (
    SIGNAL_CANCELLATION_CASCADE, SIGNAL_LOCAL_CANCELLATION,
    SIGNAL_LOG_NEAR_ROOT, SIGNAL_STABLE,
)


def error_threshold(tolerance: float) -> float:
    """The absolute rel-err bar: ``10^-tolerance`` (tolerance=10 → 1e-10)."""
    return 10.0 ** (-tolerance)


def _correctness_sort_key(r: RegionRecord):
    # worst conditioning first; location breaks ties deterministically
    return (-r.max_cond, r.target.location)


def build_correctness_queue(regions: list[RegionRecord], tolerance: float) -> list[RegionRecord]:
    """The 4-tier correctness queue in fixed tier order."""
    thr = error_threshold(tolerance)

    tier1 = [r for r in regions if r.signal_class == SIGNAL_LOCAL_CANCELLATION]
    tier2 = [r for r in regions
             if r.signal_class == SIGNAL_CANCELLATION_CASCADE and r.max_rel_err > thr]
    tier3 = [r for r in regions
             if r.signal_class == SIGNAL_LOG_NEAR_ROOT and r.max_rel_err > thr]
    tier4 = [r for r in regions
             if r.signal_class == SIGNAL_STABLE and r.max_rel_err > thr]

    queue: list[RegionRecord] = []
    for tier in (tier1, tier2, tier3, tier4):
        queue.extend(sorted(tier, key=_correctness_sort_key))
    return queue


def build_speedup_queue(regions: list[RegionRecord], tolerance: float,
                        exclude: set[tuple[str, int, int]] | None = None) -> list[RegionRecord]:
    """Stable regions safe to demote, ranked by op_count descending.

    ``exclude`` drops region keys already queued for correctness so a region is
    never worked in both phases.
    """
    thr = error_threshold(tolerance)
    exclude = exclude or set()
    candidates = [
        r for r in regions
        if r.signal_class == SIGNAL_STABLE
        and r.predicted_rel_err_if_float <= thr
        and r.key not in exclude
    ]
    # op_count desc; location tiebreak for determinism
    return sorted(candidates, key=lambda r: (-r.op_count, r.target.location))


def build_queues(regions: list[RegionRecord], tolerance: float,
                 ) -> tuple[list[RegionRecord], list[RegionRecord]]:
    """Both queues at once; speedup excludes anything in the correctness queue."""
    correctness = build_correctness_queue(regions, tolerance)
    corr_keys = {r.key for r in correctness}
    speedup = build_speedup_queue(regions, tolerance, exclude=corr_keys)
    return correctness, speedup
