"""Ranking function — two class-driven queues (design: "Ranking function").

Correctness queue (fixed 4-tier order, drained first):

  1. All ``local_cancellation`` regions (cond > 1e15 by construction).
  2. ``cancellation_cascade`` regions with ``max_rel_err > 10^-tolerance``.
  3. ``log_near_root`` regions with ``max_rel_err > 10^-tolerance``.
  4. ``stable`` regions that surprisingly show ``max_rel_err > 10^-tolerance``.

Speedup queue: ``stable`` regions demotable to a cheaper-than-baseline rung that
still meets tolerance, ranked biggest-hardware-win first — flop-weighted
throughput when a weight table is available (Wave-3 WI3: ``div``/``log``-heavy
regions lead), else raw ``op_count`` descending.  Admission gates on
``predicted_rel_err_if_ff`` (the *loosest* cheaper
rung: ff ~14 digits < float's ~7, so ``pred_ff <= thr`` subsumes ``pred_float <=
thr``).  At high tolerance (qcdloop's 10) float never clears but ff can, so this
is what actually populates the speedup queue for a ``double→ff`` demotion; at low
tolerance (≤6) both clear and the walk demotes all the way to float.  The
per-step Validator decides how far down the walk actually settles.

Intra-tier order (design leaves it open): ``max_cond`` desc then ``location``
for a deterministic, worst-first walk.  The downstream-leverage tiebreaker
(walk the prov-var DAG) is **deferred** — not implemented here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agents.strategy.characterization import RegionRecord
from agents.strategy.models import (
    SIGNAL_CANCELLATION_CASCADE, SIGNAL_LOCAL_CANCELLATION,
    SIGNAL_LOG_NEAR_ROOT, SIGNAL_STABLE,
)

# Wave-3 WI3: which weight-table column a demotion target reads.  The speedup
# queue admits on the ff error gate (a ``double->ff`` demotion), so ``ff`` is the
# default target; ``float`` maps to the table's ``native_float`` column.
_WEIGHT_COLUMN = {"ff": "ff", "float": "native_float", "dd": "dd",
                  "double": "native_double"}


def error_threshold(tolerance: float) -> float:
    """The absolute rel-err bar: ``10^-tolerance`` (tolerance=10 → 1e-10)."""
    return 10.0 ** (-tolerance)


def load_flop_weights(path: str | Path | None) -> dict | None:
    """Load the flop-weight table (``ratio_multipliers.json``) for WI3 ordering.

    Returns the parsed dict (columns ``ff`` / ``native_float`` / ``dd`` /
    ``native_double`` → ``{op_kind: flops}``), or ``None`` when the file is
    missing/unreadable — the caller then falls back to raw ``op_count`` ordering.
    Warns once on a missing table so the fallback is never silent.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        print(f"[strategy] flop-weight table not found at {p}; speedup queue falls "
              f"back to op_count ordering (WI3 disabled)", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[strategy] flop-weight table unreadable ({exc}); speedup queue "
              f"falls back to op_count ordering (WI3 disabled)", file=sys.stderr)
        return None


def flop_weighted_score(region: RegionRecord, weights: dict,
                        target_datatype: str = "ff") -> float:
    """Flop-weighted op score of a region for a demotion to ``target_datatype``.

    ``sum(ops[op] * weight[target_column][op])`` — high-throughput regions
    (div/log-heavy) score far above add-heavy regions of equal op_count, so the
    speedup queue front-loads where hardware savings actually live (RATIO_REPORT).
    An op absent from the column defaults to weight 1 (native cost).
    """
    col = weights.get(_WEIGHT_COLUMN.get(target_datatype, target_datatype), {})
    ops = region.ops or {}
    return float(sum(cnt * col.get(op, 1) for op, cnt in ops.items()))


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
                        exclude: set[tuple[str, int, int]] | None = None,
                        *, flop_weights: dict | None = None,
                        target_datatype: str = "ff") -> list[RegionRecord]:
    """Stable regions demotable to a cheaper rung, ranked biggest-win first.

    Admits a region if it is safe at its cheapest reachable rung — ff (the loosest
    cheaper-than-double bar): ``predicted_rel_err_if_ff <= thr``.  A region that
    cannot even meet tolerance in ff can't be demoted at all, so it is excluded.
    ``exclude`` drops region keys already queued for correctness so a region is
    never worked in both phases.

    Ordering (Wave-3 WI3): when ``flop_weights`` is supplied, rank by
    flop-weighted throughput (``div``/``log``-heavy regions first — that is where
    the hardware savings concentrate, per RATIO_REPORT).  Without it, fall back to
    the historical raw ``op_count desc`` — a strict superset-compatible default.
    ``location`` breaks ties for determinism either way.
    """
    thr = error_threshold(tolerance)
    exclude = exclude or set()
    candidates = [
        r for r in regions
        if r.signal_class == SIGNAL_STABLE
        and r.predicted_rel_err_if_ff <= thr
        and r.key not in exclude
    ]
    if flop_weights:
        key = lambda r: (-flop_weighted_score(r, flop_weights, target_datatype),
                         r.target.location)
    else:
        key = lambda r: (-r.op_count, r.target.location)
    return sorted(candidates, key=key)


def build_queues(regions: list[RegionRecord], tolerance: float,
                 *, flop_weights: dict | None = None,
                 target_datatype: str = "ff",
                 ) -> tuple[list[RegionRecord], list[RegionRecord]]:
    """Both queues at once; speedup excludes anything in the correctness queue."""
    correctness = build_correctness_queue(regions, tolerance)
    corr_keys = {r.key for r in correctness}
    speedup = build_speedup_queue(regions, tolerance, exclude=corr_keys,
                                  flop_weights=flop_weights,
                                  target_datatype=target_datatype)
    return correctness, speedup
