"""Whole-TU-only Strategy walk (``strategy_mode="tu_only"``, Phase-2.1).

The Phase-2.1 walk is deliberately *mechanical*: it enumerates the integrals from
the characterization report and, for each, drives a per-precision **whole-TU
precision flip** through an injected measure provider (``tu_measure_fn`` on the
state) and the tolerance gate (:mod:`agents.patcher.flip_gate`).  There is **no
Patcher LLM**, no region walk, and no chain walk — those live on the ``"region"``
path (retained for Phase-2.2 region-level demotion, never invoked here).

This is the same recipe the L-measure scripts (``runs/qcdloop/phase1_lmeasure.py``
+ ``phase2_lmeasure.py``) run, lifted into the agentic pipeline so the whole-TU
route is exercised end-to-end by the same Strategy the region walk uses.  The
qcdloop-specific build/oracle/measure is **not** here — it is injected as
``tu_measure_fn`` so this module (and all of ``agents/strategy``) stays generic
(feedback_no_placeholder_patterns).

Provider contract (``tu_measure_fn(integral, target) -> dict``):

    target == "baseline"        -> {"built": bool, "baseline_digits": float|None}
    target in {"dd","float","ff"} -> {"built": bool,
                                      "baseline_digits": float|None,
                                      "candidate_digits": float|None,
                                      "log_tail": str}

The provider owns caching (vanilla baseline + dd oracle + per-group flip builds are
built once and reused across integrals sharing a group).

**Two phases**, mirroring the region walk's correctness→speedup split:

* **Correctness (UPSHIFT).**  For an integral below the bar at double, try ``dd``.
  ``baseline >= bar`` → ``tu_no_flip_needed`` (double already clears; no flip).
  Built + clears bar → ``tu_accepted`` (routed to dd).  Otherwise the integral
  stays at double and the speedup phase still considers float/ff for it.
* **Speedup (DOWNSHIFT).**  For *every* integral, walk ``float`` then ``ff`` in
  order; the first that clears the bar wins (routed to that precision).  ``float``
  is a candidate only when the report's ``predicted_rel_err_if_float`` signal makes
  it plausible (the same ``error_threshold(tolerance)`` gate the region walk uses);
  ``ff`` is always a candidate (the workhorse per prior L-measure runs).

The final routing per integral is the widest accepted correctness precision if the
speedup phase found nothing narrower, else the narrowest accepted speedup precision
(a downshift always beats staying at the correctness precision when it clears the
bar — that is the whole point of the speedup phase).
"""

from __future__ import annotations

from agents.patcher.flip_gate import GateInputs, LiftDirection, evaluate
from agents.strategy.ranking import error_threshold

# Iteration-log statuses — distinct so downstream analysis separates cleanly.
TU_ACCEPTED = "tu_accepted"
TU_REJECTED_BELOW_TOL = "tu_rejected_below_tolerance"
TU_BUILD_FAILED = "tu_build_failed"
TU_NO_FLIP_NEEDED = "tu_no_flip_needed"

# The precision targets, by phase.  dd is the sole correctness (upshift) target;
# float then ff is the speedup (downshift) walk order (cheapest/narrowest first).
_CORRECTNESS_TARGET = "dd"
_SPEEDUP_WALK = ("float", "ff")


def status_for(decision, built: bool) -> str:
    """Map a :class:`~agents.patcher.flip_gate.GateDecision` to a log status."""
    if decision.accept:
        return TU_ACCEPTED
    if decision.no_flip_needed:
        return TU_NO_FLIP_NEEDED
    if not built:
        return TU_BUILD_FAILED
    return TU_REJECTED_BELOW_TOL


def float_is_candidate(pred_rel_err_if_float: float | None, tolerance: float,
                       *, report_prunes: bool = True) -> bool:
    """Whether ``float`` is worth attempting for an integral (report-signal prune).

    Deterministic gate on the report's numeric signal — the same comparison the
    region walk uses (``predicted_rel_err_if_float`` vs ``error_threshold(tol)``,
    see agent.py's ``_float_rung_ok``).  A missing/None signal fails **open** (float
    is attempted) so a stale report never silently drops the float rung.  With
    ``report_prunes`` off (the emergency kill-switch) float is always attempted.
    """
    if not report_prunes:
        return True
    if pred_rel_err_if_float is None:
        return True
    return pred_rel_err_if_float <= error_threshold(tolerance)
