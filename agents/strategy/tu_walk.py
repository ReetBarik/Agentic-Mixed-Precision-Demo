"""Whole-TU-only Strategy walk (``strategy_mode="tu_only"``, Phase-2.1).

The Phase-2.1 walk is deliberately *mechanical*: it enumerates the integrals from
the characterization report and, for each, drives a per-precision **whole-TU
precision flip** through an injected measure provider (``tu_measure_fn`` on the
state) and the tolerance gate (:mod:`agents.patcher.flip_gate`).  There is **no
Patcher LLM**, no region walk, and no chain walk — those live on the ``"region"``
path (retained for Phase-2.2 region-level demotion, never invoked here).

The qcdloop-specific build/oracle/measure is **not** here — it is injected as
``tu_measure_fn`` (the qcdloop provider is ``runs/qcdloop/tu_provider.py``) so
this module (and all of ``agents/strategy``) stays generic
(feedback_no_placeholder_patterns).

Provider contract (``tu_measure_fn(integral, target) -> dict``):

    target == "baseline"               -> {"built": bool, "baseline_digits": float|None}
    target in {"dd","qf","float","ff"} -> {"built": bool,
                                           "baseline_digits": float|None,
                                           "candidate_digits": float|None,
                                           "log_tail": str}

The provider owns caching (vanilla baseline + dd oracle + per-group flip builds are
built once and reused across integrals sharing a group).

**Two phases**, mirroring the region walk's correctness→speedup split:

* **Correctness (UPSHIFT).**  For an integral below the bar at double, walk
  :data:`CORRECTNESS_WALK` — ``qf`` then ``dd`` — and take the FIRST that clears;
  a qf accept short-circuits the dd attempt entirely.  ``baseline >= bar`` →
  ``tu_no_flip_needed`` (double already clears; no flip).  Otherwise the integral
  stays at double and the speedup phase still considers float/ff for it.

  A range-unsafe integral skips the ``qf`` rung and goes straight to ``dd`` — qf is
  fp32-ranged, so it cannot rescue a value that overflows float (models.FP32_FAMILY).
* **Speedup (DOWNSHIFT).**  For *every* integral, walk ``float`` then ``ff`` in
  order; the first that clears the bar wins (routed to that precision).  ``float``
  is a candidate only when the report's ``predicted_rel_err_if_float`` signal makes
  it plausible (the same ``error_threshold(tolerance)`` gate the region walk uses);
  ``ff`` is otherwise always a candidate (the workhorse per prior L-measure runs).

  A range-unsafe integral skips BOTH speedup rungs — float and ff are both
  fp32-family — exactly as it skips ``qf`` on the correctness side.  The
  ``predicted_rel_err`` signal cannot substitute for this: it models ACCURACY and
  is blind to over/underflow.

The final routing per integral is the CHEAPEST accepted correctness precision if the
speedup phase found nothing narrower, else the narrowest accepted speedup precision
(a downshift always beats staying at the correctness precision when it clears the
bar — that is the whole point of the speedup phase).

Cheapest, not widest: with qf and dd both on the correctness walk, "widest accepted"
would prefer dd over qf whenever both were recorded, which inverts the intent — the
point of the qf rung is to avoid paying for dd.  In practice the walk short-circuits
on the first accept so only one correctness precision is ever recorded, but the
tie-break is stated as cheapest so the rule stays correct if that ever changes.
"""

from __future__ import annotations

from agents.patcher.flip_gate import GateInputs, LiftDirection, evaluate
from agents.strategy.ranking import error_threshold

# Iteration-log statuses — distinct so downstream analysis separates cleanly.
TU_ACCEPTED = "tu_accepted"
TU_REJECTED_BELOW_TOL = "tu_rejected_below_tolerance"
TU_BUILD_FAILED = "tu_build_failed"
TU_NO_FLIP_NEEDED = "tu_no_flip_needed"

# The precision targets, by phase — both CHEAPEST-FIRST, first accept wins.
#
# Correctness (upshift) walks qf then dd.  qf (~29 digits, 4xFP32) is the cheap
# alternative to dd on the fp32-heavy silicon this targets, so trying it first
# makes correctness a *cheapest-sufficient* search rather than a jump straight to
# the most expensive rung.  dd remains the backstop: an integral qf cannot carry
# still reaches dd in the same pass.
#
# Walking qf-first also means STOP #ZZ (a dd accept is never downshifted) needs no
# relaxation — there is never a dd accept to undo, because qf is evaluated before
# dd is ever attempted.
#
# Speedup (downshift) walks float then ff, likewise cheapest/narrowest first.
CORRECTNESS_WALK: tuple[str, ...] = ("qf", "dd")
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
