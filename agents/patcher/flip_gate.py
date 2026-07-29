"""Precision-flip acceptance gate (Phase-1 deliverable 5; Phase-2 deliverable 6).

**Tolerance contract.** The pipeline's acceptance criterion is a single user-supplied
*minimum precise-digit bar* — :attr:`StrategyConfig.tolerance` (default 10.0).  "Acceptable
accuracy" means exactly one thing: the delivered worst-case (p100) precise digits clear that
bar.  The gate decides against ``tolerance``, not against the raw-double baseline: a flip is
worth producing when it *reaches the bar*, and precision headroom above the bar is not a
resource to be preserved.  (The old baseline-preserving rule wasted that headroom and — as
``PHASE_2_FF_LANDED_2026-07-29.md`` §3.3 showed — rejected every downshift on workloads whose
double baseline already sits above the bar; that framing is gone.)

``candidate_digits`` and ``baseline_digits`` are p100 = the *minimum* precise digits over all
samples/components (what ``_min_digits`` computes in the L-measure).  The bar is checked
against that worst case, never a mean.

The decision shape depends on :class:`LiftDirection`:

* **UPSHIFT** (Phase-1, double→dd/wider) — the flip *buys accuracy*:

  1. ``baseline_digits >= bar`` → **no_flip_needed**: raw double already clears the bar, so
     the flip is a no-op.  This is a *terminal-good* state (nothing to fix) but it is **not an
     accept** — it produces no flip TU.  Distinguish it downstream via
     :attr:`GateDecision.no_flip_needed`.
  2. built AND ``candidate_digits >= bar`` → **accept**: the flip clears the bar (whether or
     not it also lifts).  Clearing tolerance is sufficient.
  3. built AND ``lift > margin`` → **accept**: strict improvement, still under the bar —
     best-effort progress toward tolerance.
  4. otherwise → **reject**: build failed, or no lift and the bar not cleared.

* **DOWNSHIFT** (Phase-2, double→float/ff) — the flip *buys speed* and only has to stay at
  the bar:

  1. built AND ``candidate_digits >= bar`` → **accept**: the bar is cleared; a negative lift
     is fine, that precision was headroom.
  2. otherwise → **reject**: build failed, or the candidate dropped below the bar.

  The two-target downshift walk (float, then ff) evaluates each target against the bar
  independently; the caller routes to the first target that accepts.

No per-integral special cases in either direction: the same predicate decides every
candidate.  A rejected/no-op integral falls back to the raw-double baseline — the caller
routes it to the baseline binary.

``margin`` is a non-negative strictness buffer layered *on top of* the tolerance bar: the
effective bar is ``tolerance + margin`` (and UPSHIFT's strict-improvement branch demands
``lift > margin``).  Default ``margin=0.0`` → the bar is exactly ``tolerance``.  It is not a
special-case bucket — it applies uniformly.

The gate is a pure decision over already-measured numbers; it runs no build and no
measurement itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiftDirection(str, Enum):
    """Which way a flip moves precision, selecting the acceptance rule.

    * :data:`UPSHIFT` — a *wider* target (Phase-1 double→dd): accept when it reaches the
      tolerance bar, or (below the bar) when it strictly gains digits; no-op when the
      baseline already clears the bar.
    * :data:`DOWNSHIFT` — a *narrower* target (Phase-2 double→float/ff): accept iff it still
      clears the tolerance bar, trading headroom for speed.
    """

    UPSHIFT = "upshift"
    DOWNSHIFT = "downshift"


# Reason-string prefixes — stable tokens callers can match on.
_R_NO_FLIP = "no_flip_needed"
_R_ACCEPT = "accept"
_R_REJECT_BUILD = "build_failed"
_R_REJECT_UNMEASURABLE = "unmeasurable"
_R_REJECT_BELOW_TOL = "below_tolerance"


@dataclass(frozen=True)
class GateInputs:
    """The measured facts the gate decides over, for one integral.

    ``tolerance`` is REQUIRED — every caller has it via ``StrategyConfig.tolerance``.  There
    is no silent default: a missing tolerance is a construction error (fail loud), by design.
    """

    integral: str
    built: bool
    baseline_digits: float | None      # raw-double p100 precise digits (None = no measure)
    candidate_digits: float | None     # flip p100 precise digits (None = not built/no measure)
    tolerance: float                   # the acceptance bar (required)


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for one integral.

    ``accept`` is True only when the flip TU should be produced.  ``no_flip_needed`` is a
    *separate* terminal-good state (the baseline already clears the bar): ``accept`` is False
    but the integral is not a failure — it simply stays at raw double as a no-op.  Both accept
    and no_flip_needed are "good"; they differ in whether a flip TU is emitted.
    """

    integral: str
    accept: bool
    lift: float | None                 # candidate - baseline (None if unmeasurable)
    reason: str
    no_flip_needed: bool = False

    @property
    def terminal_good(self) -> bool:
        """True for either good outcome — accepted flip OR baseline already at the bar."""
        return self.accept or self.no_flip_needed


def evaluate(inp: GateInputs, *, margin: float = 0.0,
             direction: LiftDirection = LiftDirection.UPSHIFT) -> GateDecision:
    """Apply the tolerance-contract acceptance rule to one integral.

    The effective bar is ``inp.tolerance + margin``.  See the module docstring for the full
    per-direction decision.  ``inp.tolerance`` is required; passing ``None`` fails loud.
    """
    if inp.tolerance is None:
        raise ValueError(f"{inp.integral}: tolerance is required (no silent default)")
    bar = inp.tolerance + margin

    if direction is LiftDirection.DOWNSHIFT:
        return _evaluate_downshift(inp, bar)
    return _evaluate_upshift(inp, bar, margin)


def _evaluate_upshift(inp: GateInputs, bar: float, margin: float) -> GateDecision:
    # 1. baseline already clears the bar -> no-op (terminal-good, not an accept).
    if inp.baseline_digits is not None and inp.baseline_digits >= bar:
        return GateDecision(
            inp.integral, accept=False, lift=None, no_flip_needed=True,
            reason=f"{_R_NO_FLIP}: raw double already clears tolerance "
                   f"({inp.baseline_digits:.4f} >= {bar}) — no flip produced")
    # 2. must have built to accept anything below.
    if not inp.built:
        return GateDecision(inp.integral, accept=False, lift=None,
                            reason=f"{_R_REJECT_BUILD}: per-integral flip TU did not compile")
    if inp.baseline_digits is None or inp.candidate_digits is None:
        return GateDecision(
            inp.integral, accept=False, lift=None,
            reason=f"{_R_REJECT_UNMEASURABLE}: baseline={inp.baseline_digits} "
                   f"candidate={inp.candidate_digits} (no digit measure)")
    lift = inp.candidate_digits - inp.baseline_digits
    # 3. candidate clears the bar -> accept (with or without lift).
    if inp.candidate_digits >= bar:
        return GateDecision(
            inp.integral, accept=True, lift=lift,
            reason=f"{_R_ACCEPT}: built + candidate clears tolerance "
                   f"({inp.candidate_digits:.4f} >= {bar}), lift {lift:+.4f} digits "
                   f"({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f})")
    # 4. below the bar but a strict improvement -> accept (best-effort progress).
    if lift > margin:
        return GateDecision(
            inp.integral, accept=True, lift=lift,
            reason=f"{_R_ACCEPT}: built + strict lift {lift:+.4f} digits "
                   f"({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f}) > {margin} "
                   f"(still under tolerance {bar}, best-effort progress)")
    # 5. below the bar and no lift -> reject.
    return GateDecision(
        inp.integral, accept=False, lift=lift,
        reason=f"{_R_REJECT_BELOW_TOL}: built but candidate {inp.candidate_digits:.4f} "
               f"< tolerance {bar} and lift {lift:+.4f} <= {margin} "
               f"— falls back to raw double (dd-insufficient)")


def _evaluate_downshift(inp: GateInputs, bar: float) -> GateDecision:
    if not inp.built:
        return GateDecision(inp.integral, accept=False, lift=None,
                            reason=f"{_R_REJECT_BUILD}: per-integral flip TU did not compile")
    if inp.candidate_digits is None:
        return GateDecision(
            inp.integral, accept=False, lift=None,
            reason=f"{_R_REJECT_UNMEASURABLE}: candidate={inp.candidate_digits} "
                   f"(no digit measure)")
    lift = (inp.candidate_digits - inp.baseline_digits
            if inp.baseline_digits is not None else None)
    if inp.candidate_digits >= bar:
        return GateDecision(
            inp.integral, accept=True, lift=lift,
            reason=f"{_R_ACCEPT}: built + downshift clears tolerance "
                   f"({inp.candidate_digits:.4f} >= {bar}) — precision above bar is headroom")
    return GateDecision(
        inp.integral, accept=False, lift=lift,
        reason=f"{_R_REJECT_BELOW_TOL}: built but downshift candidate "
               f"{inp.candidate_digits:.4f} < tolerance {bar} "
               f"— falls back to raw double (target too narrow / ill-conditioned)")


def evaluate_all(inputs, *, margin: float = 0.0,
                 direction: LiftDirection = LiftDirection.UPSHIFT) -> dict:
    """Evaluate every integral; return ``{integral: GateDecision}``.  Convenience over
    :func:`evaluate` for a whole measurement pass — the acceptance rule is unchanged.  Each
    :class:`GateInputs` carries its own (required) ``tolerance``."""
    return {i.integral: evaluate(i, margin=margin, direction=direction) for i in inputs}
