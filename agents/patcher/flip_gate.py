"""Precision-flip acceptance gate (Phase-1 deliverable 5; Phase-2 deliverable 6).

Decision 3 (Reet): the acceptance rule is **uniform across every integral** — a flip
candidate is accepted iff

    (1) its per-integral flip TU **built** clean, AND
    (2) its ``candidate_digits - baseline_digits`` clears the acceptance threshold for the
        flip's *direction* versus the raw-double baseline.

The threshold shape depends on :class:`LiftDirection`, because upshift and downshift make
opposite promises about accuracy:

* **UPSHIFT** (Phase-1, double→dd): the flip *buys accuracy*, so it must **gain** digits —
  ``lift > margin`` (default ``margin=0.0`` → strictly positive lift).  A dd-insufficient
  integral (B15/B16/BIN*) that builds clean but does not lift (its cancellation exceeds dd's
  budget, so the narrowed result is no more accurate than raw double) is **rejected** by
  rule (2) — exactly the false-positive guard the design (§3, §5.3) and STOP #A demand.
  B14 (already ~13.19 digits) is rejected the same way, with no B14-specific code.
* **DOWNSHIFT** (Phase-2, double→float): the flip *buys speed* and only promises to
  **preserve** accuracy, so it must **not lose** digits — ``lift >= -margin`` (default
  ``margin=0.0`` → accept a precision-neutral float, ``lift == 0.0``, and any incidental
  gain; reject only a genuine precision *loss*).  A well-conditioned integral holds its
  digits at float and is accepted; an ill-conditioned one loses digits and is rejected back
  to raw double.

No special-case buckets in either direction: the same predicate decides B10, B14, and B16
under UPSHIFT and B1..B11 under DOWNSHIFT.  A rejected integral falls back to the raw-double
baseline (Decision 4) — the caller routes it to the baseline binary.

The gate is a pure decision over already-measured numbers; it runs no build and no
measurement itself.  ``margin`` is a non-negative minimum-lift (upshift) / maximum-loss
(downshift) tolerance, kept so a caller can demand a stricter threshold without changing the
rule's shape.  It is NOT a special-case bucket — it applies uniformly to every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LiftDirection(str, Enum):
    """Which way a flip moves precision, selecting the acceptance-threshold shape.

    * :data:`UPSHIFT` — the flip targets a *wider* precision (Phase-1 double→dd); it must
      **gain** digits (``lift > margin``).
    * :data:`DOWNSHIFT` — the flip targets a *narrower* precision (Phase-2 double→float); it
      must **preserve** digits (``lift >= -margin``), trading accuracy-neutrality for speed.
    """

    UPSHIFT = "upshift"
    DOWNSHIFT = "downshift"


@dataclass(frozen=True)
class GateInputs:
    """The measured facts the gate decides over, for one integral."""

    integral: str
    built: bool
    baseline_digits: float | None      # raw-double precise digits (None = no measurement)
    candidate_digits: float | None     # flip (dd-internal, narrowed) precise digits


@dataclass(frozen=True)
class GateDecision:
    """The gate's verdict for one integral."""

    integral: str
    accept: bool
    lift: float | None                 # candidate - baseline (None if unmeasurable)
    reason: str


def evaluate(inp: GateInputs, *, margin: float = 0.0,
             direction: LiftDirection = LiftDirection.UPSHIFT) -> GateDecision:
    """Apply the uniform build-AND-lift rule (Decision 3) to one integral.

    Accept iff ``inp.built``, both digit measures exist, and the ``lift`` clears the
    threshold for ``direction``:

    * ``UPSHIFT`` — ``lift > margin`` (default ``margin=0.0`` → strictly positive gain).
    * ``DOWNSHIFT`` — ``lift >= -margin`` (default ``margin=0.0`` → precision-preserving:
      a float that holds accuracy, ``lift == 0.0``, is accepted; only a genuine loss is
      rejected).

    Every other outcome — build failed or a missing measurement — is a reject with a
    specific reason.  No per-integral special cases: the same predicate decides every
    integral within a direction.
    """
    if not inp.built:
        return GateDecision(inp.integral, accept=False, lift=None,
                            reason="build_failed: per-integral flip TU did not compile")
    if inp.baseline_digits is None or inp.candidate_digits is None:
        return GateDecision(
            inp.integral, accept=False, lift=None,
            reason=f"unmeasurable: baseline={inp.baseline_digits} "
                   f"candidate={inp.candidate_digits} (no digit measure)")
    lift = inp.candidate_digits - inp.baseline_digits
    if direction is LiftDirection.DOWNSHIFT:
        # Speed flip: accept iff accuracy is preserved (lift not below -margin).
        if lift >= -margin:
            return GateDecision(
                inp.integral, accept=True, lift=lift,
                reason=f"accept: built + downshift preserves precision, lift {lift:+.4f} "
                       f"digits ({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f}) "
                       f">= {-margin}")
        return GateDecision(
            inp.integral, accept=False, lift=lift,
            reason=f"precision_loss: built but downshift lift {lift:+.4f} digits "
                   f"({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f}) < {-margin} "
                   f"— falls back to raw double (float too narrow / ill-conditioned)")
    # UPSHIFT: accuracy flip — must gain digits.
    if lift > margin:
        return GateDecision(
            inp.integral, accept=True, lift=lift,
            reason=f"accept: built + lift {lift:+.4f} digits "
                   f"({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f}) > {margin}")
    return GateDecision(
        inp.integral, accept=False, lift=lift,
        reason=f"no_lift: built but lift {lift:+.4f} digits "
               f"({inp.baseline_digits:.4f} -> {inp.candidate_digits:.4f}) <= {margin} "
               f"— falls back to raw double (dd-insufficient / already-accurate)")


def evaluate_all(inputs, *, margin: float = 0.0,
                 direction: LiftDirection = LiftDirection.UPSHIFT) -> dict:
    """Evaluate every integral; return ``{integral: GateDecision}``.  Convenience over
    :func:`evaluate` for a whole measurement pass — the acceptance rule is unchanged."""
    return {i.integral: evaluate(i, margin=margin, direction=direction) for i in inputs}
