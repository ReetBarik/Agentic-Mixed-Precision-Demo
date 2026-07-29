"""Phase-1 template-argument promotion — acceptance gate (deliverable 5).

Decision 3 (Reet): the acceptance rule is **uniform across every integral** — a flip
candidate is accepted iff

    (1) its per-integral dd TU **built** clean, AND
    (2) it **lifts** the integral's precise digits by strictly more than zero versus the
        raw-double baseline (``candidate_digits - baseline_digits > 0.0``).

No special-case buckets: a dd-insufficient integral (B15/B16/BIN*) that *builds* clean but
does not *lift* (its cancellation exceeds dd's ~32-digit budget, so the narrowed result is
no more accurate than raw double) is **rejected** by rule (2) — exactly the false-positive
guard the design (§3, §5.3) and STOP #A demand.  B14 (already ~13.19 digits, nothing to
lift) is rejected the same way, with no B14-specific code.  A rejected integral falls back
to the raw-double baseline (Decision 4) — the caller routes it to the baseline binary.

The gate is a pure decision over already-measured numbers; it runs no build and no
measurement itself.  The margin is a parameter (default ``0.0`` = "strictly positive"),
kept so a caller can demand a minimum lift without changing the rule's shape.  It is NOT a
special-case bucket — it applies uniformly to every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass


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


def evaluate(inp: GateInputs, *, margin: float = 0.0) -> GateDecision:
    """Apply the uniform build-AND-lift rule (Decision 3) to one integral.

    Accept iff ``inp.built`` and both digit measures exist and
    ``candidate_digits - baseline_digits > margin`` (default ``margin=0.0`` → strictly
    positive lift).  Every other outcome — build failed, a missing measurement, or a
    non-positive lift — is a reject with a specific reason.  No per-integral special
    cases: the same predicate decides B10, B14, and B16 alike.
    """
    if not inp.built:
        return GateDecision(inp.integral, accept=False, lift=None,
                            reason="build_failed: per-integral dd TU did not compile")
    if inp.baseline_digits is None or inp.candidate_digits is None:
        return GateDecision(
            inp.integral, accept=False, lift=None,
            reason=f"unmeasurable: baseline={inp.baseline_digits} "
                   f"candidate={inp.candidate_digits} (no digit measure)")
    lift = inp.candidate_digits - inp.baseline_digits
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


def evaluate_all(inputs, *, margin: float = 0.0) -> dict:
    """Evaluate every integral; return ``{integral: GateDecision}``.  Convenience over
    :func:`evaluate` for a whole measurement pass — the acceptance rule is unchanged."""
    return {i.integral: evaluate(i, margin=margin) for i in inputs}
