"""Per-target retry walk — the mechanical vocabulary walk (design: "Retry policy").

A ``RetryWalk`` is a small state machine the main loop drives one step at a time:

    walk = RetryWalk(record, mode="correctness", tolerance=10.0)
    while (intent := walk.propose(rationale_id)) is not None:
        # main loop runs intent through Patcher (+ Validator on ok)
        walk.resolve(accepted=..., genuine_reject=...)
    result = walk.result()

``genuine_reject`` means Patcher returned ``ok`` AND the Validator rejected — the
only condition under which a DD attempt proves a physics ceiling (P6a).  A
Patcher-side failure (build/nan/crash/llm) at the DD rung is *not* genuine and
yields ``dd_untested``.

Correctness mode (walk up, per design):

  1. Promote baseline → each higher ladder rung in cost order.  First rung that
     clears → ``cleared`` at that rung.  (In the fixed-report workflow every
     region's baseline is ``double`` so the only precision rung is ``dd``.)
  2. At ``dd`` with a genuine reject → try algorithmic rewrite(s): ``kahan`` for
     cancellation_cascade, the identity catalog for local_cancellation.
  3. A rewrite that clears → ``cleared`` (rewrite kept at baseline precision,
     orthogonal to precision per Q2).  All rewrites reject → accept DD as-is,
     ``dd_ceiling``.
  4. A Patcher failure at the ``dd`` rung (no genuine test) → ``dd_untested``,
     DD not installed, region stays at baseline.

Speedup mode (walk down): demote installed → next cheaper rung; each accept
lowers the installed rung and continues; first reject backs off one and stops
(``settled`` at the last accepted rung).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.strategy.characterization import RegionRecord
from agents.strategy.models import (
    IDENTITY_CATALOG, INTENT_CORRECTNESS, INTENT_SPEEDUP, LADDER,
    SIGNAL_CANCELLATION_CASCADE, SIGNAL_LOCAL_CANCELLATION,
    VIA_PLAIN, VIA_REGIONAL,
    RemediationIntent, TRANSITION_KINDS, next_down,
)


def _ladder_index(precision: str) -> int:
    return LADDER.index(precision)

_DD = "dd"


@dataclass
class WalkResult:
    """Terminal outcome of a per-target walk."""

    status: str                       # cleared | dd_ceiling | dd_untested | settled | exhausted
    final_precision: str
    accepted_intent: RemediationIntent | None = None
    rewrite_accepted: bool = False
    attempted_rewrites: list[str] = field(default_factory=list)
    ceiling_kind: str | None = None   # dd_ceiling | dd_untested | None


def _rewrites_for(signal_class: str) -> list[tuple[str, str | None]]:
    """Ordered (kind, identity) rewrites to try at the DD ceiling for a class."""
    if signal_class == SIGNAL_CANCELLATION_CASCADE:
        return [("reformulate-kahan", None)]
    if signal_class == SIGNAL_LOCAL_CANCELLATION:
        return [("reformulate-identity", ident) for ident in IDENTITY_CATALOG]
    # log_near_root / stable: design specifies no rewrite → straight to ceiling.
    return []


class RetryWalk:
    def __init__(self, record: RegionRecord, mode: str, tolerance: float,
                 baseline: str = "double", floor: str | None = None,
                 float_via: str = VIA_PLAIN):
        if mode not in (INTENT_CORRECTNESS, INTENT_SPEEDUP):
            raise ValueError(f"unknown walk mode {mode!r}")
        if float_via not in (VIA_PLAIN, VIA_REGIONAL):
            raise ValueError(f"unknown float_via {float_via!r}")
        self.record = record
        self.mode = mode
        self.tolerance = tolerance
        self.baseline = baseline
        self.installed = baseline
        # Speedup floor (design "Speedup floor rule"): the lowest precision this
        # region may be demoted to, because a promoted cascade chain still claims
        # one of its lines at that precision.  None → no floor (down to float).
        self.floor = floor
        # How the speedup walk realizes a ``float`` demotion (Wave 2):
        #   VIA_PLAIN    — non-templated region: a bare ``double`` token exists, so
        #                  the single-step ladder (double→ff→float) reaches float via
        #                  the Patcher's plain-edit ``ff-to-float`` rung (historical).
        #   VIA_REGIONAL — template-typed region (no bare ``double`` token): float is
        #                  reachable ONLY by generating a float-specialized shim, so
        #                  the walk tries ``double-to-float`` DIRECTLY (cheapest rung
        #                  first) and falls back to ``double-to-ff`` — both tagged
        #                  ``via="regional"`` so the Patcher routes to the float / ff
        #                  integrators.  Replaces the Wave-1 "settle, skip float"
        #                  gate (CALIBRATION.md §Bug 4).
        # Correctness walks never target float, so this is inert there.
        self.float_via = float_via
        # Regional speedup plan: demotion targets below ``double`` in cost order
        # (cheapest first); the first that the Validator accepts wins.
        self._regional_plan: list[str] = ["float", "ff"]
        self._regional_i = 0

        # correctness: higher rungs reachable from baseline via a supported kind
        base_i = LADDER.index(baseline)
        self._up_targets = [
            lvl for lvl in LADDER[base_i + 1:]
            if f"{baseline}-to-{lvl}" in TRANSITION_KINDS
        ]
        self._up_i = 0
        self._phase = "precision"                     # precision | reformulate
        self._rewrites = _rewrites_for(record.signal_class)
        self._rw_i = 0
        self.attempted_rewrites: list[str] = []

        self._pending: RemediationIntent | None = None   # last proposed, awaiting resolve
        self._pending_is_dd = False
        self._pending_is_rewrite = False
        self._result: WalkResult | None = None

    # -- proposal ---------------------------------------------------------
    def propose(self, rationale_id: str) -> RemediationIntent | None:
        """Next intent to try, or None once the walk has terminated."""
        if self._result is not None:
            return None
        if self._pending is not None:
            raise RuntimeError("propose() called before resolve()")
        intent = (self._propose_correctness(rationale_id)
                  if self.mode == INTENT_CORRECTNESS
                  else self._propose_speedup(rationale_id))
        self._pending = intent
        return intent

    def _propose_correctness(self, rationale_id: str) -> RemediationIntent | None:
        if self._phase == "precision":
            if self._up_i < len(self._up_targets):
                target_level = self._up_targets[self._up_i]
                kind = f"{self.installed}-to-{target_level}"
                self._pending_is_dd = (target_level == _DD)
                self._pending_is_rewrite = False
                return RemediationIntent(
                    target=self.record.target, kind=kind,
                    intent=INTENT_CORRECTNESS, current_precision=self.installed,
                    rationale_id=rationale_id)
            # no more precision rungs → move to reformulate phase
            self._phase = "reformulate"
        # reformulate phase
        if self._rw_i < len(self._rewrites):
            kind, identity = self._rewrites[self._rw_i]
            self._pending_is_dd = False
            self._pending_is_rewrite = True
            return RemediationIntent(
                target=self.record.target, kind=kind, intent=INTENT_CORRECTNESS,
                current_precision=self.installed, rationale_id=rationale_id,
                identity=identity)
        # nothing left to try → terminal (ceiling or exhausted, decided in resolve)
        self._finish_correctness_no_more()
        return None

    def _propose_speedup(self, rationale_id: str) -> RemediationIntent | None:
        if self.float_via == VIA_REGIONAL:
            return self._propose_speedup_regional(rationale_id)
        return self._propose_speedup_plain(rationale_id)

    def _propose_speedup_plain(self, rationale_id: str) -> RemediationIntent | None:
        """Non-templated region: single-step cost ladder (double→ff→float), float
        reached via the Patcher's plain-edit ``-to-float`` rung (the historical
        path — kept unchanged for regions that carry a bare ``double`` token)."""
        target_level = next_down(self.installed)
        if target_level is None or f"{self.installed}-to-{target_level}" not in TRANSITION_KINDS:
            self._result = WalkResult(status="settled", final_precision=self.installed)
            return None
        # required_by floor: never demote a line below the precision a promoted
        # cascade chain still requires of it (safe upper bound — extra precision
        # can only help).  Settle at the current rung rather than dropping through.
        if self.floor is not None and _ladder_index(target_level) < _ladder_index(self.floor):
            self._result = WalkResult(status="settled", final_precision=self.installed)
            return None
        self._pending_is_dd = False
        self._pending_is_rewrite = False
        return RemediationIntent(
            target=self.record.target, kind=f"{self.installed}-to-{target_level}",
            intent=INTENT_SPEEDUP, current_precision=self.installed,
            rationale_id=rationale_id, via=VIA_PLAIN)

    def _propose_speedup_regional(self, rationale_id: str) -> RemediationIntent | None:
        """Template-typed region (Wave 2): try the demotion targets below the
        baseline cheapest-first (``float`` then ``ff``) via the LLM/regional
        integrators.  The first the Validator accepts wins (cheapest passing
        precision); on reject, advance to the next (more expensive) target.

        This is what makes ``double->float`` reachable on template code — it is
        proposed DIRECTLY (a skip transition), not gated off as in Wave 1 — while
        the ``double->ff`` fallback preserves the demotions Wave 1 already won.
        """
        while self._regional_i < len(self._regional_plan):
            target_level = self._regional_plan[self._regional_i]
            kind = f"{self.baseline}-to-{target_level}"
            # respect the cascade-chain floor and the kind vocabulary
            if (kind not in TRANSITION_KINDS or
                    (self.floor is not None
                     and _ladder_index(target_level) < _ladder_index(self.floor))):
                self._regional_i += 1
                continue
            self._pending_is_dd = False
            self._pending_is_rewrite = False
            return RemediationIntent(
                target=self.record.target, kind=kind, intent=INTENT_SPEEDUP,
                current_precision=self.baseline, rationale_id=rationale_id,
                via=VIA_REGIONAL)
        self._result = WalkResult(status="settled", final_precision=self.installed)
        return None

    # -- resolution -------------------------------------------------------
    def resolve(self, accepted: bool, genuine_reject: bool = False) -> None:
        """Feed back the outcome of the pending intent.

        ``genuine_reject`` = Patcher ok + Validator reject (matters only at the
        DD rung, for dd_ceiling vs dd_untested).
        """
        if self._pending is None:
            raise RuntimeError("resolve() called without a pending intent")
        intent = self._pending
        self._pending = None
        if self.mode == INTENT_CORRECTNESS:
            self._resolve_correctness(intent, accepted, genuine_reject)
        else:
            self._resolve_speedup(intent, accepted)

    def _resolve_correctness(self, intent: RemediationIntent, accepted: bool,
                             genuine_reject: bool) -> None:
        if accepted:
            # cleared — a precision rung or a rewrite got through. A rewrite
            # layers on top of the retained DD (Q2), so its final precision is
            # the installed rung (dd after a genuine DD reject), not the baseline.
            final = self.installed if self._pending_is_rewrite else intent.kind.split("-to-")[-1]
            if self._pending_is_rewrite:
                self.attempted_rewrites.append(intent.identity or "kahan")
            self._result = WalkResult(
                status="cleared", final_precision=final, accepted_intent=intent,
                rewrite_accepted=self._pending_is_rewrite,
                attempted_rewrites=list(self.attempted_rewrites))
            return

        if self._pending_is_dd and not genuine_reject:
            # P6a: DD never honestly tested → untested, DD not installed
            self._result = WalkResult(
                status="dd_untested", final_precision=self.installed,
                ceiling_kind="dd_untested",
                attempted_rewrites=list(self.attempted_rewrites))
            return

        if self._pending_is_rewrite:
            self.attempted_rewrites.append(intent.identity or "kahan")
            self._rw_i += 1
            return  # next propose() tries the next rewrite (or finishes)

        if self._pending_is_dd:
            # genuine DD reject (Patcher ok + Validator reject): DD is retained on
            # the branch as the ceiling candidate; rewrites layer on top of it.
            self.installed = _DD
        # a precision rung rejected → next rung
        self._up_i += 1

    def _finish_correctness_no_more(self) -> None:
        """No precision rung cleared and no rewrite cleared."""
        # We reach here only after the DD rung was attempted. If a genuine DD
        # reject happened, DD is the accepted ceiling; otherwise it was untested.
        # dd_untested is handled inline in _resolve_correctness; reaching here
        # means either a genuine DD reject (→ dd_ceiling) or no dd rung existed.
        if _DD in self._up_targets:
            self._result = WalkResult(
                status="dd_ceiling", final_precision=_DD, ceiling_kind="dd_ceiling",
                rewrite_accepted=False, attempted_rewrites=list(self.attempted_rewrites))
        else:
            # no dd rung reachable (e.g. float baseline, float-to-dd unsupported)
            self._result = WalkResult(
                status="exhausted", final_precision=self.installed,
                attempted_rewrites=list(self.attempted_rewrites))

    def _resolve_speedup(self, intent: RemediationIntent, accepted: bool) -> None:
        if self.float_via == VIA_REGIONAL:
            # Cheapest-first search: the first accepted target is the cheapest
            # passing precision → install it and stop.  A reject advances to the
            # next (more expensive) candidate in the plan.
            if accepted:
                self.installed = intent.kind.split("-to-")[-1]
                self._result = WalkResult(status="settled", final_precision=self.installed)
            else:
                self._regional_i += 1
            return
        if accepted:
            self.installed = intent.kind.split("-to-")[-1]
            # continue: next propose() tries the next cheaper rung
        else:
            # first reject → back off one, stop at last accepted (== installed)
            self._result = WalkResult(status="settled", final_precision=self.installed)

    # -- result -----------------------------------------------------------
    def result(self) -> WalkResult:
        if self._result is None:
            raise RuntimeError("walk has not terminated")
        return self._result

    @property
    def done(self) -> bool:
        return self._result is not None
