"""Greedy sequential-layering solver core (Phase 2e Stage 1).

Consumes a ranked candidate queue (``queue.build_queue``) and drives the
accept/revert walk locked by Reet 2026-07-24, under the **regression-relative
gate** (Reet 2026-07-24, Stage-2 prep — replaces the Stage-1 absolute p100≥6):

    baseline_min = whole-app p100 of the UNPATCHED tree (measured once at start)
    for each candidate in rank order (float < ff < dd):
        if its region already resolved at a cheaper rung -> skip
        apply the patch ON TOP OF the accumulated tree (not the baseline)
        build the whole app + run the whole-app validator
        accept  (cand_min >= baseline_min - margin) -> keep; region resolved
        reject  (cand_min <  baseline_min - margin)  -> revert this one patch
    terminate when the queue is exhausted.

No joint re-measurement, no strategy combining, no re-characterization between
accepts (v1 keeps it simple — PLAN_overview §Loop semantics).

**Regression-relative gate (Reet's call, Stage-2 prep).**  A candidate is accepted
iff it does not *worsen* the whole-app worst-case precise-digits (p100) by more
than ``margin`` digits vs the double baseline measured at solve start:

    accept  ⇔  candidate.min_precise_digits >= baseline.min_precise_digits - margin

``margin`` is the same 0.5-digit figure the Validator's own accept verdict bundles
as its regression guard (:data:`agents.validator.validate.DEFAULT_MAX_REGRESSION`)
— reused here, not re-invented.  This replaces the Stage-1 *absolute* ``p100 >= 6``
gate, which was structurally unsatisfiable when the target integral is itself the
whole-app global-min hotspot (B12's double-precision cancellation floor is 3.69 <
6, a physics ceiling no candidate can lift — see SOLVER_STAGE1_B12.md).  The safety
statement "float where possible, dd where necessary, never make it worse" is a
*regression* claim, not a claim that double hits 6 on every integral; the
regression-relative gate encodes exactly that.

``candidate.min_precise_digits`` / ``baseline.min_precise_digits`` are read
DIRECTLY from the Validator's measurement (``verdict["candidate"]``/
``["current"]``), NOT its bundled accept verdict — the solver owns the decision.

**STOP-and-flag discipline is preserved** for structural unimplementability: if the
baseline itself cannot be scored (crash / NaN / no min), there is no reference to
gate against, so the solver stops and flags rather than guessing.  (A *low* but
well-defined baseline is no longer a stop — that is the whole point of going
regression-relative.)

All side effects are injected (``apply_fn`` / ``validate_fn`` / ``revert_fn`` /
``head_fn``) so the walk logic is unit-testable without a real build.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.solver.queue import Candidate
from agents.validator.validate import DEFAULT_MAX_REGRESSION

# Regression-relative accept margin (digits) — reuse the Validator's regression
# guard figure so the solver and the bundled validate() verdict agree on "how much
# worse is still acceptable".
DEFAULT_MARGIN = DEFAULT_MAX_REGRESSION
BASELINE_PRECISION = "double"

# Phase 2f — chain_dd positive-lift gate (Reet 2026-07-24).  A chain-scoped dd
# promotion must EARN its cost: accept iff it lifts the whole-app p100 by at least
# this many digits vs the accumulated-min BEFORE this candidate (not the fixed
# baseline — a later chain riding a prior chain's lift would otherwise pass for free).
# Symmetric with the 0.5-digit regression margin; rejects FP-noise "improvements".
LIFT_MARGIN = 0.5


# ---------------------------------------------------------------------------
# Injected-callable return shapes
# ---------------------------------------------------------------------------
@dataclass
class ApplyResult:
    """What ``apply_fn`` returns after invoking the Patcher for one candidate."""
    ok: bool
    candidate_sha: Optional[str] = None
    patcher_status: Optional[str] = None
    gate_binary: Optional[str] = None
    gate_tree_hash: Optional[str] = None
    error: Optional[dict] = None
    wall_sec: float = 0.0


@dataclass
class ValidateResult:
    """What ``validate_fn`` returns after building + scoring the whole app."""
    cand_min: float                 # random-battery p100 (the GATE metric)
    curr_min: float                 # unpatched baseline p100 (== vanilla, constant)
    combined_cand_min: Optional[float] = None   # random+tail worst-case (telemetry)
    verdict: Optional[str] = None               # validator's own accept/reject
    verdict_reason: Optional[str] = None
    wall_sec: float = 0.0


# ---------------------------------------------------------------------------
# Per-candidate outcome + whole-run result
# ---------------------------------------------------------------------------
ACCEPTED = "accepted"
REJECTED = "rejected"
SKIPPED_RESOLVED = "skipped_region_resolved"
APPLY_FAILED = "apply_failed"
STOPPED_GATE_UNIMPLEMENTABLE = "stopped_gate_unimplementable"


@dataclass
class CandidateOutcome:
    candidate: Candidate
    outcome: str
    min_before: Optional[float] = None   # accumulated p100 before this candidate
    min_after: Optional[float] = None    # accumulated-with-candidate p100
    reason: str = ""
    candidate_sha: Optional[str] = None
    patcher_status: Optional[str] = None
    validator_verdict: Optional[str] = None
    combined_min_after: Optional[float] = None
    wall_sec: float = 0.0
    # Phase 2f: chain_dd rejection sub-classification ("chain_no_lift" |
    # "chain_regression"), None for single-region candidates + all accepts.
    reason_tag: Optional[str] = None


@dataclass
class SolveResult:
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    baseline_min: Optional[float] = None      # vanilla whole-app p100 (the reference)
    final_min: Optional[float] = None         # accumulated p100 at the end
    final_head: Optional[str] = None
    # region_id -> landed rung ("double" for regions that never accepted a rung)
    region_final: dict = field(default_factory=dict)
    margin: float = DEFAULT_MARGIN            # regression-relative accept margin (digits)
    stopped: Optional[str] = None             # non-None => hard stop reason
    stop_detail: str = ""

    @property
    def accept_threshold(self) -> Optional[float]:
        """The regression-relative accept bar: ``baseline_min - margin``.

        ``None`` until the baseline is measured (first successful validate)."""
        if self.baseline_min is None:
            return None
        return self.baseline_min - self.margin

    @property
    def accepted(self) -> list[CandidateOutcome]:
        return [o for o in self.outcomes if o.outcome == ACCEPTED]

    @property
    def rejected(self) -> list[CandidateOutcome]:
        return [o for o in self.outcomes if o.outcome == REJECTED]

    def precision_distribution(self) -> dict:
        dist = {"double": 0, "float": 0, "ff": 0, "dd": 0}
        for rung in self.region_final.values():
            dist[rung] = dist.get(rung, 0) + 1
        return dist


def solve(queue: list[Candidate], *,
          apply_fn: Callable[[Candidate, str], ApplyResult],
          validate_fn: Callable[[str, Optional[str], Optional[str]], ValidateResult],
          revert_fn: Callable[[str], None],
          head_fn: Callable[[], str],
          margin: float = DEFAULT_MARGIN,
          all_region_ids: Optional[set] = None,
          on_event: Optional[Callable[[CandidateOutcome], None]] = None) -> SolveResult:
    """Run the greedy walk over ``queue`` under the regression-relative gate.

    ``margin`` is the regression-relative accept margin (digits): a candidate is
    accepted iff ``cand_min >= baseline_min - margin`` where ``baseline_min`` is the
    unpatched whole-app p100 measured once at solve start.  ``all_region_ids``
    (optional) seeds ``region_final`` so every region — even ones with no candidate
    — is reported at ``double``.  ``on_event`` is called with each
    ``CandidateOutcome`` as it is decided (live progress).
    """
    result = SolveResult(margin=margin)
    for rid in (all_region_ids or set()):
        result.region_final.setdefault(rid, BASELINE_PRECISION)
    for c in queue:
        result.region_final.setdefault(c.region_id, BASELINE_PRECISION)

    resolved: set = set()
    accumulated_min: Optional[float] = None   # p100 at current HEAD

    def emit(o: CandidateOutcome) -> None:
        result.outcomes.append(o)
        if on_event is not None:
            on_event(o)

    for cand in queue:
        if cand.region_id in resolved:
            emit(CandidateOutcome(
                candidate=cand, outcome=SKIPPED_RESOLVED,
                min_before=accumulated_min,
                reason=f"region already at '{result.region_final[cand.region_id]}' "
                       f"(cheaper rung accepted)"))
            continue

        parent = head_fn()
        ar = apply_fn(cand, parent)
        if not ar.ok:
            # The Patcher resets the tree to `parent` itself on any non-ok status.
            emit(CandidateOutcome(
                candidate=cand, outcome=APPLY_FAILED,
                min_before=accumulated_min,
                reason=_error_reason(ar),
                patcher_status=ar.patcher_status, wall_sec=ar.wall_sec))
            continue

        vr = validate_fn(ar.candidate_sha, ar.gate_binary, ar.gate_tree_hash)

        # Establish the vanilla baseline from the first successful validate
        # (validator's `current` is always the unpatched tree) — the reference the
        # regression-relative gate compares every candidate against.  STOP-and-flag
        # only when the baseline cannot be scored at all (crash / NaN / no min):
        # there is then no reference to gate against.  A *low* but well-defined
        # baseline is fine — the whole point of the regression-relative gate is to
        # not penalize the solver for an ill-conditioning floor it cannot fix.
        if result.baseline_min is None:
            result.baseline_min = vr.curr_min
            accumulated_min = vr.curr_min
            if not _scoreable(vr.curr_min):
                revert_fn(parent)
                result.stopped = STOPPED_GATE_UNIMPLEMENTABLE
                result.stop_detail = (
                    f"baseline whole-app min_precise_digits={vr.curr_min!r} is "
                    f"unscoreable (crash / NaN / no min); the regression-relative "
                    f"gate has no reference to compare candidates against")
                emit(CandidateOutcome(
                    candidate=cand, outcome=STOPPED_GATE_UNIMPLEMENTABLE,
                    min_before=vr.curr_min, min_after=vr.cand_min,
                    reason=result.stop_detail, candidate_sha=ar.candidate_sha,
                    patcher_status=ar.patcher_status,
                    validator_verdict=vr.verdict,
                    combined_min_after=vr.combined_cand_min,
                    wall_sec=ar.wall_sec + vr.wall_sec))
                break

        min_before = accumulated_min
        # Phase 2f: chain_dd uses a POSITIVE-lift gate vs the accumulated-min BEFORE
        # this candidate (each chain must earn its own dd cost); every other rung uses
        # the regression-relative gate vs the fixed baseline (unchanged, byte-for-byte).
        if cand.is_chain:
            lift_bar = (min_before if min_before is not None
                        else result.baseline_min) + LIFT_MARGIN
            accept = _scoreable(vr.cand_min) and vr.cand_min >= lift_bar
            accept_reason = (f"chain lift: p100 {vr.cand_min:.4f} >= accumulated "
                             f"{min_before:.4f} + lift {LIFT_MARGIN} = {lift_bar:.4f}")
        else:
            threshold = result.accept_threshold   # baseline_min - margin (fixed)
            accept = _scoreable(vr.cand_min) and vr.cand_min >= threshold
            accept_reason = (f"p100 {vr.cand_min:.4f} >= baseline "
                             f"{result.baseline_min:.4f} - margin {margin} = {threshold:.4f}"
                             if _scoreable(vr.cand_min) else "")

        if accept:
            resolved.add(cand.region_id)
            result.region_final[cand.region_id] = cand.rung
            accumulated_min = vr.cand_min
            emit(CandidateOutcome(
                candidate=cand, outcome=ACCEPTED,
                min_before=min_before, min_after=vr.cand_min,
                reason=accept_reason,
                candidate_sha=ar.candidate_sha,
                patcher_status=ar.patcher_status, validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))
        else:
            revert_fn(parent)  # drop this one patch; HEAD stays at `parent`
            shown = f"{vr.cand_min:.4f}" if _scoreable(vr.cand_min) else repr(vr.cand_min)
            reason_tag = None
            if cand.is_chain:
                # regression vs baseline -> chain_regression; otherwise merely no lift.
                reason_tag = ("chain_regression"
                              if _scoreable(vr.cand_min)
                              and vr.cand_min < result.baseline_min - margin
                              else "chain_no_lift")
                reason = (f"chain {reason_tag}: p100 {shown} < accumulated "
                          f"{min_before:.4f} + lift {LIFT_MARGIN} = {lift_bar:.4f}")
            else:
                threshold = result.accept_threshold
                reason = (f"p100 {shown} < baseline {result.baseline_min:.4f} "
                          f"- margin {margin} = {threshold:.4f}")
            emit(CandidateOutcome(
                candidate=cand, outcome=REJECTED,
                min_before=min_before, min_after=vr.cand_min,
                reason=reason, reason_tag=reason_tag,
                candidate_sha=ar.candidate_sha, patcher_status=ar.patcher_status,
                validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))

    result.final_head = head_fn()
    result.final_min = accumulated_min if accumulated_min is not None else result.baseline_min
    return result


def _scoreable(x: Optional[float]) -> bool:
    """True iff ``x`` is a usable min_precise_digits (not None / NaN / inf).

    A candidate or baseline whose whole-app p100 came back ``None`` (validator
    could not score) or non-finite (NaN/inf from a crash) carries no comparable
    number for the regression-relative gate."""
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _error_reason(ar: ApplyResult) -> str:
    if ar.error:
        kind = ar.error.get("kind", "?")
        detail = (ar.error.get("detail") or "")[:200]
        return f"patcher {ar.patcher_status} ({kind}): {detail}"
    return f"patcher {ar.patcher_status}"
