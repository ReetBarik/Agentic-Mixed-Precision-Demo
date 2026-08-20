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
accepts (v1 keeps it simple — README §Loop semantics).

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
    cand_min: float                 # whole-app random-battery p100 (the GATE metric)
    curr_min: float                 # unpatched whole-app baseline p100 (== vanilla, constant)
    combined_cand_min: Optional[float] = None   # random+tail worst-case (telemetry)
    verdict: Optional[str] = None               # validator's own accept/reject
    verdict_reason: Optional[str] = None
    wall_sec: float = 0.0
    # Phase 2f kernel-scope: per-integral p100 for the candidate and current baseline
    # ({integral: min_precise_digits}).  When a candidate carries ``target_kernel=K``
    # the solver gates on ``cand_per_kernel[K]`` vs ``curr_per_kernel[K]`` — the
    # candidate's own kernel floor — instead of the whole-app min.  Empty/None => the
    # whole-app gate (existing behaviour) is used.
    cand_per_kernel: dict = field(default_factory=dict)
    curr_per_kernel: dict = field(default_factory=dict)

    def cand_scope_min(self, kernel: Optional[str]) -> Optional[float]:
        """Candidate p100 at ``kernel`` scope (``None`` => whole-app ``cand_min``)."""
        if kernel is None:
            return self.cand_min
        return self.cand_per_kernel.get(kernel)

    def curr_scope_min(self, kernel: Optional[str]) -> Optional[float]:
        """Baseline p100 at ``kernel`` scope (``None`` => whole-app ``curr_min``)."""
        if kernel is None:
            return self.curr_min
        return self.curr_per_kernel.get(kernel)


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
    # "chain_regression" | "kernel_scope_unmeasured"), None for single-region
    # candidates gated whole-app + all accepts.
    reason_tag: Optional[str] = None
    # Phase 2f kernel-scope: the kernel this candidate was gated against (None =>
    # whole-app).  ``min_before``/``min_after`` are reported at THIS scope, so the
    # per-kernel lift is ``min_after - min_before`` when ``target_kernel`` is set.
    target_kernel: Optional[str] = None


@dataclass
class SolveResult:
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    baseline_min: Optional[float] = None      # vanilla whole-app p100 (the reference)
    final_min: Optional[float] = None         # accumulated whole-app p100 at the end
    final_head: Optional[str] = None
    # region_id -> landed rung ("double" for regions that never accepted a rung)
    region_final: dict = field(default_factory=dict)
    margin: float = DEFAULT_MARGIN            # regression-relative accept margin (digits)
    stopped: Optional[str] = None             # non-None => hard stop reason
    stop_detail: str = ""
    # Phase 2f kernel-scope: per-kernel baseline + final p100, captured once at solve
    # start (baseline) and updated as candidates on that kernel accept (final).  A
    # candidate targeting kernel K is gated against K's own floor here, not the
    # whole-app min pinned by whichever kernel is worst.  Empty when no candidate
    # carried a ``target_kernel`` (pure whole-app run — existing behaviour).
    baseline_by_kernel: dict = field(default_factory=dict)
    final_by_kernel: dict = field(default_factory=dict)

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
    accumulated_min: Optional[float] = None   # whole-app p100 at current HEAD
    # Phase 2f kernel-scope: per-kernel accumulated p100 at current HEAD (seeded from
    # the per-kernel baseline the first successful validate reports).  A candidate
    # targeting kernel K rides only K's own accumulated lift — chains on different
    # kernels never ride each other's lifts.
    accumulated_by_kernel: dict = {}

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
            # Seed per-kernel baselines from the same validate (whole-app run already
            # scored every integral; this is a free by-product of _score).
            result.baseline_by_kernel = dict(vr.curr_per_kernel or {})
            accumulated_by_kernel = dict(vr.curr_per_kernel or {})
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

        # Phase 2f kernel-scope: gate a candidate carrying ``target_kernel=K`` against
        # K's own floor (baseline + accumulated) — not the whole-app min pinned by
        # whichever kernel is worst.  ``kernel is None`` => whole-app scope, and every
        # scoped value below collapses to its whole-app counterpart, so the gate,
        # reason strings, and reported minima are byte-for-byte the pre-2f behaviour.
        kernel = cand.target_kernel
        cand_scope = vr.cand_scope_min(kernel)
        baseline_scope = (result.baseline_by_kernel.get(kernel)
                          if kernel is not None else result.baseline_min)
        before_scope = (accumulated_by_kernel.get(kernel)
                        if kernel is not None else accumulated_min)
        min_before = before_scope

        # A kernel-scoped candidate whose kernel floor could not be measured (kernel
        # absent from the per-kernel arrays) has no scoped reference; reject-and-flag —
        # a visible measurement gap, non-fatal so the walk continues (Stage-1's job is
        # to surface these), never a silent fall-back to whole-app gating.
        if kernel is not None and (baseline_scope is None or not _scoreable(cand_scope)):
            revert_fn(parent)
            reason = (f"kernel_scope_unmeasured: kernel {kernel!r} min_precise_digits "
                      f"unavailable (cand={cand_scope!r}, baseline={baseline_scope!r}) — "
                      f"cannot gate at kernel scope")
            emit(CandidateOutcome(
                candidate=cand, outcome=REJECTED,
                min_before=min_before, min_after=cand_scope,
                reason=reason, reason_tag="kernel_scope_unmeasured",
                target_kernel=kernel,
                candidate_sha=ar.candidate_sha, patcher_status=ar.patcher_status,
                validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))
            continue

        ktag = f"[kernel={kernel}] " if kernel is not None else ""
        # chain_dd uses a POSITIVE-lift gate vs the accumulated-min BEFORE this
        # candidate (each chain must earn its own dd cost); every other rung uses the
        # regression-relative gate vs the fixed baseline.
        if cand.is_chain:
            lift_bar = (before_scope if before_scope is not None
                        else baseline_scope) + LIFT_MARGIN
            accept = _scoreable(cand_scope) and cand_scope >= lift_bar
            accept_reason = (f"{ktag}chain lift: p100 {cand_scope:.4f} >= accumulated "
                             f"{min_before:.4f} + lift {LIFT_MARGIN} = {lift_bar:.4f}")
        else:
            threshold = baseline_scope - margin   # scoped baseline - margin
            accept = _scoreable(cand_scope) and cand_scope >= threshold
            accept_reason = (f"{ktag}p100 {cand_scope:.4f} >= baseline "
                             f"{baseline_scope:.4f} - margin {margin} = {threshold:.4f}"
                             if _scoreable(cand_scope) else "")

        if accept:
            resolved.add(cand.region_id)
            result.region_final[cand.region_id] = cand.rung
            accumulated_min = vr.cand_min
            # A patch shifts the whole app's per-kernel mins; refresh from the measured
            # candidate arrays so a later same-kernel candidate rides only this kernel's
            # own lift (and cross-kernel candidates keep their untouched floors).
            if vr.cand_per_kernel:
                accumulated_by_kernel = dict(vr.cand_per_kernel)
            elif kernel is not None:
                accumulated_by_kernel[kernel] = cand_scope
            emit(CandidateOutcome(
                candidate=cand, outcome=ACCEPTED,
                min_before=min_before, min_after=cand_scope,
                reason=accept_reason, target_kernel=kernel,
                candidate_sha=ar.candidate_sha,
                patcher_status=ar.patcher_status, validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))
        else:
            revert_fn(parent)  # drop this one patch; HEAD stays at `parent`
            shown = f"{cand_scope:.4f}" if _scoreable(cand_scope) else repr(cand_scope)
            reason_tag = None
            if cand.is_chain:
                # regression vs (scoped) baseline -> chain_regression; else merely no lift.
                reason_tag = ("chain_regression"
                              if _scoreable(cand_scope)
                              and cand_scope < baseline_scope - margin
                              else "chain_no_lift")
                reason = (f"{ktag}chain {reason_tag}: p100 {shown} < accumulated "
                          f"{min_before:.4f} + lift {LIFT_MARGIN} = {lift_bar:.4f}")
            else:
                threshold = baseline_scope - margin
                reason = (f"{ktag}p100 {shown} < baseline {baseline_scope:.4f} "
                          f"- margin {margin} = {threshold:.4f}")
            emit(CandidateOutcome(
                candidate=cand, outcome=REJECTED,
                min_before=min_before, min_after=cand_scope,
                reason=reason, reason_tag=reason_tag, target_kernel=kernel,
                candidate_sha=ar.candidate_sha, patcher_status=ar.patcher_status,
                validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))

    result.final_head = head_fn()
    result.final_min = accumulated_min if accumulated_min is not None else result.baseline_min
    result.final_by_kernel = dict(accumulated_by_kernel)
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
