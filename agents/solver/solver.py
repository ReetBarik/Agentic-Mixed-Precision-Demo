"""Greedy sequential-layering solver core (Phase 2e Stage 1).

Consumes a ranked candidate queue (``queue.build_queue``) and drives the
accept/revert walk locked by Reet 2026-07-24:

    for each candidate in rank order (float < ff < dd):
        if its region already resolved at a cheaper rung -> skip
        apply the patch ON TOP OF the accumulated tree (not the baseline)
        build the whole app + run the whole-app validator
        accept  (min_precise_digits >= gate) -> keep; region resolved; new baseline
        reject  (min_precise_digits <  gate) -> revert this one patch; continue
    terminate when the queue is exhausted.

No joint re-measurement, no strategy combining, no re-characterization between
accepts (v1 keeps it simple — PLAN_overview §Loop semantics).

**The gate is the raw p100 metric, deliberately NOT the validator's accept
verdict.** ``verdict["candidate"]["min_precise_digits"]`` is the worst-case
precise-decimal-digits across the random battery (p100 = min across samples). The
task locks the bar at **6.0**.  The Validator's own verdict additionally bundles a
0.5-digit *regression* guard vs the baseline (~8.84 digits on qcdloop), which
would reject any candidate that legitimately spends precision down toward 6 — so
the solver reads the metric and decides itself.  See SOLVER_STAGE1 report.

All side effects are injected (``apply_fn`` / ``validate_fn`` / ``revert_fn`` /
``head_fn``) so the walk logic is unit-testable without a real build.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from agents.solver.queue import Candidate

DEFAULT_GATE = 6.0
BASELINE_PRECISION = "double"


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


@dataclass
class SolveResult:
    outcomes: list[CandidateOutcome] = field(default_factory=list)
    baseline_min: Optional[float] = None      # vanilla whole-app p100
    final_min: Optional[float] = None         # accumulated p100 at the end
    final_head: Optional[str] = None
    # region_id -> landed rung ("double" for regions that never accepted a rung)
    region_final: dict = field(default_factory=dict)
    gate: float = DEFAULT_GATE
    stopped: Optional[str] = None             # non-None => hard stop reason
    stop_detail: str = ""

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
          gate: float = DEFAULT_GATE,
          all_region_ids: Optional[set] = None,
          on_event: Optional[Callable[[CandidateOutcome], None]] = None) -> SolveResult:
    """Run the greedy walk over ``queue``.

    ``all_region_ids`` (optional) seeds ``region_final`` so every region — even
    ones with no candidate — is reported at ``double``.  ``on_event`` is called
    with each ``CandidateOutcome`` as it is decided (live progress).
    """
    result = SolveResult(gate=gate)
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
        # (validator's `current` is always the unpatched tree).  If the baseline
        # itself is below the gate, the p100>=gate rule is structurally
        # unimplementable on this workload -> STOP and flag (PLAN 2e §Gate).
        if result.baseline_min is None:
            result.baseline_min = vr.curr_min
            accumulated_min = vr.curr_min
            if vr.curr_min < gate:
                revert_fn(parent)
                result.stopped = STOPPED_GATE_UNIMPLEMENTABLE
                result.stop_detail = (
                    f"baseline whole-app min_precise_digits={vr.curr_min:.4f} "
                    f"< gate={gate}; no candidate can satisfy p100>=gate when the "
                    f"unpatched tree already fails it")
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
        if vr.cand_min >= gate:
            resolved.add(cand.region_id)
            result.region_final[cand.region_id] = cand.rung
            accumulated_min = vr.cand_min
            emit(CandidateOutcome(
                candidate=cand, outcome=ACCEPTED,
                min_before=min_before, min_after=vr.cand_min,
                reason="p100 >= gate", candidate_sha=ar.candidate_sha,
                patcher_status=ar.patcher_status, validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))
        else:
            revert_fn(parent)  # drop this one patch; HEAD stays at `parent`
            emit(CandidateOutcome(
                candidate=cand, outcome=REJECTED,
                min_before=min_before, min_after=vr.cand_min,
                reason=f"p100 {vr.cand_min:.4f} < gate {gate}",
                candidate_sha=ar.candidate_sha, patcher_status=ar.patcher_status,
                validator_verdict=vr.verdict,
                combined_min_after=vr.combined_cand_min,
                wall_sec=ar.wall_sec + vr.wall_sec))

    result.final_head = head_fn()
    result.final_min = accumulated_min if accumulated_min is not None else result.baseline_min
    return result


def _error_reason(ar: ApplyResult) -> str:
    if ar.error:
        kind = ar.error.get("kind", "?")
        detail = (ar.error.get("detail") or "")[:200]
        return f"patcher {ar.patcher_status} ({kind}): {detail}"
    return f"patcher {ar.patcher_status}"
