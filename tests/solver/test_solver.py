"""Unit tests — solver greedy accept/revert loop (Phase 2e).

A tiny in-memory harness stands in for git + Patcher + Validator: the fake tree
tracks HEAD as a commit counter, ``apply_fn`` "commits" a new sha, ``revert_fn``
resets HEAD, and ``validate_fn`` returns a scripted p100 per (region_id, rung).
"""

import pytest

from agents.solver.queue import Candidate
from agents.solver.solver import (
    ACCEPTED, APPLY_FAILED, REJECTED, SKIPPED_RESOLVED,
    STOPPED_GATE_UNIMPLEMENTABLE, ApplyResult, ValidateResult, solve,
)

BASELINE = 8.84   # qcdloop-like vanilla whole-app p100


def _cand(region_id, rung, de=1e-7, bde=1e-13):
    return Candidate(region_id=region_id, rung=rung, kind=f"double-to-{rung}",
                     intent="speedup", via="regional",
                     delta_effective=de, baseline_delta_effective=bde)


class FakeHarness:
    """Scriptable apply/validate/revert/head with a commit-counter 'git'."""

    def __init__(self, cand_min_by_key, *, baseline=BASELINE, fail_keys=()):
        # cand_min_by_key: {(region_id, rung): p100_of_accumulated_tree_with_it}
        self.cand_min = cand_min_by_key
        self.baseline = baseline
        self.fail_keys = set(fail_keys)
        self._head = "sha0"
        self._n = 0
        self.applied = []      # keys apply_fn was called for
        self.reverted = []     # parent shas revert_fn was called for

    def head(self):
        return self._head

    def apply(self, cand, parent):
        key = (cand.region_id, cand.rung)
        self.applied.append(key)
        if key in self.fail_keys:
            # patcher resets tree itself; HEAD stays at parent
            return ApplyResult(ok=False, patcher_status="llm_gen_failed",
                               error={"kind": "llm", "detail": "boom"})
        self._n += 1
        self._head = f"sha{self._n}"          # patcher committed on top
        return ApplyResult(ok=True, candidate_sha=self._head,
                           patcher_status="ok", gate_binary="/b", gate_tree_hash="h")

    def validate(self, candidate_sha, gate_binary, gate_tree_hash):
        # find which candidate produced candidate_sha (last applied)
        key = self.applied[-1]
        cm = self.cand_min[key]
        return ValidateResult(cand_min=cm, curr_min=self.baseline,
                              combined_cand_min=cm, verdict="?", verdict_reason="?")

    def revert(self, parent):
        self.reverted.append(parent)
        self._head = parent

    def run(self, queue, gate=6.0):
        return solve(queue, apply_fn=self.apply, validate_fn=self.validate,
                     revert_fn=self.revert, head_fn=self.head, gate=gate)


# --- accept layering ----------------------------------------------------------
def test_accept_keeps_commit_and_advances_head():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 7.5})
    res = h.run(q)
    assert res.accepted and res.accepted[0].candidate.region_id == "A.h:10"
    assert res.final_head == "sha1"          # kept
    assert h.reverted == []                  # nothing reverted
    assert res.region_final["A.h:10"] == "float"
    assert res.final_min == 7.5


def test_two_independent_accepts_layer_sequentially():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    h = FakeHarness({("A.h:10", "float"): 7.5, ("B.h:20", "float"): 7.0})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [ACCEPTED, ACCEPTED]
    # min_before threads: first sees baseline, second sees first's accumulated min
    assert res.outcomes[0].min_before == pytest.approx(BASELINE)
    assert res.outcomes[1].min_before == pytest.approx(7.5)
    assert res.final_min == 7.0
    assert res.precision_distribution()["float"] == 2


# --- reject revert ------------------------------------------------------------
def test_reject_reverts_and_leaves_region_unresolved():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 5.0})   # below gate 6.0
    res = h.run(q)
    assert res.rejected and not res.accepted
    assert h.reverted == ["sha0"]                 # reset to parent
    assert res.final_head == "sha0"
    assert res.region_final["A.h:10"] == "double"
    assert res.final_min == pytest.approx(BASELINE)


def test_float_reject_then_ff_accept_same_region():
    # float fails the gate, ff (next rung) holds -> region lands at ff
    q = [_cand("A.h:10", "float"), _cand("A.h:10", "ff")]
    h = FakeHarness({("A.h:10", "float"): 5.0, ("A.h:10", "ff"): 7.2})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [REJECTED, ACCEPTED]
    assert res.region_final["A.h:10"] == "ff"
    assert h.reverted == ["sha0"]                 # only the float attempt reverted


# --- first-accept-per-region-wins --------------------------------------------
def test_cheaper_rung_accept_skips_remaining_rungs_of_region():
    # float holds -> ff for the same region must be skipped, never applied
    q = [_cand("A.h:10", "float"), _cand("A.h:10", "ff")]
    h = FakeHarness({("A.h:10", "float"): 7.5, ("A.h:10", "ff"): 8.0})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [ACCEPTED, SKIPPED_RESOLVED]
    assert ("A.h:10", "ff") not in h.applied      # never built
    assert res.region_final["A.h:10"] == "float"


# --- gate boundary ------------------------------------------------------------
def test_gate_is_inclusive_at_exactly_six():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 6.0})   # exactly the gate
    res = h.run(q, gate=6.0)
    assert res.accepted


def test_gate_rejects_just_below_six():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 5.9999})
    res = h.run(q, gate=6.0)
    assert res.rejected


# --- apply failure ------------------------------------------------------------
def test_apply_failure_is_recorded_not_reverted_by_solver():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    h = FakeHarness({("B.h:20", "float"): 7.0},
                    fail_keys={("A.h:10", "float")})
    res = h.run(q)
    assert res.outcomes[0].outcome == APPLY_FAILED
    assert res.outcomes[0].patcher_status == "llm_gen_failed"
    assert h.reverted == []                        # patcher owns its own reset
    assert res.outcomes[1].outcome == ACCEPTED     # walk continues
    assert res.region_final["A.h:10"] == "double"


# --- structural-unimplementable stop -----------------------------------------
def test_baseline_below_gate_stops_and_flags():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    h = FakeHarness({("A.h:10", "float"): 3.0}, baseline=3.5)  # baseline < gate 6
    res = h.run(q, gate=6.0)
    assert res.stopped == STOPPED_GATE_UNIMPLEMENTABLE
    assert res.outcomes[-1].outcome == STOPPED_GATE_UNIMPLEMENTABLE
    assert h.reverted == ["sha0"]                  # first candidate rolled back
    assert len(res.outcomes) == 1                  # walk halted, B.h:20 untried
    assert ("B.h:20", "float") not in h.applied


# --- empty queue --------------------------------------------------------------
def test_empty_queue_no_builds():
    h = FakeHarness({})
    res = h.run([])
    assert res.outcomes == []
    assert res.baseline_min is None
    assert h.applied == []


def test_all_regions_seeded_at_double():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 7.0})
    res = solve(q, apply_fn=h.apply, validate_fn=h.validate, revert_fn=h.revert,
                head_fn=h.head, gate=6.0, all_region_ids={"A.h:10", "C.h:99"})
    assert res.region_final["C.h:99"] == "double"   # region with no candidate
    assert res.region_final["A.h:10"] == "float"
