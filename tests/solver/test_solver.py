"""Unit tests — solver greedy accept/revert loop (Phase 2e).

A tiny in-memory harness stands in for git + Patcher + Validator: the fake tree
tracks HEAD as a commit counter, ``apply_fn`` "commits" a new sha, ``revert_fn``
resets HEAD, and ``validate_fn`` returns a scripted p100 per (region_id, rung).

Gate semantics (Stage-2 prep): **regression-relative** — a candidate is accepted
iff ``cand_min >= baseline_min - margin`` where ``baseline_min`` is the unpatched
whole-app p100 measured once at solve start and ``margin`` defaults to 0.5.
"""

import pytest

from agents.solver.queue import Candidate
from agents.solver.solver import (
    ACCEPTED, APPLY_FAILED, DEFAULT_MARGIN, REJECTED, SKIPPED_RESOLVED,
    STOPPED_GATE_UNIMPLEMENTABLE, ApplyResult, ValidateResult, solve,
)

BASELINE = 8.84   # qcdloop-like vanilla whole-app p100
MARGIN = DEFAULT_MARGIN            # 0.5
THRESHOLD = BASELINE - MARGIN      # 8.34 — the accept bar at the default baseline


def _cand(region_id, rung, de=1e-7, bde=1e-13):
    return Candidate(region_id=region_id, rung=rung, kind=f"double-to-{rung}",
                     intent="speedup", via="regional",
                     delta_effective=de, baseline_delta_effective=bde)


def _chain_cand(chain_id, predicted_lift=10.0, de=1e-30, bde=1e-4):
    return Candidate(region_id=chain_id, rung="chain_dd", kind="double-to-dd",
                     intent="correctness", via="chain",
                     delta_effective=de, baseline_delta_effective=bde,
                     chain_lines=(("f.h", 10, 10),), predicted_lift=predicted_lift)


class FakeHarness:
    """Scriptable apply/validate/revert/head with a commit-counter 'git'."""

    def __init__(self, cand_min_by_key, *, baseline=BASELINE, fail_keys=()):
        # cand_min_by_key: {(region_id, rung): p100_of_accumulated_tree_with_it}
        self.cand_min = cand_min_by_key
        self.baseline = baseline          # curr_min the validator reports (None => unscoreable)
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
        cm = self.cand_min.get(key)
        return ValidateResult(cand_min=cm, curr_min=self.baseline,
                              combined_cand_min=cm, verdict="?", verdict_reason="?")

    def revert(self, parent):
        self.reverted.append(parent)
        self._head = parent

    def run(self, queue, margin=MARGIN):
        return solve(queue, apply_fn=self.apply, validate_fn=self.validate,
                     revert_fn=self.revert, head_fn=self.head, margin=margin)


# --- accept layering ----------------------------------------------------------
def test_accept_keeps_commit_and_advances_head():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 8.5})    # >= threshold 8.34
    res = h.run(q)
    assert res.accepted and res.accepted[0].candidate.region_id == "A.h:10"
    assert res.final_head == "sha1"          # kept
    assert h.reverted == []                  # nothing reverted
    assert res.region_final["A.h:10"] == "float"
    assert res.final_min == 8.5
    assert res.baseline_min == pytest.approx(BASELINE)
    assert res.accept_threshold == pytest.approx(THRESHOLD)


def test_two_independent_accepts_layer_sequentially():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    h = FakeHarness({("A.h:10", "float"): 8.5, ("B.h:20", "float"): 8.4})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [ACCEPTED, ACCEPTED]
    # min_before threads: first sees baseline, second sees first's accumulated min
    assert res.outcomes[0].min_before == pytest.approx(BASELINE)
    assert res.outcomes[1].min_before == pytest.approx(8.5)
    assert res.final_min == 8.4
    assert res.precision_distribution()["float"] == 2


# --- reject revert ------------------------------------------------------------
def test_reject_reverts_and_leaves_region_unresolved():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 7.0})   # below threshold 8.34
    res = h.run(q)
    assert res.rejected and not res.accepted
    assert h.reverted == ["sha0"]                 # reset to parent
    assert res.final_head == "sha0"
    assert res.region_final["A.h:10"] == "double"
    assert res.final_min == pytest.approx(BASELINE)


def test_float_reject_then_ff_accept_same_region():
    # float regresses below threshold, ff (next rung) holds -> region lands at ff
    q = [_cand("A.h:10", "float"), _cand("A.h:10", "ff")]
    h = FakeHarness({("A.h:10", "float"): 7.0, ("A.h:10", "ff"): 8.5})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [REJECTED, ACCEPTED]
    assert res.region_final["A.h:10"] == "ff"
    assert h.reverted == ["sha0"]                 # only the float attempt reverted


# --- first-accept-per-region-wins --------------------------------------------
def test_cheaper_rung_accept_skips_remaining_rungs_of_region():
    # float holds -> ff for the same region must be skipped, never applied
    q = [_cand("A.h:10", "float"), _cand("A.h:10", "ff")]
    h = FakeHarness({("A.h:10", "float"): 8.5, ("A.h:10", "ff"): 8.6})
    res = h.run(q)
    assert [o.outcome for o in res.outcomes] == [ACCEPTED, SKIPPED_RESOLVED]
    assert ("A.h:10", "ff") not in h.applied      # never built
    assert res.region_final["A.h:10"] == "float"


# --- gate boundary (regression-relative) --------------------------------------
def test_gate_is_inclusive_at_exactly_the_threshold():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): THRESHOLD})   # exactly baseline - margin
    res = h.run(q)
    assert res.accepted


def test_gate_rejects_just_below_the_threshold():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): THRESHOLD - 1e-4})
    res = h.run(q)
    assert res.rejected


def test_margin_widens_the_accept_bar():
    # A candidate 1 digit below baseline is rejected at margin 0.5 but accepted at 1.5.
    q = [_cand("A.h:10", "float")]
    below = BASELINE - 1.0
    assert FakeHarness({("A.h:10", "float"): below}).run(q, margin=0.5).rejected
    assert FakeHarness({("A.h:10", "float"): below}).run(q, margin=1.5).accepted


# --- low-but-defined baseline is NOT a stop (the whole point) ------------------
def test_low_baseline_does_not_stop_and_admits_harmless_candidate():
    # B12-like: baseline 3.69 (a physics cancellation floor, well below any absolute
    # 6.0).  Regression-relative admits a candidate that leaves the floor untouched.
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 3.69}, baseline=3.69)
    res = h.run(q)
    assert res.stopped is None
    assert res.accepted                      # 3.69 >= 3.69 - 0.5
    assert res.baseline_min == pytest.approx(3.69)


def test_low_baseline_still_rejects_a_worsening_candidate():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 2.5}, baseline=3.69)  # 2.5 < 3.19
    res = h.run(q)
    assert res.stopped is None
    assert res.rejected


# --- apply failure ------------------------------------------------------------
def test_apply_failure_is_recorded_not_reverted_by_solver():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    h = FakeHarness({("B.h:20", "float"): 8.5},
                    fail_keys={("A.h:10", "float")})
    res = h.run(q)
    assert res.outcomes[0].outcome == APPLY_FAILED
    assert res.outcomes[0].patcher_status == "llm_gen_failed"
    assert h.reverted == []                        # patcher owns its own reset
    assert res.outcomes[1].outcome == ACCEPTED     # walk continues
    assert res.region_final["A.h:10"] == "double"


# --- structural-unimplementable stop (unscoreable baseline) -------------------
def test_unscoreable_baseline_stops_and_flags():
    q = [_cand("A.h:10", "float"), _cand("B.h:20", "float")]
    # baseline (curr_min) comes back None => nothing to gate against
    h = FakeHarness({("A.h:10", "float"): 3.0}, baseline=None)
    res = h.run(q)
    assert res.stopped == STOPPED_GATE_UNIMPLEMENTABLE
    assert res.outcomes[-1].outcome == STOPPED_GATE_UNIMPLEMENTABLE
    assert h.reverted == ["sha0"]                  # first candidate rolled back
    assert len(res.outcomes) == 1                  # walk halted, B.h:20 untried
    assert ("B.h:20", "float") not in h.applied


def test_nan_baseline_stops_and_flags():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 3.0}, baseline=float("nan"))
    res = h.run(q)
    assert res.stopped == STOPPED_GATE_UNIMPLEMENTABLE


# --- empty queue --------------------------------------------------------------
def test_empty_queue_no_builds():
    h = FakeHarness({})
    res = h.run([])
    assert res.outcomes == []
    assert res.baseline_min is None
    assert res.accept_threshold is None
    assert h.applied == []


def test_all_regions_seeded_at_double():
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 8.5})
    res = solve(q, apply_fn=h.apply, validate_fn=h.validate, revert_fn=h.revert,
                head_fn=h.head, margin=MARGIN, all_region_ids={"A.h:10", "C.h:99"})
    assert res.region_final["C.h:99"] == "double"   # region with no candidate
    assert res.region_final["A.h:10"] == "float"


# --- Phase 2f: chain_dd positive-lift gate ------------------------------------
def test_chain_accepts_on_own_lift():
    # First candidate: accumulated == baseline; a chain lifting >= baseline + 0.5 accepts.
    q = [_chain_cand("cascade1")]
    h = FakeHarness({("cascade1", "chain_dd"): BASELINE + 1.0})
    res = h.run(q)
    assert len(res.accepted) == 1
    assert res.region_final["cascade1"] == "chain_dd"
    assert res.final_min == BASELINE + 1.0
    assert "chain lift" in res.accepted[0].reason


def test_chain_no_lift_rejected():
    # cand_min within [baseline-0.5, baseline+0.5): not a regression, but no real lift.
    q = [_chain_cand("cascade1")]
    h = FakeHarness({("cascade1", "chain_dd"): BASELINE + 0.2})
    res = h.run(q)
    assert len(res.rejected) == 1
    assert res.rejected[0].reason_tag == "chain_no_lift"
    assert h.reverted == ["sha0"]                 # patch reverted to parent


def test_chain_regression_rejected():
    q = [_chain_cand("cascade1")]
    h = FakeHarness({("cascade1", "chain_dd"): BASELINE - 2.0})   # < baseline - margin
    res = h.run(q)
    assert res.rejected[0].reason_tag == "chain_regression"


def test_chain_ride_along_is_closed():
    # chain1 lifts baseline -> baseline+1 (accepted); chain2 adds NOTHING beyond that,
    # so its cand_min == accumulated -> bar = accumulated + 0.5 -> rejected (no_lift).
    # This is the fix for the literal-fixed-baseline hole (a free ride on chain1's lift).
    q = [_chain_cand("cascade1"), _chain_cand("cascade2")]
    h = FakeHarness({("cascade1", "chain_dd"): BASELINE + 1.0,
                     ("cascade2", "chain_dd"): BASELINE + 1.0})
    res = h.run(q)
    assert res.region_final["cascade1"] == "chain_dd"      # accepted
    assert "cascade2" not in {o.candidate.region_id for o in res.accepted}
    c2 = [o for o in res.rejected if o.candidate.region_id == "cascade2"][0]
    assert c2.reason_tag == "chain_no_lift"


def test_chain_second_stacks_when_it_adds_lift():
    # chain2 DOES add >= 0.5 beyond chain1's accumulated -> both accepted (stacking).
    q = [_chain_cand("cascade1"), _chain_cand("cascade2")]
    h = FakeHarness({("cascade1", "chain_dd"): BASELINE + 1.0,
                     ("cascade2", "chain_dd"): BASELINE + 1.6})   # +0.6 over accumulated
    res = h.run(q)
    assert {o.candidate.region_id for o in res.accepted} == {"cascade1", "cascade2"}
    assert res.final_min == BASELINE + 1.6


def test_single_region_gate_unchanged_by_chain_addition():
    # A plain single-region candidate still uses the regression-relative baseline gate
    # and its reason string is unchanged (byte-for-byte "baseline ... - margin").
    q = [_cand("A.h:10", "float")]
    h = FakeHarness({("A.h:10", "float"): 8.5})   # >= THRESHOLD 8.34
    res = h.run(q)
    assert len(res.accepted) == 1
    assert res.accepted[0].reason_tag is None
    assert "baseline" in res.accepted[0].reason and "margin" in res.accepted[0].reason
