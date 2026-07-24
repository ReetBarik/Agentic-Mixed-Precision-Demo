"""P6 dispatch table — all Patcher statuses (design: "P6")."""

import pytest

from agents.patcher import result as R
from agents.strategy.dispatch import DISPATCH, dispatch


def test_all_statuses_present():
    assert set(DISPATCH) == {
        "ok", "build_failed", "runtime_nan", "runtime_crashed",
        "llm_gen_failed", "patch_apply_failed", "timeout", "commit_failed",
        "empty_candidate", "patch_inapplicable", "promotion_no_op",
        "write_truncation",
    }


def test_dispatch_covers_every_patcher_status():
    # The guard that would have caught the Phase 2c drift: the dispatch table must
    # handle exactly the statuses the Patcher can return — no missing key (a real
    # status the loop would crash on) and no stale one.
    assert set(DISPATCH) == set(R.STATUSES)


def test_patch_inapplicable_is_benign_advance():
    # A `-to-float` rung inapplicable to template code: advance the walk, but do
    # NOT tag it a strategy_bug, count it vs budget, or (see agent) bump the DR
    # streak.  It is not a signal about the intent — the transition just doesn't
    # apply to this region.
    e = dispatch("patch_inapplicable")
    assert e.action == "advance"
    assert e.log_tag == "patch_inapplicable"
    assert e.counts_budget is False
    assert e.is_reject is False


def test_empty_candidate_advances_non_fatal():
    # gen+build ok but candidate == parent: benign no-op, NOT fatal commit_failed.
    e = dispatch("empty_candidate")
    assert e.action == "advance"          # advances the walk, run continues
    assert e.is_reject and e.counts_budget
    assert e.log_tag == "empty_candidate"


def test_ok_hands_to_validator():
    e = dispatch("ok")
    assert e.action == "validate" and not e.is_reject and e.counts_budget


def test_bucket_a_advances_and_counts_budget():
    for status, tag in [("build_failed", "compile"),
                        ("runtime_nan", "runtime_nan"),
                        ("runtime_crashed", "runtime_crash")]:
        e = dispatch(status)
        assert e.action == "advance"
        assert e.is_reject and e.counts_budget
        assert e.log_tag == tag


def test_llm_gen_failed_is_terminal_and_free():
    e = dispatch("llm_gen_failed")
    assert e.action == "advance_terminal"     # P6b: never retried this run
    assert e.log_tag == "llm_capacity"
    assert e.counts_budget is False           # doesn't count vs budget


def test_promotion_no_op_is_terminal_and_free():
    # Phase 2c: an empty promotion payload is a deterministic, rung-independent
    # structural miss — terminal for the intent (like llm_gen_failed), git-only so
    # free vs budget, but a real reject (unlike patch_inapplicable).
    e = dispatch("promotion_no_op")
    assert e.action == "advance_terminal"     # no rung would promote anything
    assert e.log_tag == "promotion_no_op"
    assert e.counts_budget is False
    assert e.is_reject is True


def test_write_truncation_is_terminal_and_free():
    # Phase 2d-B: an upcast that truncates every landing back to caller precision is a
    # deterministic, rung-fixed inert promotion — terminal for the intent (a wider rung
    # truncates identically), git-only so free vs budget, and a real reject.
    e = dispatch("write_truncation")
    assert e.action == "advance_terminal"
    assert e.log_tag == "write_truncation"
    assert e.counts_budget is False
    assert e.is_reject is True


def test_patch_apply_failed_is_strategy_bug_and_free():
    e = dispatch("patch_apply_failed")
    assert e.action == "skip_intent"
    assert e.log_tag == "strategy_bug"
    assert e.counts_budget is False


def test_timeout_retries_once():
    e = dispatch("timeout")
    assert e.action == "retry_once" and e.log_tag == "timeout"


def test_commit_failed_is_fatal():
    e = dispatch("commit_failed")
    assert e.action == "fatal" and e.log_tag == "fatal"


def test_non_ok_statuses_flag_dd_untested():
    # P6a: any Patcher failure at the DD rung means DD was never honestly tested.
    for status in ["build_failed", "runtime_nan", "runtime_crashed",
                   "llm_gen_failed", "patch_apply_failed", "timeout",
                   "empty_candidate", "patch_inapplicable", "promotion_no_op",
                   "write_truncation"]:
        assert dispatch(status).dd_untested is True
    assert dispatch("ok").dd_untested is False


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        dispatch("teleported")
