"""P6 dispatch table — all Patcher statuses (design: "P6")."""

import pytest

from agents.strategy.dispatch import DISPATCH, dispatch


def test_all_statuses_present():
    assert set(DISPATCH) == {
        "ok", "build_failed", "runtime_nan", "runtime_crashed",
        "llm_gen_failed", "patch_apply_failed", "timeout", "commit_failed",
        "empty_candidate",
    }


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
                   "empty_candidate"]:
        assert dispatch(status).dd_untested is True
    assert dispatch("ok").dd_untested is False


def test_unknown_status_raises():
    with pytest.raises(ValueError):
        dispatch("teleported")
