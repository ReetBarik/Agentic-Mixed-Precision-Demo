"""Delta-primary verdict gate (agents.validator.validate._decide)."""

from agents.validator.validate import _decide


def test_identity_patch_accepts():
    # candidate == current (delta 0) always accepts, even at a low absolute min.
    verdict, reason = _decide(9.2, 9.2, 0.5)
    assert verdict == "accept" and reason == "accept"


def test_small_noise_within_tolerance_accepts():
    # a fraction-of-a-digit wobble under tolerance is fine.
    verdict, reason = _decide(9.0, 9.2, 0.5)
    assert verdict == "accept" and reason == "accept"


def test_regression_beyond_tolerance_rejects():
    verdict, reason = _decide(7.0, 9.2, 0.5)
    assert verdict == "reject" and reason == "regression"


def test_improvement_accepts():
    # a patch that improves precision is trivially accepted.
    verdict, reason = _decide(15.0, 9.2, 0.5)
    assert verdict == "accept" and reason == "accept"


def test_delta_primary_ignores_absolute_floor_by_default():
    # low absolute min but no regression -> accept (floor is None in delta mode).
    verdict, reason = _decide(2.0, 2.0, 0.5)
    assert verdict == "accept" and reason == "accept"


def test_floor_marks_insufficient_fix():
    # optional absolute expectation: no regression but below floor -> not enough.
    verdict, reason = _decide(9.2, 9.2, 0.5, floor=12.0)
    assert verdict == "reject" and reason == "insufficient_fix"


def test_regression_beats_floor_reason():
    # a real regression is reported as regression even when a floor is set.
    verdict, reason = _decide(6.0, 9.2, 0.5, floor=12.0)
    assert verdict == "reject" and reason == "regression"
