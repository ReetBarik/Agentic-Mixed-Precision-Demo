"""Deliverable 5/6 — acceptance gate, tolerance contract.

The gate decides against a user-supplied precise-digit bar (``tolerance``), not against the
raw-double baseline.  UPSHIFT (double->dd) accepts when it reaches the bar or (below the bar)
strictly gains, and is a no-op when the baseline already clears the bar.  DOWNSHIFT
(double->float/ff) accepts iff the candidate still clears the bar — headroom above it is
disposable.  ``margin`` is a strictness buffer: the effective bar is ``tolerance + margin``.
"""

from __future__ import annotations

import pytest

from agents.patcher.flip_gate import (
    GateInputs, LiftDirection, evaluate, evaluate_all)

TOL = 10.0


# --------------------------------------------------------------------------- #
# UPSHIFT (Phase-1, double -> dd)
# --------------------------------------------------------------------------- #

def test_upshift_baseline_clears_tol_is_no_flip_needed():
    # Well-conditioned dd candidate: raw double already >= tol -> no-op, not accept.
    d = evaluate(GateInputs("B", True, baseline_digits=12.0, candidate_digits=15.9,
                            tolerance=TOL))
    assert d.accept is False
    assert d.no_flip_needed is True
    assert d.terminal_good is True
    assert "no_flip_needed" in d.reason


def test_upshift_candidate_clears_tol_accepts():
    # baseline below tol, candidate clears tol -> accept (regardless of lift size).
    d = evaluate(GateInputs("B10", True, baseline_digits=1.5, candidate_digits=15.9,
                            tolerance=TOL))
    assert d.accept is True
    assert d.no_flip_needed is False
    assert d.lift == pytest.approx(15.9 - 1.5)


def test_upshift_below_tol_but_strict_lift_accepts():
    # baseline and candidate both < tol, but candidate strictly improves -> accept.
    d = evaluate(GateInputs("Bx", True, baseline_digits=2.0, candidate_digits=5.0,
                            tolerance=TOL))
    assert d.accept is True
    assert d.lift == pytest.approx(3.0)


def test_upshift_below_tol_no_lift_rejects():
    # dd-insufficient: builds, below tol, no lift -> reject back to raw double.
    d = evaluate(GateInputs("B16", True, baseline_digits=0.0, candidate_digits=0.0,
                            tolerance=TOL))
    assert d.accept is False
    assert d.no_flip_needed is False
    assert "below_tolerance" in d.reason


def test_upshift_below_tol_negative_lift_rejects():
    d = evaluate(GateInputs("Bx", True, baseline_digits=5.0, candidate_digits=4.0,
                            tolerance=TOL))
    assert d.accept is False
    assert d.lift == pytest.approx(-1.0)


def test_upshift_build_failed_rejects():
    d = evaluate(GateInputs("B12", False, baseline_digits=None, candidate_digits=None,
                            tolerance=TOL))
    assert d.accept is False
    assert "build_failed" in d.reason


def test_upshift_build_failed_but_baseline_clears_tol_is_no_flip():
    # If the baseline already clears the bar, a failed build is moot — it's a no-op.
    d = evaluate(GateInputs("B", False, baseline_digits=12.0, candidate_digits=None,
                            tolerance=TOL))
    assert d.no_flip_needed is True
    assert d.accept is False


def test_upshift_unmeasurable_rejects():
    d = evaluate(GateInputs("Bz", True, baseline_digits=2.0, candidate_digits=None,
                            tolerance=TOL))
    assert d.accept is False
    assert "unmeasurable" in d.reason


# --------------------------------------------------------------------------- #
# DOWNSHIFT (Phase-2, double -> float/ff)
# --------------------------------------------------------------------------- #

def test_downshift_candidate_clears_tol_accepts_even_with_loss():
    # ff delivers 10.5 digits, baseline was 15.9 -> lift negative but clears tol -> accept.
    d = evaluate(GateInputs("B1", True, baseline_digits=15.9, candidate_digits=10.5,
                            tolerance=TOL), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is True
    assert d.lift == pytest.approx(10.5 - 15.9)
    assert "clears tolerance" in d.reason


def test_downshift_below_tol_rejects():
    # float delivers 3 digits under a 10-digit bar -> reject back to raw double.
    d = evaluate(GateInputs("B3", True, baseline_digits=15.9, candidate_digits=3.0,
                            tolerance=TOL), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is False
    assert "below_tolerance" in d.reason


def test_downshift_build_failed_rejects():
    d = evaluate(GateInputs("B4", False, baseline_digits=None, candidate_digits=None,
                            tolerance=TOL), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is False
    assert "build_failed" in d.reason


def test_downshift_no_baseline_still_decides_on_candidate():
    # Downshift accept is a bar check on the candidate; baseline only informs lift display.
    d = evaluate(GateInputs("B", True, baseline_digits=None, candidate_digits=11.0,
                            tolerance=TOL), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is True
    assert d.lift is None


# --------------------------------------------------------------------------- #
# margin as a strictness buffer on top of the bar
# --------------------------------------------------------------------------- #

def test_margin_raises_the_bar_upshift():
    # Small lift (0.3) with candidate 10.2 near the bar: at margin 0.0 the candidate clears
    # tol 10.0 -> accept; at margin 0.5 the bar is 10.5 (candidate below) AND the 0.3 lift is
    # <= margin, so the strict-improvement branch also declines -> reject.
    lo = evaluate(GateInputs("B", True, 9.9, 10.2, tolerance=TOL), margin=0.0)
    hi = evaluate(GateInputs("B", True, 9.9, 10.2, tolerance=TOL), margin=0.5)
    assert lo.accept is True
    assert hi.accept is False


def test_margin_raises_the_bar_downshift():
    lo = evaluate(GateInputs("B", True, 15.9, 10.2, tolerance=TOL), margin=0.0,
                  direction=LiftDirection.DOWNSHIFT)
    hi = evaluate(GateInputs("B", True, 15.9, 10.2, tolerance=TOL), margin=0.5,
                  direction=LiftDirection.DOWNSHIFT)
    assert lo.accept is True
    assert hi.accept is False


# --------------------------------------------------------------------------- #
# tolerance is required (fail loud, no silent default)
# --------------------------------------------------------------------------- #

def test_tolerance_required_fails_loud():
    with pytest.raises(ValueError, match="tolerance is required"):
        evaluate(GateInputs("B", True, 1.0, 15.9, tolerance=None))


# --------------------------------------------------------------------------- #
# direction contrast + evaluate_all
# --------------------------------------------------------------------------- #

def test_same_numbers_opposite_verdict_by_direction():
    # candidate below the bar: UPSHIFT rejects (no lift), DOWNSHIFT rejects (below bar too).
    # candidate above the bar with a loss: DOWNSHIFT accepts, UPSHIFT no-op (baseline clears).
    up = evaluate(GateInputs("B", True, 15.9, 11.0, tolerance=TOL),
                  direction=LiftDirection.UPSHIFT)
    down = evaluate(GateInputs("B", True, 15.9, 11.0, tolerance=TOL),
                    direction=LiftDirection.DOWNSHIFT)
    assert up.no_flip_needed is True     # baseline 15.9 already clears tol -> upshift no-op
    assert down.accept is True           # candidate 11.0 clears tol -> downshift accept


def test_upshift_is_the_default_direction():
    d = evaluate(GateInputs("B10", True, 1.5, 15.9, tolerance=TOL))
    same = evaluate(GateInputs("B10", True, 1.5, 15.9, tolerance=TOL),
                    direction=LiftDirection.UPSHIFT)
    assert d == same


def test_evaluate_all():
    res = evaluate_all([
        GateInputs("B10", True, 1.5, 15.9, tolerance=TOL),   # accept (clears tol)
        GateInputs("B14", True, 13.19, 13.19, tolerance=TOL),  # no-op (baseline clears)
        GateInputs("B16", True, 0.0, 0.0, tolerance=TOL),    # reject (below tol, no lift)
        GateInputs("B12", False, None, None, tolerance=TOL),  # reject (build failed)
    ])
    assert res["B10"].accept is True
    assert res["B14"].no_flip_needed is True and res["B14"].accept is False
    assert res["B16"].accept is False and res["B16"].no_flip_needed is False
    assert res["B12"].accept is False
