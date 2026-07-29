"""Deliverable 5/6 — acceptance gate (build AND lift), uniform (Decision 3).

Upshift (Phase-1, double->dd) must GAIN digits; downshift (Phase-2, double->float) must
PRESERVE them (lift >= 0).  Same predicate shape, direction-selected threshold.
"""

from __future__ import annotations

from agents.patcher.flip_gate import (
    GateInputs, LiftDirection, evaluate, evaluate_all)


def test_built_and_positive_lift_accepts():
    d = evaluate(GateInputs("B10", built=True, baseline_digits=1.5, candidate_digits=15.9))
    assert d.accept is True
    assert d.lift == 15.9 - 1.5


def test_build_failed_rejects():
    d = evaluate(GateInputs("B12", built=False, baseline_digits=None, candidate_digits=None))
    assert d.accept is False
    assert "build_failed" in d.reason


def test_zero_lift_rejects_uniformly():
    # B14-like: already accurate, no lift. Rejected by the SAME rule, no special case.
    d = evaluate(GateInputs("B14", built=True, baseline_digits=13.19, candidate_digits=13.19))
    assert d.accept is False
    assert "no_lift" in d.reason
    assert d.lift == 0.0


def test_negative_lift_rejects():
    d = evaluate(GateInputs("Bx", built=True, baseline_digits=10.0, candidate_digits=9.0))
    assert d.accept is False
    assert d.lift == -1.0


def test_dd_insufficient_builds_but_no_lift_rejected():
    # B16/BIN*: clean dd build, but cancellation > dd budget -> narrowed == raw double.
    d = evaluate(GateInputs("B16", built=True, baseline_digits=0.0, candidate_digits=0.0))
    assert d.accept is False
    assert "no_lift" in d.reason


def test_unmeasurable_rejects():
    d = evaluate(GateInputs("Bz", built=True, baseline_digits=None, candidate_digits=5.0))
    assert d.accept is False
    assert "unmeasurable" in d.reason


def test_margin_is_a_uniform_threshold_not_a_bucket():
    # A 0.4-digit lift is accepted at margin 0.0 but rejected at margin 0.5 — same rule,
    # different threshold, applied to every integral identically.
    lo = evaluate(GateInputs("B", True, 10.0, 10.4), margin=0.0)
    hi = evaluate(GateInputs("B", True, 10.0, 10.4), margin=0.5)
    assert lo.accept is True
    assert hi.accept is False


def test_evaluate_all():
    res = evaluate_all([
        GateInputs("B10", True, 1.5, 15.9),
        GateInputs("B14", True, 13.19, 13.19),
        GateInputs("B12", False, None, None),
    ])
    assert res["B10"].accept is True
    assert res["B14"].accept is False
    assert res["B12"].accept is False


# --------------------------------------------------------------------------- #
# deliverable 6 (Phase-2) — downshift direction
# --------------------------------------------------------------------------- #

def test_downshift_accepts_precision_preserving_flip():
    # A well-conditioned integral holds its digits at float (lift 0.0) -> accepted.
    d = evaluate(GateInputs("B1", True, 15.9, 15.9), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is True
    assert d.lift == 0.0
    assert "downshift preserves precision" in d.reason


def test_downshift_accepts_incidental_gain():
    d = evaluate(GateInputs("B2", True, 15.0, 15.2), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is True


def test_downshift_rejects_precision_loss():
    # An ill-conditioned integral loses digits at float -> rejected back to raw double.
    d = evaluate(GateInputs("B3", True, 15.9, 12.0), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is False
    assert "precision_loss" in d.reason
    assert d.lift == 12.0 - 15.9


def test_downshift_margin_tightens_tolerance():
    # margin 0.5 tolerates a 0.3-digit loss but not a 0.8-digit one.
    ok = evaluate(GateInputs("B", True, 15.9, 15.6), margin=0.5,
                  direction=LiftDirection.DOWNSHIFT)
    bad = evaluate(GateInputs("B", True, 15.9, 15.1), margin=0.5,
                   direction=LiftDirection.DOWNSHIFT)
    assert ok.accept is True
    assert bad.accept is False


def test_downshift_build_failed_still_rejects():
    d = evaluate(GateInputs("B4", False, None, None), direction=LiftDirection.DOWNSHIFT)
    assert d.accept is False
    assert "build_failed" in d.reason


def test_same_numbers_opposite_verdict_by_direction():
    # lift 0.0: UPSHIFT rejects (no gain), DOWNSHIFT accepts (preserved).
    up = evaluate(GateInputs("B", True, 15.9, 15.9), direction=LiftDirection.UPSHIFT)
    down = evaluate(GateInputs("B", True, 15.9, 15.9), direction=LiftDirection.DOWNSHIFT)
    assert up.accept is False
    assert down.accept is True


def test_upshift_is_the_default_direction():
    # Back-compat: no direction arg == UPSHIFT (Phase-1 behavior unchanged).
    d = evaluate(GateInputs("B10", True, 1.5, 15.9))
    assert d.accept is True
    same = evaluate(GateInputs("B10", True, 1.5, 15.9), direction=LiftDirection.UPSHIFT)
    assert d == same
