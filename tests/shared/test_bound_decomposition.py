"""Layer-0 tests for the shared bound-decomposition arithmetic (Phase 2f).

These lock the extracted Item 6/7 math so the analysis script and the live pipeline
selection cannot silently drift apart.
"""

from __future__ import annotations

import math

import pytest

from agents.shared.bound_decomposition import (
    U_DOUBLE, U_DD, DD_DIGITS, DD_LIFT_ORDERS, TIGHT_LO, TIGHT_HI,
    predict, _predict, chain_row, chain_tightness, chain_is_computed,
    chain_predicted_lift,
)


def _chain(sens, measured, *, chain_id="c0", lines=(("f.h", 10),)):
    return {
        "chain_id": chain_id,
        "signal_class": "cancellation_cascade",
        "n": len(lines),
        "chain": [{"file": f, "line_start": ln} for f, ln in lines],
        "ops": {"sub": 3},
        "max_cond": 1e2,
        "max_sensitivity": sens,
        "max_rel_err": measured,
    }


def test_predict_scales_by_unit_roundoff():
    p = predict(1.0e6)
    assert p["predicted_rel_err_if_double"] == U_DOUBLE * 1.0e6
    assert p["predicted_rel_err_if_dd"] == U_DD * 1.0e6
    # dd is ~15.95 orders tighter than double for the same sensitivity.
    ratio = p["predicted_rel_err_if_double"] / p["predicted_rel_err_if_dd"]
    assert math.isclose(math.log10(ratio), DD_LIFT_ORDERS, abs_tol=0.01)


def test_predict_alias_is_the_public_fn():
    assert _predict is predict


def test_chain_row_tightness_is_predicted_double_over_measured():
    sens = 1.0e8
    pd = U_DOUBLE * sens
    measured = pd  # perfectly tight
    row = chain_row(_chain(sens, measured))
    assert row["tightness_double_over_measured"] == pytest.approx(1.0)
    assert row["chain_lines"] == ["f.h:10"]
    assert row["max_sensitivity"] == sens


def test_chain_row_zero_measured_gives_none_tightness():
    row = chain_row(_chain(1.0e8, 0.0))
    assert row["tightness_double_over_measured"] is None
    assert chain_tightness(row) is None


def test_chain_is_computed_band():
    sens = 1.0e8
    pd = U_DOUBLE * sens
    # tightness exactly 1 -> COMPUTED
    assert chain_is_computed(chain_row(_chain(sens, pd)))
    # tightness at the low edge (TIGHT_LO) -> COMPUTED (inclusive)
    assert chain_is_computed(chain_row(_chain(sens, pd / TIGHT_LO)))
    # tightness at the high edge (TIGHT_HI) -> COMPUTED (inclusive)
    assert chain_is_computed(chain_row(_chain(sens, pd / TIGHT_HI)))
    # tightness far below TIGHT_LO (loose bound / analytic) -> NOT computed
    assert not chain_is_computed(chain_row(_chain(sens, pd / (TIGHT_LO / 100))))
    # tightness far above TIGHT_HI (over-predicting) -> NOT computed
    assert not chain_is_computed(chain_row(_chain(sens, pd / (TIGHT_HI * 100))))
    # no measured error -> NOT computed (nothing to explain)
    assert not chain_is_computed(chain_row(_chain(sens, 0.0)))


def test_chain_predicted_lift_tight_chain_is_capped_lift():
    # A tight chain at ~5 measured digits: measured = 1e-5.
    sens = 1.0e-5 / U_DOUBLE
    row = chain_row(_chain(sens, 1.0e-5))
    lift = chain_predicted_lift(row)
    # digits_now = 5; digits_after_dd = min(DD_DIGITS, -log10(U_dd*sens)).
    digits_after = min(DD_DIGITS, -math.log10(U_DD * sens))
    assert lift == pytest.approx(digits_after - 5.0, abs=1e-9)
    assert lift > 0.0


def test_chain_predicted_lift_monotone_in_measured_error():
    # More cancellation (larger measured error) at fixed sens -> larger lift.
    sens = 1.0e12
    small = chain_predicted_lift(chain_row(_chain(sens, 1.0e-8)))
    large = chain_predicted_lift(chain_row(_chain(sens, 1.0e-3)))
    assert large > small


def test_chain_predicted_lift_zero_when_no_measured_error():
    assert chain_predicted_lift(chain_row(_chain(1.0e8, 0.0))) == 0.0


def test_chain_predicted_lift_never_exceeds_dd_digits():
    # Catastrophic cancellation: huge measured error, huge sens.
    row = chain_row(_chain(1.0e20, 1.0e-1))
    assert 0.0 <= chain_predicted_lift(row) <= DD_DIGITS
