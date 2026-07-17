"""Precise-digits formula: edge cases + Decimal-oracle vs fast-path agreement."""

import math
import random
from decimal import Decimal

from agents.validator.precise_digits import (
    MAX_DIGITS, MAX_DIGITS_F, precise_digits, precise_digits_fast,
)


def _D(x):
    return Decimal(x)


# ---- edge cases (Decimal oracle) ------------------------------------------

def test_exact_match_is_max():
    assert precise_digits(_D("1.5"), _D("1.5")) == MAX_DIGITS
    assert precise_digits(_D(0), _D(0)) == MAX_DIGITS  # both zero


def test_true_zero_nonzero_err_is_zero():
    assert precise_digits(_D("1e-9"), _D(0)) == 0


def test_error_exceeds_true_is_zero():
    # |cand - ref| > |ref|  ->  0 digits
    assert precise_digits(_D("3.0"), _D("1.0")) == 0


def test_below_dd_resolution_is_max():
    # relative error 2**-110 < 2**-106  ->  max digits
    ref = _D(1)
    err = _D(2) ** -110
    assert precise_digits(ref + err, ref) == MAX_DIGITS


def test_ten_digit_error_reads_ten():
    # rel err = 1e-10  ->  exactly 10 precise digits
    ref = _D("1.0")
    d = precise_digits(ref * (1 + _D("1e-10")), ref)
    assert abs(float(d) - 10.0) < 1e-9


def test_ref_scale_effectively_zero_is_max():
    # both |true| and |err| below 1e-30 * ref_scale -> treated as zero -> max
    d = precise_digits(_D("1e-50"), _D("2e-50"), ref_scale=_D("1.0"))
    assert d == MAX_DIGITS
    # without ref_scale, the same near-zero terms read as ~0.3 noise digits
    # (rel err 0.5) — which is exactly the _ieps50 artifact ref_scale rescues.
    assert precise_digits(_D("1e-50"), _D("2e-50")) < 1


def test_max_digits_value():
    assert abs(float(MAX_DIGITS) - 106 * math.log10(2)) < 1e-9
    assert abs(MAX_DIGITS_F - 106 * math.log10(2)) < 1e-12


# ---- fast float path mirrors the Decimal oracle ---------------------------

def test_fast_matches_oracle_on_double_vs_dd():
    rng = random.Random(20260717)
    worst = 0.0
    for _ in range(4000):
        # a "true" DD value: hi + lo (lo ~ 1e-16 * hi)
        hi = rng.uniform(-1e6, 1e6) or 1.0
        lo = hi * rng.uniform(-1e-16, 1e-16)
        ref = Decimal(hi) + Decimal(lo)
        # a candidate double a few ulps off (spans ~0..16 digits of agreement)
        rel = 10 ** rng.uniform(-16, -1)
        cand = hi * (1 + (rel if rng.random() < 0.5 else -rel))
        d_oracle = float(precise_digits(Decimal(cand), ref))
        d_fast = precise_digits_fast(cand, 0.0, hi, lo)
        worst = max(worst, abs(d_oracle - d_fast))
    # fast path uses double arithmetic for the cancellation; agreement well
    # within 0.01 digit is all the verdict needs.
    assert worst < 0.01, f"worst oracle-vs-fast gap {worst}"


def test_fast_dd_self_comparison_is_max():
    # candidate == reference (DD vs DD): err == 0 -> max digits
    hi, lo = 3.14159, 3.14159e-17
    assert precise_digits_fast(hi, lo, hi, lo) == MAX_DIGITS_F
