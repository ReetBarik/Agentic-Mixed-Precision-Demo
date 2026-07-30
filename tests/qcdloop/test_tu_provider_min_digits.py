"""``tu_provider._min_digits`` ref_scale plumbing (PHASE_2_TU_E2E_REFSCALE).

The whole-TU walk's per-integral digit metric must match the Validator's
``agents.validator.validate._score`` convention: each sample carries a
``ref_scale`` (the max ``|DD-reference component|`` across that sample's six
coeffs) so a component whose DD reference is an analytic zero against the sample
scale reports at the DD cap (:data:`MAX_DIGITS_F`) rather than as spurious
0-digit roundoff noise.

Motivating diagnosis: ``runs/qcdloop/PHASE_2_B15_TWO_LIMB_TRACE_2026-07-30.md``
— B14/B16 route ``double`` only because ``_min_digits`` was called without
``ref_scale``; every genuine (double-resolvable) component already agrees with
the oracle to ~15 digits.
"""

from __future__ import annotations

from array import array

from agents.validator.coeffs import N_COMPONENTS
from agents.validator.precise_digits import MAX_DIGITS_F, ZERO_REF_TOL
from runs.qcdloop.tu_provider import _min_digits


def _arrays(samples):
    """Build a ``(hi, lo)`` flat-array pair from a list of per-sample component lists.

    ``samples[i]`` is a list of ``N_COMPONENTS`` ``(hi, lo)`` pairs.
    """
    total = len(samples)
    hi = array("d", bytes(8 * total * N_COMPONENTS))
    lo = array("d", bytes(8 * total * N_COMPONENTS))
    for s, comps in enumerate(samples):
        assert len(comps) == N_COMPONENTS
        base = s * N_COMPONENTS
        for c, (h, l) in enumerate(comps):
            hi[base + c] = h
            lo[base + c] = l
    return (hi, lo)


def _one_sample(comps):
    return {"X": _arrays([comps])}


def test_ref_scale_rescues_analytic_zero_component():
    """A component ~1e-30 of the sample scale, disagreeing at roundoff, must report
    MAX_DIGITS (not ~0.0) once ref_scale is supplied — the B14/B16 restoration."""
    # sample scale ~ 1.0 (comp0); comp1 is a genuine analytic zero at ~1e-30 whose
    # candidate/oracle disagree completely (rel err ~1) — pure roundoff noise.
    oracle = _one_sample([
        (1.0, 0.0),            # comp0: the scale-setting genuine value
        (2.0e-30, 0.0),        # comp1: analytic zero vs scale (2e-30 / 1.0 < 1e-24)
        (0.5, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    ])
    candidate = _one_sample([
        (1.0, 0.0),            # comp0 matches exactly -> MAX
        (-9.0e-30, 0.0),       # comp1 totally disagrees at the noise floor
        (0.5, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    ])
    d = _min_digits(candidate, oracle, "X", 1)
    # ref_scale = max|oracle| = 1.0; 2e-30 < ZERO_REF_TOL*1.0 -> comp1 is analytic zero
    assert 2.0e-30 < ZERO_REF_TOL * 1.0
    assert d == MAX_DIGITS_F, (
        f"analytic-zero component must report the DD cap, got {d}")


def test_without_ref_scale_the_same_component_would_read_zero():
    """Guard: the same noise-floor disagreement WITHOUT ref_scale reads as 0.0.

    Documents exactly what the plumbing fixes (this is the pre-fix behavior, called
    at the primitive level — NOT how ``_min_digits`` is invoked anymore)."""
    from agents.validator.precise_digits import precise_digits_fast
    # comp1: candidate/oracle disagree completely at ~1e-30 -> rel err ~1 -> 0 digits
    d_no_scale = precise_digits_fast(-9.0e-30, 0.0, 2.0e-30, 0.0)
    assert d_no_scale == 0.0
    d_with_scale = precise_digits_fast(-9.0e-30, 0.0, 2.0e-30, 0.0, ref_scale=1.0)
    assert d_with_scale == MAX_DIGITS_F


def test_genuine_signal_unaffected_by_ref_scale():
    """A genuine (double-resolvable) component keeps its real digit count — ref_scale
    must NOT inflate genuine signal (only rescue sub-scale analytic zeros)."""
    # comp0 candidate off by 1e-9 relative -> ~9 digits; well above ZERO_REF_TOL*scale.
    oracle = _one_sample([
        (1.0, 0.0), (0.7, 0.0), (0.3, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    ])
    candidate = _one_sample([
        (1.0 + 1.0e-9, 0.0),   # comp0: rel err 1e-9 -> ~9 digits (genuine)
        (0.7, 0.0), (0.3, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
    ])
    d = _min_digits(candidate, oracle, "X", 1)
    assert 8.5 < d < 9.5, f"genuine ~9-digit signal must survive, got {d}"


def test_ref_scale_is_per_sample_from_oracle():
    """ref_scale is computed per-sample from the ORACLE values (not the candidate),
    matching validate._score.  A candidate with a huge spurious component must not
    change the scale used to judge the analytic zero."""
    oracle = {
        "X": _arrays([
            # sample 0: scale 1.0, comp1 analytic zero at 1e-30
            [(1.0, 0.0), (1.0e-30, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            # sample 1: scale 1e6, comp1 analytic zero at 1e-20 (< 1e-24*1e6 = 1e-18)
            [(1.0e6, 0.0), (1.0e-20, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        ])
    }
    candidate = {
        "X": _arrays([
            [(1.0, 0.0), (5.0e-30, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(1.0e6, 0.0), (7.0e-20, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        ])
    }
    # candidate's spurious comp1 values are NOT used for the scale; both samples'
    # comp1 are analytic zeros against their oracle scales -> both rescued -> MAX.
    d = _min_digits(candidate, oracle, "X", 2)
    assert 1.0e-20 < ZERO_REF_TOL * 1.0e6   # sample-1 zero is judged against 1e6, not the candidate
    assert d == MAX_DIGITS_F, f"per-sample oracle-scale rescue failed, got {d}"


def test_missing_integral_returns_none():
    """Unchanged contract: an integral absent from either dict yields None."""
    oracle = _one_sample([(1.0, 0.0)] * N_COMPONENTS)
    candidate = _one_sample([(1.0, 0.0)] * N_COMPONENTS)
    assert _min_digits(candidate, oracle, "MISSING", 1) is None
    assert _min_digits({}, oracle, "X", 1) is None
