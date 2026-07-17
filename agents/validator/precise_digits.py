"""Precise-digits metric for the Validator.

For one output component, how many correct decimal digits does a ``candidate``
value carry relative to a double-double ``reference`` (the ground truth)?

    d = -log10( |candidate - reference| / |reference| )

capped at DD's ~106-bit ceiling (``106 * log10(2) ≈ 31.9`` digits) and with the
edge cases spelled out in :func:`precise_digits`.  All arithmetic is in
:class:`decimal.Decimal` so the subtraction against the ~31-digit reference does
not lose precision (a plain ``double`` candidate reconstructs exactly via
``Decimal(float)``; a DD reference reconstructs exactly as ``Decimal(hi) +
Decimal(lo)``).
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext, localcontext

# Plenty of head-room for exact reconstruction (hi+lo) and a stable ln().
getcontext().prec = 60

# Float mirror of MAX_DIGITS for the fast path (106 * log10(2)).
MAX_DIGITS_F = 106.0 * math.log10(2.0)  # ≈ 31.9089
_DD_MIN_REL_ERR_F = 2.0 ** -106
_ZERO_REL_TOL_F = 1e-15

# DD's ~106-bit mantissa ceiling: 106 * log10(2).  A candidate that matches the
# reference to (or beyond) DD's own resolution is reported at this cap — you
# cannot claim more correct digits than the reference itself carries.
MAX_DIGITS = Decimal(106) * (Decimal(2).ln() / Decimal(10).ln())  # ≈ 31.9089

# DD's minimum representable *relative* error ≈ 2**-106.  A relative error below
# this means the candidate agrees with the reference to DD precision → max digits.
_DD_MIN_REL_ERR = Decimal(2) ** -106

# Per-sample "effectively zero" band.  ``ref_scale`` is the characteristic
# magnitude of the sample (the max |component| across a sample's coeffs — see
# validate._score).  A component whose reference AND error are both below
# ``ZERO_REL_TOL * ref_scale`` carries no signal above the sample's own
# double-precision resolution: it is a numeric/physics zero (e.g. the imaginary
# part of a purely-real integral, where DD reads ~1e-42 and double roundoff
# ~1e-28 against an ~1e-11 real coeff), so the relative-error metric on it is
# meaningless and we report it at the cap.  ~1e-15 is a few ulps above double
# eps (2.2e-16): tight enough that a genuinely-computed small term (even one
# only a few digits correct) stays out of the band, loose enough to swallow
# roundoff-around-zero.  Subsumes the old fixed 1e-30 floor for ~1e-50 artifacts
# (with ref_scale ~ O(1) those sit far inside 1e-15 * ref_scale).
ZERO_REL_TOL = 1e-15


def effectively_zero(true_abs, err_abs, ref_scale) -> bool:
    """Both ``|reference|`` and ``|error|`` below ``ZERO_REL_TOL * |ref_scale|``.

    Single source of truth for the effectively-zero band, shared by the metric
    (:func:`precise_digits` / :func:`precise_digits_fast`) and the scorer's
    zeroed-component count.  ``ref_scale is None`` → always ``False`` (no band).
    Accepts ``float`` or :class:`~decimal.Decimal` uniformly.
    """
    if ref_scale is None:
        return False
    thresh = _ZERO_REL_TOL_F * abs(float(ref_scale))
    return float(true_abs) < thresh and float(err_abs) < thresh


def precise_digits(
    candidate: Decimal,
    reference: Decimal,
    *,
    ref_scale: Decimal | None = None,
) -> Decimal:
    """Correct decimal digits of ``candidate`` vs the DD ``reference``.

    Both inputs are exact :class:`~decimal.Decimal` values (see module docstring).
    Returns a ``Decimal`` in ``[0, MAX_DIGITS]``.  Edge cases, in priority order:

    * ``|err| == 0`` (candidate == reference, incl. both zero) → ``MAX_DIGITS``.
    * ``ref_scale`` given and both ``|true|`` and ``|err|`` below
      ``ZERO_REL_TOL * ref_scale`` → ``MAX_DIGITS`` (effectively-zero band).
    * ``|true| == 0`` and ``err != 0`` → ``0``.
    * relative error below DD's min representable (``2**-106``) → ``MAX_DIGITS``.
    * ``|err| > |true|`` (relative error ≥ 1) → ``0``.
    * otherwise ``-log10(|err|/|true|)``, clamped to ``[0, MAX_DIGITS]``.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        err = abs(candidate - reference)
        true = abs(reference)

        # candidate == reference exactly (covers both-zero) → max.
        if err == 0:
            return MAX_DIGITS

        # Effectively-zero term: both magnitudes below the per-sample rel band.
        if effectively_zero(true, err, ref_scale):
            return MAX_DIGITS

        # |true| == 0 with a nonzero error → no correct digits.
        if true == 0:
            return Decimal(0)

        rel = err / true

        # Candidate matches to (or beyond) DD's own resolution → max.
        if rel < _DD_MIN_REL_ERR:
            return MAX_DIGITS

        # Error swamps the value → no correct digits.
        if rel >= 1:
            return Decimal(0)

        d = -(rel.ln() / Decimal(10).ln())
        if d < 0:
            return Decimal(0)
        if d > MAX_DIGITS:
            return MAX_DIGITS
        return d


def precise_digits_fast(
    cand_hi: float, cand_lo: float,
    ref_hi: float, ref_lo: float,
    *,
    ref_scale: float | None = None,
) -> float:
    """Fast float mirror of :func:`precise_digits` for bulk (100k-scale) use.

    Each value is carried as a ``(hi, lo)`` double pair — a vanilla candidate is
    ``(double, 0.0)``; a DD value is its two words.  Computing the error as
    ``(cand_hi - ref_hi) + (cand_lo - ref_lo)`` keeps the leading cancellation
    exact enough that ``d`` is accurate to well under 0.01 digit, which is all the
    verdict needs (the gating minimum sits at badly-conditioned, low-digit
    components where float is exact; high-digit components are capped anyway).

    Cross-checked against the Decimal :func:`precise_digits` oracle in the tests.
    Same edge-case priority order; returns a ``float`` in ``[0, MAX_DIGITS_F]``.
    """
    err = abs((cand_hi - ref_hi) + (cand_lo - ref_lo))
    true = abs(ref_hi + ref_lo)

    if err == 0.0:
        return MAX_DIGITS_F
    if effectively_zero(true, err, ref_scale):
        return MAX_DIGITS_F
    if true == 0.0:
        return 0.0
    rel = err / true
    if rel < _DD_MIN_REL_ERR_F:
        return MAX_DIGITS_F
    if rel >= 1.0:
        return 0.0
    d = -math.log10(rel)
    if d < 0.0:
        return 0.0
    if d > MAX_DIGITS_F:
        return MAX_DIGITS_F
    return d
