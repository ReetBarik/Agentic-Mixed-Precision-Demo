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
_REF_SCALE_FLOOR_F = 1e-30

# DD's ~106-bit mantissa ceiling: 106 * log10(2).  A candidate that matches the
# reference to (or beyond) DD's own resolution is reported at this cap — you
# cannot claim more correct digits than the reference itself carries.
MAX_DIGITS = Decimal(106) * (Decimal(2).ln() / Decimal(10).ln())  # ≈ 31.9089

# DD's minimum representable *relative* error ≈ 2**-106.  A relative error below
# this means the candidate agrees with the reference to DD precision → max digits.
_DD_MIN_REL_ERR = Decimal(2) ** -106

# ref_scale "effectively zero" band: both |true| and |err| below 1e-30 * ref_scale
# are treated as a physics zero (avoids e.g. an _ieps50 ~1e-50 term whose tiny
# absolute noise would otherwise read as 0 digits).
_REF_SCALE_FLOOR = Decimal("1e-30")


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
      ``1e-30 * ref_scale`` → ``MAX_DIGITS`` (effectively-zero band).
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

        # Effectively-zero physics term: both magnitudes below the ref_scale floor.
        if ref_scale is not None:
            thresh = _REF_SCALE_FLOOR * abs(ref_scale)
            if true < thresh and err < thresh:
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
    if ref_scale is not None:
        thresh = _REF_SCALE_FLOOR_F * abs(ref_scale)
        if true < thresh and err < thresh:
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
