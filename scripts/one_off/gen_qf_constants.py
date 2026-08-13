#!/usr/bin/env python3
"""Generate the QF-encoded Chebyshev / Bernoulli coefficient tables for
``third_party/include/kokkosMaths_qf.h``.

The QF analogue of the upstream ``scripts/gen_ff_constants`` that produced
``kokkosMaths_ff.h``'s tables, with one deliberate difference in the SOURCE.

**Source of truth is the DD table, at full DD width.**  ``kokkosMaths_dd.h``
stores each coefficient as a ``(hi_bits, lo_bits)`` pair of IEEE-754 binary64
patterns whose exact sum carries ~31 decimal digits.  QF resolves ~28.9 digits,
which is *more* than a single ``double`` (~15.9) — so splitting from the ``double``
approximation alone would throw away ~13 digits QF can actually hold and cap the
table at double accuracy.  We therefore reconstruct the exact rational ``hi + lo``
and split *that* into four FP32 words.  (The FF generator could ignore this: ff
resolves ~14 digits, comfortably inside a single double, so its Dekker split of
the source double value lost nothing.)

**Splitting** mirrors ``QuadFloat(double)`` and upstream ``gen_qf_constants.cpp``:
successive round-to-nearest FP32 extraction of the running residual,

    f0 = fp32(r);  r -= f0
    f1 = fp32(r);  r -= f1
    f2 = fp32(r);  r -= f2
    f3 = fp32(r)

so ``f0+f1+f2+f3`` reconstructs the source to ~4x24 = 96 bits.  All arithmetic is
exact :class:`fractions.Fraction`; the FP32 rounding is implemented here rather
than delegated to ``float()`` so no intermediate double rounding can creep in
(the residual after ``f0`` can need more than 53 bits to represent exactly).

Usage:
    python3 scripts/one_off/gen_qf_constants.py [--check]

Writes the two C++ table bodies to stdout.  ``--check`` instead re-reads the
tables already in ``kokkosMaths_qf.h`` and reports the reconstruction error of
each entry against the DD source, which is the regression form worth keeping.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from fractions import Fraction
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
DD_HEADER = _REPO / "runs" / "qcdloop_headers_full" / "kokkosMaths_dd.h"
QF_HEADER = _REPO / "third_party" / "include" / "kokkosMaths_qf.h"

# FP32 format constants.
_F32_MANT_BITS = 24          # significand bits including the implicit leading 1
_F32_MIN_EXP = -126          # minimum normal exponent
_F32_SUBNORMAL_MIN_EXP = _F32_MIN_EXP - (_F32_MANT_BITS - 1)   # -149


def _u64_to_float(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _exact(x: float) -> Fraction:
    return Fraction(x)


def _pow2(e: int) -> Fraction:
    """Exact 2**e as a Fraction, for any sign of ``e``."""
    return Fraction(1 << e) if e >= 0 else Fraction(1, 1 << -e)


def round_to_fp32(r: Fraction) -> float:
    """Exact round-to-nearest-even of a rational to IEEE-754 binary32.

    Implemented over integers so a residual needing more than 53 bits is not
    silently double-rounded through a Python ``float``.  Handles subnormals and
    overflow-to-infinity; the caller treats an infinity as a hard error.
    """
    if r == 0:
        return 0.0
    sign = -1 if r < 0 else 1
    r = abs(r)

    # Binade: the largest e with 2^e <= r, computed exactly on the integer parts.
    # bit_length() is floor(log2(x)) + 1, so the difference lands within 1 of the
    # true exponent; a single exact comparison fixes the off-by-one.
    e = r.numerator.bit_length() - r.denominator.bit_length()
    if _pow2(e) > r:
        e -= 1

    # The significand holds _F32_MANT_BITS bits with its leading bit at 2^e, so the
    # ulp is 2^(e - 23) — floored at the subnormal ulp for tiny values.
    ulp_exp = max(e - (_F32_MANT_BITS - 1), _F32_SUBNORMAL_MIN_EXP)

    # scaled = r / 2^ulp_exp, exactly, as a Fraction; round half to even.
    scaled = r / _pow2(ulp_exp)
    q, rem = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * rem
    if twice > scaled.denominator or (twice == scaled.denominator and (q & 1)):
        q += 1

    # Reassemble.  q is at most 2^24, so float(q) is exact and ldexp is exact.
    val = math.ldexp(float(q), ulp_exp)
    try:
        return struct.unpack("<f", struct.pack("<f", sign * val))[0]
    except OverflowError:
        # Past FLT_MAX: IEEE round-to-nearest carries to infinity.  struct raises
        # instead of saturating, so name the result explicitly; split_qf treats an
        # infinity as a hard error rather than emitting a bogus limb.
        return math.inf if sign > 0 else -math.inf


def split_qf(value: Fraction) -> tuple[list[float], Fraction]:
    """Split an exact rational into four FP32 words; return (words, residual)."""
    words: list[float] = []
    r = value
    for _ in range(4):
        w = round_to_fp32(r)
        if w in (float("inf"), float("-inf")):
            raise OverflowError(f"FP32 overflow splitting {float(value)!r}")
        words.append(w)
        r = r - _exact(w)
    return words, r


_DD_ENTRY = re.compile(
    r"ql::ddfun::make_dd\(0x([0-9a-fA-F]+)ULL,\s*0x([0-9a-fA-F]+)ULL\)")


def read_dd_table(text: str, fn_name: str, count: int) -> list[Fraction]:
    """Exact values of the ``count`` entries of ``kokkosMaths_dd.h``'s ``fn_name``."""
    start = text.index(f"static T {fn_name}(int i)")
    body = text[start:]
    entries = _DD_ENTRY.findall(body)[:count]
    if len(entries) != count:
        raise RuntimeError(
            f"{fn_name}: found {len(entries)} make_dd entries, expected {count}")
    out = []
    for hi_s, lo_s in entries:
        hi = _u64_to_float(int(hi_s, 16))
        lo = _u64_to_float(int(lo_s, 16))
        out.append(_exact(hi) + _exact(lo))
    return out


def emit_table(values: list[Fraction], label: str) -> str:
    lines = []
    n = len(values)
    for i, v in enumerate(values):
        words, resid = split_qf(v)
        bits = [_f32_bits(w) for w in words]
        approx = float(v)
        comma = "," if i < n - 1 else " "
        lines.append(
            f"                ql::qfun::make_qf({bits[0]:#010x}U, {bits[1]:#010x}U, "
            f"{bits[2]:#010x}U, {bits[3]:#010x}U){comma}  "
            f"// {label}[{i}]{'' if i > 9 else ' '}  (~ {approx!r})")
    return "\n".join(lines)


def rel_err(values: list[Fraction]) -> list[float]:
    out = []
    for v in values:
        _, resid = split_qf(v)
        out.append(0.0 if v == 0 else abs(float(resid / v)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report reconstruction error instead of emitting tables")
    args = ap.parse_args()

    dd_text = DD_HEADER.read_text()
    cheb = read_dd_table(dd_text, "_C", 43)
    bern = read_dd_table(dd_text, "_B", 25)

    if args.check:
        for label, vals in (("C", cheb), ("B", bern)):
            errs = rel_err(vals)
            worst = max(errs)
            print(f"{label}: n={len(vals)} worst_rel_err={worst:.3e}")
            for i, e in enumerate(errs):
                if e > 1e-28:
                    print(f"  {label}[{i}] rel_err={e:.3e} value={float(vals[i]):.6e}")
        return 0

    print("// ---- Chebyshev (43) ----")
    print(emit_table(cheb, "C"))
    print()
    print("// ---- Bernoulli (25) ----")
    print(emit_table(bern, "B"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
