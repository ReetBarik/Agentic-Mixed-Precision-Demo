// SPDX-License-Identifier: LicenseRef-DHB-License
//
// Copyright (c) 2024 David H. Bailey — DDFUN v04 (original algorithms)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical translation of dd_math.hpp (this repo's
// Kokkos C++ port of DDFUN v04) from double-double (2×FP64) to
// float-float (2×FP32). Function inventory, algorithm choices, and
// coefficient tables descend from DDFUN v04 by David H. Bailey.
//
// FP32-specific modifications (input narrowing, splitter constant
// 8193.0f = 2^13+1, joint sin/cos doublings, Taylor branches for
// |a|<0.5 in sinh/cosh/atanh, direct exp scaling and scaled-splitter
// guards at the multiply/divide primitives to avoid splitter
// overflow, nint magic-constant replacement) are documented in
// PORT_NOTES.md. These modifications fall under DHB-License §3
// (grant-back) and are governed by the same terms as the original.
//
// See LICENSES/LicenseRef-DHB-License.txt for the full license text
// and NOTICE.md for the per-file license mapping.

#pragma once

// Float-float real arithmetic — Kokkos::Experimental::FloatFloat.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Mechanically ported from dd_math.hpp (this repo's Kokkos C++ port of DDFUN
// v04 by David H. Bailey), swapping 2×FP64 for 2×FP32. See PORT_NOTES.md for
// the FP32-specific fixes.
//
// Precision: ~14.4 decimal digits (24-bit FP32 mantissa × 2 = 48 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// Naming conventions (T0.4/T2.0, for eventual upstreaming to Kokkos) mirror
// dd_math.hpp:
//   * Type + math live under namespace Kokkos::Experimental so an upstream PR is
//     a mechanical move rather than a rewrite.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions FloatFloat_pi(), FloatFloat_e(), ...
//   * The former bit-pattern constructor became the static factory
//     FloatFloat::from_bits(hi, lo).
//   * Every single/double-return math function is additionally re-exposed under
//     namespace Kokkos at the bottom of this header (forwarding overloads,
//     mirroring impl/Kokkos_QuadPrecisionMath.hpp) so Kokkos::exp(ff) works
//     identically to Kokkos::exp(double)/Kokkos::exp(__float128). Math functions
//     are also ADL-findable via the argument's namespace (Kokkos::Experimental).
//     add/subtract/multiply/divide are NOT re-exposed under Kokkos — they are for
//     operators + explicit ADL only.

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef __CUDA_ARCH__
#  include <iomanip>
#  include <ostream>
#endif

namespace Kokkos {
namespace Experimental {

// ============================================================
// Forward declarations
// ============================================================
struct FloatFloat;
KOKKOS_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b);
KOKKOS_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b);
KOKKOS_INLINE_FUNCTION FloatFloat negate(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat abs(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n);
KOKKOS_INLINE_FUNCTION FloatFloat exp(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat log(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b);
KOKKOS_INLINE_FUNCTION void   sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y);
KOKKOS_INLINE_FUNCTION void   sincos(FloatFloat a, FloatFloat& x, FloatFloat& y);
KOKKOS_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y);

// ============================================================
// FloatFloat struct
// ============================================================
struct FloatFloat {
    float hi;
    float lo;

    KOKKOS_INLINE_FUNCTION FloatFloat() : hi(0.0f), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(float h) : hi(h), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(float h, float l) : hi(h), lo(l) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(double h) : hi((float)h), lo((float)(h - (double)(float)h)) {}
    KOKKOS_INLINE_FUNCTION FloatFloat(const FloatFloat& o) : hi(o.hi), lo(o.lo) {}
    KOKKOS_INLINE_FUNCTION FloatFloat& operator=(const FloatFloat& o) { hi=o.hi; lo=o.lo; return *this; }

    // Factory: build a FloatFloat from the IEEE-754 bit patterns of its two
    // components. Safe on host (memcpy) and device (__int_as_float). Replaces
    // the former free bit-pattern constructor function make_ff().
    static KOKKOS_INLINE_FUNCTION FloatFloat from_bits(uint32_t hi_bits, uint32_t lo_bits) {
        float h, l;
#ifndef __CUDA_ARCH__
        std::memcpy(&h, &hi_bits, sizeof(float));
        std::memcpy(&l, &lo_bits, sizeof(float));
#else
        h = __int_as_float(static_cast<int>(hi_bits));
        l = __int_as_float(static_cast<int>(lo_bits));
#endif
        return FloatFloat(h, l);
    }

    KOKKOS_INLINE_FUNCTION FloatFloat operator-() const { return negate(*this); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator+(FloatFloat b) const { return add(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator-(FloatFloat b) const { return subtract(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator*(FloatFloat b) const { return multiply(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator/(FloatFloat b) const { return divide(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator*(float b)  const { return multiply_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator/(float b)  const { return divide_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator+(float b)  const { return add(*this, FloatFloat(b)); }
    KOKKOS_INLINE_FUNCTION FloatFloat operator-(float b)  const { return subtract(*this, FloatFloat(b)); }

    KOKKOS_INLINE_FUNCTION FloatFloat& operator+=(FloatFloat b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator-=(FloatFloat b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator*=(FloatFloat b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator/=(FloatFloat b) { *this = *this / b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator+=(float b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator-=(float b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator*=(float b) { *this = multiply_scalar(*this, b); return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloat& operator/=(float b) { *this = divide_scalar(*this, b); return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(FloatFloat b) const { return hi==b.hi && lo==b.lo; }
    KOKKOS_INLINE_FUNCTION bool operator!=(FloatFloat b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(FloatFloat b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator>(FloatFloat b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator<=(FloatFloat b) const { return !(b < *this); }
    KOKKOS_INLINE_FUNCTION bool operator>=(FloatFloat b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION FloatFloat operator+(float a, FloatFloat b) { return add(FloatFloat(a), b); }
KOKKOS_INLINE_FUNCTION FloatFloat operator-(float a, FloatFloat b) { return subtract(FloatFloat(a), b); }
KOKKOS_INLINE_FUNCTION FloatFloat operator*(float a, FloatFloat b) { return multiply_scalar(b, a); }
KOKKOS_INLINE_FUNCTION FloatFloat operator/(float a, FloatFloat b) { return divide(FloatFloat(a), b); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const FloatFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
// Auto-generated by scripts/gen_ff_constants.cpp -- do not edit by hand.
// Route A: round_to_nearest_FF(Bailey FP64 hi+lo pair).
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_pi          () { return FloatFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU); } // pi
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_e           () { return FloatFloat::from_bits(0x402df854U, 0x33b14577U); } // e
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_log2        () { return FloatFloat::from_bits(0x3f317218U, 0xb102e308U); } // ln(2)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_log10       () { return FloatFloat::from_bits(0x40135d8eU, 0xb309555dU); } // ln(10)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_sqrt2       () { return FloatFloat::from_bits(0x3fb504f3U, 0x32cfe77aU); } // sqrt(2)
KOKKOS_INLINE_FUNCTION FloatFloat FloatFloat_euler_gamma () { return FloatFloat::from_bits(0x3f13c468U, 0xb1e4127aU); } // Euler gamma

// ============================================================
// Primitive arithmetic
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat negate(FloatFloat a) {
    return FloatFloat(-a.hi, -a.lo);
}

// TwoSum (Knuth)
KOKKOS_INLINE_FUNCTION FloatFloat add(FloatFloat a, FloatFloat b) {
    float t1 = a.hi + b.hi;
    float e  = t1 - a.hi;
    float t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat subtract(FloatFloat a, FloatFloat b) {
    float t1 = a.hi - b.hi;
    float e  = t1 - a.hi;
    float t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return FloatFloat(hi, lo);
}

// TwoProduct (Dekker splitting). Splitter = 2^13 + 1 for FP32 (24-bit mantissa).
KOKKOS_INLINE_FUNCTION FloatFloat multiply(FloatFloat a, FloatFloat b) {
    const float split = 8193.0f;
    // B9: scaled splitter — the same overflow B8 fixed in divide() below, at a
    // third site (see divide()'s comment for the full derivation). BOTH operands
    // are split here, so both need the guard: for |x.hi| > FLT_MAX / (split + 1)
    // ≈ 4.15e34 the product x.hi * split overflows to ±inf, and then
    // x1 = conx - (conx - x.hi) = inf - inf = NaN poisons the whole result.
    // Surfaced by B6: erfc's multiply(sqrt(pi), exp(z^2)) walks into it at
    // z ≈ 8.93 (PORT_NOTES §4i). B6 dodged the exposure by reformulating; this
    // closes it at the primitive.
    //
    // Unscale by MULTIPLYING by 2^64 per scaled operand, applied SEPARATELY —
    // never by 1/(sa*sb). When both operands are scaled, sa*sb = 2^-128 is
    // subnormal and 1/(sa*sb) overflows to +inf, which would replace one garbage
    // answer with another. Two exact power-of-two multiplies have no such
    // intermediate.
    //
    // Scope: this fixes products that are REPRESENTABLE but were reached through
    // a hazard-band operand. It does not change what happens when the product
    // itself overflows FP32 — multiply() already returned NaN there long before
    // any splitter scaling (c11 = inf makes a1*b1 - c11 = -inf and then
    // e = t1 - c11 = NaN), as multiply(1e30, 1e30) shows with both operands well
    // below the band. Both operands in the band always implies such an overflow
    // (|a.hi*b.hi| > 1.7e69), so that corner stays non-finite; it lands on
    // (inf, -inf) rather than NaN, which is neither better nor worse. Fixing
    // overflow semantics is a separate question from fixing the splitter.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = Kokkos::fabs(a.hi) > kSplitOverflowThresh;
    const bool  hb = Kokkos::fabs(b.hi) > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    a = FloatFloat(a.hi * sa, a.lo * sa);
    b = FloatFloat(b.hi * sb, b.lo * sb);
    float cona = a.hi * split, conb = b.hi * split;
    float a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    float a2 = a.hi - a1,            b2 = b.hi - b1;
    float c11 = a.hi * b.hi;
    float c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2  = a.hi * b.lo + a.lo * b.hi;
    float t1  = c11 + c2;
    float e   = t1 - c11;
    float t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    float hi  = t1 + t2;
    float lo  = t2 - (hi - t1);
    // B9: unscale. ua/ub are exact powers of two (or 1.0f on the non-hazard path,
    // where `hi * 1.0f * 1.0f == hi` bit-for-bit).
    return FloatFloat(hi * ua * ub, lo * ua * ub);
}

KOKKOS_INLINE_FUNCTION FloatFloat divide(FloatFloat a, FloatFloat b) {
    // §B8 extension: a NON-FINITE DIVISOR must never reach the Dekker splitter.
    // B8's divisor guard below classifies on |b.hi| > kSplitOverflowThresh, which
    // is TRUE for b.hi = ±inf, so it scales by 2^-64 — and inf * 2^-64 is still
    // inf. The splitter then computes conb = inf * split = inf and
    // b1 = conb - (conb - b.hi) = inf - inf = NaN, poisoning a quotient that IEEE
    // defines perfectly well: x / ±inf = ±0 for finite x. divide(1.0f, inf)
    // returned NaN where IEEE 754-2019 §6.1 requires +0.
    //
    // The whole answer is the bare float quotient a.hi / b.hi, which is exact for
    // every non-finite divisor and needs no FF arithmetic at all:
    //   finite / ±inf  ->  ±0, correctly signed (including ±0 / ±inf, where IEEE
    //                      preserves the sign of the zero numerator)
    //   ±inf   / ±inf  ->  NaN (invalid), unchanged
    //   anything/ NaN  ->  NaN, unchanged
    // Returning the quotient itself, rather than a copysign(0, ...) form, is what
    // keeps the inf/inf case NaN — copysign would flatten it to a signed zero.
    // lo = 0.0f because the result is exact, matching B10's Zone C convention.
    //
    // Early return, so this short-circuits ahead of every splitter below by
    // construction. Divide-by-ZERO is deliberately NOT touched here: b.hi = 0 is
    // finite, falls through, and keeps B10's Zone C ±inf semantics.
    if (!Kokkos::isfinite(b.hi)) return FloatFloat(a.hi / b.hi, 0.0f);
    const float split = 8193.0f;
    // B8: the Dekker splitter below computes conb = b.hi * split (line ~"conb =")
    // to extract b.hi's high half. For |b.hi| > FLT_MAX / (split + 1) ≈ 4.15e34
    // that product overflows to ±inf, then b1 = conb - (conb - b.hi) = inf - inf
    // = NaN, which poisons the whole quotient. This bites log()'s Newton
    // iteration (divisor exp(b) ≈ a) for x ≳ e^79.7 ≈ 1.5e34, feeding NaN back
    // into exp() and hanging its Taylor loop at the 60-iteration cap (surfaced by
    // the B4 investigation — see TEST_SUITE_PLAN.md §B4/§B8).
    //
    // Fix: when b.hi is in the overflow-hazard band, pre-scale the divisor down
    // by an exact power of two (2^-64: no FP rounding, so b's full FF precision is
    // preserved), run the unchanged Dekker split, then unscale the quotient. Since
    // q = a / (b·s) = (a/b) / s, the true quotient a/b is recovered by MULTIPLYING
    // q by the same down-scale factor s. 2^-64 gives ample headroom: the largest
    // |b.hi| ≈ FLT_MAX = 2^128 maps to 2^64, and 2^64 * split ≈ 2^77 ≪ FLT_MAX.
    // Mirrors PORT_NOTES §4a's power-of-2 scaling pattern for exp's final scaling
    // (same bug class, different site — the splitter, not exp's scaling).
    //
    // B10: the SECOND splitter run in this function splits the QUOTIENT ESTIMATE
    // s1 = a.hi / b.hi via `cona = s1 * split`, and |s1| can land in the same
    // 4.15e34 band with BOTH operands far below it — divide(1e35f, 1.0f) and
    // divide(1e30f, 1e-5f) returned NaN with no large operand anywhere. Three
    // zones, classified from the actual value of s1:
    //
    //   Zone A  |s1| <= kSplitOverflowThresh. Safe; untouched, bit-identical.
    //   Zone B  kSplitOverflowThresh < |s1| < inf. The splitter overflows but the
    //           true quotient IS representable. Pre-scale the NUMERATOR down by
    //           the exact power of two 2^-32, run the unchanged body, unscale the
    //           quotient by 2^32. Was NaN; now correct.
    //   Zone C  |s1| == inf. The true quotient overflows FP32, so NO scaling
    //           recovers a finite answer. Return the correctly-signed ±inf
    //           instead of NaN.
    //
    // Zone C is a deliberate SEMANTIC upgrade (an honest overflow signal instead
    // of a NaN poison), not a bit-identical fix. Unlike B8 and B9, B10 cannot
    // promise "correct everywhere in the band", because part of the band has no
    // representable answer at all. That is inherent to division, not a shortcut.
    //
    // Classifying on s1 itself is overflow-safe by construction, which is why no
    // `|a.hi|` vs `|b.hi| * thresh` product is needed: IEEE division never traps,
    // delivers a correctly-signed ±inf exactly when the quotient exceeds FLT_MAX,
    // and yields NaN only for 0/0 and inf/inf — both of which fail
    // `fabs(s1) > thresh` and fall through to the unchanged body, where they
    // produced NaN before and still do.
    //
    // Why scale the NUMERATOR: the divisor is B8's to scale, and s1 is what has to
    // shrink. Multiplying `a` by an exact power of two scales s1, every Dekker
    // intermediate, and the final (hi, lo) by exactly that factor, so the result
    // is recovered exactly. Watch the direction, as in B8/B9:
    // q = (a*sn) / (b*s) = (a/b) * (sn/s), so recovering a/b means MULTIPLYING q
    // by s/sn — applied as two separate exact factors, never as one reciprocal.
    //
    // Why 2^-32 rather than B8/B9's 2^-64: this scale lands on the NUMERATOR,
    // which in Zone B can be as small as |b.hi| * 4.15e34 ~ 2^-34 (b.hi subnormal),
    // so an over-aggressive scale risks flushing a.lo into the subnormal range.
    // 2^-32 keeps ~19 binades of splitter headroom (|s1| <= 2^128 maps to 2^96 and
    // 2^96 * split ~ 2^109 << FLT_MAX) while leaving a.lo >= 2^-90 in that worst
    // corner, ~36 binades clear of 2^-126. 2^-64 would leave under 4.
    //
    // Interaction with B8's divisor scale: the two guards are mutually exclusive.
    // B8 fires only for |b.hi| > T, which already forces |a.hi/b.hi| <= FLT_MAX/T
    // = split+1 = 8194; after B8's 2^-64 the estimate is |s1| <= 8194 * 2^64
    // ~ 1.5e23, eleven orders below T. So B10 can never fire on top of B8. The
    // unscale is still written as two independent factors so it would compose
    // correctly if it ever could.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const float s = (Kokkos::fabs(b.hi) > kSplitOverflowThresh)
                        ? ldexpf(1.0f, -64) : 1.0f;
    b = FloatFloat(b.hi * s, b.lo * s);
    float s1 = a.hi / b.hi;
    float un = 1.0f;                                     // B10 numerator unscale
    if (Kokkos::fabs(s1) > kSplitOverflowThresh) {       // not Zone A
        if (Kokkos::isinf(s1)) return FloatFloat(s1, 0.0f);        // Zone C
        const float sn = ldexpf(1.0f, -32);                        // Zone B
        un = ldexpf(1.0f, 32);
        a  = FloatFloat(a.hi * sn, a.lo * sn);
        s1 = s1 * sn;   // exact, and identical to recomputing (a.hi*sn) / b.hi
    }
    float cona = s1 * split, conb = b.hi * split;
    float a1   = cona - (cona - s1), b1 = conb - (conb - b.hi);
    float a2   = s1 - a1,            b2 = b.hi - b1;
    float c11  = s1 * b.hi;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = s1 * b.lo;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float t12  = t1 + t2;
    float t22  = t2 - (t12 - t1);
    float t11  = a.hi - t12;
    e = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float s2   = (t11 + t21) / b.hi;
    float hi   = s1 + s2;
    float lo   = s2 - (hi - s1);
    // B8/B10: unscale — recover a/b from (a·un⁻¹)/(b·s) by multiplying by s and by
    // un, as two separate exact powers of two. Both are 1.0f on the non-hazard
    // path, where `hi * 1.0f * 1.0f == hi` bit-for-bit.
    return FloatFloat(hi * s * un, lo * s * un);
}

KOKKOS_INLINE_FUNCTION FloatFloat multiply_scalar(FloatFloat a, float b) {
    const float split = 8193.0f;
    // B9: scaled splitter, identical in shape to multiply() above — both a.hi and
    // the scalar b are split, so both are guarded. Reachable from the public
    // operator*(FloatFloat, float) / operator*(float, FloatFloat) with either side
    // in the hazard band; the in-header callers all pass small scalars, but `a`
    // is unconstrained. Same separate-unscale rule as multiply().
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = Kokkos::fabs(a.hi) > kSplitOverflowThresh;
    const bool  hb = Kokkos::fabs(b)    > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    a = FloatFloat(a.hi * sa, a.lo * sa);
    b = b * sb;
    float cona = a.hi * split, conb = b * split;
    float a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    float a2   = a.hi - a1,            b2 = b - b1;
    float c11  = a.hi * b;
    float c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    float c2   = a.lo * b;
    float t1   = c11 + c2;
    float e    = t1 - c11;
    float t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    return FloatFloat(hi * ua * ub, lo * ua * ub);   // B9 unscale, see multiply()
}

KOKKOS_INLINE_FUNCTION FloatFloat divide_scalar(FloatFloat a, float b) {
    // §B8 extension: non-finite divisor, divide()'s guard at the scalar site. The
    // divisor guard below scales by 2^-64 for |b| > kSplitOverflowThresh, which
    // b = ±inf satisfies while staying inf, so conb = inf * split = inf and
    // b1 = inf - inf = NaN. Same bare-quotient answer, same reasoning; the full
    // derivation lives at divide() and is not repeated here.
    if (!Kokkos::isfinite(b)) return FloatFloat(a.hi / b, 0.0f);
    const float split = 8193.0f;
    // B9: scaled splitter on the DIVISOR — B8's divide() fix at the scalar site.
    // For |b| > FLT_MAX / (split + 1) ≈ 4.15e34, conb = b * split overflows and
    // b1 = inf - inf = NaN. Pre-scale b down by the exact power of two 2^-64, run
    // the unchanged split, then recover a/b from a/(b·s) by MULTIPLYING the
    // quotient by s (B8's sign-error warning applies verbatim). Reachable from
    // the public operator/(FloatFloat, float).
    //
    // Scaling b down inflates the quotient estimate t1 = a.hi / b by 2^64, which
    // cannot itself reach the band: the guard only fires for |b| > 4.15e34, so
    // |t1| < FLT_MAX / 4.15e34 = 8193 beforehand and < 8193·2^64 ≈ 1.5e23 after —
    // eleven orders below the 4.15e34 threshold.
    //
    // B10: the quotient-estimate guard, divide()'s fix at the scalar site. t1 is
    // this function's `s1` — the estimate that `cona = t1 * split` splits — and it
    // reaches the hazard band on its own with a perfectly ordinary scalar
    // (divide_scalar(1e35f, 2.0f) returned NaN). Same three zones, same
    // classify-on-the-quotient-itself rule, same 2^-32 numerator scale, same
    // Zone C ±inf-instead-of-NaN semantic upgrade, same mutual exclusivity with
    // the divisor guard above. The full derivation lives at divide(); it is not
    // repeated here.
    //
    // No shared helper for the two sites: the operand types differ (FloatFloat vs
    // float divisor) and each guard is five lines inline at a hot arithmetic
    // primitive — the same trade B9 made across its four splitter sites.
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const float s = (Kokkos::fabs(b) > kSplitOverflowThresh) ? ldexpf(1.0f, -64) : 1.0f;
    b = b * s;
    float t1 = a.hi / b;
    float un = 1.0f;                                     // B10 numerator unscale
    if (Kokkos::fabs(t1) > kSplitOverflowThresh) {       // not Zone A
        if (Kokkos::isinf(t1)) return FloatFloat(t1, 0.0f);        // Zone C
        const float sn = ldexpf(1.0f, -32);                        // Zone B
        un = ldexpf(1.0f, 32);
        a  = FloatFloat(a.hi * sn, a.lo * sn);
        t1 = t1 * sn;   // exact, and identical to recomputing (a.hi*sn) / b
    }
    float cona = t1 * split, conb = b * split;
    float a1   = cona - (cona - t1), b1 = conb - (conb - b);
    float a2   = t1 - a1,            b2 = b - b1;
    float t12  = t1 * b;
    float t22  = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    float t11  = a.hi - t12;
    float e    = t11 - a.hi;
    float t21  = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    float t2   = (t11 + t21) / b;
    float hi   = t1 + t2;
    float lo   = t2 - (hi - t1);
    // B9/B10: unscale — two separate exact powers of two, both 1.0f off-hazard.
    return FloatFloat(hi * s * un, lo * s * un);
}

// Exact product of two floats
KOKKOS_INLINE_FUNCTION FloatFloat two_prod(float fa, float fb) {
    const float split = 8193.0f;
    // B9: scaled splitter, multiply()'s guard at the bare-float site. sqrt() —
    // the only in-header caller — passes t2 ≈ sqrt(a.hi) ≤ 1.9e19, far below the
    // band, so this is unreachable from inside the header today. Guarded anyway:
    // two_prod is a public free function in Kokkos::Experimental, the fix is the
    // same four lines, and an exactness primitive silently returning NaN is a bad
    // trap to leave armed. Same separate-unscale rule as multiply().
    const float kSplitOverflowThresh = 4.1528233e34f; // FLT_MAX / (split + 1)
    const bool  ha = Kokkos::fabs(fa) > kSplitOverflowThresh;
    const bool  hb = Kokkos::fabs(fb) > kSplitOverflowThresh;
    const float sa = ha ? ldexpf(1.0f, -64) : 1.0f;
    const float sb = hb ? ldexpf(1.0f, -64) : 1.0f;
    const float ua = ha ? ldexpf(1.0f,  64) : 1.0f;
    const float ub = hb ? ldexpf(1.0f,  64) : 1.0f;
    fa = fa * sa;
    fb = fb * sb;
    float cona = fa * split, conb = fb * split;
    float a1   = cona - (cona - fa), b1 = conb - (conb - fb);
    float a2   = fa - a1,            b2 = fb - b1;
    float s1   = fa * fb;
    float s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return FloatFloat(s1 * ua * ub, s2 * ua * ub);   // B9 unscale, see multiply()
}

// ============================================================
// Basic math
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat abs(FloatFloat a) {
    return (a.hi >= 0.0f) ? a : FloatFloat(-a.hi, -a.lo);
}

// Nearest integer. The DD-style magic-constant trick (using a 2^47 FF constant)
// is fragile in FP32: ULP at 2^47 is 2^24, much larger than typical integer
// inputs, so the FF lo component must rescue the precision and ties land on
// the wrong side. Instead, do the rounding in FP64 — FF values are bounded by
// 2^48 and fit exactly in FP64's 53-bit mantissa, where the magic-constant
// trick is well-conditioned.
KOKKOS_INLINE_FUNCTION FloatFloat round_to_nearest_int(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    double total = (double)a.hi + (double)a.lo;
    if (Kokkos::fabs(total) >= 1.40737488355328e14 /* 2^47 */) {
        Kokkos::printf("FFNINT: argument too large\n");
        return FloatFloat(0.0f);
    }
    const double T52 = 4.503599627370496e15; // 2^52
    double rounded = (total > 0.0) ? (total + T52) - T52 : (total - T52) + T52;
    float hi = (float)rounded;
    float lo = (float)(rounded - (double)hi);
    return FloatFloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION FloatFloat sqrt(FloatFloat a) {
    if (a.hi == 0.0f) return FloatFloat(0.0f);
    if (a.hi < 0.0f) {
        Kokkos::printf("FFSQRT: negative argument\n");
        return FloatFloat(0.0f);
    }
    float t1 = 1.0f / Kokkos::sqrt(a.hi);
    float t2 = a.hi * t1;
    FloatFloat s0 = two_prod(t2, t2);
    FloatFloat s1 = subtract(a, s0);
    float t3  = 0.5f * s1.hi * t1;
    return add(FloatFloat(t2), FloatFloat(t3));
}

// Integer power
KOKKOS_INLINE_FUNCTION FloatFloat pow_int(FloatFloat a, int n) {
    const float cl2 = 1.4426950408889633f;
    if (a.hi == 0.0f) {
        if (n >= 0) return FloatFloat(0.0f);
        Kokkos::printf("FFNPWR: zero base with negative exponent\n");
        return FloatFloat(0.0f);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return FloatFloat(1.0f);
    if (nn == 1) return (n > 0) ? a : divide(FloatFloat(1.0f), a);
    if (nn == 2) { FloatFloat r = multiply(a,a); return (n>0) ? r : divide(FloatFloat(1.0f),r); }
    int mn = (int)(cl2 * Kokkos::log((float)nn) + 1.0f + 1.0e-6f);
    FloatFloat s0 = a, s2 = FloatFloat(1.0f);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = multiply(s2, s0);
        kn = kk;
        if (j < mn) s0 = multiply(s0, s0);
    }
    if (n < 0) s2 = divide(FloatFloat(1.0f), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat exp(FloatFloat a) {
    const int nq = 4;
    const float eps = 1.0e-15f;
    FloatFloat al2 = FloatFloat_log2();
    // FP32 finite range: |x| < ln(3.4e38) ~= 88.7
    if (a.hi >= 88.0f) {
        Kokkos::printf("FFEXP: argument too large\n");
        return FloatFloat(0.0f);
    }
    if (a.hi <= -88.0f) return FloatFloat(0.0f);

    FloatFloat s0 = divide(a, al2);
    FloatFloat s1 = round_to_nearest_int(s0);
    float t1  = s1.hi;
    int nz    = (int)(t1 + Kokkos::copysign(1.0e-6f, t1));
    s0 = subtract(a, multiply(al2, s1));

    if (s0.hi == 0.0f) {
        return FloatFloat(ldexpf(1.0f, nz));
    }
    // Scale down by 2^nq then square nq times
    s1 = multiply_scalar(s0, ldexpf(1.0f, -nq));
    FloatFloat s2 = FloatFloat(1.0f), s3 = FloatFloat(1.0f);
    for (int l1 = 1; l1 <= 60; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (float)l1);
        s0 = add(s3, s2);
        s3 = s0;
        if (Kokkos::fabs(s2.hi) <= eps * Kokkos::fabs(s3.hi)) break;
        if (l1 == 60) { Kokkos::printf("FFEXP: iteration limit\n"); return FloatFloat(0.0f); }
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, s3);

    // Final scaling by 2^nz is exact in FP32 (power-of-2 multiplication does
    // not round). Going through multiply_scalar would compute b*8193 inside Dekker
    // splitting, which overflows for nz >= 115 (i.e. a > ~79) — the cause of
    // the previous NaN outputs at the high end of the input range.
    float pow2 = ldexpf(1.0f, nz);
    return FloatFloat(s3.hi * pow2, s3.lo * pow2);
}

KOKKOS_INLINE_FUNCTION FloatFloat log(FloatFloat a) {
    if (a.hi <= 0.0f) {
        Kokkos::printf("FFLOG: non-positive argument\n");
        return FloatFloat(0.0f);
    }
    // Initial approximation then 2 Newton steps (FP32 base gives ~6 digits, doubles per iter -> 24 -> 48 bits)
    FloatFloat b = FloatFloat(Kokkos::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        FloatFloat s0 = exp(b);
        FloatFloat s1 = subtract(a, s0);
        FloatFloat s2 = divide(s1, s0);
        b = add(b, s2);
    }
    return b;
}

KOKKOS_INLINE_FUNCTION FloatFloat log2(FloatFloat a) {
    return divide(log(a), FloatFloat_log2());
}

KOKKOS_INLINE_FUNCTION FloatFloat log10(FloatFloat a) {
    return divide(log(a), FloatFloat_log10());
}

KOKKOS_INLINE_FUNCTION FloatFloat log1p(FloatFloat a) {
    return log(add(FloatFloat(1.0f), a));
}

KOKKOS_INLINE_FUNCTION FloatFloat exp2(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log2()));
}

KOKKOS_INLINE_FUNCTION FloatFloat exp10(FloatFloat a) {
    return exp(multiply(a, FloatFloat_log10()));
}

KOKKOS_INLINE_FUNCTION FloatFloat expm1(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 0.5f) {
        return subtract(exp(a), FloatFloat(1.0f));
    }
    // Taylor: a + a^2/2! + a^3/3! + ...
    FloatFloat sum = a, term = a;
    for (int k = 2; k <= 30; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < 1.0e-15f * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// Track sin and cos jointly through nq doublings — avoids the sqrt(1-cos^2)
// recovery step, which loses relative precision when sin is near zero
// (i.e. when the answer most needs precision).
KOKKOS_INLINE_FUNCTION void sincos(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    const int itrmx = 100, nq = 4;
    const float eps = 1.0e-15f;
    if (a.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    if (a.hi >= 1.0e30f) {
        Kokkos::printf("FFCSSNR: argument too large\n");
        x = FloatFloat(0.0f); y = FloatFloat(0.0f); return;
    }
    FloatFloat pi2 = multiply_scalar(FloatFloat_pi(), 2.0f);
    FloatFloat s1  = divide(a, pi2);
    FloatFloat s2  = round_to_nearest_int(s1);
    FloatFloat s3  = subtract(a, multiply(pi2, s2));
    if (s3.hi == 0.0f) { x = FloatFloat(1.0f); y = FloatFloat(0.0f); return; }
    float scale = 1.0f / (float)(1 << nq);
    FloatFloat r  = multiply_scalar(s3, scale);   // r = s3 / 2^nq, |r| < pi/2^nq
    FloatFloat r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + r^5/5! - ...
    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    FloatFloat sin_r = r,             cos_r  = FloatFloat(1.0f);
    FloatFloat sterm = r,             cterm  = FloatFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (Kokkos::fabs(sterm.hi) < eps * Kokkos::fabs(sin_r.hi) &&
            Kokkos::fabs(cterm.hi) < eps) break;
        if (k == itrmx) { Kokkos::printf("FFCSSNR: iteration limit\n"); return; }
    }

    // Doubling: sin(2x) = 2 sin x cos x, cos(2x) = cos^2 x - sin^2 x
    for (int j = 0; j < nq; ++j) {
        FloatFloat new_sin = multiply_scalar(multiply(sin_r, cos_r), 2.0f);
        FloatFloat new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    x = cos_r; y = sin_r;
}

KOKKOS_INLINE_FUNCTION FloatFloat sin(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION FloatFloat cos(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION FloatFloat tan(FloatFloat a) {
    FloatFloat c, s; sincos(a, c, s); return divide(s, c);
}

// Angle of point (x, y) = atan2(y, x)
KOKKOS_INLINE_FUNCTION FloatFloat angle(FloatFloat x, FloatFloat y) {
    FloatFloat pi = FloatFloat_pi();
    if (x.hi == 0.0f && y.hi == 0.0f) return FloatFloat(0.0f);
    if (x.hi == 0.0f) return (y.hi > 0.0f) ? multiply_scalar(pi, 0.5f) : multiply_scalar(pi, -0.5f);
    if (y.hi == 0.0f) return (x.hi > 0.0f) ? FloatFloat(0.0f) : pi;
    FloatFloat r = sqrt(add(multiply(x,x), multiply(y,y)));
    FloatFloat nx = divide(x, r), ny = divide(y, r);
    FloatFloat a = FloatFloat(Kokkos::atan2(ny.hi, nx.hi));
    bool use_x = (Kokkos::fabs(nx.hi) <= Kokkos::fabs(ny.hi));
    FloatFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        FloatFloat sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        FloatFloat corr;
        if (use_x) {
            corr = divide(subtract(target, cos_a), sin_a);
            a = subtract(a, corr);
        } else {
            corr = divide(subtract(target, sin_a), cos_a);
            a = add(a, corr);
        }
    }
    return a;
}

KOKKOS_INLINE_FUNCTION FloatFloat asin(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFASIN: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(t, a);
}
KOKKOS_INLINE_FUNCTION FloatFloat acos(FloatFloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFACOS: argument out of range\n");
        return FloatFloat(0.0f);
    }
    FloatFloat t = sqrt(subtract(FloatFloat(1.0f), multiply(a, a)));
    return angle(a, t);
}
KOKKOS_INLINE_FUNCTION FloatFloat atan(FloatFloat a) {
    return angle(FloatFloat(1.0f), a);
}
KOKKOS_INLINE_FUNCTION FloatFloat atan2(FloatFloat y, FloatFloat x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

KOKKOS_INLINE_FUNCTION void sinhcosh(FloatFloat a, FloatFloat& x, FloatFloat& y) {
    // Taylor series for |a| < 0.5 — avoids the (e^a - e^-a)/2 cancellation
    // when a is small (both exponentials approach 1, leading bits cancel).
    if (Kokkos::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sinh_sum = a,             sinh_term = a;
        FloatFloat cosh_sum = FloatFloat(1.0f),  cosh_term = FloatFloat(1.0f);
        for (int k = 1; k <= 30; ++k) {
            sinh_term = divide_scalar(multiply(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = add(sinh_sum, sinh_term);
            cosh_term = divide_scalar(multiply(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = add(cosh_sum, cosh_term);
            if (Kokkos::fabs(sinh_term.hi) < 1.0e-15f * Kokkos::fabs(sinh_sum.hi) &&
                Kokkos::fabs(cosh_term.hi) < 1.0e-15f) break;
        }
        x = cosh_sum; y = sinh_sum;
        return;
    }
    FloatFloat s0 = exp(a);
    FloatFloat s1 = divide(FloatFloat(1.0f), s0);
    x = multiply_scalar(add(s0, s1), 0.5f);
    y = multiply_scalar(subtract(s0, s1), 0.5f);
}

KOKKOS_INLINE_FUNCTION FloatFloat sinh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION FloatFloat cosh(FloatFloat a) {
    FloatFloat c, s; sinhcosh(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION FloatFloat tanh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(tanh(negate(a)));
    FloatFloat e = expm1(multiply_scalar(a, 2.0f));
    return divide(e, add(e, FloatFloat(2.0f)));
}

KOKKOS_INLINE_FUNCTION FloatFloat asinh(FloatFloat a) {
    if (a.hi < 0.0f) return negate(asinh(negate(a)));
    return log(add(a, sqrt(add(multiply(a, a), FloatFloat(1.0f)))));
}
KOKKOS_INLINE_FUNCTION FloatFloat acosh(FloatFloat a) {
    if (a.hi < 1.0f) { Kokkos::printf("FFACOSH: argument < 1\n"); return FloatFloat(0.0f); }
    FloatFloat t1 = subtract(multiply(a, a), FloatFloat(1.0f));
    return log(add(a, sqrt(t1)));
}
KOKKOS_INLINE_FUNCTION FloatFloat atanh(FloatFloat a) {
    if (Kokkos::fabs(a.hi) >= 1.0f) { Kokkos::printf("FFATANH: |argument| >= 1\n"); return FloatFloat(0.0f); }
    // Taylor for |a|<0.5 avoids calling log (which loses precision when its
    // argument is close to 1). All terms positive — no cancellation.
    if (Kokkos::fabs(a.hi) < 0.5f) {
        FloatFloat a2 = multiply(a, a);
        FloatFloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr  = multiply(pwr, a2);
            FloatFloat term = divide_scalar(pwr, (float)(2*k + 1));
            sum  = add(sum, term);
            if (Kokkos::fabs(term.hi) < 1.0e-15f * Kokkos::fabs(sum.hi)) break;
        }
        return sum;
    }
    // For 0.5 <= |a| < 1, log((1+a)/(1-a)) is well-conditioned (ratio >= 3).
    FloatFloat t1 = add(FloatFloat(1.0f), a);
    FloatFloat t2 = subtract(FloatFloat(1.0f), a);
    return multiply_scalar(log(divide(t1, t2)), 0.5f);
}

// ============================================================
// Multi-argument operations
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat pow(FloatFloat a, FloatFloat b) {
    if (a.hi <= 0.0f) {
        if (a.hi == 0.0f && b.hi > 0.0f) return FloatFloat(0.0f);
        Kokkos::printf("FFPOW: non-positive base\n");
        return FloatFloat(0.0f);
    }
    return exp(multiply(log(a), b));
}

KOKKOS_INLINE_FUNCTION FloatFloat hypot(FloatFloat a, FloatFloat b) {
    return sqrt(add(multiply(a, a), multiply(b, b)));
}

KOKKOS_INLINE_FUNCTION FloatFloat ceil(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat floor(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat trunc(FloatFloat a);
KOKKOS_INLINE_FUNCTION FloatFloat round(FloatFloat a);

KOKKOS_INLINE_FUNCTION FloatFloat fmod(FloatFloat a, FloatFloat b) {
    FloatFloat q = divide(a, b);
    FloatFloat qt = trunc(q);
    return subtract(a, multiply(b, qt));
}

KOKKOS_INLINE_FUNCTION FloatFloat remainder(FloatFloat a, FloatFloat b) {
    FloatFloat q = divide(a, b);
    FloatFloat qn = round_to_nearest_int(q);
    return subtract(a, multiply(b, qn));
}

KOKKOS_INLINE_FUNCTION FloatFloat copysign(FloatFloat a, FloatFloat b) {
    FloatFloat r = abs(a);
    if (b.hi < 0.0f || (b.hi == 0.0f && b.lo < 0.0f)) return negate(r);
    return r;
}

KOKKOS_INLINE_FUNCTION FloatFloat fmax(FloatFloat a, FloatFloat b) {
    return (a > b) ? a : b;
}
KOKKOS_INLINE_FUNCTION FloatFloat fmin(FloatFloat a, FloatFloat b) {
    return (a < b) ? a : b;
}
KOKKOS_INLINE_FUNCTION FloatFloat fdim(FloatFloat a, FloatFloat b) {
    return (a > b) ? subtract(a, b) : FloatFloat(0.0f);
}
KOKKOS_INLINE_FUNCTION FloatFloat fma(FloatFloat a, FloatFloat b, FloatFloat c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat floor(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n > a) return subtract(n, FloatFloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION FloatFloat ceil(FloatFloat a) {
    FloatFloat n = round_to_nearest_int(a);
    if (n < a) return add(n, FloatFloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION FloatFloat trunc(FloatFloat a) {
    return (a.hi >= 0.0f) ? floor(a) : ceil(a);
}
KOKKOS_INLINE_FUNCTION FloatFloat round(FloatFloat a) {
    return round_to_nearest_int(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

// Internal helper (not part of the FF API surface): the SUM of the asymptotic
// erfc expansion, A&S 7.1.23,
//     erfc(z) = e^{-z^2}/sqrt(pi) * sum_k term_k,
//     term_0 = 1/|z|,  term_k = term_{k-1} * (-(2k-1))/(2 z^2),
// evaluated with optimal truncation. The series is DIVERGENT: its terms shrink
// to ~e^{-z^2} and then grow again, so summing past the smallest term strictly
// loses accuracy and a relative-eps exit can never be the primary one. Both the
// eps exit and the k cap are secondary; the smallest-term test below is primary.
// At FP32 the k cap is also a hard overflow guard — see B5's note in erf().
//
// Only the SUM is returned, not erfc: the two callers need different scalings.
// erf() (B5) sees only |z| in [3.5, 6], where e^{z^2} <= 4.3e15, and divides by
// it; erfc() (B6) runs out to the FP32 underflow floor |z| ~ 10.05, where e^{z^2}
// ~ 7.3e43 would first overflow the Dekker splitter (|z| ~ 8.93) and then trip
// exp()'s hard |arg| >= 88 guard (|z| ~ 9.38) — so erfc() multiplies by e^{-z^2}
// instead. Keeping the scaling at the call sites lets erf()'s arithmetic stay
// bit-for-bit what B5 shipped while erfc() gets the overflow-safe form.
//
// Preconditions: az = |z| > 0 and z2 = z*z, both finite.
KOKKOS_INLINE_FUNCTION FloatFloat erfc_asymptotic_sum(FloatFloat az, FloatFloat z2) {
    const float eps = 1.0e-14f;   // ~ FF relative resolution (2^-46); finer never fires
    FloatFloat two_z2 = multiply_scalar(z2, 2.0f);
    FloatFloat term = divide(FloatFloat(1.0f), az), sum = term;
    float prev_mag = Kokkos::fabs(term.hi);
    for (int k = 1; k <= 60; ++k) {
        FloatFloat next = divide(multiply_scalar(term, -(2.0f*k - 1.0f)), two_z2);
        float mag = Kokkos::fabs(next.hi);
        if (mag > prev_mag) break;                 // smallest term reached -> stop
        sum = add(sum, next);
        term = next; prev_mag = mag;
        if (mag <= eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat erf(FloatFloat z) {
    // B5 (FF): repair both erf branches for FP32's narrow exponent range. The DD
    // port (dd_math.hpp:670) computes each Taylor/asymptotic term from a separately
    // accumulated numerator (t2) and denominator (t3). At FP64 those intermediates
    // stay finite; at FP32 (max ~3.4e38) they overflow long before the loop's fixed
    // 60-iteration cap, so erf returned NaN across the whole smooth range ~[1.9, 6]
    // (probe: [2,6) was 100% NaN). Two independent failures were compounding:
    //   (a) Taylor branch: t2 = 2^k*z^{2k+1} and t3 = (2k+1)!! each overflow FP32
    //       around k~26-31 for |z| in [2,4); the loop never converged first.
    //   (b) Asymptotic branch: the erfc asymptotic series is *divergent*, so the
    //       relative-eps test never fired; the loop ran to k=60 where (2k-1)!!
    //       overflows FP32 -> inf/inf = NaN. (This is the FF sibling of DD's B3.)
    // Fix: (1) accumulate each term from the previous via its running ratio, which
    // keeps every intermediate O(term) and thus finite; (2) truncate the divergent
    // asymptotic series at its smallest term (standard optimal truncation); and
    // (3) lower the Taylor->asymptotic switchover to 3.5 (see kTaylorMax below) so
    // each branch is used only where it is both convergent and overflow-safe.
    // Series identities: A&S 7.1.6 (Taylor) and A&S 7.1.23 (asymptotic erfc).
    const float eps = 1.0e-14f; // ~ FF relative resolution (2^-46); finer never fires
    if (z.hi == 0.0f) return FloatFloat(0.0f);
    const float large = 6.0f; // erfc(6) ~= 2e-17 << FF resolution -> erf saturates
    if (z.hi >  large) return FloatFloat( 1.0f);
    if (z.hi < -large) return FloatFloat(-1.0f);

    FloatFloat z2 = multiply(z, z);
    int sign = (z.hi >= 0.0f) ? 1 : -1;
    FloatFloat az = abs(z);

    // Switchover derivation (probe over |z| in [1,6], incremental recurrence):
    // Taylor stays >= FF precision and converges within the iteration cap up to
    // |z| ~ 4; the asymptotic erf (erf = 1 - tiny erfc) clears the 8.45-digit gate
    // for |z| >~ 3. 3.5 sits inside both windows with margin.
    const float kTaylorMax = 3.5f;

    if (Kokkos::fabs(z.hi) < kTaylorMax) {
        // Taylor (A&S 7.1.6): erf(z) = (2/sqrt(pi)) e^{-z^2} sum_k term_k,
        //   term_0 = |z|,  term_k = term_{k-1} * (2 z^2) / (2k+1).
        // All terms positive; the running ratio keeps each O(<= sum) so no FP32
        // overflow (sum <= (sqrt(pi)/2) e^{z^2}, ~3.8e15 at |z|=6).
        FloatFloat two_z2 = multiply_scalar(z2, 2.0f);
        FloatFloat term = az, sum = az;
        for (int k = 1; k <= 100; ++k) {
            term = divide_scalar(multiply(term, two_z2), 2.0f*k + 1.0f);
            FloatFloat sumnew = add(sum, term);
            if (Kokkos::fabs(term.hi) <= eps * Kokkos::fabs(sumnew.hi)) { sum = sumnew; break; }
            sum = sumnew;
        }
        FloatFloat result = divide(multiply_scalar(sum, 2.0f),
                                   multiply(sqrt(FloatFloat_pi()), exp(z2)));
        return (sign > 0) ? result : negate(result);
    } else {
        // Asymptotic (A&S 7.1.23), optimal truncation — see erfc_asymptotic_sum()
        // above, which B6 factored out of this branch so erfc() can invoke the
        // identical series directly. The scaling stays here: |z| <= 6 on this
        // path, so e^{z^2} <= 4.3e15 and the divide is safe (erfc()'s
        // multiply-by-e^{-z^2} form is the one that has to reach |z| ~ 10). Then
        // erf = 1 - erfc; erfc is ~7.4e-7 at the |z| = 3.5 seam and smaller above,
        // so that subtraction is benign for erf (it is NOT for erfc itself — see
        // erfc() below).
        FloatFloat sum = erfc_asymptotic_sum(az, z2);
        FloatFloat erfc_val = divide(sum, multiply(sqrt(FloatFloat_pi()), exp(z2)));
        FloatFloat erf_val  = subtract(FloatFloat(1.0f), erfc_val);
        return (sign > 0) ? erf_val : negate(erf_val);
    }
}

// erfc — direct asymptotic expansion for z >= 5.75, 1 - erf(z) below that.
//
// B6: `erfc(z) = subtract(FloatFloat(1.0f), erf(z))` for ALL z is catastrophic
// cancellation as erf(z) -> 1. erf is accurate to u^2 RELATIVE to 1, so the
// difference carries an absolute error ~u^2 and erfc's relative error is
// u^2/erfc(z) — a loss of exactly log10(1/erfc(z)) digits, by construction, at
// every precision. The FP32 flavour of the DD sibling (B2, dd_math.hpp:erfc).
//
// What FF does differently from DD is the TOP of the range. DD's erf saturates
// at |z| = 8.5, well past where its asymptotic branch (live from 6.0) has pushed
// erfc's answer into erf's lo word, so DD's shipped erfc degraded to a flat
// ~16-digit shelf. FF's erf saturates to exactly ±1 at |z| > 6.0, only 2.5 above
// its own 3.5 switchover, so FF's shipped erfc does not get a shelf — it falls
// off a CLIFF: measured 7.60 digits at z = 6.00 and exactly 0.00 (erfc returns
// +0, relative error 1) at every z >= 6.02. That cliff, not a slow ramp, is what
// pinned the ff_accuracy_test erfc row's min at -0.00.
//
// Fix: for z >= kDirectMin, evaluate erfc directly from the SAME asymptotic
// series erf() uses (erfc_asymptotic_sum, A&S 7.1.23) and never form 1 - erf.
// Negative z needs no such path: erfc(-|z|) = 2 - erfc(|z|) is ~2, so
// subtract(1, erf) is benign there and already scores the 14-digit report cap.
//
// Threshold derivation. Same rule as B2: the cut goes where the direct series
// stops being worse than the fallback it replaces at EVERY measured point, not
// on average. Below the cut the fallback wins outright — the series' optimal-
// truncation floor is only 4.04 digits at z = 3.0 and 5.46 at z = 3.5, against
// the fallback's ~8 there. The two curves cross around z ~ 4.0, but the
// fallback's value is pure roundoff LUCK (its digit count is set by where
// erfc(z) happens to land in erf's 24-bit lo word), so it scatters: over
// [5.4, 6.0] its mean is 7.80 while individual points spike to 13.40. The direct
// series is smooth and climbs ~4 digits per unit z through that scatter band
// (7.09 at z = 4.0, 11.00 at 5.0, 13.18 at 5.5, then the 14-digit report cap).
// Enumerating EVERY representable float (not a grid — the fallback's scatter is
// per-float) over [5.4, 6.001], 1,260,389 inputs, exactly 2 regress, the highest
// at z = 5.6338043 by 0.106 digit. kDirectMin = 5.75 sits 0.116 above that, and
// over all 7,235,176 representable floats in [5.7, 10.3] NO input scores worse
// than the shipped 1 - erf code did.
//
// Rejected alternative — kDirectMin = 4.9, which a 0.00005-spaced grid says is
// clean (1 regressing point in 35,001 over [4.85, 6.6], worth 0.020 digit). It
// is not: exhaustive float enumeration finds the fallback's lucky spikes that a
// grid steps over. Grid sampling is the wrong instrument for a function whose
// error is roundoff-scattered; B2 used a 0.0005 grid because DD's fallback had a
// genuine 53-bit shelf with a narrow envelope, which FF's 24-bit lo word does
// not give.
//
// The pointwise rule is NOT free here, unlike at DD. B2 could apply it at no
// cost because DD's row mean was flat to +/-0.03 digit across its whole
// candidate window. FF's is not: the predicted ff_accuracy_test random-domain
// mean rises monotonically with a lower cut — 11.84 with no direct path, 11.96
// at 5.75, 12.26 at 5.0, 12.36 at 4.5-4.0 (the plateau), 12.21 at 3.0. So 5.75
// leaves ~0.40 digit of mean on the table versus the mean-optimal ~4.0. That is
// the deliberate trade: 4.0 regresses 814 of 6201 measured points by up to 1.93
// digits, and a measured regression is not worth 0.40 of a mean that already
// clears its gate by 3.5.
//
// KNOWN LIMITATION, not closed here (inherited verbatim from B2 DEV1). Below
// the cut the 1 - erf cancellation is a ramp, not a cliff, so it costs digits
// well under any usable threshold for THIS series. Exhaustive per-float window
// means against a quadmath oracle: [2, 3] 10.32, [3, 3.2] 8.77, [3.2, 3.5] 8.03,
// [3.5, 4] 6.26, [4, 4.5] 7.69, [4.5, 5] 7.81, [5, 5.5] 7.81, [5.5, 5.75] 7.81 —
// i.e. under the 8.45 test gate at scattered points from z ~ 2.94 and at
// substantially every point from z ~ 3.2 up to kDirectMin. A direct asymptotic
// path cannot help there: its optimal-truncation floor is 4.04 digits at z = 3.0
// and 5.46 at z = 3.5, at or below the subtract it would replace. (The global
// trough, 5.459 digits at z = 3.5, is not cancellation at all — it is erf()'s own
// Taylor->asymptotic seam, where the two paths return the identical value.)
// Closing that band needs a different algorithm (a Lentz continued fraction,
// A&S 7.1.14, or a triple-float erf). The ff_accuracy_test row gates on the MEAN
// and passes at 11.97 vs 8.45; the pointwise band is a separate, open concern.
KOKKOS_INLINE_FUNCTION FloatFloat erfc(FloatFloat z) {
    // See derivation above. Sits below erf()'s own |z| > 6.0 saturation, so the
    // cliff that saturation creates is never reached through erfc.
    const float kDirectMin = 5.75f;
    // Above this, erfc(z) rounds to +0 in FP32 and +0 is the only representable
    // answer, so honest underflow beats evaluating the series. Derivation:
    // erfc(z) ~ e^{-z^2}/(z*sqrt(pi)) crosses FLT_TRUE_MIN/2 = 7.0e-46 at
    // z ~ 10.05 (oracle: erfc(10.04) = 9.3e-46, erfc(10.06) = 6.2e-46), and the
    // series below independently first returns exactly 0 at z = 10.04. The clamp
    // also keeps z*z away from the range where the corpus (which feeds erfc
    // FLT_MAX) would make z2 = +inf and poison the series into NaN, and where
    // exp() would print its "argument too large" diagnostic.
    const float kUnderflowMax = 10.05f;
    if (z.hi >= kDirectMin) {
        if (!(z.hi <= kUnderflowMax)) return FloatFloat(0.0f);  // also catches +inf/NaN
        FloatFloat z2 = multiply(z, z);
        FloatFloat sum = erfc_asymptotic_sum(z, z2);
        // e^{-z^2}, not 1/e^{z^2}: forming sqrt(pi)*e^{z^2} and dividing by it
        // failed twice over when this was written. (a) The Dekker splitter
        // overflowed: multiply() computes conb = b.hi*8193 with b.hi = e^{z^2},
        // which passes FLT_MAX/(8193+1) = 4.15e34 at z = 8.93, giving
        // inf - inf = NaN. That was B8 / PORT_NOTES §4d's bug at multiply()'s
        // splitter, which B8 did not reach. It is FIXED as of B9 / §4j, which
        // scaled the splitter at multiply() and its siblings — so (a) no longer
        // bites, and this sighting is in fact what motivated B9. (b) still
        // stands on its own, and is sufficient by itself to keep this form: ff
        // exp() hits its hard |arg| >= 88 guard (FP32's finite range) at
        // z = 9.38 and returns 0 AND prints "FFEXP: argument too large", so the
        // quotient is 1/0. Measured pre-B9: the divide form returned NaN for
        // every z >= 8.95 and printed from z >= 9.38; post-B9 only the printing
        // and the 1/0 remain. The multiply form has neither failure and reaches
        // the underflow floor, so it stays.
        FloatFloat emz2;
        if (z2.hi < 88.0f) {
            emz2 = exp(negate(z2));
        } else {
            // Same |arg| >= 88 guard: quarter the argument (max |z^2/4| = 25.25
            // at kUnderflowMax) and square twice. Costs ~2 bits of exp's relative
            // error, which is moot here — every z past 9.38 has e^{-z^2} below
            // FLT_MIN, so the result is subnormal and losing bits anyway.
            emz2 = exp(divide_scalar(negate(z2), 4.0f));
            emz2 = multiply(emz2, emz2);
            emz2 = multiply(emz2, emz2);
        }
        return divide(multiply(sum, emz2), sqrt(FloatFloat_pi()));
    }
    return subtract(FloatFloat(1.0f), erf(z));
}

// gamma — Lanczos approximation
KOKKOS_INLINE_FUNCTION FloatFloat tgamma(FloatFloat a) {
    if (a.hi < 0.5f) {
        FloatFloat pi = FloatFloat_pi();
        FloatFloat sin_pi_a = sin(multiply(pi, a));
        return divide(pi, multiply(sin_pi_a, tgamma(subtract(FloatFloat(1.0f), a))));
    }
    // B7: Lanczos g=7 coefficients promoted from `float` to `double`. Stored as
    // `float` literals, each coefficient was truncated to FP32's ~7-digit ceiling,
    // capping tgamma at ~6 digits regardless of the enclosing FF arithmetic (a
    // mechanical DD->FF port artifact: the `double` literals of dd_math.hpp:730-738
    // were erroneously given `f` suffixes). As `double`, each FloatFloat(cN) call
    // below invokes the FloatFloat(double) constructor (ff_math.hpp:94), which splits
    // to a full FF pair (hi=(float)d, lo=(float)(d-(double)hi)) — ~14 digits, the FF
    // resolution floor. double (53-bit) exceeds FF (48-bit), so the split is
    // FF-exact. Coefficient set: Godfrey g=7 (P. Godfrey 2001; same values as
    // dd_math.hpp's DD tgamma / Boost.Math / Wikipedia "Lanczos approximation").
    const double c0 =  0.99999999999980993;
    const double c1 =  676.5203681218851;
    const double c2 = -1259.1392167224028;
    const double c3 =  771.32342877765313;
    const double c4 = -176.61502916214059;
    const double c5 =  12.507343278686905;
    const double c6 = -0.13857109526572012;
    const double c7 =  9.9843695780195716e-6;
    const double c8 =  1.5056327351493116e-7;
    FloatFloat x = subtract(a, FloatFloat(1.0f));
    FloatFloat t = add(x, FloatFloat(7.5f));
    FloatFloat s = FloatFloat(c0);
    s = add(s, divide(FloatFloat(c1), add(x, FloatFloat(1.0f))));
    s = add(s, divide(FloatFloat(c2), add(x, FloatFloat(2.0f))));
    s = add(s, divide(FloatFloat(c3), add(x, FloatFloat(3.0f))));
    s = add(s, divide(FloatFloat(c4), add(x, FloatFloat(4.0f))));
    s = add(s, divide(FloatFloat(c5), add(x, FloatFloat(5.0f))));
    s = add(s, divide(FloatFloat(c6), add(x, FloatFloat(6.0f))));
    s = add(s, divide(FloatFloat(c7), add(x, FloatFloat(7.0f))));
    s = add(s, divide(FloatFloat(c8), add(x, FloatFloat(8.0f))));
    // B7: sqrt(2*pi) leading factor — likewise `double` (was `2.5...f`), so it
    // splits to a full FF pair instead of capping the whole product at ~7 digits.
    FloatFloat two_pi_sqrt = FloatFloat(2.5066282746310002);
    return multiply(multiply(two_pi_sqrt, s),
                 multiply(pow(t, add(x, FloatFloat(0.5f))), exp(negate(t))));
}

// Bessel J0 via series
KOKKOS_INLINE_FUNCTION FloatFloat bessel_j0(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = FloatFloat(1.0f), sum = FloatFloat(1.0f);
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k*k));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_j1(FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat x2 = multiply_scalar(multiply(x, x), -0.25f);
    FloatFloat term = multiply_scalar(x, 0.5f), sum = term;
    for (int k = 1; k <= 60; ++k) {
        term = divide_scalar(multiply(term, x2), (float)(k * (k+1)));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_jn(int n, FloatFloat x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    FloatFloat j0 = bessel_j0(x), j1 = bessel_j1(x);
    FloatFloat jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        FloatFloat jp1 = subtract(multiply_scalar(divide(j_cur, x), 2.0f*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

KOKKOS_INLINE_FUNCTION FloatFloat bessel_y0(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j0 = bessel_j0(x);
    return multiply(two_over_pi, multiply(j0, log(multiply_scalar(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION FloatFloat bessel_y1(FloatFloat x) {
    FloatFloat two_over_pi = divide_scalar(FloatFloat(2.0f), FloatFloat_pi().hi);
    FloatFloat j1 = bessel_j1(x);
    return multiply(two_over_pi, multiply(j1, log(multiply_scalar(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION FloatFloat bessel_yn(int n, FloatFloat x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    FloatFloat y0 = bessel_y0(x), y1 = bessel_y1(x);
    FloatFloat ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        FloatFloat yp1 = subtract(multiply_scalar(divide(y_cur, x), 2.0f*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

KOKKOS_INLINE_FUNCTION FloatFloat zeta(FloatFloat s) {
    if (s.hi <= 1.0f) { Kokkos::printf("FFZETA: s <= 1\n"); return FloatFloat(0.0f); }
    const int N = 30;
    FloatFloat sum = FloatFloat(0.0f);
    for (int k = 1; k <= N; ++k)
        sum = add(sum, exp(multiply(negate(s), log(FloatFloat((float)k)))));
    FloatFloat tail = divide(exp(multiply(subtract(FloatFloat(1.0f), s), log(FloatFloat((float)N)))),
                         subtract(s, FloatFloat(1.0f)));
    return add(sum, tail);
}

KOKKOS_INLINE_FUNCTION FloatFloat expint(FloatFloat x) {
    FloatFloat eg = FloatFloat_euler_gamma();
    FloatFloat sum = add(eg, log(abs(x)));
    FloatFloat term = x;
    for (int k = 1; k <= 60; ++k) {
        sum = add(sum, divide_scalar(term, (float)(k * k)));
        term = multiply(term, x);
        if (Kokkos::fabs(term.hi) * 1e-15f < Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION FloatFloat incgamma(FloatFloat a, FloatFloat x) {
    const float eps = 1.0e-15f;
    FloatFloat term = divide(exp(negate(x)), a);
    FloatFloat sum  = term;
    for (int k = 1; k <= 60; ++k) {
        term = multiply(term, divide(x, add(a, FloatFloat((float)k))));
        sum  = add(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return multiply(sum, exp(multiply(a, log(x))));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp's __float128 overloads (and
// dd_math.hpp's block) so user code can call Kokkos::exp(ff) identically to
// Kokkos::exp(double)/Kokkos::exp(__float128). One-line forwards to the
// Kokkos::Experimental implementations.
// NOTE: add/subtract/multiply/divide are deliberately NOT forwarded here — those
// are reached via operators and explicit ADL, not as Kokkos::add etc.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat abs(Experimental::FloatFloat x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sqrt(Experimental::FloatFloat x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp(Experimental::FloatFloat x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp2(Experimental::FloatFloat x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat exp10(Experimental::FloatFloat x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat expm1(Experimental::FloatFloat x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log(Experimental::FloatFloat x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log2(Experimental::FloatFloat x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log10(Experimental::FloatFloat x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat log1p(Experimental::FloatFloat x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sin(Experimental::FloatFloat x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cos(Experimental::FloatFloat x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tan(Experimental::FloatFloat x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asin(Experimental::FloatFloat x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acos(Experimental::FloatFloat x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan(Experimental::FloatFloat x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atan2(Experimental::FloatFloat y, Experimental::FloatFloat x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat sinh(Experimental::FloatFloat x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat cosh(Experimental::FloatFloat x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tanh(Experimental::FloatFloat x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat asinh(Experimental::FloatFloat x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat acosh(Experimental::FloatFloat x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat atanh(Experimental::FloatFloat x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat pow(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat hypot(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmod(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat remainder(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat copysign(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmax(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fmin(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fdim(Experimental::FloatFloat a, Experimental::FloatFloat b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat fma(Experimental::FloatFloat a, Experimental::FloatFloat b, Experimental::FloatFloat c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat ceil(Experimental::FloatFloat x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat floor(Experimental::FloatFloat x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat round(Experimental::FloatFloat x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat trunc(Experimental::FloatFloat x) { return Experimental::trunc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erf(Experimental::FloatFloat x)   { return Experimental::erf(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat erfc(Experimental::FloatFloat x)  { return Experimental::erfc(x); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat tgamma(Experimental::FloatFloat x){ return Experimental::tgamma(x); }
// clang-format on
}  // namespace Kokkos
