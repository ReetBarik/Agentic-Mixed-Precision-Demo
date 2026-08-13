// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (original algorithms;
//   Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical port of the QD 2.3.24 quad-double package
// (qd/src/qd_real.cpp and qd/include/qd/qd_inline.h) from four-word FP64
// (quad-double, ~212-bit significand) to four-word FP32 (quad-float,
// "QuadFloat", ~96-bit significand, ~29 decimal digits, u = 2^-96). The
// algorithm structure — Priest renormalization (renorm_4), Hida-Li-Bailey
// sloppy/ieee addition, sloppy multiplication, long-division, and Heron
// square-root — descends directly from QD 2.3.24. Every non-trivial routine
// cites its QD source location.
//
// LICENSE LINEAGE (per docs/TEST_SUITE_PLAN.md §"Phase 2/3 open question — FF
// and QF port lineage", T3.0a kickoff): QF is modeled on QD, NOT on DDFUN, so
// it inherits QD's LBNL-BSD-License (triple-authored Hida/Li/Bailey, LBNL
// *institutional* copyright, commercial contact ipo@lbl.gov / TTD@lbl.gov) —
// which is a DIFFERENT license than the DHB-License that governs dd_math.hpp /
// ff_math.hpp (Bailey personal copyright, DDFUN provenance). The header here
// therefore carries LicenseRef-LBNL-BSD-License, not LicenseRef-DHB-License.
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.
//
// FP32-specific porting notes (splitter reuse, Newton/Heron iteration counts,
// sloppy_add safety at the narrower FP32 exponent, constant generation) are
// documented in docs/PORT_NOTES_QF.md.

#pragma once

// Quad-float real arithmetic — Kokkos::Experimental::QuadFloat.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Ported from QD 2.3.24 (Hida-Li-Bailey), swapping 4×FP64 for 4×FP32. Reuses
// the FP32 error-free-transform primitives (Knuth twoSum, Dekker twoProduct)
// already validated for the FF backend in third_party/include/ff_math.hpp
// (T2.1); the QD-style by-reference forms below are bit-identical to those.
//
// Precision: ~28.9 decimal digits (24-bit FP32 mantissa × 4 = ~96 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.
//
// Naming conventions (T0.4/T2.0, for eventual upstreaming to Kokkos) mirror
// dd_math.hpp / ff_math.hpp:
//   * Type + math live under namespace Kokkos::Experimental.
//   * Arithmetic free functions use STL-style names (add/subtract/multiply/
//     divide/negate) and are also reachable through operator overloads.
//   * Constants are free functions QuadFloat_pi(), QuadFloat_e(), ...
//   * The bit-pattern constructor is the static factory
//     QuadFloat::from_bits(f0, f1, f2, f3).
//   * Single-output math functions are re-exposed under namespace Kokkos at the
//     bottom of this header (T3.0a exposes abs/sqrt; T3.0b adds the
//     transcendentals). add/subtract/multiply/divide are NOT re-exposed under
//     Kokkos — operators + explicit ADL only, same posture as DD/FF.
//
// SCOPE (T3.0a): arithmetic + renormalization — renorm_4, add (sloppy +
// ieee), subtract, negate, abs, multiply, divide, sqrt, round_to_nearest_int,
// pow_int, and the six math constants.
// SCOPE (T3.0b): transcendentals appended after the arithmetic block — exp/
// expm1/exp2/exp10, log/log1p/log2/log10, sin/cos/sincos/tan, asin/acos/atan/
// atan2, sinh/cosh/tanh, asinh/acosh/atanh, pow, hypot, fmod, remainder,
// copysign, fmax/fmin/fdim/fma, ceil/floor/round/trunc. Ported from QD 2.3.24
// qd_real.cpp; the table-based QD transcendentals (inv_fact exp, pi/1024 sin/cos
// tables) are re-expressed in the table-free divide-by-k / joint-doubling style
// of dd_math.hpp / ff_math.hpp per the T3.0b task + PORT_NOTES §3a — see the
// block header at the exp/log section for the full source-fidelity note.
// SCOPE (T3.0c): complex QF lives in qf_complex.hpp (QuadFloatComplex),
// mirroring ff_complex.hpp / dd_complex.hpp. QD 2.3.24 ships no complex header,
// so that layer is modeled on this repo's own complex files, not on QD.

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
struct QuadFloat;
KOKKOS_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b);
KOKKOS_INLINE_FUNCTION QuadFloat divide_scalar(QuadFloat a, float b);
KOKKOS_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b);
KOKKOS_INLINE_FUNCTION QuadFloat negate(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat abs(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat sqr(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n);
// T3.0b transcendentals (forward decls — struct-independent, but several call
// each other, so declare the whole family up front).
KOKKOS_INLINE_FUNCTION QuadFloat exp(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat log(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat pow(QuadFloat a, QuadFloat b);
KOKKOS_INLINE_FUNCTION void      sincos(QuadFloat a, QuadFloat& sin_a, QuadFloat& cos_a);
KOKKOS_INLINE_FUNCTION void      sinhcosh(QuadFloat a, QuadFloat& sinh_a, QuadFloat& cosh_a);
KOKKOS_INLINE_FUNCTION QuadFloat angle(QuadFloat x, QuadFloat y);
KOKKOS_INLINE_FUNCTION QuadFloat ceil(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat floor(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat trunc(QuadFloat a);
KOKKOS_INLINE_FUNCTION QuadFloat round(QuadFloat a);

// ============================================================
// Error-free transforms (FP32).
// Bit-identical to the primitives validated for FF in ff_math.hpp (T2.1);
// expressed here in QD's by-reference form (QD 2.3.24 qd/include/qd/inline.h).
// ============================================================

// fl(a+b) and err, assuming |a| >= |b|.  QD inline.h:35-39 (quick_two_sum).
KOKKOS_INLINE_FUNCTION float qf_quick_two_sum(float a, float b, float& err) {
    float s = a + b;
    err = b - (s - a);
    return s;
}

// fl(a+b) and err (Knuth TwoSum, no ordering assumption).  QD inline.h:49-55.
// Mirror of the twoSum inside ff_math.hpp add() (ff_math.hpp:174-181).
KOKKOS_INLINE_FUNCTION float qf_two_sum(float a, float b, float& err) {
    float s  = a + b;
    float bb = s - a;
    err = (a - (s - bb)) + (b - bb);
    return s;
}

// fl(a*b) and err (Dekker TwoProduct via Veltkamp split).  QD inline.h:85-99.
// Splitter 8193.0f = 2^13+1 for the 24-bit FP32 mantissa — the same constant
// used and validated for FF (ff_math.hpp:194, two_prod ff_math.hpp:266-274).
// Empirically exact over |operands| <= 1e6 (scripts/gen_qf_constants harness /
// scripts/test_qfmul.cpp). Large-magnitude splitter overflow (QD's
// _QD_SPLIT_THRESH branch, inline.h:66-83) is NOT ported — see PORT_NOTES_QF.
KOKKOS_INLINE_FUNCTION float qf_two_prod(float a, float b, float& err) {
    const float split = 8193.0f;
    float cona = a * split, conb = b * split;
    float a1 = cona - (cona - a), b1 = conb - (conb - b);
    float a2 = a - a1,            b2 = b - b1;
    float p  = a * b;
    err = ((a1 * b1 - p) + a1 * b2 + a2 * b1) + a2 * b2;
    return p;
}

// fl(a*a) and err.  QD inline.h:101-113 (two_sqr).
KOKKOS_INLINE_FUNCTION float qf_two_sqr(float a, float& err) {
    const float split = 8193.0f;
    float con = a * split;
    float hi  = con - (con - a);
    float lo  = a - hi;
    float q   = a * a;
    err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
    return q;
}

// three_sum / three_sum2.  QD inline.h:192-204.
KOKKOS_INLINE_FUNCTION void qf_three_sum(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = qf_two_sum(t2, t3, c);
}
KOKKOS_INLINE_FUNCTION void qf_three_sum2(float& a, float& b, float& c) {
    float t1, t2, t3;
    t1 = qf_two_sum(a, b, t2);
    a  = qf_two_sum(c, t1, t3);
    b  = t2 + t3;
}

// ============================================================
// Renormalization (Priest normalization — Hida-Li-Bailey Algorithm 3)
// ============================================================

// Length-4 renormalization: collapse a 4-word unnormalized expansion to a
// non-overlapping length-4 QuadFloat.  Port of qd::renorm(c0,c1,c2,c3),
// QD 2.3.24 qd/include/qd/qd_inline.h:95-125.  Used by divide() and
// round_to_nearest_int().
KOKKOS_INLINE_FUNCTION void renorm(float& c0, float& c1, float& c2, float& c3) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (Kokkos::isinf(c0)) return;

    s0 = qf_quick_two_sum(c2, c3, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;
    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f)
            s2 = qf_quick_two_sum(s2, c3, s3);
        else
            s1 = qf_quick_two_sum(s1, c3, s2);
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f)
            s1 = qf_quick_two_sum(s1, c3, s2);
        else
            s0 = qf_quick_two_sum(s0, c3, s1);
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// Length-5 -> length-4 renormalization (the task's "renorm_4"): collapse a
// 5-word unnormalized accumulator (the natural output width of add/multiply/
// multiply_scalar/divide) to a non-overlapping length-4 QuadFloat.  Port of
// qd::renorm(c0,c1,c2,c3,c4), QD 2.3.24 qd_inline.h:127-177.
KOKKOS_INLINE_FUNCTION void renorm_4(float& c0, float& c1, float& c2,
                                     float& c3, float& c4) {
    float s0, s1, s2 = 0.0f, s3 = 0.0f;
    if (Kokkos::isinf(c0)) return;

    s0 = qf_quick_two_sum(c3, c4, c4);
    s0 = qf_quick_two_sum(c2, s0, c3);
    s0 = qf_quick_two_sum(c1, s0, c2);
    c0 = qf_quick_two_sum(c0, s0, c1);

    s0 = c0;
    s1 = c1;

    if (s1 != 0.0f) {
        s1 = qf_quick_two_sum(s1, c2, s2);
        if (s2 != 0.0f) {
            s2 = qf_quick_two_sum(s2, c3, s3);
            if (s3 != 0.0f)
                s3 += c4;
            else
                s2 = qf_quick_two_sum(s2, c4, s3);
        } else {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        }
    } else {
        s0 = qf_quick_two_sum(s0, c2, s1);
        if (s1 != 0.0f) {
            s1 = qf_quick_two_sum(s1, c3, s2);
            if (s2 != 0.0f)
                s2 = qf_quick_two_sum(s2, c4, s3);
            else
                s1 = qf_quick_two_sum(s1, c4, s2);
        } else {
            s0 = qf_quick_two_sum(s0, c3, s1);
            if (s1 != 0.0f)
                s1 = qf_quick_two_sum(s1, c4, s2);
            else
                s0 = qf_quick_two_sum(s0, c4, s1);
        }
    }
    c0 = s0; c1 = s1; c2 = s2; c3 = s3;
}

// ============================================================
// QuadFloat struct
// ============================================================
struct QuadFloat {
    float f0, f1, f2, f3;

    KOKKOS_INLINE_FUNCTION QuadFloat() : f0(0.0f), f1(0.0f), f2(0.0f), f3(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloat(float x) : f0(x), f1(0.0f), f2(0.0f), f3(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloat(float a0, float a1, float a2, float a3)
        : f0(a0), f1(a1), f2(a2), f3(a3) {}

    // Faithfully encode an FP64 value by successive FP32 splitting (Route-A,
    // length-4 analogue of ff_math.hpp's ffloat(double)). A double carries 53
    // bits, so two words suffice; the remaining words fall to 0 after the split.
    KOKKOS_INLINE_FUNCTION QuadFloat(double x) {
        double r = x;
        float  c0 = (float)r; r -= (double)c0;
        float  c1 = (float)r; r -= (double)c1;
        float  c2 = (float)r; r -= (double)c2;
        float  c3 = (float)r;
        f0 = c0; f1 = c1; f2 = c2; f3 = c3;
    }

    // LOCAL PATCH (QF integration 2026-08-13): int ctor — not upstream. Twin of
    // ff_math.hpp's 4f21245 patch. int -> QuadFloat is otherwise ambiguous: both
    // QuadFloat(float) and QuadFloat(double) are viable via a user-defined
    // conversion from int. Delegates to the double ctor so ints beyond float's
    // 2^24 exact range do not truncate at conversion. NOT explicit, so the scalar
    // comparison overloads below can still see an int literal.
    // Evidence: B3m.h / B4m.h "conversion from 'int' to 'QuadFloat' is ambiguous"
    // (e.g. `ql::Imag(r13) == 0`) in the per-group QF flip TU.
    KOKKOS_INLINE_FUNCTION QuadFloat(int h) : QuadFloat((double)h) {}

    KOKKOS_INLINE_FUNCTION QuadFloat(const QuadFloat& o)
        : f0(o.f0), f1(o.f1), f2(o.f2), f3(o.f3) {}
    KOKKOS_INLINE_FUNCTION QuadFloat& operator=(const QuadFloat& o) {
        f0=o.f0; f1=o.f1; f2=o.f2; f3=o.f3; return *this;
    }

    KOKKOS_INLINE_FUNCTION float operator[](int i) const {
        return (i==0)?f0:(i==1)?f1:(i==2)?f2:f3;
    }

    // Factory: build a QuadFloat from the IEEE-754 bit patterns of its four
    // FP32 components. Safe on host (memcpy) and device (__int_as_float).
    static KOKKOS_INLINE_FUNCTION QuadFloat from_bits(uint32_t b0, uint32_t b1,
                                                      uint32_t b2, uint32_t b3) {
        float a0, a1, a2, a3;
#ifndef __CUDA_ARCH__
        std::memcpy(&a0, &b0, sizeof(float));
        std::memcpy(&a1, &b1, sizeof(float));
        std::memcpy(&a2, &b2, sizeof(float));
        std::memcpy(&a3, &b3, sizeof(float));
#else
        a0 = __int_as_float(static_cast<int>(b0));
        a1 = __int_as_float(static_cast<int>(b1));
        a2 = __int_as_float(static_cast<int>(b2));
        a3 = __int_as_float(static_cast<int>(b3));
#endif
        return QuadFloat(a0, a1, a2, a3);
    }

    KOKKOS_INLINE_FUNCTION QuadFloat operator-() const { return negate(*this); }
    // LOCAL PATCH (QF integration 2026-08-13): unary operator+() — not upstream.
    // Twin of dd_math.hpp 44c1ec4 / ff_math.hpp 3ab4aa6. Template-arg promotion
    // rewrites double expressions to QuadFloat in place, so a source-level `+x`
    // must survive the rewrite.
    // Evidence: B2m.h `TOutput(+p3sq + m3sq - m4sq)` — "no match for 'operator+'
    // (operand type is 'const QuadFloat')" in the per-group QF flip TU.
    KOKKOS_INLINE_FUNCTION QuadFloat operator+() const { return *this; }
    // LOCAL PATCH (QF integration 2026-08-13): QuadFloat -> int conversion — not
    // upstream. Twin of dd_math.hpp 44c1ec4 / ff_math.hpp 45197be. Routes via the
    // FOUR-limb sum (not f0 alone) so the low words still contribute near an
    // exact-int boundary.
    // Evidence: B3m.h `ir14 = <QuadFloat expr>` where ir14 is int — "cannot convert
    // 'QuadFloat' to 'int' in assignment" in the per-group QF flip TU.
    // PAIRED with the scalar comparison overloads below — see the note there.
    KOKKOS_INLINE_FUNCTION operator int() const {
        return (int)((double)f0 + (double)f1 + (double)f2 + (double)f3);
    }
    KOKKOS_INLINE_FUNCTION QuadFloat operator+(QuadFloat b) const { return add(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator-(QuadFloat b) const { return subtract(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator*(QuadFloat b) const { return multiply(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator/(QuadFloat b) const { return divide(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator*(float b) const { return multiply_scalar(*this, b); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator+(float b) const { return add(*this, QuadFloat(b)); }
    KOKKOS_INLINE_FUNCTION QuadFloat operator-(float b) const { return subtract(*this, QuadFloat(b)); }

    KOKKOS_INLINE_FUNCTION QuadFloat& operator+=(QuadFloat b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator-=(QuadFloat b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator*=(QuadFloat b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloat& operator/=(QuadFloat b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(QuadFloat b) const {
        return f0==b.f0 && f1==b.f1 && f2==b.f2 && f3==b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator!=(QuadFloat b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(QuadFloat b) const {
        if (f0 != b.f0) return f0 < b.f0;
        if (f1 != b.f1) return f1 < b.f1;
        if (f2 != b.f2) return f2 < b.f2;
        return f3 < b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator>(QuadFloat b) const {
        if (f0 != b.f0) return f0 > b.f0;
        if (f1 != b.f1) return f1 > b.f1;
        if (f2 != b.f2) return f2 > b.f2;
        return f3 > b.f3;
    }
    KOKKOS_INLINE_FUNCTION bool operator<=(QuadFloat b) const { return !(*this > b); }
    KOKKOS_INLINE_FUNCTION bool operator>=(QuadFloat b) const { return !(*this < b); }

    // LOCAL PATCH (QF integration 2026-08-13): scalar-float comparisons — not
    // upstream. Twin of dd_math.hpp 44c1ec4 / ff_math.hpp f12d8bf, and the float
    // (not double) family for the same reason ff uses float: QuadFloat has BOTH a
    // float and a double converting ctor, so declaring both families would make
    // `qf == 0` ambiguous all over again.
    //
    // PAIRED WITH operator int() ABOVE — do not re-apply one without the other.
    // With operator int() in scope a bare `qf == 0` is ambiguous between
    // operator==(QuadFloat) (promote the literal via QuadFloat(int)) and built-in
    // int==int (demote *this via operator int()); both are user-defined
    // conversions. These overloads take the literal by a standard floating-integral
    // conversion, which outranks both, so they resolve it and win.
    KOKKOS_INLINE_FUNCTION bool operator==(float b) const { return *this == QuadFloat(b); }
    KOKKOS_INLINE_FUNCTION bool operator!=(float b) const { return *this != QuadFloat(b); }
    KOKKOS_INLINE_FUNCTION bool operator<(float b)  const { return *this <  QuadFloat(b); }
    KOKKOS_INLINE_FUNCTION bool operator>(float b)  const { return *this >  QuadFloat(b); }
    KOKKOS_INLINE_FUNCTION bool operator<=(float b) const { return *this <= QuadFloat(b); }
    KOKKOS_INLINE_FUNCTION bool operator>=(float b) const { return *this >= QuadFloat(b); }
};

KOKKOS_INLINE_FUNCTION QuadFloat operator+(float a, QuadFloat b) { return add(QuadFloat(a), b); }
KOKKOS_INLINE_FUNCTION QuadFloat operator-(float a, QuadFloat b) { return subtract(QuadFloat(a), b); }
KOKKOS_INLINE_FUNCTION QuadFloat operator*(float a, QuadFloat b) { return multiply_scalar(b, a); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const QuadFloat& d) {
    os << "[" << std::setprecision(8) << std::scientific
       << d.f0 << ", " << d.f1 << ", " << d.f2 << ", " << d.f3 << "]";
    return os;
}
#endif

// ============================================================
// Constants (4×FP32 bit patterns).
// Auto-generated by scripts/gen_qf_constants.cpp — do not edit by hand.
// Successive-splitting of a 113-bit __float128 constant into 4 FP32 words;
// reconstruction rel_err < 6e-31 for every entry (well below u = 2^-96).
// ============================================================
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_pi          () { return QuadFloat::from_bits(0x40490fdbU, 0xb3bbbd2eU, 0xa7772cedU, 0x19cc5170U); } // pi
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_e           () { return QuadFloat::from_bits(0x402df854U, 0x33b14577U, 0xa7559541U, 0x1ae2b101U); } // e
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_log2        () { return QuadFloat::from_bits(0x3f317218U, 0xb102e308U, 0xa4ca86c4U, 0x186ce601U); } // ln(2)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_log10       () { return QuadFloat::from_bits(0x40135d8eU, 0xb309555dU, 0xa69f48adU, 0x9a129d48U); } // ln(10)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_sqrt2       () { return QuadFloat::from_bits(0x3fb504f3U, 0x32cfe77aU, 0xa65bdd34U, 0x989d9323U); } // sqrt(2)
KOKKOS_INLINE_FUNCTION QuadFloat QuadFloat_euler_gamma () { return QuadFloat::from_bits(0x3f13c468U, 0xb1e4127aU, 0x24f49a38U, 0x97e03f7fU); } // Euler gamma

// ============================================================
// Negation / absolute value
// ============================================================

// QD qd_inline.h:438-440 (operator-).
KOKKOS_INLINE_FUNCTION QuadFloat negate(QuadFloat a) {
    return QuadFloat(-a.f0, -a.f1, -a.f2, -a.f3);
}

// QD qd_inline.h:788-790 (abs).
KOKKOS_INLINE_FUNCTION QuadFloat abs(QuadFloat a) {
    return (a.f0 < 0.0f) ? negate(a) : a;
}

// ============================================================
// Addition
// ============================================================

// IEEE-style addition (satisfies the IEEE error bound).  Port of
// qd_real::ieee_add, QD 2.3.24 qd_inline.h:286-336, plus the quick_three_accum
// helper it calls (qd_inline.h:261-282).  NOT the default (QD builds with
// QD_IEEE_ADD off by default); provided for parity with the DD/FF story and for
// callers that need the tighter bound.
KOKKOS_INLINE_FUNCTION float qf_quick_three_accum(float& a, float& b, float c) {
    float s;
    bool za, zb;
    s = qf_two_sum(b, c, b);
    s = qf_two_sum(a, s, a);
    za = (a != 0.0f);
    zb = (b != 0.0f);
    if (za && zb) return s;
    if (!zb) { b = a; a = s; }
    else     { a = s; }
    return 0.0f;
}

KOKKOS_INLINE_FUNCTION QuadFloat ieee_add(QuadFloat a, QuadFloat b) {
    int i, j, k;
    float s, t;
    float u, v;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    i = j = k = 0;
    if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) u = a[i++]; else u = b[j++];
    if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) v = a[i++]; else v = b[j++];

    u = qf_quick_two_sum(u, v, v);

    while (k < 4) {
        if (i >= 4 && j >= 4) {
            x[k] = u;
            if (k < 3) x[++k] = v;
            break;
        }
        if (i >= 4)                                    t = b[j++];
        else if (j >= 4)                               t = a[i++];
        else if (Kokkos::fabs(a[i]) > Kokkos::fabs(b[j])) t = a[i++];
        else                                           t = b[j++];

        s = qf_quick_three_accum(u, v, t);
        if (s != 0.0f) x[k++] = s;
    }
    for (k = i; k < 4; k++) x[3] += a[k];
    for (k = j; k < 4; k++) x[3] += b[k];

    renorm(x[0], x[1], x[2], x[3]);
    return QuadFloat(x[0], x[1], x[2], x[3]);
}

// Cray-style ("sloppy") addition — the QD DEFAULT (QD_IEEE_ADD undefined).
// Port of qd_real::sloppy_add, QD 2.3.24 qd_inline.h:338-405 (the active,
// data-dependency-minimized form; QD documents it as identical to the
// commented two_sum version at qd_inline.h:339-354).  This is what
// add()/operator+ dispatch to, matching QD's out-of-the-box configuration.
// Safe at FP32: correctness rests on the inputs being non-overlapping
// expansions, not on the exponent range (see PORT_NOTES_QF).
KOKKOS_INLINE_FUNCTION QuadFloat sloppy_add(QuadFloat a, QuadFloat b) {
    float s0, s1, s2, s3;
    float t0, t1, t2, t3;
    float v0, v1, v2, v3;
    float u0, u1, u2, u3;
    float w0, w1, w2, w3;

    s0 = a.f0 + b.f0;  s1 = a.f1 + b.f1;  s2 = a.f2 + b.f2;  s3 = a.f3 + b.f3;

    v0 = s0 - a.f0;    v1 = s1 - a.f1;    v2 = s2 - a.f2;    v3 = s3 - a.f3;
    u0 = s0 - v0;      u1 = s1 - v1;      u2 = s2 - v2;      u3 = s3 - v3;
    w0 = a.f0 - u0;    w1 = a.f1 - u1;    w2 = a.f2 - u2;    w3 = a.f3 - u3;

    u0 = b.f0 - v0;    u1 = b.f1 - v1;    u2 = b.f2 - v2;    u3 = b.f3 - v3;

    t0 = w0 + u0;      t1 = w1 + u1;      t2 = w2 + u2;      t3 = w3 + u3;

    s1 = qf_two_sum(s1, t0, t0);
    qf_three_sum(s2, t0, t1);
    qf_three_sum2(s3, t0, t2);
    t0 = t0 + t1 + t3;

    renorm_4(s0, s1, s2, s3, t0);
    return QuadFloat(s0, s1, s2, s3);
}

// QD default dispatch (qd_inline.h:408-414): operator+ -> sloppy_add.
KOKKOS_INLINE_FUNCTION QuadFloat add(QuadFloat a, QuadFloat b) {
    return sloppy_add(a, b);
}

// QD qd_inline.h:459-461 (operator-): a - b == a + (-b).
KOKKOS_INLINE_FUNCTION QuadFloat subtract(QuadFloat a, QuadFloat b) {
    return add(a, negate(b));
}

// ============================================================
// Multiplication
// ============================================================

// quad-float * float.  Port of operator*(qd_real, double),
// QD 2.3.24 qd_inline.h:490-514.
KOKKOS_INLINE_FUNCTION QuadFloat multiply_scalar(QuadFloat a, float b) {
    float p0, p1, p2, p3;
    float q0, q1, q2;
    float s0, s1, s2, s3, s4;

    p0 = qf_two_prod(a.f0, b, q0);
    p1 = qf_two_prod(a.f1, b, q1);
    p2 = qf_two_prod(a.f2, b, q2);
    p3 = a.f3 * b;

    s0 = p0;
    s1 = qf_two_sum(q0, p1, s2);
    qf_three_sum(s2, q1, p2);
    qf_three_sum2(q1, q2, p3);
    s3 = q1;
    s4 = q2 + p2;

    renorm_4(s0, s1, s2, s3, s4);
    return QuadFloat(s0, s1, s2, s3);
}

// Exact multiplication by a power of two (no rounding).  QD qd_inline.h:485-487.
KOKKOS_INLINE_FUNCTION QuadFloat mul_pwr2(QuadFloat a, float b) {
    return QuadFloat(a.f0 * b, a.f1 * b, a.f2 * b, a.f3 * b);
}

// quad-float * quad-float — "sloppy" multiplication, the QD DEFAULT
// (QD builds with QD_SLOPPY_MUL on).  16 partial products a_i*b_j; the leading
// 6 (weights u^0..u^2) are formed with exact two_prod, the O(u^3) terms are
// folded in scalar, and terms of weight u^4 and higher are dropped before
// renorm_4 collapses the length-5 accumulator to length-4.  Port of
// qd_real::sloppy_mul, QD 2.3.24 qd_inline.h:567-599.
KOKKOS_INLINE_FUNCTION QuadFloat multiply(QuadFloat a, QuadFloat b) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3, q4, q5;
    float t0, t1;
    float s0, s1, s2;

    p0 = qf_two_prod(a.f0, b.f0, q0);

    p1 = qf_two_prod(a.f0, b.f1, q1);
    p2 = qf_two_prod(a.f1, b.f0, q2);

    p3 = qf_two_prod(a.f0, b.f2, q3);
    p4 = qf_two_prod(a.f1, b.f1, q4);
    p5 = qf_two_prod(a.f2, b.f0, q5);

    // Start accumulation.
    qf_three_sum(p1, p2, q0);

    // Six-three sum of (p2, q1, q2) and (p3, p4, p5).
    qf_three_sum(p2, q1, q2);
    qf_three_sum(p3, p4, p5);
    s0 = qf_two_sum(p2, p3, t0);
    s1 = qf_two_sum(q1, p4, t1);
    s2 = q2 + p5;
    s1 = qf_two_sum(s1, t0, t0);
    s2 += (t0 + t1);

    // O(u^3) terms: the nine remaining cross-products, folded in scalar.
    s1 += a.f0*b.f3 + a.f1*b.f2 + a.f2*b.f1 + a.f3*b.f0 + q0 + q3 + q4 + q5;
    renorm_4(p0, p1, s0, s1, s2);
    return QuadFloat(p0, p1, s0, s1);
}

// quad-float ^ 2.  Port of sqr(qd_real), QD 2.3.24 qd_inline.h:674-715.
KOKKOS_INLINE_FUNCTION QuadFloat sqr(QuadFloat a) {
    float p0, p1, p2, p3, p4, p5;
    float q0, q1, q2, q3;
    float s0, s1;
    float t0, t1;

    p0 = qf_two_sqr(a.f0, q0);
    p1 = qf_two_prod(2.0f * a.f0, a.f1, q1);
    p2 = qf_two_prod(2.0f * a.f0, a.f2, q2);
    p3 = qf_two_sqr(a.f1, q3);

    p1 = qf_two_sum(q0, p1, q0);

    q0 = qf_two_sum(q0, q1, q1);
    p2 = qf_two_sum(p2, p3, p3);

    s0 = qf_two_sum(q0, p2, t0);
    s1 = qf_two_sum(q1, p3, t1);

    s1 = qf_two_sum(s1, t0, t0);
    t0 += t1;

    s1 = qf_quick_two_sum(s1, t0, t0);
    p2 = qf_quick_two_sum(s0, s1, t1);
    p3 = qf_quick_two_sum(t1, t0, q0);

    p4 = 2.0f * a.f0 * a.f3;
    p5 = 2.0f * a.f1 * a.f2;

    p4 = qf_two_sum(p4, p5, p5);
    q2 = qf_two_sum(q2, q3, q3);

    t0 = qf_two_sum(p4, q2, t1);
    t1 = t1 + p5 + q3;

    p3 = qf_two_sum(p3, t0, p4);
    p4 = p4 + q0 + t1;

    renorm_4(p0, p1, p2, p3, p4);
    return QuadFloat(p0, p1, p2, p3);
}

// ============================================================
// Division
// ============================================================

// quad-float / quad-float — "sloppy" long division, the QD DEFAULT
// (QD builds with QD_SLOPPY_DIV on).  Port of qd_real::sloppy_div,
// QD 2.3.24 qd/src/qd_real.cpp:693-712.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// divide as "Newton iteration, initial reciprocal from FP32 division, 3
// iterations". QD 2.3.24's qd_real::div is NOT Newton — it is classical long
// division: each quotient digit q_k = r[0]/b[0] contributes ~24 fresh bits
// (q0~24, q1~48, q2~72, q3~96), and the residual r is refined by a full
// QuadFloat multiply-subtract between digits. Four digits reach the ~96-bit
// QuadFloat width; the accurate variant adds a fifth digit + length-5 renorm.
// This header ports QD's actual routine and cites it; the discrepancy with the
// task text is recorded in PORT_NOTES_QF and the T3.0a report.
KOKKOS_INLINE_FUNCTION QuadFloat divide(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3;
    QuadFloat r;

    q0 = a.f0 / b.f0;
    r = subtract(a, multiply_scalar(b, q0));

    q1 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q1));

    q2 = r.f0 / b.f0;
    r = subtract(r, multiply_scalar(b, q2));

    q3 = r.f0 / b.f0;

    renorm(q0, q1, q2, q3);
    return QuadFloat(q0, q1, q2, q3);
}

// Long division by a plain FP32 scalar.  Same four-quotient-digit structure as
// divide() above and the same closing renorm; the only change is the residual
// correction. When the divisor is a single float, q_i * b is ONE exact product
// (qf_two_prod gives it as a non-overlapping (p, e) pair, so (p, e, 0, 0) is a
// valid QuadFloat), where divide() has to run a full four-word multiply_scalar.
//
// Motivation: the Taylor loops in exp/expm1/sincos/sinhcosh/atanh divide by a
// small loop integer on every iteration. Routing that through divide()
// promotes the integer to a QuadFloat and pays the full four-word division —
// the dominant cost in every QF transcendental. DD and FF have carried a
// divide_scalar since the DDFUN port; QF did not, and this closes that gap.
//
// Accuracy: identical algorithm to divide(), and the residual correction here
// is exact rather than merely faithful, so this is not a precision/speed
// trade. Validated against qf_accuracy_test / qf_property_test.
KOKKOS_INLINE_FUNCTION QuadFloat divide_scalar(QuadFloat a, float b) {
    float q0, q1, q2, q3, p, e;
    QuadFloat r;

    q0 = a.f0 / b;  p = qf_two_prod(q0, b, e);  r = subtract(a, QuadFloat(p, e, 0.0f, 0.0f));
    q1 = r.f0 / b;  p = qf_two_prod(q1, b, e);  r = subtract(r, QuadFloat(p, e, 0.0f, 0.0f));
    q2 = r.f0 / b;  p = qf_two_prod(q2, b, e);  r = subtract(r, QuadFloat(p, e, 0.0f, 0.0f));

    q3 = r.f0 / b;

    renorm(q0, q1, q2, q3);
    return QuadFloat(q0, q1, q2, q3);
}

// Accurate long division (five quotient digits + length-5 renorm).  Port of
// qd_real::accurate_div, QD 2.3.24 qd_real.cpp:714-736. Not the default;
// provided for parity with QD and for tight-bound callers (T3.4).
KOKKOS_INLINE_FUNCTION QuadFloat divide_accurate(QuadFloat a, QuadFloat b) {
    float q0, q1, q2, q3, q4;
    QuadFloat r;

    q0 = a.f0 / b.f0;  r = subtract(a, multiply_scalar(b, q0));
    q1 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q1));
    q2 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q2));
    q3 = r.f0 / b.f0;  r = subtract(r, multiply_scalar(b, q3));
    q4 = r.f0 / b.f0;

    renorm_4(q0, q1, q2, q3, q4);
    return QuadFloat(q0, q1, q2, q3);
}

// ============================================================
// Square root
// ============================================================

// QuadFloat square root — Heron's method (a Newton iteration on x^2 - a),
// each step doubling the number of correct digits: y = (1/2)(x + a/x).
// Port of fsqrt / sqrt(qd_real), QD 2.3.24 qd/src/qd_real.cpp:738-785.
//
// NOTE ON ALGORITHM (source-fidelity, rule 6): the T3.0a task text describes
// sqrt as "Newton iteration, initial reciprocal from FP32 division, same
// posture as divide" — that is the Karp reciprocal-Newton variant used by
// dd_real::sqrt (QD dd_real.cpp:47-72). QD 2.3.24's qd_real::sqrt is instead
// Heron's method (fsqrt), which needs a full QuadFloat divide per step. This
// header ports QD's actual qd_real::sqrt. Iteration count: the FP32 seed
// sqrt(a.f0) is accurate to ~24 bits; Heron doubles precision per step
// (24 -> 48 -> 96, saturating at the ~96-bit QuadFloat width), so 3 iterations
// suffice. QD's loop runs up to 10 with an early-out convergence test; the port
// keeps that structure with eps = 2^-96 so it stops after ~3 on real inputs.
KOKKOS_INLINE_FUNCTION QuadFloat sqrt(QuadFloat a) {
    if (a.f0 == 0.0f && a.f1 == 0.0f && a.f2 == 0.0f && a.f3 == 0.0f)
        return QuadFloat(0.0f);
    if (a.f0 < 0.0f) {
        Kokkos::printf("QFSQRT: negative argument\n");
        return QuadFloat(0.0f);
    }

    const float eps = 1.2621774e-29f; // ~= 2^-96, QuadFloat unit roundoff
    const QuadFloat half(0.5f);

    QuadFloat x = QuadFloat(Kokkos::sqrt(a.f0));  // ~24-bit FP32 seed
    for (int i = 0; i < 10; ++i) {
        QuadFloat y    = multiply(half, add(x, divide(a, x)));
        QuadFloat diff = subtract(x, y);
        x = y;
        float e = Kokkos::fabs(((diff.f3 + diff.f2) + diff.f1) + diff.f0);
        if (e < Kokkos::fabs(x.f0) * eps)
            return x;
    }
    return x;
}

// ============================================================
// Nearest integer
// ============================================================

// Nearest FP32-int of a single float (round-half-away-from-zero via floor).
// QD 2.3.24 qd/include/qd/inline.h:116-120 (nint(double)), at FP32.
// NOTE (PORT_NOTES_QF): QD's nint does NOT use the 2^(2p-1) magic-constant
// trick that broke FF's ffnint at FP32 (PORT_NOTES §4b); it uses floor(d+0.5),
// which is well-conditioned at every FP32 magnitude, so the FF bug does not
// recur here.
KOKKOS_INLINE_FUNCTION float qf_nint(float d) {
    if (d == Kokkos::floor(d)) return d;
    return Kokkos::floor(d + 0.5f);
}

// Nearest integer of a QuadFloat, component-wise with half-integer tie
// corrections and a final renorm.  Port of nint(qd_real),
// QD 2.3.24 qd/src/qd_real.cpp:48-86.
KOKKOS_INLINE_FUNCTION QuadFloat round_to_nearest_int(QuadFloat a) {
    float x0, x1, x2, x3;
    x0 = qf_nint(a.f0);
    x1 = x2 = x3 = 0.0f;

    if (x0 == a.f0) {
        x1 = qf_nint(a.f1);
        if (x1 == a.f1) {
            x2 = qf_nint(a.f2);
            if (x2 == a.f2) {
                x3 = qf_nint(a.f3);
            } else {
                if (Kokkos::fabs(x2 - a.f2) == 0.5f && a.f3 < 0.0f) x2 -= 1.0f;
            }
        } else {
            if (Kokkos::fabs(x1 - a.f1) == 0.5f && a.f2 < 0.0f) x1 -= 1.0f;
        }
    } else {
        if (Kokkos::fabs(x0 - a.f0) == 0.5f && a.f1 < 0.0f) x0 -= 1.0f;
    }

    renorm(x0, x1, x2, x3);
    return QuadFloat(x0, x1, x2, x3);
}

// ============================================================
// Integer power
// ============================================================

// a^n by binary exponentiation using sqr.  Port of pow(qd_real, int),
// QD 2.3.24 qd/src/qd_real.cpp:568-598.
KOKKOS_INLINE_FUNCTION QuadFloat pow_int(QuadFloat a, int n) {
    if (n == 0) return QuadFloat(1.0f);

    QuadFloat r = a;              // odd-case multiplier
    QuadFloat s = QuadFloat(1.0f); // running answer
    int N = (n < 0) ? -n : n;

    if (N > 1) {
        while (N > 0) {
            if (N % 2 == 1) s = multiply(s, r);
            N /= 2;
            if (N > 0) r = sqr(r);
        }
    } else {
        s = r;
    }

    if (n < 0) return divide(QuadFloat(1.0f), s);
    return s;
}

// ============================================================
// Exp / Log family
// ============================================================
//
// SOURCE-FIDELITY NOTE (transcendental block, rule 6). QD 2.3.24's actual
// transcendentals in qd/src/qd_real.cpp are TABLE-BASED: exp (qd_real.cpp:925)
// uses a 15-entry inv_fact[] Taylor table plus 16 squarings; sin/cos/sincos
// (qd_real.cpp:2136-2360) reduce mod pi/2 then mod pi/1024 and look up 256-entry
// sin_table/cos_table of sin/cos(k*pi/1024). The QF port instead uses the
// TABLE-FREE structure of the sibling dd_math.hpp / ff_math.hpp headers
// (divide-by-k Taylor, joint sin/cos doublings), for three reasons: (1) the
// T3.0b task text + PORT_NOTES.md §3a explicitly direct "joint sin/cos
// doublings" and a divide-by-k Taylor with "more terms than QD's FP64 version";
// (2) 256-entry 4xFP32 tables are device-hostile (constant memory / register
// pressure under CUDA) whereas dd/ff are the designated portable multi-word
// references; (3) it keeps QF byte-consistent in *style* with its DD/FF
// siblings. Each function below still cites the QD 2.3.24 routine it mirrors
// mathematically, and flags where it follows dd/ff structure instead. The
// argument-reduction and Newton skeletons (log-Newton, atan2-Newton) ARE ported
// faithfully from QD and cited. sin_table/cos_table/inv_fact are therefore NOT
// generated — see docs/PORT_NOTES_QF.md.

// e^a.  Mathematical mirror of exp(qd_real), QD 2.3.24 qd_real.cpp:925-983
// (same reduce-by-m*log2 / scale-by-2^-nq / Taylor / square-nq-times skeleton),
// but with the table-free divide-by-k Taylor of dd_math.hpp:345 / ff_math.hpp:347
// rather than QD's inv_fact[] table.
//
// EXP TERM-COUNT DERIVATION (T3.0b deliverable). After reduction |s0| <= log2/2
// = 0.347; scaling by 2^-nq gives |r| <= 0.347/2^nq. The divide-by-k Taylor
// e^r = sum_k r^k/k! must reach the QF unit roundoff u = 2^-96 ~= 1.3e-29:
// need |r|^N / N! < u.  With nq = 6 (|r| <= 5.4e-3): N = 11 terms suffice
// (5.4e-3^11 / 11! ~= 1e-30 < u).  QD's FP64 exp uses nq = 16 squarings + a
// 15-entry inv_fact table for its ~64-digit target; QF needs FAR fewer terms
// (11 vs QD's effective ~9 at a 4096x finer reduction) and NO factorial table,
// because dividing by k each step accumulates 1/k! directly.  nq = 6 (matching
// dd_math.hpp) balances squaring cost against Taylor length; the loop caps at 60
// and, unlike ff_math.hpp:376, does NOT return 0 on the cap (that FF behavior
// is a latent stall bug — see PORT_NOTES_QF; here we proceed with the best sum).
KOKKOS_INLINE_FUNCTION QuadFloat exp(QuadFloat a) {
    const int   nq  = 6;
    // eps is deliberately COARSER than QF's resolution u = 2^-96 ~= 1.3e-29.
    // ff_math.hpp used eps = 1e-15f finer than FloatFloat resolution 3.55e-15,
    // which made the term never fall below eps*sum -> spurious stalls / 0-returns
    // (the FF exp-eps bug). 1e-28f > u keeps convergence reachable at QF width.
    const float eps = 1.0e-28f;
    QuadFloat al2 = QuadFloat_log2();
    // FP32 finite range: e^a overflows FP32 (~3.4e38) at a ~= 88.7; underflows
    // to 0 below a ~= -88.  Mirror ff_math.hpp:352 guards (QD uses +-709 for FP64).
    if (a.f0 >= 88.0f) {
        Kokkos::printf("QFEXP: argument too large\n");
        return QuadFloat(0.0f);
    }
    if (a.f0 <= -88.0f) return QuadFloat(0.0f);

    QuadFloat s0 = divide(a, al2);
    QuadFloat s1 = round_to_nearest_int(s0);
    float t1  = s1.f0;
    int   nz  = (int)(t1 + Kokkos::copysign(1.0e-6f, t1));
    s0 = subtract(a, multiply(al2, s1));            // |s0| <= log2/2

    if (s0.f0 == 0.0f && s0.f1 == 0.0f) {
        return QuadFloat(ldexpf(1.0f, nz));         // result = 2^nz exactly
    }
    // Scale down by 2^nq (exact via mul_pwr2, no Dekker splitter), Taylor, then
    // square nq times: e^r squared nq times = e^(2^nq r) = e^s0.
    s1 = mul_pwr2(s0, ldexpf(1.0f, -nq));           // r = s0 / 2^nq
    QuadFloat s2 = QuadFloat(1.0f), s3 = QuadFloat(1.0f);  // s2 = term, s3 = sum
    for (int l1 = 1; l1 <= 60; ++l1) {
        s0 = multiply(s2, s1);
        s2 = divide_scalar(s0, (float)l1);      // term = r^l1 / l1!
        s3 = add(s3, s2);
        if (Kokkos::fabs(s2.f0) <= eps * Kokkos::fabs(s3.f0)) break;
        // NOTE: no return-0 on l1 == 60 (see header comment); fall through with s3.
    }
    for (int i = 0; i < nq; ++i) s3 = multiply(s3, s3);

    // Final scaling by 2^nz.  PORT_NOTES §4a: power-of-2 multiplication is exact
    // in FP32 and must NOT go through multiply_scalar (which would compute
    // 8193*2^nz inside Dekker splitting and overflow FP32 for nz >= 116).  Scale
    // each component directly.  This is also QD's approach (ldexp(s, m),
    // qd_real.cpp:982, which is component-wise std::ldexp).
    float pow2 = ldexpf(1.0f, nz);
    return QuadFloat(s3.f0 * pow2, s3.f1 * pow2, s3.f2 * pow2, s3.f3 * pow2);
}

// log(a) via Newton's iteration on f(x) = exp(x) - a.  Faithful port of
// log(qd_real), QD 2.3.24 qd_real.cpp:986-1011: seed x = log(a.f0), then
// x <- x + a*exp(-x) - 1 three times (Newton ~doubles correct digits per step;
// FP32 seed ~24 bits -> 48 -> 96, saturating at QF width on the 3rd).  Same
// three-step structure as dd_math.hpp:380.
KOKKOS_INLINE_FUNCTION QuadFloat log(QuadFloat a) {
    if (a.f0 <= 0.0f) {
        Kokkos::printf("QFLOG: non-positive argument\n");
        return QuadFloat(0.0f);
    }
    if (a.f0 == 1.0f && a.f1 == 0.0f && a.f2 == 0.0f && a.f3 == 0.0f)
        return QuadFloat(0.0f);
    QuadFloat x = QuadFloat(Kokkos::log(a.f0));     // ~24-bit FP32 seed
    for (int k = 0; k < 3; ++k) {
        // x = x + a*exp(-x) - 1   (QD qd_real.cpp:1007-1009)
        x = subtract(add(x, multiply(a, exp(negate(x)))), QuadFloat(1.0f));
    }
    return x;
}

// log10(a) = log(a) / log(10).  QD qd_real.cpp:1025 (log10 = log(a)/_log10).
KOKKOS_INLINE_FUNCTION QuadFloat log10(QuadFloat a) {
    return divide(log(a), QuadFloat_log10());
}

// log2(a) = log(a) / log(2).  QD has no log2; composition (cf. dd_math.hpp:396).
KOKKOS_INLINE_FUNCTION QuadFloat log2(QuadFloat a) {
    return divide(log(a), QuadFloat_log2());
}

// log1p(a) = log(1 + a).  QD has no log1p; composition (cf. dd_math.hpp:404).
KOKKOS_INLINE_FUNCTION QuadFloat log1p(QuadFloat a) {
    return log(add(QuadFloat(1.0f), a));
}

// exp2(a) = e^(a*ln2).  QD has no exp2; composition (cf. dd_math.hpp:409).
KOKKOS_INLINE_FUNCTION QuadFloat exp2(QuadFloat a) {
    return exp(multiply(a, QuadFloat_log2()));
}

// exp10(a) = e^(a*ln10).  QD has no exp10; composition (cf. dd_math.hpp:413).
KOKKOS_INLINE_FUNCTION QuadFloat exp10(QuadFloat a) {
    return exp(multiply(a, QuadFloat_log10()));
}

// expm1(a) = e^a - 1.  QD has no expm1; Taylor for |a| <= 0.5 to avoid the
// e^a - 1 cancellation near 0, else exp(a) - 1 (cf. dd_math.hpp:417 / ff:424).
KOKKOS_INLINE_FUNCTION QuadFloat expm1(QuadFloat a) {
    const float eps = 1.0e-28f;
    if (Kokkos::fabs(a.f0) > 0.5f) {
        return subtract(exp(a), QuadFloat(1.0f));
    }
    QuadFloat sum = a, term = a;
    for (int k = 2; k <= 60; ++k) {
        term = divide_scalar(multiply(term, a), (float)k);
        sum  = add(sum, term);
        if (Kokkos::fabs(term.f0) < eps * Kokkos::fabs(sum.f0)) break;
    }
    return sum;
}

// ============================================================
// Trig — joint sin/cos through the doublings (PORT_NOTES §3a)
// ============================================================

// sincos(a): writes sin_a = sin(a), cos_a = cos(a).  Mathematical mirror of
// sincos(qd_real), QD 2.3.24 qd_real.cpp:2298-2360 (same reduce-mod-2pi
// skeleton), BUT structured like ff_math.hpp:445 / dd_math.hpp:439 — a
// divide-by-k Taylor on r = s3/2^nq followed by nq angle-doublings — instead of
// QD's pi/1024 table lookup (see block header).  PORT_NOTES §3a: sin and cos are
// tracked JOINTLY through the doublings (sin(2x)=2 sin x cos x,
// cos(2x)=cos^2 x - sin^2 x) so no sqrt(1-cos^2) recovery loses relative
// precision near multiples of pi.  QF's 4-word _2pi (accurate to ~2^-96) makes
// the mod-2pi reduction good enough that near-pi sin/cos are distinguishable
// from noise (the T3.6 goal FF §5 could not reach with 2-word pi).
KOKKOS_INLINE_FUNCTION void sincos(QuadFloat a, QuadFloat& sin_a, QuadFloat& cos_a) {
    const int   itrmx = 100, nq = 5;
    const float eps = 1.0e-28f;
    if (a.f0 == 0.0f) { sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return; }
    if (Kokkos::fabs(a.f0) >= 1.0e30f) {
        Kokkos::printf("QFCSSNR: argument too large\n");
        sin_a = QuadFloat(0.0f); cos_a = QuadFloat(0.0f); return;
    }
    // Reduce mod 2pi (QD qd_real.cpp:2306-2308: z = nint(a/2pi); t = a - 2pi*z).
    QuadFloat pi2 = mul_pwr2(QuadFloat_pi(), 2.0f);   // 2pi, exact from 4-word pi
    QuadFloat s1  = divide(a, pi2);
    QuadFloat s2  = round_to_nearest_int(s1);
    QuadFloat s3  = subtract(a, multiply(pi2, s2));   // |s3| <= pi
    if (s3.f0 == 0.0f) { sin_a = QuadFloat(0.0f); cos_a = QuadFloat(1.0f); return; }

    QuadFloat r  = mul_pwr2(s3, ldexpf(1.0f, -nq));   // r = s3 / 2^nq, |r| < pi/2^nq
    QuadFloat r2 = multiply(r, r);

    // sin(r) = r - r^3/3! + ... ; cos(r) = 1 - r^2/2! + ...
    QuadFloat sin_r = r,             cos_r = QuadFloat(1.0f);
    QuadFloat sterm = r,             cterm = QuadFloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = divide_scalar(multiply(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = add(sin_r, sterm);
        cterm = divide_scalar(multiply(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = add(cos_r, cterm);
        if (Kokkos::fabs(sterm.f0) < eps * Kokkos::fabs(sin_r.f0) &&
            Kokkos::fabs(cterm.f0) < eps) break;
        // No return on itrmx (converges in ~9 terms at nq=5); fall through.
    }

    // Doublings: sin(2x)=2 sin x cos x, cos(2x)=cos^2 x - sin^2 x (PORT_NOTES §3a).
    for (int j = 0; j < nq; ++j) {
        QuadFloat new_sin = mul_pwr2(multiply(sin_r, cos_r), 2.0f);
        QuadFloat new_cos = subtract(multiply(cos_r, cos_r), multiply(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }
    sin_a = sin_r; cos_a = cos_r;
}

// tan(a) = sin(a)/cos(a).  QD qd_real.cpp:2473 (sincos then s/c).
KOKKOS_INLINE_FUNCTION QuadFloat sin(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return s;
}
KOKKOS_INLINE_FUNCTION QuadFloat cos(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return c;
}
KOKKOS_INLINE_FUNCTION QuadFloat tan(QuadFloat a) {
    QuadFloat s, c; sincos(a, s, c); return divide(s, c);
}

// angle(x, y) = atan2(y, x).  Mathematical mirror of atan2(qd_real,qd_real),
// QD 2.3.24 qd_real.cpp:2393-2460: normalize (x,y) onto the unit circle, seed
// with the FP32 std::atan2, then Newton-refine z += (y - sin z)/cos z (or the
// cos variant when |x|>|y|), 3 iterations.  dd_math.hpp:497 uses the same
// structure; the joint sincos above supplies (sin z, cos z) per iteration.
KOKKOS_INLINE_FUNCTION QuadFloat angle(QuadFloat x, QuadFloat y) {
    QuadFloat pi = QuadFloat_pi();
    if (x.f0 == 0.0f && y.f0 == 0.0f) return QuadFloat(0.0f);
    if (x.f0 == 0.0f) return (y.f0 > 0.0f) ? mul_pwr2(pi, 0.5f) : mul_pwr2(pi, -0.5f);
    if (y.f0 == 0.0f) return (x.f0 > 0.0f) ? QuadFloat(0.0f) : pi;
    QuadFloat r  = sqrt(add(multiply(x, x), multiply(y, y)));
    QuadFloat nx = divide(x, r), ny = divide(y, r);
    QuadFloat a  = QuadFloat(Kokkos::atan2(ny.f0, nx.f0));   // FP32 seed
    bool use_x = (Kokkos::fabs(nx.f0) <= Kokkos::fabs(ny.f0));
    QuadFloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        QuadFloat sin_a, cos_a;
        sincos(a, sin_a, cos_a);
        if (use_x) {
            // Newton on cos: z' = z - (x - cos z)/(-sin z) -> a -= (target-cos)/sin
            a = subtract(a, divide(subtract(target, cos_a), sin_a));
        } else {
            a = add(a, divide(subtract(target, sin_a), cos_a));
        }
    }
    return a;
}

// asin(a) = atan2(a, sqrt(1-a^2)).  QD qd_real.cpp:2479.
KOKKOS_INLINE_FUNCTION QuadFloat asin(QuadFloat a) {
    if (Kokkos::fabs(a.f0) > 1.0f) {
        Kokkos::printf("QFASIN: argument out of range\n");
        return QuadFloat(0.0f);
    }
    QuadFloat t = sqrt(subtract(QuadFloat(1.0f), multiply(a, a)));
    return angle(t, a);
}
// acos(a) = atan2(sqrt(1-a^2), a).  QD qd_real.cpp:2494.
KOKKOS_INLINE_FUNCTION QuadFloat acos(QuadFloat a) {
    if (Kokkos::fabs(a.f0) > 1.0f) {
        Kokkos::printf("QFACOS: argument out of range\n");
        return QuadFloat(0.0f);
    }
    QuadFloat t = sqrt(subtract(QuadFloat(1.0f), multiply(a, a)));
    return angle(a, t);
}
// atan(a) = atan2(a, 1).  QD qd_real.cpp:2389.
KOKKOS_INLINE_FUNCTION QuadFloat atan(QuadFloat a) {
    return angle(QuadFloat(1.0f), a);
}
// atan2(y, x) = angle(x, y).  QD qd_real.cpp:2393 (STL argument order).
KOKKOS_INLINE_FUNCTION QuadFloat atan2(QuadFloat y, QuadFloat x) {
    return angle(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

// sinhcosh(a): writes sinh_a, cosh_a.  Mathematical mirror of sinh/cosh(qd_real),
// QD 2.3.24 qd_real.cpp:2509-2545.  For small |a|, (e^a - e^-a)/2 cancels
// (both exponentials -> 1), so use a direct Taylor for sinh (PORT_NOTES §3b);
// cosh is well-conditioned and taken from the exponentials.  QD's Taylor
// threshold is 0.05; this port keeps ff_math.hpp:553's 0.5 instead — the
// exp-method relative error is ~u/|a|, i.e. digits_lost ~= log10(1/|a|), which
// is ~0.3 digits at |a|=0.5 and ~1.3 digits at QD's 0.05.  At QF's 29-digit
// budget the larger 0.5 threshold (wider Taylor coverage) is the safer choice;
// see docs/PORT_NOTES_QF.md §"sinh/cosh threshold".
KOKKOS_INLINE_FUNCTION void sinhcosh(QuadFloat a, QuadFloat& sinh_a, QuadFloat& cosh_a) {
    const float eps = 1.0e-28f;
    if (Kokkos::fabs(a.f0) < 0.5f) {
        QuadFloat a2 = multiply(a, a);
        QuadFloat sinh_sum = a,             sinh_term = a;
        QuadFloat cosh_sum = QuadFloat(1.0f), cosh_term = QuadFloat(1.0f);
        for (int k = 1; k <= 60; ++k) {
            sinh_term = divide_scalar(multiply(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = add(sinh_sum, sinh_term);
            cosh_term = divide_scalar(multiply(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = add(cosh_sum, cosh_term);
            if (Kokkos::fabs(sinh_term.f0) < eps * Kokkos::fabs(sinh_sum.f0) &&
                Kokkos::fabs(cosh_term.f0) < eps) break;
        }
        sinh_a = sinh_sum; cosh_a = cosh_sum;
        return;
    }
    QuadFloat s0 = exp(a);
    QuadFloat s1 = divide(QuadFloat(1.0f), s0);
    cosh_a = mul_pwr2(add(s0, s1), 0.5f);
    sinh_a = mul_pwr2(subtract(s0, s1), 0.5f);
}

KOKKOS_INLINE_FUNCTION QuadFloat sinh(QuadFloat a) {
    QuadFloat s, c; sinhcosh(a, s, c); return s;
}
KOKKOS_INLINE_FUNCTION QuadFloat cosh(QuadFloat a) {
    QuadFloat s, c; sinhcosh(a, s, c); return c;
}
// tanh(a) via expm1(2a)/(expm1(2a)+2) with odd reflection — avoids dividing two
// nearly-equal large exponentials.  ff_math.hpp:580 (QD qd_real.cpp:2547 divides
// the exponentials directly; the expm1 form is better-conditioned near 0).
KOKKOS_INLINE_FUNCTION QuadFloat tanh(QuadFloat a) {
    if (a.f0 < 0.0f) return negate(tanh(negate(a)));
    QuadFloat e = expm1(mul_pwr2(a, 2.0f));
    return divide(e, add(e, QuadFloat(2.0f)));
}

// asinh(a) = log(a + sqrt(a^2 + 1)).  QD qd_real.cpp:2576.  Odd reflection
// (ff_math.hpp:586) keeps the log argument >= 1 for negative a.
KOKKOS_INLINE_FUNCTION QuadFloat asinh(QuadFloat a) {
    if (a.f0 < 0.0f) return negate(asinh(negate(a)));
    return log(add(a, sqrt(add(multiply(a, a), QuadFloat(1.0f)))));
}
// acosh(a) = log(a + sqrt(a^2 - 1)).  QD qd_real.cpp:2580.
KOKKOS_INLINE_FUNCTION QuadFloat acosh(QuadFloat a) {
    if (a.f0 < 1.0f) { Kokkos::printf("QFACOSH: argument < 1\n"); return QuadFloat(0.0f); }
    return log(add(a, sqrt(subtract(multiply(a, a), QuadFloat(1.0f)))));
}
// atanh(a).  QD qd_real.cpp:2589 is 0.5*log((1+a)/(1-a)) only; this port adds a
// Taylor branch for |a| < 0.5 (PORT_NOTES §3c, ff_math.hpp:595) — all-positive
// terms, no cancellation, and avoids log() evaluated near 1.
KOKKOS_INLINE_FUNCTION QuadFloat atanh(QuadFloat a) {
    if (Kokkos::fabs(a.f0) >= 1.0f) { Kokkos::printf("QFATANH: |argument| >= 1\n"); return QuadFloat(0.0f); }
    const float eps = 1.0e-28f;
    if (Kokkos::fabs(a.f0) < 0.5f) {
        QuadFloat a2 = multiply(a, a);
        QuadFloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr = multiply(pwr, a2);
            QuadFloat term = divide_scalar(pwr, (float)(2*k + 1));
            sum = add(sum, term);
            if (Kokkos::fabs(term.f0) < eps * Kokkos::fabs(sum.f0)) break;
        }
        return sum;
    }
    QuadFloat t1 = add(QuadFloat(1.0f), a);
    QuadFloat t2 = subtract(QuadFloat(1.0f), a);
    return mul_pwr2(log(divide(t1, t2)), 0.5f);
}

// ============================================================
// Power / multi-argument
// ============================================================

// pow(a, b) = e^(b log a).  QD qd_real.cpp:655 (pow(qd,qd) = exp(b*log(a))).
KOKKOS_INLINE_FUNCTION QuadFloat pow(QuadFloat a, QuadFloat b) {
    if (a.f0 <= 0.0f) {
        if (a.f0 == 0.0f && b.f0 > 0.0f) return QuadFloat(0.0f);
        Kokkos::printf("QFPOW: non-positive base\n");
        return QuadFloat(0.0f);
    }
    return exp(multiply(log(a), b));
}

// hypot(a, b) = sqrt(a^2 + b^2).  QD has no hypot; composition (cf. dd:604).
KOKKOS_INLINE_FUNCTION QuadFloat hypot(QuadFloat a, QuadFloat b) {
    return sqrt(add(multiply(a, a), multiply(b, b)));
}

// fmod(a, b) = a - b*trunc(a/b).  QD qd_real.cpp:2598 (fmod = a - b*aint(a/b);
// aint = trunc, qd_inline.h:975).
KOKKOS_INLINE_FUNCTION QuadFloat fmod(QuadFloat a, QuadFloat b) {
    QuadFloat q  = divide(a, b);
    QuadFloat qt = trunc(q);
    return subtract(a, multiply(b, qt));
}

// remainder(a, b) = a - b*nint(a/b).  QD drem, qd_real.cpp:2462
// (n = nint(a/b); a - n*b).
KOKKOS_INLINE_FUNCTION QuadFloat remainder(QuadFloat a, QuadFloat b) {
    QuadFloat q  = divide(a, b);
    QuadFloat qn = round_to_nearest_int(q);
    return subtract(a, multiply(b, qn));
}

// copysign / fmax / fmin / fdim / fma — no QD analogue; componentwise (cf. dd).
KOKKOS_INLINE_FUNCTION QuadFloat copysign(QuadFloat a, QuadFloat b) {
    QuadFloat r = abs(a);
    if (b.f0 < 0.0f || (b.f0 == 0.0f && b.f1 < 0.0f)) return negate(r);
    return r;
}
KOKKOS_INLINE_FUNCTION QuadFloat fmax(QuadFloat a, QuadFloat b) { return (a > b) ? a : b; }
KOKKOS_INLINE_FUNCTION QuadFloat fmin(QuadFloat a, QuadFloat b) { return (a < b) ? a : b; }
KOKKOS_INLINE_FUNCTION QuadFloat fdim(QuadFloat a, QuadFloat b) {
    return (a > b) ? subtract(a, b) : QuadFloat(0.0f);
}
KOKKOS_INLINE_FUNCTION QuadFloat fma(QuadFloat a, QuadFloat b, QuadFloat c) {
    return add(multiply(a, b), c);
}

// ============================================================
// Rounding — component-wise floor/ceil with renorm (QD's, not FF's nint form)
// ============================================================

// floor(a).  Faithful port of floor(qd_real), QD 2.3.24 qd_real.cpp:136-157:
// floor each component while the previous is integral; renorm.
KOKKOS_INLINE_FUNCTION QuadFloat floor(QuadFloat a) {
    float x0, x1, x2, x3;
    x1 = x2 = x3 = 0.0f;
    x0 = Kokkos::floor(a.f0);
    if (x0 == a.f0) {
        x1 = Kokkos::floor(a.f1);
        if (x1 == a.f1) {
            x2 = Kokkos::floor(a.f2);
            if (x2 == a.f2) x3 = Kokkos::floor(a.f3);
        }
        renorm(x0, x1, x2, x3);
        return QuadFloat(x0, x1, x2, x3);
    }
    return QuadFloat(x0, x1, x2, x3);
}
// ceil(a).  Faithful port of ceil(qd_real), QD 2.3.24 qd_real.cpp:159-180.
KOKKOS_INLINE_FUNCTION QuadFloat ceil(QuadFloat a) {
    float x0, x1, x2, x3;
    x1 = x2 = x3 = 0.0f;
    x0 = Kokkos::ceil(a.f0);
    if (x0 == a.f0) {
        x1 = Kokkos::ceil(a.f1);
        if (x1 == a.f1) {
            x2 = Kokkos::ceil(a.f2);
            if (x2 == a.f2) x3 = Kokkos::ceil(a.f3);
        }
        renorm(x0, x1, x2, x3);
        return QuadFloat(x0, x1, x2, x3);
    }
    return QuadFloat(x0, x1, x2, x3);
}
// trunc(a) = (a >= 0) ? floor(a) : ceil(a).  QD aint, qd_inline.h:975.
KOKKOS_INLINE_FUNCTION QuadFloat trunc(QuadFloat a) {
    return (a.f0 >= 0.0f) ? floor(a) : ceil(a);
}
// round(a) = round_to_nearest_int(a) (QD nint, qd_real.cpp:96, T3.0a).
KOKKOS_INLINE_FUNCTION QuadFloat round(QuadFloat a) {
    return round_to_nearest_int(a);
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0 convention)
// ============================================================
// Mirrors impl/Kokkos_QuadPrecisionMath.hpp / dd_math.hpp / ff_math.hpp so
// user code can call Kokkos::sqrt(qf) identically to Kokkos::sqrt(double).
// T3.0a exposes the arithmetic-tier single-output functions only; T3.0b adds
// the transcendentals. add/subtract/multiply/divide are deliberately NOT
// forwarded (operators + explicit ADL only, same posture as DD/FF).
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat abs(Experimental::QuadFloat x)   { return Experimental::abs(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sqrt(Experimental::QuadFloat x)  { return Experimental::sqrt(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp(Experimental::QuadFloat x)   { return Experimental::exp(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp2(Experimental::QuadFloat x)  { return Experimental::exp2(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat exp10(Experimental::QuadFloat x) { return Experimental::exp10(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat expm1(Experimental::QuadFloat x) { return Experimental::expm1(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log(Experimental::QuadFloat x)   { return Experimental::log(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log2(Experimental::QuadFloat x)  { return Experimental::log2(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log10(Experimental::QuadFloat x) { return Experimental::log10(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat log1p(Experimental::QuadFloat x) { return Experimental::log1p(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sin(Experimental::QuadFloat x)   { return Experimental::sin(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat cos(Experimental::QuadFloat x)   { return Experimental::cos(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat tan(Experimental::QuadFloat x)   { return Experimental::tan(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat asin(Experimental::QuadFloat x)  { return Experimental::asin(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat acos(Experimental::QuadFloat x)  { return Experimental::acos(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atan(Experimental::QuadFloat x)  { return Experimental::atan(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atan2(Experimental::QuadFloat y, Experimental::QuadFloat x) { return Experimental::atan2(y, x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat sinh(Experimental::QuadFloat x)  { return Experimental::sinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat cosh(Experimental::QuadFloat x)  { return Experimental::cosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat tanh(Experimental::QuadFloat x)  { return Experimental::tanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat asinh(Experimental::QuadFloat x) { return Experimental::asinh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat acosh(Experimental::QuadFloat x) { return Experimental::acosh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat atanh(Experimental::QuadFloat x) { return Experimental::atanh(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat pow(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::pow(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat hypot(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::hypot(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmod(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmod(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat remainder(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::remainder(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat copysign(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::copysign(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmax(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmax(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fmin(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fmin(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fdim(Experimental::QuadFloat a, Experimental::QuadFloat b) { return Experimental::fdim(a, b); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat fma(Experimental::QuadFloat a, Experimental::QuadFloat b, Experimental::QuadFloat c) { return Experimental::fma(a, b, c); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat ceil(Experimental::QuadFloat x)  { return Experimental::ceil(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat floor(Experimental::QuadFloat x) { return Experimental::floor(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat round(Experimental::QuadFloat x) { return Experimental::round(x); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat trunc(Experimental::QuadFloat x) { return Experimental::trunc(x); }
// clang-format on
}  // namespace Kokkos
