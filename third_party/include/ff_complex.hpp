// SPDX-License-Identifier: LicenseRef-DHB-License
//
// Copyright (c) 2024 David H. Bailey — DDFUN v04 (original algorithms)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is a mechanical translation of dd_complex.hpp (this repo's
// Kokkos C++ port of DDFUN v04) from double-double (2×FP64) to
// float-float (2×FP32). Function inventory, algorithm choices, and
// coefficient tables descend from DDFUN v04 by David H. Bailey.
//
// FP32-specific modifications (input narrowing, splitter constant
// 8193.0f = 2^13+1, joint sin/cos doublings, Taylor branches for
// |a|<0.5 in sinh/cosh/atanh, direct exp scaling to avoid splitter
// overflow, nint magic-constant replacement) are documented in
// PORT_NOTES.md. These modifications fall under DHB-License §3
// (grant-back) and are governed by the same terms as the original.
//
// See LICENSES/LicenseRef-DHB-License.txt for the full license text
// and NOTICE.md for the per-file license mapping.

#pragma once

// Float-float complex arithmetic — Kokkos::Experimental::FloatFloatComplex.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on ff_math.hpp.
//
// Naming follows ff_math.hpp (T0.4/T2.0): type + math live under
// Kokkos::Experimental for eventual upstreaming. This remains a bespoke struct
// rather than Kokkos::complex<FloatFloat> — that integration is a separate
// future task.

#include <ff_math.hpp>

#ifndef __CUDA_ARCH__
#  include <ostream>
#endif

namespace Kokkos {
namespace Experimental {

// ============================================================
// FloatFloatComplex struct
// ============================================================
struct FloatFloatComplex {
    FloatFloat re;
    FloatFloat im;

    KOKKOS_INLINE_FUNCTION FloatFloatComplex() : re(0.0f), im(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex(float r)               : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex(FloatFloat r)              : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex(float r, float i)      : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex(FloatFloat r, FloatFloat i)    : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex(const FloatFloatComplex& o)    : re(o.re), im(o.im) {}
    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator=(const FloatFloatComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator=(FloatFloat r) {
        re = r; im = FloatFloat(0.0f); return *this;
    }

    KOKKOS_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex b) const {
        return FloatFloatComplex(add(re, b.re), add(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex b) const {
        return FloatFloatComplex(subtract(re, b.re), subtract(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex b) const {
        return FloatFloatComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                         add(multiply(re, b.im), multiply(im, b.re)));
    }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex b) const {
        if (b.re.hi == 0.0f && b.im.hi == 0.0f) {
            Kokkos::printf("FFCOMPLEX: division by zero\n");
            return FloatFloatComplex();
        }
        FloatFloat denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        FloatFloat inv   = divide(FloatFloat(1.0f), denom);
        return FloatFloatComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                         multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-() const {
        return FloatFloatComplex(negate(re), negate(im));
    }

    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator+=(FloatFloatComplex b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator-=(FloatFloatComplex b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator*=(FloatFloatComplex b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION FloatFloatComplex& operator/=(FloatFloatComplex b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(FloatFloatComplex b) const { return re==b.re && im==b.im; }
    KOKKOS_INLINE_FUNCTION bool operator!=(FloatFloatComplex b) const { return !(*this == b); }

    KOKKOS_INLINE_FUNCTION FloatFloat real() const { return re; }
    KOKKOS_INLINE_FUNCTION FloatFloat imag() const { return im; }
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const FloatFloatComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed FloatFloat × FloatFloatComplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(add(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(add(r, z.re), z.im); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(subtract(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(subtract(r, z.re), negate(z.im)); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(multiply(z.re, r), multiply(z.im, r)); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(multiply(r, z.re), multiply(r, z.im)); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex z, FloatFloat r) { return FloatFloatComplex(divide(z.re, r), divide(z.im, r)); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloat r, FloatFloatComplex z) { return FloatFloatComplex(r) / z; }

// ============================================================
// Mixed float × FloatFloatComplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator+(FloatFloatComplex z, float b) { return z + FloatFloat(b); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator+(float b, FloatFloatComplex z) { return FloatFloat(b) + z; }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-(FloatFloatComplex z, float b) { return z - FloatFloat(b); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator-(float b, FloatFloatComplex z) { return FloatFloat(b) - z; }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator*(FloatFloatComplex z, float b) { return z * FloatFloat(b); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator*(float b, FloatFloatComplex z) { return FloatFloat(b) * z; }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator/(FloatFloatComplex z, float b) { return z / FloatFloat(b); }
KOKKOS_INLINE_FUNCTION FloatFloatComplex operator/(float b, FloatFloatComplex z) { return FloatFloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

KOKKOS_INLINE_FUNCTION FloatFloat abs(FloatFloatComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex conj(FloatFloatComplex z) {
    return FloatFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex sqrt(FloatFloatComplex z) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return FloatFloatComplex();
    FloatFloat r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
    FloatFloat a1 = abs(z.re);
    FloatFloat s2 = multiply_scalar(add(r, a1), 0.5f);
    FloatFloat s0 = sqrt(s2);
    FloatFloat s1 = multiply_scalar(s0, 2.0f);
    FloatFloatComplex b;
    if (z.re.hi >= 0.0f) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.hi < 0.0f) b.re = negate(b.re);
        b.im = s0;
        if (z.im.hi < 0.0f) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex exp(FloatFloatComplex z) {
    FloatFloat er = exp(z.re);
    FloatFloat c, s;
    sincos(z.im, c, s);
    return FloatFloatComplex(multiply(er, c), multiply(er, s));
}

KOKKOS_INLINE_FUNCTION FloatFloatComplex log(FloatFloatComplex z) {
    FloatFloat modulus = abs(z);
    FloatFloat arg     = atan2(z.im, z.re);
    return FloatFloatComplex(log(modulus), arg);
}

KOKKOS_INLINE_FUNCTION FloatFloatComplex log10(FloatFloatComplex z) {
    FloatFloatComplex lg = log(z);
    FloatFloat ln10 = FloatFloat_log10();
    return FloatFloatComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex sin(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex cos(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return FloatFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex tan(FloatFloatComplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex asin(FloatFloatComplex z) {
    FloatFloatComplex iz  = FloatFloatComplex(negate(z.im), z.re);
    FloatFloatComplex z2  = z * z;
    FloatFloatComplex one_minus_z2 = FloatFloatComplex(FloatFloat(1.0f)) - z2;
    FloatFloatComplex sum = iz + sqrt(one_minus_z2);
    FloatFloatComplex lg  = log(sum);
    return FloatFloatComplex(lg.im, negate(lg.re));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex acos(FloatFloatComplex z) {
    FloatFloat pi_over_2 = multiply_scalar(FloatFloat_pi(), 0.5f);
    FloatFloatComplex asin_z  = asin(z);
    return FloatFloatComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex atan(FloatFloatComplex z) {
    FloatFloatComplex iz    = FloatFloatComplex(negate(z.im), z.re);
    FloatFloatComplex num   = FloatFloatComplex(FloatFloat(1.0f)) - iz;
    FloatFloatComplex den   = FloatFloatComplex(FloatFloat(1.0f)) + iz;
    FloatFloatComplex ratio = num / den;
    FloatFloatComplex lg    = log(ratio);
    return FloatFloatComplex(multiply_scalar(negate(lg.im), 0.5f), multiply_scalar(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex sinh(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return FloatFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex cosh(FloatFloatComplex z) {
    FloatFloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return FloatFloatComplex(multiply(ca, cb), multiply(sa, sb));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex tanh(FloatFloatComplex z) {
    FloatFloat T = tanh(z.re);
    FloatFloat cb, sb;
    sincos(z.im, cb, sb);
    FloatFloat T2    = multiply(T, T);
    FloatFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return FloatFloatComplex(divide(T, denom),
                     divide(multiply(multiply(sb, cb), subtract(FloatFloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex asinh(FloatFloatComplex z) {
    return log(z + sqrt(z*z + FloatFloatComplex(FloatFloat(1.0f))));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex acosh(FloatFloatComplex z) {
    return log(z + sqrt(z*z - FloatFloatComplex(FloatFloat(1.0f))));
}
KOKKOS_INLINE_FUNCTION FloatFloatComplex atanh(FloatFloatComplex z) {
    FloatFloatComplex one = FloatFloatComplex(FloatFloat(1.0f));
    FloatFloatComplex lg  = log((one + z) / (one - z));
    return FloatFloatComplex(multiply_scalar(lg.re, 0.5f), multiply_scalar(lg.im, 0.5f));
}

// ============================================================
// Complex power and polar
// ============================================================
KOKKOS_INLINE_FUNCTION FloatFloatComplex pow(FloatFloatComplex z, FloatFloatComplex w) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return FloatFloatComplex();
    return exp(w * log(z));
}

KOKKOS_INLINE_FUNCTION FloatFloatComplex polar(FloatFloat r, FloatFloat theta) {
    FloatFloat c, s;
    sincos(theta, c, s);
    return FloatFloatComplex(multiply(r, c), multiply(r, s));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0)
// ============================================================
// Mirror of ff_math.hpp: so Kokkos::exp(ffc) works identically to
// Kokkos::exp(Kokkos::complex<double>). One-line forwards. Arithmetic operators
// are reached directly / via ADL and are not re-exposed here.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::FloatFloat        abs(Experimental::FloatFloatComplex z)   { return Experimental::abs(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex conj(Experimental::FloatFloatComplex z)  { return Experimental::conj(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sqrt(Experimental::FloatFloatComplex z)  { return Experimental::sqrt(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex exp(Experimental::FloatFloatComplex z)   { return Experimental::exp(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex log(Experimental::FloatFloatComplex z)   { return Experimental::log(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex log10(Experimental::FloatFloatComplex z) { return Experimental::log10(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sin(Experimental::FloatFloatComplex z)   { return Experimental::sin(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex cos(Experimental::FloatFloatComplex z)   { return Experimental::cos(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex tan(Experimental::FloatFloatComplex z)   { return Experimental::tan(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex asin(Experimental::FloatFloatComplex z)  { return Experimental::asin(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex acos(Experimental::FloatFloatComplex z)  { return Experimental::acos(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex atan(Experimental::FloatFloatComplex z)  { return Experimental::atan(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex sinh(Experimental::FloatFloatComplex z)  { return Experimental::sinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex cosh(Experimental::FloatFloatComplex z)  { return Experimental::cosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex tanh(Experimental::FloatFloatComplex z)  { return Experimental::tanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex asinh(Experimental::FloatFloatComplex z) { return Experimental::asinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex acosh(Experimental::FloatFloatComplex z) { return Experimental::acosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex atanh(Experimental::FloatFloatComplex z) { return Experimental::atanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::FloatFloatComplex pow(Experimental::FloatFloatComplex z, Experimental::FloatFloatComplex w) { return Experimental::pow(z, w); }
// clang-format on
}  // namespace Kokkos
