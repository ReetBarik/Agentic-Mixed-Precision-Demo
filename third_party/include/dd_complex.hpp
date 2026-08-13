// SPDX-License-Identifier: LicenseRef-DHB-License
// SPDX-FileCopyrightText: Copyright (c) 2024 David H. Bailey
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// Ported from DDFUN v04:
//   https://www.davidhbailey.com/dhbsoftware/ddfun-v04.tar.gz
//   Original author: David H. Bailey (LBNL retired / UC Davis)
//   Original license: DHB-License (modified BSD-3-Clause with §3
//     grant-back clause). Full text: LICENSES/LicenseRef-DHB-License.txt
//     or https://www.davidhbailey.com/dhbsoftware/DHB-License.txt.
//
// This C++/Kokkos port is a derivative work distributed under the
// same DHB-License. See §3 of that license regarding upstream
// contribution rights. This is the complex layer; it builds on the
// double-double real arithmetic in dd_math.hpp (also DHB-License).
//
// Modifications from the original DDFUN v04 sources:
//   * Translated the complex double-double routines from Fortran-90
//     (ddfunc.f90 / the ddc* entry points) to header-only C++17.
//   * Adapted to Kokkos: every function KOKKOS_INLINE_FUNCTION for
//     host + device portability across CUDA/HIP/SYCL/OpenMP-target.
//   * Namespaced as Kokkos::Experimental::DoubleDoubleComplex (a
//     bespoke struct, not yet Kokkos::complex<DoubleDouble>) with
//     STL-style free functions and ADL-friendly re-exposure under
//     namespace Kokkos, for potential upstreaming to Kokkos.
//   * See docs/TEST_SUITE_PLAN.md "Upstreaming considerations" for
//     naming and API conventions.

#pragma once

// Double-double complex arithmetic — Kokkos::Experimental::DoubleDoubleComplex.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on dd_math.hpp.
//
// Ported from DDFUN (David H. Bailey, Lawrence Berkeley National Lab).
//
// Naming follows dd_math.hpp (T0.4): type + math live under
// Kokkos::Experimental for eventual upstreaming. This remains a bespoke struct
// rather than Kokkos::complex<DoubleDouble> — that integration is a separate
// future task.

#include <dd_math.hpp>

#ifndef __CUDA_ARCH__
#  include <ostream>
#endif

namespace Kokkos {
namespace Experimental {

// ============================================================
// DoubleDoubleComplex struct
// ============================================================
struct DoubleDoubleComplex {
    DoubleDouble re;
    DoubleDouble im;

    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex() : re(0.0), im(0.0) {}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex(double r)                     : re(r),    im(0.0) {}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex(DoubleDouble r)               : re(r),    im(0.0) {}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex(double r, double i)           : re(r),    im(i)   {}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex(DoubleDouble r, DoubleDouble i) : re(r),  im(i)   {}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex(const DoubleDoubleComplex& o) : re(o.re), im(o.im){}
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator=(const DoubleDoubleComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator=(DoubleDouble r) {
        re = r; im = DoubleDouble(0.0); return *this;
    }

    // Arithmetic
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex b) const {
        return DoubleDoubleComplex(add(re, b.re), add(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex b) const {
        return DoubleDoubleComplex(subtract(re, b.re), subtract(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return DoubleDoubleComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                         add(multiply(re, b.im), multiply(im, b.re)));
    }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex b) const {
        if (b.re.hi == 0.0 && b.im.hi == 0.0) {
            Kokkos::printf("DDCOMPLEX: division by zero\n");
            return DoubleDoubleComplex();
        }
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)
        DoubleDouble denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        DoubleDouble inv   = divide(DoubleDouble(1.0), denom);
        return DoubleDoubleComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                         multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-() const {
        return DoubleDoubleComplex(negate(re), negate(im));
    }

    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator+=(DoubleDoubleComplex b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator-=(DoubleDoubleComplex b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator*=(DoubleDoubleComplex b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex& operator/=(DoubleDoubleComplex b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(DoubleDoubleComplex b) const { return re==b.re && im==b.im; }
    KOKKOS_INLINE_FUNCTION bool operator!=(DoubleDoubleComplex b) const { return !(*this == b); }

    KOKKOS_INLINE_FUNCTION DoubleDouble real() const { return re; }
    KOKKOS_INLINE_FUNCTION DoubleDouble imag() const { return im; }
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const DoubleDoubleComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed DoubleDouble × DoubleDoubleComplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(add(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(add(r, z.re), z.im); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(subtract(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(subtract(r, z.re), negate(z.im)); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(multiply(z.re, r), multiply(z.im, r)); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(multiply(r, z.re), multiply(r, z.im)); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex z, DoubleDouble r) { return DoubleDoubleComplex(divide(z.re, r), divide(z.im, r)); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDouble r, DoubleDoubleComplex z) { return DoubleDoubleComplex(r) / z; }

// ============================================================
// Mixed double × DoubleDoubleComplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator+(DoubleDoubleComplex z, double b) { return z + DoubleDouble(b); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator+(double b, DoubleDoubleComplex z) { return DoubleDouble(b) + z; }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-(DoubleDoubleComplex z, double b) { return z - DoubleDouble(b); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator-(double b, DoubleDoubleComplex z) { return DoubleDouble(b) - z; }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator*(DoubleDoubleComplex z, double b) { return z * DoubleDouble(b); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator*(double b, DoubleDoubleComplex z) { return DoubleDouble(b) * z; }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator/(DoubleDoubleComplex z, double b) { return z / DoubleDouble(b); }
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex operator/(double b, DoubleDoubleComplex z) { return DoubleDouble(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

KOKKOS_INLINE_FUNCTION DoubleDouble abs(DoubleDoubleComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex conj(DoubleDoubleComplex z) {
    return DoubleDoubleComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex sqrt(DoubleDoubleComplex z) {
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return DoubleDoubleComplex();
    // B = sqrt((R+A1)/2) + i*sign(A2)*sqrt((R-A1)/2)  where R = |z|
    DoubleDouble r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
    DoubleDouble a1 = abs(z.re);
    DoubleDouble s2 = multiply_scalar(add(r, a1), 0.5);
    DoubleDouble s0 = sqrt(s2);
    DoubleDouble s1 = multiply_scalar(s0, 2.0);
    DoubleDoubleComplex b;
    if (z.re.hi >= 0.0) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.hi < 0.0) b.re = negate(b.re);
        b.im = s0;
        if (z.im.hi < 0.0) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex exp(DoubleDoubleComplex z) {
    DoubleDouble er = exp(z.re);
    DoubleDouble c, s;
    sincos(z.im, c, s);
    return DoubleDoubleComplex(multiply(er, c), multiply(er, s));
}

KOKKOS_INLINE_FUNCTION DoubleDoubleComplex log(DoubleDoubleComplex z) {
    DoubleDouble modulus = abs(z);
    DoubleDouble arg     = atan2(z.im, z.re); // atan2(im, re)
    return DoubleDoubleComplex(log(modulus), arg);
}

KOKKOS_INLINE_FUNCTION DoubleDoubleComplex log10(DoubleDoubleComplex z) {
    DoubleDoubleComplex lg = log(z);
    DoubleDouble ln10 = DoubleDouble_log10();
    return DoubleDoubleComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex sin(DoubleDoubleComplex z) {
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    DoubleDouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(sa, cb), multiply(ca, sb));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex cos(DoubleDoubleComplex z) {
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    DoubleDouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex tan(DoubleDoubleComplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex asin(DoubleDoubleComplex z) {
    // asin(z) = -i * log(iz + sqrt(1 - z^2))
    DoubleDoubleComplex iz  = DoubleDoubleComplex(negate(z.im), z.re);
    DoubleDoubleComplex z2  = z * z;
    DoubleDoubleComplex one_minus_z2 = DoubleDoubleComplex(DoubleDouble(1.0)) - z2;
    DoubleDoubleComplex sum = iz + sqrt(one_minus_z2);
    DoubleDoubleComplex lg  = log(sum);
    // multiply by -i: (a+bi)*(-i) = b - ai
    return DoubleDoubleComplex(lg.im, negate(lg.re));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex acos(DoubleDoubleComplex z) {
    // acos(z) = pi/2 - asin(z)
    DoubleDouble pi_over_2 = multiply_scalar(DoubleDouble_pi(), 0.5);
    DoubleDoubleComplex asin_z  = asin(z);
    return DoubleDoubleComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex atan(DoubleDoubleComplex z) {
    // atan(z) = (i/2) * log((1-iz)/(1+iz))
    DoubleDoubleComplex iz    = DoubleDoubleComplex(negate(z.im), z.re);
    DoubleDoubleComplex num   = DoubleDoubleComplex(DoubleDouble(1.0)) - iz;
    DoubleDoubleComplex den   = DoubleDoubleComplex(DoubleDouble(1.0)) + iz;
    DoubleDoubleComplex ratio = num / den;
    DoubleDoubleComplex lg    = log(ratio);
    // multiply by i/2: (a+bi)*(i/2) = (-b/2) + (a/2)*i
    return DoubleDoubleComplex(multiply_scalar(negate(lg.im), 0.5), multiply_scalar(lg.re, 0.5));
}

// ============================================================
// Complex hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex sinh(DoubleDoubleComplex z) {
    // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
    DoubleDouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(sa, cb), multiply(ca, sb));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex cosh(DoubleDoubleComplex z) {
    // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
    DoubleDouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return DoubleDoubleComplex(multiply(ca, cb), multiply(sa, sb));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex tanh(DoubleDoubleComplex z) {
    // tanh(a+bi): re = tanh(a) / (cos²(b) + tanh²(a)·sin²(b))
    //             im = sin(b)·cos(b)·(1 - tanh²(a)) / (cos²(b) + tanh²(a)·sin²(b))
    // Denominator ≥ 0 always; uses improved real tanh to avoid cancellation.
    DoubleDouble T = tanh(z.re);
    DoubleDouble cb, sb;
    sincos(z.im, cb, sb);
    DoubleDouble T2    = multiply(T, T);
    DoubleDouble denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return DoubleDoubleComplex(divide(T, denom),
                     divide(multiply(multiply(sb, cb), subtract(DoubleDouble(1.0), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex asinh(DoubleDoubleComplex z) {
    // asinh(z) = log(z + sqrt(z^2 + 1))
    return log(z + sqrt(z*z + DoubleDoubleComplex(DoubleDouble(1.0))));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex acosh(DoubleDoubleComplex z) {
    // acosh(z) = log(z + sqrt(z^2 - 1))
    return log(z + sqrt(z*z - DoubleDoubleComplex(DoubleDouble(1.0))));
}
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex atanh(DoubleDoubleComplex z) {
    // atanh(z) = (1/2)*log((1+z)/(1-z))
    DoubleDoubleComplex one = DoubleDoubleComplex(DoubleDouble(1.0));
    DoubleDoubleComplex lg  = log((one + z) / (one - z));
    return DoubleDoubleComplex(multiply_scalar(lg.re, 0.5), multiply_scalar(lg.im, 0.5));
}

// ============================================================
// Complex power and polar
// ============================================================
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex pow(DoubleDoubleComplex z, DoubleDoubleComplex w) {
    // z^w = exp(w * log(z))
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return DoubleDoubleComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r * (cos(theta) + i*sin(theta))
KOKKOS_INLINE_FUNCTION DoubleDoubleComplex polar(DoubleDouble r, DoubleDouble theta) {
    DoubleDouble c, s;
    sincos(theta, c, s);
    return DoubleDoubleComplex(multiply(r, c), multiply(r, s));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4)
// ============================================================
// Mirror of dd_math.hpp: so Kokkos::exp(ddc) works identically to
// Kokkos::exp(Kokkos::complex<double>). One-line forwards. Arithmetic operators
// are reached directly / via ADL and are not re-exposed here.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::DoubleDouble        abs(Experimental::DoubleDoubleComplex z)   { return Experimental::abs(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex conj(Experimental::DoubleDoubleComplex z)  { return Experimental::conj(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex sqrt(Experimental::DoubleDoubleComplex z)  { return Experimental::sqrt(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex exp(Experimental::DoubleDoubleComplex z)   { return Experimental::exp(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex log(Experimental::DoubleDoubleComplex z)   { return Experimental::log(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex log10(Experimental::DoubleDoubleComplex z) { return Experimental::log10(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex sin(Experimental::DoubleDoubleComplex z)   { return Experimental::sin(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex cos(Experimental::DoubleDoubleComplex z)   { return Experimental::cos(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex tan(Experimental::DoubleDoubleComplex z)   { return Experimental::tan(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex asin(Experimental::DoubleDoubleComplex z)  { return Experimental::asin(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex acos(Experimental::DoubleDoubleComplex z)  { return Experimental::acos(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex atan(Experimental::DoubleDoubleComplex z)  { return Experimental::atan(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex sinh(Experimental::DoubleDoubleComplex z)  { return Experimental::sinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex cosh(Experimental::DoubleDoubleComplex z)  { return Experimental::cosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex tanh(Experimental::DoubleDoubleComplex z)  { return Experimental::tanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex asinh(Experimental::DoubleDoubleComplex z) { return Experimental::asinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex acosh(Experimental::DoubleDoubleComplex z) { return Experimental::acosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex atanh(Experimental::DoubleDoubleComplex z) { return Experimental::atanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::DoubleDoubleComplex pow(Experimental::DoubleDoubleComplex z, Experimental::DoubleDoubleComplex w) { return Experimental::pow(z, w); }
// clang-format on
}  // namespace Kokkos
