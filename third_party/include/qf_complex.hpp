// SPDX-License-Identifier: LicenseRef-LBNL-BSD-License
//
// Copyright (c) 2003-2023 The Regents of the University of California, through
//   Lawrence Berkeley National Laboratory — QD 2.3.24 (real four-word algorithms
//   this complex layer composes on; Yozo Hida, Xiaoye S. Li, David H. Bailey)
// Modifications Copyright (c) 2026 UChicago Argonne, LLC
//
// This file is the complex layer for the QF (quad-float, 4×FP32) backend on
// branch qffunKokkos. It is a mechanical scalar-swap port of
// third_party/include/ff_complex.hpp (this repo's float-float complex layer,
// itself a DD→FF translation of dd_complex.hpp) from FloatFloat (2×FP32) to
// QuadFloat (4×FP32). Every complex algorithm — the (ac−bd)+(ad+bc)i product,
// the Kahan-style complex sqrt, exp = eˣ(cos y + i sin y), the log/atan2 polar
// decomposition, the sin/cos/sinh/cosh angle-addition formulas — descends
// structurally from ff_complex.hpp / dd_complex.hpp, and each function cites the
// ff_complex.hpp (and, where it is the deeper reference, dd_complex.hpp) line
// range it mirrors.
//
// QD 2.3.24 SHIPS NO COMPLEX HEADER. The QD 2.3.24 tarball
// (qd/include/qd/, qd/src/) contains qd_real.{h,cpp}, dd_real.{h,cpp},
// c_dd.{h,cpp}, c_qd.{h,cpp} (a C-linkage wrapper of the *real* types) and
// fpu/bits/inline headers — but no qd_complex.* or dd_complex.* anywhere (only
// NEWS/README/TODO/Fortran mention the word "complex"). QD's quad-double complex
// is left to the user to layer on. Consequently ff_complex.hpp + dd_complex.hpp
// are the SOLE algorithm references for this header; there is no QD complex
// routine to cite. (Verified by grep -ril complex over the 2.3.24 tree during
// T3.0c; recorded in docs/PORT_NOTES_QF.md.)
//
// LICENSE LINEAGE (reported for review, per T3.0c task). This header carries
// LicenseRef-LBNL-BSD-License — the SAME license as qf_math.hpp — NOT the
// LicenseRef-DHB-License that governs ff_complex.hpp / dd_complex.hpp. Two
// lineage precedents were weighed:
//   (a) follow scalar dispatch (LBNL-BSD): every arithmetic call inside this
//       header dispatches to qf_math.hpp's LBNL-BSD QuadFloat routines, so the
//       observable numerics are QD-derived, exactly the reasoning T3.0a used to
//       put qf_math.hpp under LBNL-BSD.
//   (b) follow the structural template (DHB): qf_complex.hpp is a scalar-swap of
//       ff_complex.hpp, which is DHB-licensed for its DDFUN heritage.
// Decision: (a) LBNL-BSD. Rationale: the file contains NO DDFUN/DHB-original
// arithmetic — the complex composition formulas (product, quotient, Kahan sqrt,
// Euler exp, polar log) are textbook identities, not DDFUN inventions, and every
// non-trivial numeric step is a QD-derived QuadFloat operation. Applying the DHB
// license would attribute copyright to Bailey personally and cite the wrong
// commercial contact (dhbailey@lbl.gov) for a file whose substance is the LBNL
// institutional QD package. This keeps the whole QF backend (qf_math.hpp +
// qf_complex.hpp) under one consistent license, matching T3.0a/T3.0b. Reet to
// confirm in review; see docs/PORT_NOTES_QF.md "License lineage (complex layer)".
//
// See LICENSES/LicenseRef-LBNL-BSD-License.txt for the full text and NOTICE.md
// for the per-file mapping.

#pragma once

// Quad-float complex arithmetic — Kokkos::Experimental::QuadFloatComplex.
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on qf_math.hpp (the 4×FP32 real backend, T3.0a/T3.0b).
//
// TABLE-FREE POSTURE (inherited from T3.0b, PORT_NOTES_QF §6). This header adds
// NO lookup tables — no sin_table / cos_table / inv_fact. Complex exp/sin/cos/
// sinh/cosh dispatch through qf_math.hpp's table-free real transcendentals
// (divide-by-k Taylor, joint sin/cos doublings), so they inherit T3.0b's §7
// exp term count and §8 sinh threshold. Complex-only compositions
// (exp(z) = exp(x)·(cos y + i sin y), etc.) call the real exp/cos/sin directly.
//
// Naming follows qf_math.hpp / ff_complex.hpp (T0.4/T2.0/T3.0a): type + math
// live under Kokkos::Experimental for eventual upstreaming. This remains a
// bespoke struct rather than Kokkos::complex<QuadFloat> — that integration is a
// separate future task.
//
// SINCOS / SINHCOSH OUTPUT ORDER (QF-specific gotcha, PORT_NOTES_QF §12). Unlike
// ff_math.hpp — whose sincos(a, x, y) writes x=cos, y=sin and sinhcosh(a, x, y)
// writes x=cosh, y=sinh — qf_math.hpp names its out-params sin-first:
// sincos(a, sin_a, cos_a) and sinhcosh(a, sinh_a, cosh_a). To keep each call
// site's downstream algebra byte-identical to ff_complex.hpp, this header passes
// the local (cos, sin) / (cosh, sinh) variables in SWAPPED positional order,
// i.e. sincos(a, s, c) and sinhcosh(a, sh, ch). The local variable meanings
// (c=cos, s=sin, ...) then match ff_complex.hpp exactly.

#include <qf_math.hpp>

#ifndef __CUDA_ARCH__
#  include <ostream>
#endif

namespace Kokkos {
namespace Experimental {

// ============================================================
// QuadFloatComplex struct
// ============================================================
// Members named re/im (mirroring ff_complex.hpp / dd_complex.hpp verbatim so the
// demo's mqr(i).re / .im field access and the real()/imag() accessors coexist).
// The task's shorthand "{ QuadFloat real, imag; }" denotes the two QuadFloat
// components; re/im is the faithful structural match.
struct QuadFloatComplex {
    QuadFloat re;
    QuadFloat im;

    KOKKOS_INLINE_FUNCTION QuadFloatComplex() : re(0.0f), im(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex(float r)                  : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex(QuadFloat r)             : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex(float r, float i)         : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex(QuadFloat r, QuadFloat i) : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex(const QuadFloatComplex& o): re(o.re), im(o.im) {}
    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator=(const QuadFloatComplex& o) {
        re = o.re; im = o.im; return *this;
    }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator=(QuadFloat r) {
        re = r; im = QuadFloat(0.0f); return *this;
    }

    // ff_complex.hpp:61-83 (operator+/-/*// and unary -).
    KOKKOS_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex b) const {
        return QuadFloatComplex(add(re, b.re), add(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex b) const {
        return QuadFloatComplex(subtract(re, b.re), subtract(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i  (ff_complex.hpp:67-70)
        return QuadFloatComplex(subtract(multiply(re, b.re), multiply(im, b.im)),
                                add(multiply(re, b.im), multiply(im, b.re)));
    }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex b) const {
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)  (ff_complex.hpp:71-80)
        if (b.re.f0 == 0.0f && b.im.f0 == 0.0f) {
            Kokkos::printf("QFCOMPLEX: division by zero\n");
            return QuadFloatComplex();
        }
        QuadFloat denom = add(multiply(b.re, b.re), multiply(b.im, b.im));
        QuadFloat inv   = divide(QuadFloat(1.0f), denom);
        return QuadFloatComplex(multiply(add(multiply(re, b.re), multiply(im, b.im)), inv),
                                multiply(subtract(multiply(im, b.re), multiply(re, b.im)), inv));
    }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-() const {
        return QuadFloatComplex(negate(re), negate(im));
    }

    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator+=(QuadFloatComplex b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator-=(QuadFloatComplex b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator*=(QuadFloatComplex b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION QuadFloatComplex& operator/=(QuadFloatComplex b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(QuadFloatComplex b) const { return re==b.re && im==b.im; }
    KOKKOS_INLINE_FUNCTION bool operator!=(QuadFloatComplex b) const { return !(*this == b); }

    KOKKOS_INLINE_FUNCTION QuadFloat real() const { return re; }
    KOKKOS_INLINE_FUNCTION QuadFloat imag() const { return im; }
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const QuadFloatComplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed QuadFloat × QuadFloatComplex arithmetic  (ff_complex.hpp:104-114)
// ============================================================
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(add(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(add(r, z.re), z.im); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(subtract(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(subtract(r, z.re), negate(z.im)); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(multiply(z.re, r), multiply(z.im, r)); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(multiply(r, z.re), multiply(r, z.im)); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex z, QuadFloat r) { return QuadFloatComplex(divide(z.re, r), divide(z.im, r)); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloat r, QuadFloatComplex z) { return QuadFloatComplex(r) / z; }

// ============================================================
// Mixed float × QuadFloatComplex arithmetic  (ff_complex.hpp:116-126)
// ============================================================
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator+(QuadFloatComplex z, float b) { return z + QuadFloat(b); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator+(float b, QuadFloatComplex z) { return QuadFloat(b) + z; }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-(QuadFloatComplex z, float b) { return z - QuadFloat(b); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator-(float b, QuadFloatComplex z) { return QuadFloat(b) - z; }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator*(QuadFloatComplex z, float b) { return z * QuadFloat(b); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator*(float b, QuadFloatComplex z) { return QuadFloat(b) * z; }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator/(QuadFloatComplex z, float b) { return z / QuadFloat(b); }
KOKKOS_INLINE_FUNCTION QuadFloatComplex operator/(float b, QuadFloatComplex z) { return QuadFloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

// abs(z) = |z| = sqrt(re²+im²).  ff_complex.hpp:132-134 / dd_complex.hpp:145-147.
KOKKOS_INLINE_FUNCTION QuadFloat abs(QuadFloatComplex z) {
    return sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
}
// norm(z) = |z|² = re²+im² (std::norm convention; the squared magnitude, no
// sqrt).  ff_complex.hpp / dd_complex.hpp ship no norm(); this is the textbook
// definition (cf. abs above without the sqrt), added per the T3.0c op inventory.
KOKKOS_INLINE_FUNCTION QuadFloat norm(QuadFloatComplex z) {
    return add(multiply(z.re, z.re), multiply(z.im, z.im));
}
// arg(z) = atan2(im, re) (std::arg convention; the polar angle).  ff_complex.hpp
// / dd_complex.hpp expose this only inside log() (ff_complex.hpp:174); surfaced
// as a standalone op per the T3.0c inventory. Uses qf_math.hpp atan2(y, x).
KOKKOS_INLINE_FUNCTION QuadFloat arg(QuadFloatComplex z) {
    return atan2(z.im, z.re);
}
// conj(z) = re - im·i.  ff_complex.hpp:135-137 / dd_complex.hpp:148-150.
KOKKOS_INLINE_FUNCTION QuadFloatComplex conj(QuadFloatComplex z) {
    return QuadFloatComplex(z.re, negate(z.im));
}

// ============================================================
// Complex square root  (Kahan-style; ff_complex.hpp:142-160 / dd_complex.hpp:155-174)
// ============================================================
// B = sqrt((R+|re|)/2) + i·sign(im)·sqrt((R-|re|)/2), R = |z|, arranged to avoid
// cancellation. The ½ and 2 are FP32-exact literals used via multiply_scalar
// (PORT_NOTES §3-lift), matching ff_complex.hpp:146-148.
KOKKOS_INLINE_FUNCTION QuadFloatComplex sqrt(QuadFloatComplex z) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return QuadFloatComplex();
    QuadFloat r  = sqrt(add(multiply(z.re, z.re), multiply(z.im, z.im)));
    QuadFloat a1 = abs(z.re);
    QuadFloat s2 = multiply_scalar(add(r, a1), 0.5f);
    QuadFloat s0 = sqrt(s2);
    QuadFloat s1 = multiply_scalar(s0, 2.0f);
    QuadFloatComplex b;
    if (z.re.f0 >= 0.0f) {
        b.re = s0;
        b.im = divide(z.im, s1);
    } else {
        b.re = divide(z.im, s1);
        if (b.re.f0 < 0.0f) b.re = negate(b.re);
        b.im = s0;
        if (z.im.f0 < 0.0f) b.im = negate(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
// exp(z) = eˣ·(cos y + i·sin y), z = x + iy.  ff_complex.hpp:165-170 /
// dd_complex.hpp:179-184. Real exp + joint sincos are the table-free QF
// transcendentals (T3.0b §6/§7). NOTE the swapped sincos args (header §sincos):
// qf sincos writes (sin, cos), so pass (s, c) to keep c=cos(y), s=sin(y).
KOKKOS_INLINE_FUNCTION QuadFloatComplex exp(QuadFloatComplex z) {
    QuadFloat er = exp(z.re);
    QuadFloat c, s;
    sincos(z.im, s, c);
    return QuadFloatComplex(multiply(er, c), multiply(er, s));
}

// log(z) = log|z| + i·arg(z).  ff_complex.hpp:172-176 / dd_complex.hpp:186-190.
KOKKOS_INLINE_FUNCTION QuadFloatComplex log(QuadFloatComplex z) {
    QuadFloat modulus = abs(z);
    QuadFloat argument = atan2(z.im, z.re);
    return QuadFloatComplex(log(modulus), argument);
}

// log10(z) = log(z)/ln(10).  ff_complex.hpp:178-182 / dd_complex.hpp:192-196.
KOKKOS_INLINE_FUNCTION QuadFloatComplex log10(QuadFloatComplex z) {
    QuadFloatComplex lg = log(z);
    QuadFloat ln10 = QuadFloat_log10();
    return QuadFloatComplex(divide(lg.re, ln10), divide(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
// sin(a+bi) = sin(a)·cosh(b) + i·cos(a)·sinh(b).  ff_complex.hpp:187-192 /
// dd_complex.hpp:201-207. Swapped sincos/sinhcosh args (header §sincos): local
// ca=cos(a), sa=sin(a), cb=cosh(b), sb=sinh(b).
KOKKOS_INLINE_FUNCTION QuadFloatComplex sin(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return QuadFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cos(a+bi) = cos(a)·cosh(b) - i·sin(a)·sinh(b).  ff_complex.hpp:193-198 /
// dd_complex.hpp:208-214.
KOKKOS_INLINE_FUNCTION QuadFloatComplex cos(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sincos(z.re, sa, ca);
    sinhcosh(z.im, sb, cb);
    return QuadFloatComplex(multiply(ca, cb), negate(multiply(sa, sb)));
}
// tan(z) = sin(z)/cos(z).  ff_complex.hpp:199-201 / dd_complex.hpp:215-217.
KOKKOS_INLINE_FUNCTION QuadFloatComplex tan(QuadFloatComplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
// asin(z) = -i·log(iz + sqrt(1 - z²)).  ff_complex.hpp:206-213 /
// dd_complex.hpp:222-231. iz built by literal-lifted components; the 1 promoted
// via QuadFloat(1.0f) inside the single-arg QuadFloatComplex ctor (imag→0).
KOKKOS_INLINE_FUNCTION QuadFloatComplex asin(QuadFloatComplex z) {
    QuadFloatComplex iz  = QuadFloatComplex(negate(z.im), z.re);
    QuadFloatComplex z2  = z * z;
    QuadFloatComplex one_minus_z2 = QuadFloatComplex(QuadFloat(1.0f)) - z2;
    QuadFloatComplex sum = iz + sqrt(one_minus_z2);
    QuadFloatComplex lg  = log(sum);
    return QuadFloatComplex(lg.im, negate(lg.re));  // × (-i): (a+bi)(-i) = b - ai
}
// acos(z) = π/2 - asin(z).  ff_complex.hpp:214-218 / dd_complex.hpp:232-237.
KOKKOS_INLINE_FUNCTION QuadFloatComplex acos(QuadFloatComplex z) {
    QuadFloat pi_over_2 = multiply_scalar(QuadFloat_pi(), 0.5f);
    QuadFloatComplex asin_z = asin(z);
    return QuadFloatComplex(subtract(pi_over_2, asin_z.re), negate(asin_z.im));
}
// atan(z) = (i/2)·log((1 - iz)/(1 + iz)).  ff_complex.hpp:219-226 /
// dd_complex.hpp:238-247.
KOKKOS_INLINE_FUNCTION QuadFloatComplex atan(QuadFloatComplex z) {
    QuadFloatComplex iz    = QuadFloatComplex(negate(z.im), z.re);
    QuadFloatComplex num   = QuadFloatComplex(QuadFloat(1.0f)) - iz;
    QuadFloatComplex den   = QuadFloatComplex(QuadFloat(1.0f)) + iz;
    QuadFloatComplex ratio = num / den;
    QuadFloatComplex lg    = log(ratio);
    // × (i/2): (a+bi)(i/2) = (-b/2) + (a/2)i
    return QuadFloatComplex(multiply_scalar(negate(lg.im), 0.5f), multiply_scalar(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
// sinh(a+bi) = sinh(a)·cos(b) + i·cosh(a)·sin(b).  ff_complex.hpp:231-236 /
// dd_complex.hpp:252-258. Swapped args: ca=cosh(a), sa=sinh(a), cb=cos(b),
// sb=sin(b).
KOKKOS_INLINE_FUNCTION QuadFloatComplex sinh(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return QuadFloatComplex(multiply(sa, cb), multiply(ca, sb));
}
// cosh(a+bi) = cosh(a)·cos(b) + i·sinh(a)·sin(b).  ff_complex.hpp:237-242 /
// dd_complex.hpp:259-265.
KOKKOS_INLINE_FUNCTION QuadFloatComplex cosh(QuadFloatComplex z) {
    QuadFloat ca, sa, cb, sb;
    sinhcosh(z.re, sa, ca);
    sincos(z.im, sb, cb);
    return QuadFloatComplex(multiply(ca, cb), multiply(sa, sb));
}
// tanh(a+bi): re = T/(cos²b + T²·sin²b), im = sin b·cos b·(1-T²)/(...),
// T = tanh(a).  ff_complex.hpp:243-251 / dd_complex.hpp:266-277. Denominator ≥ 0;
// uses the improved real tanh to avoid cancellation. Swapped sincos args:
// cb=cos(b), sb=sin(b).
KOKKOS_INLINE_FUNCTION QuadFloatComplex tanh(QuadFloatComplex z) {
    QuadFloat T = tanh(z.re);
    QuadFloat cb, sb;
    sincos(z.im, sb, cb);
    QuadFloat T2    = multiply(T, T);
    QuadFloat denom = add(multiply(cb, cb), multiply(T2, multiply(sb, sb)));
    return QuadFloatComplex(divide(T, denom),
                            divide(multiply(multiply(sb, cb), subtract(QuadFloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
// asinh(z) = log(z + sqrt(z² + 1)).  ff_complex.hpp:256-258 / dd_complex.hpp:282-285.
KOKKOS_INLINE_FUNCTION QuadFloatComplex asinh(QuadFloatComplex z) {
    return log(z + sqrt(z*z + QuadFloatComplex(QuadFloat(1.0f))));
}
// acosh(z) = log(z + sqrt(z² - 1)).  ff_complex.hpp:259-261 / dd_complex.hpp:286-289.
KOKKOS_INLINE_FUNCTION QuadFloatComplex acosh(QuadFloatComplex z) {
    return log(z + sqrt(z*z - QuadFloatComplex(QuadFloat(1.0f))));
}
// atanh(z) = ½·log((1 + z)/(1 - z)).  ff_complex.hpp:262-266 / dd_complex.hpp:290-295.
KOKKOS_INLINE_FUNCTION QuadFloatComplex atanh(QuadFloatComplex z) {
    QuadFloatComplex one = QuadFloatComplex(QuadFloat(1.0f));
    QuadFloatComplex lg  = log((one + z) / (one - z));
    return QuadFloatComplex(multiply_scalar(lg.re, 0.5f), multiply_scalar(lg.im, 0.5f));
}

// ============================================================
// Complex power and polar
// ============================================================
// pow(z, w) = exp(w·log(z)).  ff_complex.hpp:271-274 / dd_complex.hpp:300-304.
KOKKOS_INLINE_FUNCTION QuadFloatComplex pow(QuadFloatComplex z, QuadFloatComplex w) {
    if (z.re.f0 == 0.0f && z.im.f0 == 0.0f) return QuadFloatComplex();
    return exp(w * log(z));
}

// polar(r, theta) = r·(cos θ + i·sin θ).  ff_complex.hpp:276-280 /
// dd_complex.hpp:306-311. Swapped sincos args: c=cos(θ), s=sin(θ).
KOKKOS_INLINE_FUNCTION QuadFloatComplex polar(QuadFloat r, QuadFloat theta) {
    QuadFloat c, s;
    sincos(theta, s, c);
    return QuadFloatComplex(multiply(r, c), multiply(r, s));
}

} // namespace Experimental
} // namespace Kokkos

// ============================================================
// Re-exposure under namespace Kokkos (T0.4/T2.0/T3.0a convention)
// ============================================================
// Extends the qf_math.hpp Kokkos re-exposure block for the complex ops, so
// Kokkos::exp(qfc) works identically to Kokkos::exp(Kokkos::complex<double>).
// One-line forwards; this does NOT duplicate the real forwards in qf_math.hpp
// (different argument type → distinct overloads). Arithmetic operators are
// reached directly / via ADL and are not re-exposed here.
namespace Kokkos {
// clang-format off
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat        abs(Experimental::QuadFloatComplex z)   { return Experimental::abs(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat        norm(Experimental::QuadFloatComplex z)  { return Experimental::norm(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloat        arg(Experimental::QuadFloatComplex z)   { return Experimental::arg(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex conj(Experimental::QuadFloatComplex z)  { return Experimental::conj(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex sqrt(Experimental::QuadFloatComplex z)  { return Experimental::sqrt(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex exp(Experimental::QuadFloatComplex z)   { return Experimental::exp(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex log(Experimental::QuadFloatComplex z)   { return Experimental::log(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex log10(Experimental::QuadFloatComplex z) { return Experimental::log10(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex sin(Experimental::QuadFloatComplex z)   { return Experimental::sin(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex cos(Experimental::QuadFloatComplex z)   { return Experimental::cos(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex tan(Experimental::QuadFloatComplex z)   { return Experimental::tan(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex asin(Experimental::QuadFloatComplex z)  { return Experimental::asin(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex acos(Experimental::QuadFloatComplex z)  { return Experimental::acos(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex atan(Experimental::QuadFloatComplex z)  { return Experimental::atan(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex sinh(Experimental::QuadFloatComplex z)  { return Experimental::sinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex cosh(Experimental::QuadFloatComplex z)  { return Experimental::cosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex tanh(Experimental::QuadFloatComplex z)  { return Experimental::tanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex asinh(Experimental::QuadFloatComplex z) { return Experimental::asinh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex acosh(Experimental::QuadFloatComplex z) { return Experimental::acosh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex atanh(Experimental::QuadFloatComplex z) { return Experimental::atanh(z); }
KOKKOS_INLINE_FUNCTION Experimental::QuadFloatComplex pow(Experimental::QuadFloatComplex z, Experimental::QuadFloatComplex w) { return Experimental::pow(z, w); }
// clang-format on
}  // namespace Kokkos
