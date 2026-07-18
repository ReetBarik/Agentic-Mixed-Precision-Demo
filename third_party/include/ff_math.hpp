#pragma once
// Float-float real arithmetic — namespace quad::ffun
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Mechanically ported from dd_math.hpp (DDFUN by David H. Bailey).
//
// Precision: ~14.4 decimal digits (24-bit FP32 mantissa × 2 = 48 bits).
// Range: bounded by FP32 (~3.4e38), much tighter than FP64.

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef __CUDA_ARCH__
#  include <iomanip>
#  include <ostream>
#endif

namespace quad {
namespace ffun {

// ============================================================
// Forward declarations
// ============================================================
struct ffloat;
KOKKOS_INLINE_FUNCTION ffloat ffadd(ffloat a, ffloat b);
KOKKOS_INLINE_FUNCTION ffloat ffsub(ffloat a, ffloat b);
KOKKOS_INLINE_FUNCTION ffloat ffmul(ffloat a, ffloat b);
KOKKOS_INLINE_FUNCTION ffloat ffdiv(ffloat a, ffloat b);
KOKKOS_INLINE_FUNCTION ffloat ffmulf(ffloat a, float b);
KOKKOS_INLINE_FUNCTION ffloat ffdivf(ffloat a, float b);
KOKKOS_INLINE_FUNCTION ffloat ffneg(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat abs(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat sqrt(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat ffnint(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat powi(ffloat a, int n);
KOKKOS_INLINE_FUNCTION ffloat exp(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat log(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat pow(ffloat a, ffloat b);
KOKKOS_INLINE_FUNCTION void   sinhcosh(ffloat a, ffloat& x, ffloat& y);
KOKKOS_INLINE_FUNCTION void   sincos(ffloat a, ffloat& x, ffloat& y);
KOKKOS_INLINE_FUNCTION ffloat ffang(ffloat x, ffloat y);

// ============================================================
// ffloat struct
// ============================================================
struct ffloat {
    float hi;
    float lo;

    KOKKOS_INLINE_FUNCTION ffloat() : hi(0.0f), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION ffloat(float h) : hi(h), lo(0.0f) {}
    KOKKOS_INLINE_FUNCTION ffloat(float h, float l) : hi(h), lo(l) {}
    KOKKOS_INLINE_FUNCTION ffloat(double h) : hi((float)h), lo((float)(h - (double)(float)h)) {}
    KOKKOS_INLINE_FUNCTION ffloat(const ffloat& o) : hi(o.hi), lo(o.lo) {}
    KOKKOS_INLINE_FUNCTION ffloat& operator=(const ffloat& o) { hi=o.hi; lo=o.lo; return *this; }

    KOKKOS_INLINE_FUNCTION ffloat operator-() const { return ffneg(*this); }
    KOKKOS_INLINE_FUNCTION ffloat operator+(ffloat b) const { return ffadd(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator-(ffloat b) const { return ffsub(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator*(ffloat b) const { return ffmul(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator/(ffloat b) const { return ffdiv(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator*(float b)  const { return ffmulf(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator/(float b)  const { return ffdivf(*this, b); }
    KOKKOS_INLINE_FUNCTION ffloat operator+(float b)  const { return ffadd(*this, ffloat(b)); }
    KOKKOS_INLINE_FUNCTION ffloat operator-(float b)  const { return ffsub(*this, ffloat(b)); }

    KOKKOS_INLINE_FUNCTION ffloat& operator+=(ffloat b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator-=(ffloat b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator*=(ffloat b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator/=(ffloat b) { *this = *this / b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator+=(float b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator-=(float b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator*=(float b) { *this = ffmulf(*this, b); return *this; }
    KOKKOS_INLINE_FUNCTION ffloat& operator/=(float b) { *this = ffdivf(*this, b); return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(ffloat b) const { return hi==b.hi && lo==b.lo; }
    KOKKOS_INLINE_FUNCTION bool operator!=(ffloat b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(ffloat b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator>(ffloat b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator<=(ffloat b) const { return !(b < *this); }
    KOKKOS_INLINE_FUNCTION bool operator>=(ffloat b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION ffloat operator+(float a, ffloat b) { return ffadd(ffloat(a), b); }
KOKKOS_INLINE_FUNCTION ffloat operator-(float a, ffloat b) { return ffsub(ffloat(a), b); }
KOKKOS_INLINE_FUNCTION ffloat operator*(float a, ffloat b) { return ffmulf(b, a); }
KOKKOS_INLINE_FUNCTION ffloat operator/(float a, ffloat b) { return ffdiv(ffloat(a), b); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const ffloat& d) {
    os << "[" << std::setprecision(8) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
KOKKOS_INLINE_FUNCTION ffloat make_ff(uint32_t hi_bits, uint32_t lo_bits) {
    float hi, lo;
#ifndef __CUDA_ARCH__
    std::memcpy(&hi, &hi_bits, sizeof(float));
    std::memcpy(&lo, &lo_bits, sizeof(float));
#else
    hi = __int_as_float(static_cast<int>(hi_bits));
    lo = __int_as_float(static_cast<int>(lo_bits));
#endif
    return ffloat(hi, lo);
}

// Auto-generated by scripts/gen_ff_constants.cpp -- do not edit by hand.
// Route A: round_to_nearest_FF(Bailey FP64 hi+lo pair).
KOKKOS_INLINE_FUNCTION ffloat ff_pi          () { return make_ff(0x40490fdbU, 0xb3bbbd2eU); } // pi
KOKKOS_INLINE_FUNCTION ffloat ff_e           () { return make_ff(0x402df854U, 0x33b14577U); } // e
KOKKOS_INLINE_FUNCTION ffloat ff_log2        () { return make_ff(0x3f317218U, 0xb102e308U); } // ln(2)
KOKKOS_INLINE_FUNCTION ffloat ff_log10       () { return make_ff(0x40135d8eU, 0xb309555dU); } // ln(10)
KOKKOS_INLINE_FUNCTION ffloat ff_sqrt2       () { return make_ff(0x3fb504f3U, 0x32cfe77aU); } // sqrt(2)
KOKKOS_INLINE_FUNCTION ffloat ff_euler_gamma () { return make_ff(0x3f13c468U, 0xb1e4127aU); } // Euler gamma

// ============================================================
// Primitive arithmetic
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat ffneg(ffloat a) {
    return ffloat(-a.hi, -a.lo);
}

// TwoSum (Knuth)
KOKKOS_INLINE_FUNCTION ffloat ffadd(ffloat a, ffloat b) {
    float t1 = a.hi + b.hi;
    float e  = t1 - a.hi;
    float t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return ffloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION ffloat ffsub(ffloat a, ffloat b) {
    float t1 = a.hi - b.hi;
    float e  = t1 - a.hi;
    float t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    float hi = t1 + t2;
    float lo = t2 - (hi - t1);
    return ffloat(hi, lo);
}

// TwoProduct (Dekker splitting). Splitter = 2^13 + 1 for FP32 (24-bit mantissa).
KOKKOS_INLINE_FUNCTION ffloat ffmul(ffloat a, ffloat b) {
    const float split = 8193.0f;
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
    return ffloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION ffloat ffdiv(ffloat a, ffloat b) {
    const float split = 8193.0f;
    float s1   = a.hi / b.hi;
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
    return ffloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION ffloat ffmulf(ffloat a, float b) {
    const float split = 8193.0f;
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
    return ffloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION ffloat ffdivf(ffloat a, float b) {
    const float split = 8193.0f;
    float t1   = a.hi / b;
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
    return ffloat(hi, lo);
}

// Exact product of two floats
KOKKOS_INLINE_FUNCTION ffloat ffmulff(float fa, float fb) {
    const float split = 8193.0f;
    float cona = fa * split, conb = fb * split;
    float a1   = cona - (cona - fa), b1 = conb - (conb - fb);
    float a2   = fa - a1,            b2 = fb - b1;
    float s1   = fa * fb;
    float s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return ffloat(s1, s2);
}

// ============================================================
// Basic math
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat abs(ffloat a) {
    return (a.hi >= 0.0f) ? a : ffloat(-a.hi, -a.lo);
}

// Nearest integer. The DD-style magic-constant trick (using a 2^47 FF constant)
// is fragile in FP32: ULP at 2^47 is 2^24, much larger than typical integer
// inputs, so the FF lo component must rescue the precision and ties land on
// the wrong side. Instead, do the rounding in FP64 — FF values are bounded by
// 2^48 and fit exactly in FP64's 53-bit mantissa, where the magic-constant
// trick is well-conditioned.
KOKKOS_INLINE_FUNCTION ffloat ffnint(ffloat a) {
    if (a.hi == 0.0f) return ffloat(0.0f);
    double total = (double)a.hi + (double)a.lo;
    if (Kokkos::fabs(total) >= 1.40737488355328e14 /* 2^47 */) {
        Kokkos::printf("FFNINT: argument too large\n");
        return ffloat(0.0f);
    }
    const double T52 = 4.503599627370496e15; // 2^52
    double rounded = (total > 0.0) ? (total + T52) - T52 : (total - T52) + T52;
    float hi = (float)rounded;
    float lo = (float)(rounded - (double)hi);
    return ffloat(hi, lo);
}

KOKKOS_INLINE_FUNCTION ffloat sqrt(ffloat a) {
    if (a.hi == 0.0f) return ffloat(0.0f);
    if (a.hi < 0.0f) {
        Kokkos::printf("FFSQRT: negative argument\n");
        return ffloat(0.0f);
    }
    float t1 = 1.0f / Kokkos::sqrt(a.hi);
    float t2 = a.hi * t1;
    ffloat s0 = ffmulff(t2, t2);
    ffloat s1 = ffsub(a, s0);
    float t3  = 0.5f * s1.hi * t1;
    return ffadd(ffloat(t2), ffloat(t3));
}

// Integer power
KOKKOS_INLINE_FUNCTION ffloat powi(ffloat a, int n) {
    const float cl2 = 1.4426950408889633f;
    if (a.hi == 0.0f) {
        if (n >= 0) return ffloat(0.0f);
        Kokkos::printf("FFNPWR: zero base with negative exponent\n");
        return ffloat(0.0f);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return ffloat(1.0f);
    if (nn == 1) return (n > 0) ? a : ffdiv(ffloat(1.0f), a);
    if (nn == 2) { ffloat r = ffmul(a,a); return (n>0) ? r : ffdiv(ffloat(1.0f),r); }
    int mn = (int)(cl2 * Kokkos::log((float)nn) + 1.0f + 1.0e-6f);
    ffloat s0 = a, s2 = ffloat(1.0f);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = ffmul(s2, s0);
        kn = kk;
        if (j < mn) s0 = ffmul(s0, s0);
    }
    if (n < 0) s2 = ffdiv(ffloat(1.0f), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat exp(ffloat a) {
    const int nq = 4;
    const float eps = 1.0e-15f;
    ffloat al2 = ff_log2();
    // FP32 finite range: |x| < ln(3.4e38) ~= 88.7
    if (a.hi >= 88.0f) {
        Kokkos::printf("FFEXP: argument too large\n");
        return ffloat(0.0f);
    }
    if (a.hi <= -88.0f) return ffloat(0.0f);

    ffloat s0 = ffdiv(a, al2);
    ffloat s1 = ffnint(s0);
    float t1  = s1.hi;
    int nz    = (int)(t1 + Kokkos::copysign(1.0e-6f, t1));
    s0 = ffsub(a, ffmul(al2, s1));

    if (s0.hi == 0.0f) {
        return ffloat(ldexpf(1.0f, nz));
    }
    // Scale down by 2^nq then square nq times
    s1 = ffmulf(s0, ldexpf(1.0f, -nq));
    ffloat s2 = ffloat(1.0f), s3 = ffloat(1.0f);
    for (int l1 = 1; l1 <= 60; ++l1) {
        s0 = ffmul(s2, s1);
        s2 = ffdivf(s0, (float)l1);
        s0 = ffadd(s3, s2);
        s3 = s0;
        if (Kokkos::fabs(s2.hi) <= eps * Kokkos::fabs(s3.hi)) break;
        if (l1 == 60) { Kokkos::printf("FFEXP: iteration limit\n"); return ffloat(0.0f); }
    }
    for (int i = 0; i < nq; ++i) s3 = ffmul(s3, s3);

    // Final scaling by 2^nz is exact in FP32 (power-of-2 multiplication does
    // not round). Going through ffmulf would compute b*8193 inside Dekker
    // splitting, which overflows for nz >= 115 (i.e. a > ~79) — the cause of
    // the previous NaN outputs at the high end of the input range.
    float pow2 = ldexpf(1.0f, nz);
    return ffloat(s3.hi * pow2, s3.lo * pow2);
}

KOKKOS_INLINE_FUNCTION ffloat log(ffloat a) {
    if (a.hi <= 0.0f) {
        Kokkos::printf("FFLOG: non-positive argument\n");
        return ffloat(0.0f);
    }
    // Initial approximation then 2 Newton steps (FP32 base gives ~6 digits, doubles per iter -> 24 -> 48 bits)
    ffloat b = ffloat(Kokkos::log(a.hi));
    for (int k = 0; k < 2; ++k) {
        ffloat s0 = exp(b);
        ffloat s1 = ffsub(a, s0);
        ffloat s2 = ffdiv(s1, s0);
        b = ffadd(b, s2);
    }
    return b;
}

KOKKOS_INLINE_FUNCTION ffloat log2(ffloat a) {
    return ffdiv(log(a), ff_log2());
}

KOKKOS_INLINE_FUNCTION ffloat log10(ffloat a) {
    return ffdiv(log(a), ff_log10());
}

KOKKOS_INLINE_FUNCTION ffloat log1p(ffloat a) {
    return log(ffadd(ffloat(1.0f), a));
}

KOKKOS_INLINE_FUNCTION ffloat exp2(ffloat a) {
    return exp(ffmul(a, ff_log2()));
}

KOKKOS_INLINE_FUNCTION ffloat exp10(ffloat a) {
    return exp(ffmul(a, ff_log10()));
}

KOKKOS_INLINE_FUNCTION ffloat expm1(ffloat a) {
    if (Kokkos::fabs(a.hi) > 0.5f) {
        return ffsub(exp(a), ffloat(1.0f));
    }
    // Taylor: a + a^2/2! + a^3/3! + ...
    ffloat sum = a, term = a;
    for (int k = 2; k <= 30; ++k) {
        term = ffdivf(ffmul(term, a), (float)k);
        sum  = ffadd(sum, term);
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
KOKKOS_INLINE_FUNCTION void sincos(ffloat a, ffloat& x, ffloat& y) {
    const int itrmx = 100, nq = 4;
    const float eps = 1.0e-15f;
    if (a.hi == 0.0f) { x = ffloat(1.0f); y = ffloat(0.0f); return; }
    if (a.hi >= 1.0e30f) {
        Kokkos::printf("FFCSSNR: argument too large\n");
        x = ffloat(0.0f); y = ffloat(0.0f); return;
    }
    ffloat pi2 = ffmulf(ff_pi(), 2.0f);
    ffloat s1  = ffdiv(a, pi2);
    ffloat s2  = ffnint(s1);
    ffloat s3  = ffsub(a, ffmul(pi2, s2));
    if (s3.hi == 0.0f) { x = ffloat(1.0f); y = ffloat(0.0f); return; }
    float scale = 1.0f / (float)(1 << nq);
    ffloat r  = ffmulf(s3, scale);   // r = s3 / 2^nq, |r| < pi/2^nq
    ffloat r2 = ffmul(r, r);

    // sin(r) = r - r^3/3! + r^5/5! - ...
    // cos(r) = 1 - r^2/2! + r^4/4! - ...
    ffloat sin_r = r,             cos_r  = ffloat(1.0f);
    ffloat sterm = r,             cterm  = ffloat(1.0f);
    for (int k = 1; k <= itrmx; ++k) {
        sterm = ffdivf(ffmul(sterm, r2), -(float)((2*k) * (2*k + 1)));
        sin_r = ffadd(sin_r, sterm);
        cterm = ffdivf(ffmul(cterm, r2), -(float)((2*k - 1) * (2*k)));
        cos_r = ffadd(cos_r, cterm);
        if (Kokkos::fabs(sterm.hi) < eps * Kokkos::fabs(sin_r.hi) &&
            Kokkos::fabs(cterm.hi) < eps) break;
        if (k == itrmx) { Kokkos::printf("FFCSSNR: iteration limit\n"); return; }
    }

    // Doubling: sin(2x) = 2 sin x cos x, cos(2x) = cos^2 x - sin^2 x
    for (int j = 0; j < nq; ++j) {
        ffloat new_sin = ffmulf(ffmul(sin_r, cos_r), 2.0f);
        ffloat new_cos = ffsub(ffmul(cos_r, cos_r), ffmul(sin_r, sin_r));
        sin_r = new_sin;
        cos_r = new_cos;
    }

    x = cos_r; y = sin_r;
}

KOKKOS_INLINE_FUNCTION ffloat sin(ffloat a) {
    ffloat c, s; sincos(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION ffloat cos(ffloat a) {
    ffloat c, s; sincos(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION ffloat tan(ffloat a) {
    ffloat c, s; sincos(a, c, s); return ffdiv(s, c);
}

// Angle of point (x, y) = atan2(y, x)
KOKKOS_INLINE_FUNCTION ffloat ffang(ffloat x, ffloat y) {
    ffloat pi = ff_pi();
    if (x.hi == 0.0f && y.hi == 0.0f) return ffloat(0.0f);
    if (x.hi == 0.0f) return (y.hi > 0.0f) ? ffmulf(pi, 0.5f) : ffmulf(pi, -0.5f);
    if (y.hi == 0.0f) return (x.hi > 0.0f) ? ffloat(0.0f) : pi;
    ffloat r = sqrt(ffadd(ffmul(x,x), ffmul(y,y)));
    ffloat nx = ffdiv(x, r), ny = ffdiv(y, r);
    ffloat a = ffloat(Kokkos::atan2(ny.hi, nx.hi));
    bool use_x = (Kokkos::fabs(nx.hi) <= Kokkos::fabs(ny.hi));
    ffloat target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        ffloat sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        ffloat corr;
        if (use_x) {
            corr = ffdiv(ffsub(target, cos_a), sin_a);
            a = ffsub(a, corr);
        } else {
            corr = ffdiv(ffsub(target, sin_a), cos_a);
            a = ffadd(a, corr);
        }
    }
    return a;
}

KOKKOS_INLINE_FUNCTION ffloat asin(ffloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFASIN: argument out of range\n");
        return ffloat(0.0f);
    }
    ffloat t = sqrt(ffsub(ffloat(1.0f), ffmul(a, a)));
    return ffang(t, a);
}
KOKKOS_INLINE_FUNCTION ffloat acos(ffloat a) {
    if (Kokkos::fabs(a.hi) > 1.0f) {
        Kokkos::printf("FFACOS: argument out of range\n");
        return ffloat(0.0f);
    }
    ffloat t = sqrt(ffsub(ffloat(1.0f), ffmul(a, a)));
    return ffang(a, t);
}
KOKKOS_INLINE_FUNCTION ffloat atan(ffloat a) {
    return ffang(ffloat(1.0f), a);
}
KOKKOS_INLINE_FUNCTION ffloat atan2(ffloat y, ffloat x) {
    return ffang(x, y);
}

// ============================================================
// Hyperbolic
// ============================================================

KOKKOS_INLINE_FUNCTION void sinhcosh(ffloat a, ffloat& x, ffloat& y) {
    // Taylor series for |a| < 0.5 — avoids the (e^a - e^-a)/2 cancellation
    // when a is small (both exponentials approach 1, leading bits cancel).
    if (Kokkos::fabs(a.hi) < 0.5f) {
        ffloat a2 = ffmul(a, a);
        ffloat sinh_sum = a,             sinh_term = a;
        ffloat cosh_sum = ffloat(1.0f),  cosh_term = ffloat(1.0f);
        for (int k = 1; k <= 30; ++k) {
            sinh_term = ffdivf(ffmul(sinh_term, a2), (float)((2*k) * (2*k + 1)));
            sinh_sum  = ffadd(sinh_sum, sinh_term);
            cosh_term = ffdivf(ffmul(cosh_term, a2), (float)((2*k - 1) * (2*k)));
            cosh_sum  = ffadd(cosh_sum, cosh_term);
            if (Kokkos::fabs(sinh_term.hi) < 1.0e-15f * Kokkos::fabs(sinh_sum.hi) &&
                Kokkos::fabs(cosh_term.hi) < 1.0e-15f) break;
        }
        x = cosh_sum; y = sinh_sum;
        return;
    }
    ffloat s0 = exp(a);
    ffloat s1 = ffdiv(ffloat(1.0f), s0);
    x = ffmulf(ffadd(s0, s1), 0.5f);
    y = ffmulf(ffsub(s0, s1), 0.5f);
}

KOKKOS_INLINE_FUNCTION ffloat sinh(ffloat a) {
    ffloat c, s; sinhcosh(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION ffloat cosh(ffloat a) {
    ffloat c, s; sinhcosh(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION ffloat tanh(ffloat a) {
    if (a.hi < 0.0f) return ffneg(tanh(ffneg(a)));
    ffloat e = expm1(ffmulf(a, 2.0f));
    return ffdiv(e, ffadd(e, ffloat(2.0f)));
}

KOKKOS_INLINE_FUNCTION ffloat asinh(ffloat a) {
    if (a.hi < 0.0f) return ffneg(asinh(ffneg(a)));
    return log(ffadd(a, sqrt(ffadd(ffmul(a, a), ffloat(1.0f)))));
}
KOKKOS_INLINE_FUNCTION ffloat acosh(ffloat a) {
    if (a.hi < 1.0f) { Kokkos::printf("FFACOSH: argument < 1\n"); return ffloat(0.0f); }
    ffloat t1 = ffsub(ffmul(a, a), ffloat(1.0f));
    return log(ffadd(a, sqrt(t1)));
}
KOKKOS_INLINE_FUNCTION ffloat atanh(ffloat a) {
    if (Kokkos::fabs(a.hi) >= 1.0f) { Kokkos::printf("FFATANH: |argument| >= 1\n"); return ffloat(0.0f); }
    // Taylor for |a|<0.5 avoids calling log (which loses precision when its
    // argument is close to 1). All terms positive — no cancellation.
    if (Kokkos::fabs(a.hi) < 0.5f) {
        ffloat a2 = ffmul(a, a);
        ffloat sum = a, pwr = a;
        for (int k = 1; k <= 60; ++k) {
            pwr  = ffmul(pwr, a2);
            ffloat term = ffdivf(pwr, (float)(2*k + 1));
            sum  = ffadd(sum, term);
            if (Kokkos::fabs(term.hi) < 1.0e-15f * Kokkos::fabs(sum.hi)) break;
        }
        return sum;
    }
    // For 0.5 <= |a| < 1, log((1+a)/(1-a)) is well-conditioned (ratio >= 3).
    ffloat t1 = ffadd(ffloat(1.0f), a);
    ffloat t2 = ffsub(ffloat(1.0f), a);
    return ffmulf(log(ffdiv(t1, t2)), 0.5f);
}

// ============================================================
// Multi-argument operations
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat pow(ffloat a, ffloat b) {
    if (a.hi <= 0.0f) {
        if (a.hi == 0.0f && b.hi > 0.0f) return ffloat(0.0f);
        Kokkos::printf("FFPOW: non-positive base\n");
        return ffloat(0.0f);
    }
    return exp(ffmul(log(a), b));
}

KOKKOS_INLINE_FUNCTION ffloat hypot(ffloat a, ffloat b) {
    return sqrt(ffadd(ffmul(a, a), ffmul(b, b)));
}

KOKKOS_INLINE_FUNCTION ffloat ceil(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat floor(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat trunc(ffloat a);
KOKKOS_INLINE_FUNCTION ffloat round(ffloat a);

KOKKOS_INLINE_FUNCTION ffloat fmod(ffloat a, ffloat b) {
    ffloat q = ffdiv(a, b);
    ffloat qt = trunc(q);
    return ffsub(a, ffmul(b, qt));
}

KOKKOS_INLINE_FUNCTION ffloat remainder(ffloat a, ffloat b) {
    ffloat q = ffdiv(a, b);
    ffloat qn = ffnint(q);
    return ffsub(a, ffmul(b, qn));
}

KOKKOS_INLINE_FUNCTION ffloat copysign(ffloat a, ffloat b) {
    ffloat r = abs(a);
    if (b.hi < 0.0f || (b.hi == 0.0f && b.lo < 0.0f)) return ffneg(r);
    return r;
}

KOKKOS_INLINE_FUNCTION ffloat fmax(ffloat a, ffloat b) {
    return (a > b) ? a : b;
}
KOKKOS_INLINE_FUNCTION ffloat fmin(ffloat a, ffloat b) {
    return (a < b) ? a : b;
}
KOKKOS_INLINE_FUNCTION ffloat fdim(ffloat a, ffloat b) {
    return (a > b) ? ffsub(a, b) : ffloat(0.0f);
}
KOKKOS_INLINE_FUNCTION ffloat fma(ffloat a, ffloat b, ffloat c) {
    return ffadd(ffmul(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat floor(ffloat a) {
    ffloat n = ffnint(a);
    if (n > a) return ffsub(n, ffloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION ffloat ceil(ffloat a) {
    ffloat n = ffnint(a);
    if (n < a) return ffadd(n, ffloat(1.0f));
    return n;
}
KOKKOS_INLINE_FUNCTION ffloat trunc(ffloat a) {
    return (a.hi >= 0.0f) ? floor(a) : ceil(a);
}
KOKKOS_INLINE_FUNCTION ffloat round(ffloat a) {
    return ffnint(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat erf(ffloat z) {
    const float eps = 1.0e-15f;
    if (z.hi == 0.0f) return ffloat(0.0f);
    const float large = 6.0f; // erfc(6) ~= 2e-17 << FF resolution
    if (z.hi >  large) return ffloat( 1.0f);
    if (z.hi < -large) return ffloat(-1.0f);

    ffloat z2 = ffmul(z, z);
    int sign = (z.hi >= 0.0f) ? 1 : -1;
    ffloat az = abs(z);

    if (Kokkos::fabs(z.hi) < 4.0f) {
        ffloat t1 = ffloat(0.0f), t2 = az, t3 = ffloat(1.0f);
        for (int k = 0; k <= 60; ++k) {
            if (k > 0) {
                t2 = ffmulf(ffmul(z2, t2), 2.0f);
                t3 = ffmulf(t3, 2.0f*k + 1.0f);
            }
            ffloat t4 = ffdiv(t2, t3);
            ffloat t1new = ffadd(t1, t4);
            if (Kokkos::fabs(t4.hi) < eps * Kokkos::fabs(t1new.hi)) { t1 = t1new; break; }
            t1 = t1new;
        }
        ffloat result = ffmulf(ffdiv(ffmulf(t1, 2.0f),
                                ffmul(sqrt(ff_pi()), exp(z2))), 1.0f);
        return (sign > 0) ? result : ffneg(result);
    } else {
        ffloat t1 = ffloat(0.0f), t2 = ffloat(1.0f), t3 = az;
        for (int k = 0; k <= 60; ++k) {
            if (k > 0) {
                t2 = ffmulf(t2, -(2.0f*k - 1.0f));
                t3 = ffmul(t3, ffmulf(z2, 2.0f));
            }
            ffloat t4 = ffdiv(t2, t3);
            ffloat t1new = ffadd(t1, t4);
            if (Kokkos::fabs(ffdiv(t4, t1new).hi) < eps) { t1 = t1new; break; }
            t1 = t1new;
        }
        ffloat erfc_val = ffdiv(t1, ffmul(sqrt(ff_pi()), exp(z2)));
        ffloat erf_val  = ffsub(ffloat(1.0f), erfc_val);
        return (sign > 0) ? erf_val : ffneg(erf_val);
    }
}

KOKKOS_INLINE_FUNCTION ffloat erfc(ffloat z) {
    return ffsub(ffloat(1.0f), erf(z));
}

// gamma — Lanczos approximation
KOKKOS_INLINE_FUNCTION ffloat tgamma(ffloat a) {
    if (a.hi < 0.5f) {
        ffloat pi = ff_pi();
        ffloat sin_pi_a = sin(ffmul(pi, a));
        return ffdiv(pi, ffmul(sin_pi_a, tgamma(ffsub(ffloat(1.0f), a))));
    }
    const float c0 =  0.99999999999980993f;
    const float c1 =  676.5203681218851f;
    const float c2 = -1259.1392167224028f;
    const float c3 =  771.32342877765313f;
    const float c4 = -176.61502916214059f;
    const float c5 =  12.507343278686905f;
    const float c6 = -0.13857109526572012f;
    const float c7 =  9.9843695780195716e-6f;
    const float c8 =  1.5056327351493116e-7f;
    ffloat x = ffsub(a, ffloat(1.0f));
    ffloat t = ffadd(x, ffloat(7.5f));
    ffloat s = ffloat(c0);
    s = ffadd(s, ffdiv(ffloat(c1), ffadd(x, ffloat(1.0f))));
    s = ffadd(s, ffdiv(ffloat(c2), ffadd(x, ffloat(2.0f))));
    s = ffadd(s, ffdiv(ffloat(c3), ffadd(x, ffloat(3.0f))));
    s = ffadd(s, ffdiv(ffloat(c4), ffadd(x, ffloat(4.0f))));
    s = ffadd(s, ffdiv(ffloat(c5), ffadd(x, ffloat(5.0f))));
    s = ffadd(s, ffdiv(ffloat(c6), ffadd(x, ffloat(6.0f))));
    s = ffadd(s, ffdiv(ffloat(c7), ffadd(x, ffloat(7.0f))));
    s = ffadd(s, ffdiv(ffloat(c8), ffadd(x, ffloat(8.0f))));
    ffloat two_pi_sqrt = ffloat(2.5066282746310002f);
    return ffmul(ffmul(two_pi_sqrt, s),
                 ffmul(pow(t, ffadd(x, ffloat(0.5f))), exp(ffneg(t))));
}

// Bessel J0 via series
KOKKOS_INLINE_FUNCTION ffloat bessel_j0(ffloat x) {
    const float eps = 1.0e-15f;
    ffloat x2 = ffmulf(ffmul(x, x), -0.25f);
    ffloat term = ffloat(1.0f), sum = ffloat(1.0f);
    for (int k = 1; k <= 60; ++k) {
        term = ffdivf(ffmul(term, x2), (float)(k*k));
        sum  = ffadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION ffloat bessel_j1(ffloat x) {
    const float eps = 1.0e-15f;
    ffloat x2 = ffmulf(ffmul(x, x), -0.25f);
    ffloat term = ffmulf(x, 0.5f), sum = term;
    for (int k = 1; k <= 60; ++k) {
        term = ffdivf(ffmul(term, x2), (float)(k * (k+1)));
        sum  = ffadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION ffloat bessel_jn(int n, ffloat x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    ffloat j0 = bessel_j0(x), j1 = bessel_j1(x);
    ffloat jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        ffloat jp1 = ffsub(ffmulf(ffdiv(j_cur, x), 2.0f*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

KOKKOS_INLINE_FUNCTION ffloat bessel_y0(ffloat x) {
    ffloat two_over_pi = ffdivf(ffloat(2.0f), ff_pi().hi);
    ffloat j0 = bessel_j0(x);
    return ffmul(two_over_pi, ffmul(j0, log(ffmulf(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION ffloat bessel_y1(ffloat x) {
    ffloat two_over_pi = ffdivf(ffloat(2.0f), ff_pi().hi);
    ffloat j1 = bessel_j1(x);
    return ffmul(two_over_pi, ffmul(j1, log(ffmulf(x, 0.5f))));
}
KOKKOS_INLINE_FUNCTION ffloat bessel_yn(int n, ffloat x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    ffloat y0 = bessel_y0(x), y1 = bessel_y1(x);
    ffloat ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        ffloat yp1 = ffsub(ffmulf(ffdiv(y_cur, x), 2.0f*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

KOKKOS_INLINE_FUNCTION ffloat zeta(ffloat s) {
    if (s.hi <= 1.0f) { Kokkos::printf("FFZETA: s <= 1\n"); return ffloat(0.0f); }
    const int N = 30;
    ffloat sum = ffloat(0.0f);
    for (int k = 1; k <= N; ++k)
        sum = ffadd(sum, exp(ffmul(ffneg(s), log(ffloat((float)k)))));
    ffloat tail = ffdiv(exp(ffmul(ffsub(ffloat(1.0f), s), log(ffloat((float)N)))),
                         ffsub(s, ffloat(1.0f)));
    return ffadd(sum, tail);
}

KOKKOS_INLINE_FUNCTION ffloat ffexpint(ffloat x) {
    ffloat eg = ff_euler_gamma();
    ffloat sum = ffadd(eg, log(abs(x)));
    ffloat term = x;
    for (int k = 1; k <= 60; ++k) {
        sum = ffadd(sum, ffdivf(term, (float)(k * k)));
        term = ffmul(term, x);
        if (Kokkos::fabs(term.hi) * 1e-15f < Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION ffloat ffincgamma(ffloat a, ffloat x) {
    const float eps = 1.0e-15f;
    ffloat term = ffdiv(exp(ffneg(x)), a);
    ffloat sum  = term;
    for (int k = 1; k <= 60; ++k) {
        term = ffmul(term, ffdiv(x, ffadd(a, ffloat((float)k))));
        sum  = ffadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return ffmul(sum, exp(ffmul(a, log(x))));
}

} // namespace ffun
} // namespace quad
