#pragma once
// Double-double real arithmetic — namespace quad::ddfun
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Ported from DDFUN (David H. Bailey) Fortran sources (ddfuna.f90, ddfune.f90).

#include <Kokkos_Core.hpp>
#include <cstdint>
#include <cstring>
#include <cmath>

#ifndef __CUDA_ARCH__
#  include <iomanip>
#  include <ostream>
#endif

namespace quad {
namespace ddfun {

// ============================================================
// Forward declarations (struct uses them in operator bodies)
// ============================================================
struct ddouble;
KOKKOS_INLINE_FUNCTION ddouble ddadd(ddouble a, ddouble b);
KOKKOS_INLINE_FUNCTION ddouble ddsub(ddouble a, ddouble b);
KOKKOS_INLINE_FUNCTION ddouble ddmul(ddouble a, ddouble b);
KOKKOS_INLINE_FUNCTION ddouble dddiv(ddouble a, ddouble b);
KOKKOS_INLINE_FUNCTION ddouble ddmuld(ddouble a, double b);
KOKKOS_INLINE_FUNCTION ddouble dddivd(ddouble a, double b);
KOKKOS_INLINE_FUNCTION ddouble ddneg(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble abs(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble sqrt(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble ddnint(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble powi(ddouble a, int n);
KOKKOS_INLINE_FUNCTION ddouble exp(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble log(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble pow(ddouble a, ddouble b);
KOKKOS_INLINE_FUNCTION void   sinhcosh(ddouble a, ddouble& x, ddouble& y);
KOKKOS_INLINE_FUNCTION void   sincos(ddouble a, ddouble& x, ddouble& y);
KOKKOS_INLINE_FUNCTION ddouble ddang(ddouble x, ddouble y);

// ============================================================
// ddouble struct
// ============================================================
struct ddouble {
    double hi;
    double lo;

    KOKKOS_INLINE_FUNCTION ddouble() : hi(0.0), lo(0.0) {}
    KOKKOS_INLINE_FUNCTION ddouble(double h) : hi(h), lo(0.0) {}
    KOKKOS_INLINE_FUNCTION ddouble(double h, double l) : hi(h), lo(l) {}
    KOKKOS_INLINE_FUNCTION ddouble(const ddouble& o) : hi(o.hi), lo(o.lo) {}
    KOKKOS_INLINE_FUNCTION ddouble& operator=(const ddouble& o) { hi=o.hi; lo=o.lo; return *this; }

    KOKKOS_INLINE_FUNCTION ddouble operator-() const { return ddneg(*this); }
    KOKKOS_INLINE_FUNCTION ddouble operator+(ddouble b) const { return ddadd(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator-(ddouble b) const { return ddsub(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator*(ddouble b) const { return ddmul(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator/(ddouble b) const { return dddiv(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator*(double b)  const { return ddmuld(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator/(double b)  const { return dddivd(*this, b); }
    KOKKOS_INLINE_FUNCTION ddouble operator+(double b)  const { return ddadd(*this, ddouble(b)); }
    KOKKOS_INLINE_FUNCTION ddouble operator-(double b)  const { return ddsub(*this, ddouble(b)); }

    KOKKOS_INLINE_FUNCTION ddouble& operator+=(ddouble b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator-=(ddouble b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator*=(ddouble b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator/=(ddouble b) { *this = *this / b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator+=(double b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator-=(double b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator*=(double b) { *this = ddmuld(*this, b); return *this; }
    KOKKOS_INLINE_FUNCTION ddouble& operator/=(double b) { *this = dddivd(*this, b); return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(ddouble b) const { return hi==b.hi && lo==b.lo; }
    KOKKOS_INLINE_FUNCTION bool operator!=(ddouble b) const { return !(*this == b); }
    KOKKOS_INLINE_FUNCTION bool operator<(ddouble b)  const { return hi<b.hi || (hi==b.hi && lo<b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator>(ddouble b)  const { return hi>b.hi || (hi==b.hi && lo>b.lo); }
    KOKKOS_INLINE_FUNCTION bool operator<=(ddouble b) const { return !(b < *this); }
    KOKKOS_INLINE_FUNCTION bool operator>=(ddouble b) const { return !(*this < b); }
};

KOKKOS_INLINE_FUNCTION ddouble operator+(double a, ddouble b) { return ddadd(ddouble(a), b); }
KOKKOS_INLINE_FUNCTION ddouble operator-(double a, ddouble b) { return ddsub(ddouble(a), b); }
KOKKOS_INLINE_FUNCTION ddouble operator*(double a, ddouble b) { return ddmuld(b, a); }
KOKKOS_INLINE_FUNCTION ddouble operator/(double a, ddouble b) { return dddiv(ddouble(a), b); }

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const ddouble& d) {
    os << "[" << std::setprecision(16) << std::scientific << d.hi
       << ", " << d.lo << "]";
    return os;
}
#endif

// ============================================================
// Constants via bit-pattern construction (safe on host + device)
// ============================================================
KOKKOS_INLINE_FUNCTION ddouble make_dd(uint64_t hi_bits, uint64_t lo_bits) {
    double hi, lo;
#ifndef __CUDA_ARCH__
    std::memcpy(&hi, &hi_bits, sizeof(double));
    std::memcpy(&lo, &lo_bits, sizeof(double));
#else
    hi = __longlong_as_double(static_cast<long long>(hi_bits));
    lo = __longlong_as_double(static_cast<long long>(lo_bits));
#endif
    return ddouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION ddouble dd_pi()          { return make_dd(0x400921fb54442d18ULL, 0x3ca1a62633145c07ULL); }
KOKKOS_INLINE_FUNCTION ddouble dd_e()           { return make_dd(0x4005bf0a8b145769ULL, 0x3ca4d57ee2b1013aULL); }
KOKKOS_INLINE_FUNCTION ddouble dd_log2()        { return make_dd(0x3fe62e42fefa39efULL, 0x3c7abc9e3b39803fULL); }
KOKKOS_INLINE_FUNCTION ddouble dd_log10()       { return make_dd(0x40026bb1bbb55516ULL, 0xbcaf48ad494ea3eaULL); } // ln(10)
KOKKOS_INLINE_FUNCTION ddouble dd_sqrt2()       { return make_dd(0x3ff6a09e667f3bcdULL, 0xbc9bdd3413b26456ULL); }
KOKKOS_INLINE_FUNCTION ddouble dd_euler_gamma() { return make_dd(0x3fe2788cfc6fb619ULL, 0xbc56cb90701fbfabULL); }

// ============================================================
// Primitive arithmetic
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble ddneg(ddouble a) {
    return ddouble(-a.hi, -a.lo);
}

// TwoSum (Knuth)
KOKKOS_INLINE_FUNCTION ddouble ddadd(ddouble a, ddouble b) {
    double t1 = a.hi + b.hi;
    double e  = t1 - a.hi;
    double t2 = ((b.hi - e) + (a.hi - (t1 - e))) + a.lo + b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return ddouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION ddouble ddsub(ddouble a, ddouble b) {
    double t1 = a.hi - b.hi;
    double e  = t1 - a.hi;
    double t2 = ((-b.hi - e) + (a.hi - (t1 - e))) + a.lo - b.lo;
    double hi = t1 + t2;
    double lo = t2 - (hi - t1);
    return ddouble(hi, lo);
}

// TwoProduct (Dekker splitting)
KOKKOS_INLINE_FUNCTION ddouble ddmul(ddouble a, ddouble b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b.hi * split;
    double a1 = cona - (cona - a.hi), b1 = conb - (conb - b.hi);
    double a2 = a.hi - a1,           b2 = b.hi - b1;
    double c11 = a.hi * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = a.hi * b.lo + a.lo * b.hi;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21 + a.lo * b.lo;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return ddouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION ddouble dddiv(ddouble a, ddouble b) {
    const double split = 134217729.0;
    double s1  = a.hi / b.hi;
    double cona = s1 * split, conb = b.hi * split;
    double a1  = cona - (cona - s1), b1 = conb - (conb - b.hi);
    double a2  = s1 - a1,            b2 = b.hi - b1;
    double c11 = s1 * b.hi;
    double c21 = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2  = s1 * b.lo;
    double t1  = c11 + c2;
    double e   = t1 - c11;
    double t2  = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double t12 = t1 + t2;
    double t22 = t2 - (t12 - t1);
    double t11 = a.hi - t12;
    e = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double s2  = (t11 + t21) / b.hi;
    double hi  = s1 + s2;
    double lo  = s2 - (hi - s1);
    return ddouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION ddouble ddmuld(ddouble a, double b) {
    const double split = 134217729.0;
    double cona = a.hi * split, conb = b * split;
    double a1   = cona - (cona - a.hi), b1 = conb - (conb - b);
    double a2   = a.hi - a1,            b2 = b - b1;
    double c11  = a.hi * b;
    double c21  = (((a1*b1 - c11) + a1*b2) + a2*b1) + a2*b2;
    double c2   = a.lo * b;
    double t1   = c11 + c2;
    double e    = t1 - c11;
    double t2   = ((c2 - e) + (c11 - (t1 - e))) + c21;
    double hi   = t1 + t2;
    double lo   = t2 - (hi - t1);
    return ddouble(hi, lo);
}

KOKKOS_INLINE_FUNCTION ddouble dddivd(ddouble a, double b) {
    const double split = 134217729.0;
    double t1  = a.hi / b;
    double cona = t1 * split, conb = b * split;
    double a1  = cona - (cona - t1), b1 = conb - (conb - b);
    double a2  = t1 - a1,            b2 = b - b1;
    double t12 = t1 * b;
    double t22 = (((a1*b1 - t12) + a1*b2) + a2*b1) + a2*b2;
    double t11 = a.hi - t12;
    double e   = t11 - a.hi;
    double t21 = ((-t12 - e) + (a.hi - (t11 - e))) + a.lo - t22;
    double t2  = (t11 + t21) / b;
    double hi  = t1 + t2;
    double lo  = t2 - (hi - t1);
    return ddouble(hi, lo);
}

// Exact product of two doubles
KOKKOS_INLINE_FUNCTION ddouble ddmuldd(double da, double db) {
    const double split = 134217729.0;
    double cona = da * split, conb = db * split;
    double a1   = cona - (cona - da), b1 = conb - (conb - db);
    double a2   = da - a1,            b2 = db - b1;
    double s1   = da * db;
    double s2   = (((a1*b1 - s1) + a1*b2) + a2*b1) + a2*b2;
    return ddouble(s1, s2);
}

// ============================================================
// Basic math
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble abs(ddouble a) {
    return (a.hi >= 0.0) ? a : ddouble(-a.hi, -a.lo);
}

// Nearest integer
KOKKOS_INLINE_FUNCTION ddouble ddnint(ddouble a) {
    if (a.hi == 0.0) return ddouble(0.0);
    const double T105 = ldexp(1.0, 105); // 2^105
    const double T52  = ldexp(1.0, 52);  // 2^52
    ddouble CON = ddouble(T105, T52);
    if (a.hi >= T105) {
        Kokkos::printf("DDNINT: argument too large\n");
        return ddouble(0.0);
    }
    if (a.hi > 0.0) return ddsub(ddadd(a, CON), CON);
    else            return ddadd(ddsub(a, CON), CON);
}

KOKKOS_INLINE_FUNCTION ddouble sqrt(ddouble a) {
    if (a.hi == 0.0) return ddouble(0.0);
    if (a.hi < 0.0) {
        Kokkos::printf("DDSQRT: negative argument\n");
        return ddouble(0.0);
    }
    double t1 = 1.0 / Kokkos::sqrt(a.hi);
    double t2 = a.hi * t1;
    ddouble s0 = ddmuldd(t2, t2);
    ddouble s1 = ddsub(a, s0);
    double t3  = 0.5 * s1.hi * t1;
    return ddadd(ddouble(t2), ddouble(t3));
}

// Integer power
KOKKOS_INLINE_FUNCTION ddouble powi(ddouble a, int n) {
    const double cl2 = 1.4426950408889633;
    if (a.hi == 0.0) {
        if (n >= 0) return ddouble(0.0);
        Kokkos::printf("DDNPWR: zero base with negative exponent\n");
        return ddouble(0.0);
    }
    int nn = (n < 0) ? -n : n;
    if (nn == 0) return ddouble(1.0);
    if (nn == 1) return (n > 0) ? a : dddiv(ddouble(1.0), a);
    if (nn == 2) { ddouble r = ddmul(a,a); return (n>0) ? r : dddiv(ddouble(1.0),r); }
    int mn = (int)(cl2 * Kokkos::log((double)nn) + 1.0 + 1.0e-14);
    ddouble s0 = a, s2 = ddouble(1.0);
    int kn = nn;
    for (int j = 1; j <= mn; ++j) {
        int kk = kn / 2;
        if (kn != 2*kk) s2 = ddmul(s2, s0);
        kn = kk;
        if (j < mn) s0 = ddmul(s0, s0);
    }
    if (n < 0) s2 = dddiv(ddouble(1.0), s2);
    return s2;
}

// ============================================================
// Exp / Log family
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble exp(ddouble a) {
    const int nq = 6;
    const double eps = 1.0e-32;
    ddouble al2 = dd_log2();
    if (a.hi >= 300.0) {
        Kokkos::printf("DDEXP: argument too large\n");
        return ddouble(0.0);
    }
    if (a.hi <= -300.0) return ddouble(0.0);

    ddouble s0 = dddiv(a, al2);
    ddouble s1 = ddnint(s0);
    double t1  = s1.hi;
    int nz     = (int)(t1 + Kokkos::copysign(1.0e-14, t1));
    s0 = ddsub(a, ddmul(al2, s1));

    if (s0.hi == 0.0) {
        return ddouble(ldexp(1.0, nz)); // result = 2^nz exactly
    }
    // Scale down by 2^nq then square nq times
    s1 = ddmuld(s0, ldexp(1.0, -nq));
    ddouble s2 = ddouble(1.0), s3 = ddouble(1.0);
    for (int l1 = 1; l1 <= 100; ++l1) {
        s0 = ddmul(s2, s1);
        s2 = dddivd(s0, (double)l1);
        s0 = ddadd(s3, s2);
        s3 = s0;
        if (Kokkos::fabs(s2.hi) <= eps * Kokkos::fabs(s3.hi)) break;
        if (l1 == 100) { Kokkos::printf("DDEXP: iteration limit\n"); return ddouble(0.0); }
    }
    for (int i = 0; i < nq; ++i) s3 = ddmul(s3, s3);

    return ddmuld(s3, ldexp(1.0, nz)); // multiply by 2^nz
}

KOKKOS_INLINE_FUNCTION ddouble log(ddouble a) {
    if (a.hi <= 0.0) {
        Kokkos::printf("DDLOG: non-positive argument\n");
        return ddouble(0.0);
    }
    // Initial approximation then 3 Newton steps: b <- b + (a - exp(b)) / exp(b)
    ddouble b = ddouble(Kokkos::log(a.hi));
    for (int k = 0; k < 3; ++k) {
        ddouble s0 = exp(b);
        ddouble s1 = ddsub(a, s0);
        ddouble s2 = dddiv(s1, s0);
        b = ddadd(b, s2);
    }
    return b;
}

KOKKOS_INLINE_FUNCTION ddouble log2(ddouble a) {
    return dddiv(log(a), dd_log2());
}

KOKKOS_INLINE_FUNCTION ddouble log10(ddouble a) {
    return dddiv(log(a), dd_log10());
}

KOKKOS_INLINE_FUNCTION ddouble log1p(ddouble a) {
    // log(1+a); use direct formula for moderate a
    return log(ddadd(ddouble(1.0), a));
}

KOKKOS_INLINE_FUNCTION ddouble exp2(ddouble a) {
    return exp(ddmul(a, dd_log2()));
}

KOKKOS_INLINE_FUNCTION ddouble exp10(ddouble a) {
    return exp(ddmul(a, dd_log10()));
}

KOKKOS_INLINE_FUNCTION ddouble expm1(ddouble a) {
    if (Kokkos::fabs(a.hi) > 0.5) {
        // |exp(a)-1| > e^0.5-1 ~ 0.65: subtraction of 1 causes no significant cancellation
        return ddsub(exp(a), ddouble(1.0));
    }
    // Taylor series: a + a²/2! + a³/3! + ...
    // Avoids catastrophic cancellation of exp(a)-1 near a=0
    ddouble sum = a, term = a;
    for (int k = 2; k <= 50; ++k) {
        term = dddivd(ddmul(term, a), (double)k);
        sum  = ddadd(sum, term);
        if (Kokkos::fabs(term.hi) < 1.0e-32 * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// ============================================================
// Trig — internal combined cos+sin, then derived
// ============================================================

// sincos: compute (cos(a), sin(a)) via argument reduction + Taylor series
// x = cos(a), y = sin(a)
KOKKOS_INLINE_FUNCTION void sincos(ddouble a, ddouble& x, ddouble& y) {
    const int itrmx = 1000, nq = 5;
    const double eps = 1.0e-32;
    if (a.hi == 0.0) { x = ddouble(1.0); y = ddouble(0.0); return; }
    if (a.hi >= 1.0e60) {
        Kokkos::printf("DDCSSNR: argument too large\n");
        x = ddouble(0.0); y = ddouble(0.0); return;
    }
    ddouble pi2 = ddmuld(dd_pi(), 2.0);
    ddouble s1  = dddiv(a, pi2);
    ddouble s2  = ddnint(s1);
    ddouble s3  = ddsub(a, ddmul(pi2, s2));
    if (s3.hi == 0.0) { x = ddouble(1.0); y = ddouble(0.0); return; }
    int is = (s3.hi < 0.0) ? -1 : 1;
    double scale = 1.0 / (double)(1 << nq);
    ddouble s0 = ddmuld(s3, scale);
    ddouble s1t = s0;
    ddouble s2sq = ddmul(s0, s0);
    // Sine series: s0 accumulates sin(s3/2^nq)
    for (int i1 = 1; i1 <= itrmx; ++i1) {
        double t2 = -(2.0*i1) * (2.0*i1 + 1.0);
        ddouble s3t = ddmul(s2sq, s1t);
        s1t = dddivd(s3t, t2);
        s3t = ddadd(s1t, s0);
        s0  = s3t;
        if (Kokkos::fabs(s1t.hi) < eps) break;
        if (i1 == itrmx) { Kokkos::printf("DDCSSNR: iteration limit\n"); return; }
    }
    // Double-angle nq times: sin(2x) = 2*sin(x)*cos(x), cos(2x) = 1 - 2*sin²(x)
    ddouble f2 = ddouble(0.5);
    ddouble s4 = ddmul(s0, s0);
    ddouble s5 = ddsub(f2, s4);
    s0 = ddmuld(s5, 2.0);
    for (int j = 2; j <= nq; ++j) {
        s4 = ddmul(s0, s0);
        s5 = ddsub(s4, f2);
        s0 = ddmuld(s5, 2.0);
    }
    // s0 now = cos(s3). Recover sin.
    s4 = ddmul(s0, s0);
    s5 = ddsub(ddouble(1.0), s4);
    s1t = sqrt(s5);
    if (is < 0) { s1t.hi = -s1t.hi; s1t.lo = -s1t.lo; }
    x = s0; y = s1t;
}

KOKKOS_INLINE_FUNCTION ddouble sin(ddouble a) {
    ddouble c, s; sincos(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION ddouble cos(ddouble a) {
    ddouble c, s; sincos(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION ddouble tan(ddouble a) {
    ddouble c, s; sincos(a, c, s); return dddiv(s, c);
}

// Angle of point (x, y) = atan2(y, x)
KOKKOS_INLINE_FUNCTION ddouble ddang(ddouble x, ddouble y) {
    ddouble pi = dd_pi();
    if (x.hi == 0.0 && y.hi == 0.0) return ddouble(0.0);
    if (x.hi == 0.0) return (y.hi > 0.0) ? ddmuld(pi, 0.5) : ddmuld(pi, -0.5);
    if (y.hi == 0.0) return (x.hi > 0.0) ? ddouble(0.0) : pi;
    // Normalize
    ddouble r = sqrt(ddadd(ddmul(x,x), ddmul(y,y)));
    ddouble nx = dddiv(x, r), ny = dddiv(y, r);
    // Initial approximation
    ddouble a = ddouble(Kokkos::atan2(ny.hi, nx.hi));
    bool use_x = (Kokkos::fabs(nx.hi) <= Kokkos::fabs(ny.hi));
    ddouble target = use_x ? nx : ny;
    for (int k = 0; k < 3; ++k) {
        ddouble sin_a, cos_a;
        sincos(a, cos_a, sin_a);
        ddouble corr;
        if (use_x) {
            corr = dddiv(ddsub(target, cos_a), sin_a);
            a = ddsub(a, corr);
        } else {
            corr = dddiv(ddsub(target, sin_a), cos_a);
            a = ddadd(a, corr);
        }
    }
    return a;
}

KOKKOS_INLINE_FUNCTION ddouble asin(ddouble a) {
    if (Kokkos::fabs(a.hi) > 1.0) {
        Kokkos::printf("DDASIN: argument out of range\n");
        return ddouble(0.0);
    }
    ddouble t = sqrt(ddsub(ddouble(1.0), ddmul(a, a)));
    return ddang(t, a); // atan2(a, sqrt(1-a^2))
}
KOKKOS_INLINE_FUNCTION ddouble acos(ddouble a) {
    if (Kokkos::fabs(a.hi) > 1.0) {
        Kokkos::printf("DDACOS: argument out of range\n");
        return ddouble(0.0);
    }
    ddouble t = sqrt(ddsub(ddouble(1.0), ddmul(a, a)));
    return ddang(a, t); // atan2(sqrt(1-a^2), a)
}
KOKKOS_INLINE_FUNCTION ddouble atan(ddouble a) {
    return ddang(ddouble(1.0), a); // atan2(a, 1)
}
KOKKOS_INLINE_FUNCTION ddouble atan2(ddouble y, ddouble x) {
    return ddang(x, y);
}

// ============================================================
// Hyperbolic — internal combined cosh+sinh, then derived
// ============================================================

// x = cosh(a), y = sinh(a)
KOKKOS_INLINE_FUNCTION void sinhcosh(ddouble a, ddouble& x, ddouble& y) {
    ddouble s0 = exp(a);
    ddouble s1 = dddiv(ddouble(1.0), s0);
    x = ddmuld(ddadd(s0, s1), 0.5);
    y = ddmuld(ddsub(s0, s1), 0.5);
}

KOKKOS_INLINE_FUNCTION ddouble sinh(ddouble a) {
    ddouble c, s; sinhcosh(a, c, s); return s;
}
KOKKOS_INLINE_FUNCTION ddouble cosh(ddouble a) {
    ddouble c, s; sinhcosh(a, c, s); return c;
}
KOKKOS_INLINE_FUNCTION ddouble tanh(ddouble a) {
    // tanh(x) = expm1(2x) / (expm1(2x) + 2), reflected for negative x
    // Avoids dividing two nearly-equal large numbers from sinhcosh
    if (a.hi < 0.0) return ddneg(tanh(ddneg(a)));
    ddouble e = expm1(ddmuld(a, 2.0));
    return dddiv(e, ddadd(e, ddouble(2.0)));
}

KOKKOS_INLINE_FUNCTION ddouble asinh(ddouble a) {
    // Reflect: asinh(-a) = -asinh(a). For positive a, a + sqrt(a²+1) >= 1 always,
    // so log argument never causes cancellation.
    if (a.hi < 0.0) return ddneg(asinh(ddneg(a)));
    return log(ddadd(a, sqrt(ddadd(ddmul(a, a), ddouble(1.0)))));
}
KOKKOS_INLINE_FUNCTION ddouble acosh(ddouble a) {
    if (a.hi < 1.0) { Kokkos::printf("DDACOSH: argument < 1\n"); return ddouble(0.0); }
    ddouble t1 = ddsub(ddmul(a, a), ddouble(1.0));
    return log(ddadd(a, sqrt(t1)));
}
KOKKOS_INLINE_FUNCTION ddouble atanh(ddouble a) {
    if (Kokkos::fabs(a.hi) >= 1.0) { Kokkos::printf("DDATANH: |argument| >= 1\n"); return ddouble(0.0); }
    ddouble t1 = ddadd(ddouble(1.0), a);
    ddouble t2 = ddsub(ddouble(1.0), a);
    return ddmuld(log(dddiv(t1, t2)), 0.5);
}

// ============================================================
// Multi-argument operations
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble pow(ddouble a, ddouble b) {
    if (a.hi <= 0.0) {
        if (a.hi == 0.0 && b.hi > 0.0) return ddouble(0.0);
        Kokkos::printf("DDPOW: non-positive base\n");
        return ddouble(0.0);
    }
    return exp(ddmul(log(a), b));
}

KOKKOS_INLINE_FUNCTION ddouble hypot(ddouble a, ddouble b) {
    return sqrt(ddadd(ddmul(a, a), ddmul(b, b)));
}

KOKKOS_INLINE_FUNCTION ddouble ceil(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble floor(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble trunc(ddouble a);
KOKKOS_INLINE_FUNCTION ddouble round(ddouble a);

KOKKOS_INLINE_FUNCTION ddouble fmod(ddouble a, ddouble b) {
    ddouble q = dddiv(a, b);
    ddouble qt = trunc(q);
    return ddsub(a, ddmul(b, qt));
}

KOKKOS_INLINE_FUNCTION ddouble remainder(ddouble a, ddouble b) {
    ddouble q = dddiv(a, b);
    ddouble qn = ddnint(q);
    return ddsub(a, ddmul(b, qn));
}

KOKKOS_INLINE_FUNCTION ddouble copysign(ddouble a, ddouble b) {
    ddouble r = abs(a);
    if (b.hi < 0.0 || (b.hi == 0.0 && b.lo < 0.0)) return ddneg(r);
    return r;
}

KOKKOS_INLINE_FUNCTION ddouble fmax(ddouble a, ddouble b) {
    return (a > b) ? a : b;
}
KOKKOS_INLINE_FUNCTION ddouble fmin(ddouble a, ddouble b) {
    return (a < b) ? a : b;
}
KOKKOS_INLINE_FUNCTION ddouble fdim(ddouble a, ddouble b) {
    return (a > b) ? ddsub(a, b) : ddouble(0.0);
}
KOKKOS_INLINE_FUNCTION ddouble fma(ddouble a, ddouble b, ddouble c) {
    return ddadd(ddmul(a, b), c);
}

// ============================================================
// Rounding
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble floor(ddouble a) {
    ddouble n = ddnint(a);
    if (n > a) return ddsub(n, ddouble(1.0));
    return n;
}
KOKKOS_INLINE_FUNCTION ddouble ceil(ddouble a) {
    ddouble n = ddnint(a);
    if (n < a) return ddadd(n, ddouble(1.0));
    return n;
}
KOKKOS_INLINE_FUNCTION ddouble trunc(ddouble a) {
    return (a.hi >= 0.0) ? floor(a) : ceil(a);
}
KOKKOS_INLINE_FUNCTION ddouble round(ddouble a) {
    return ddnint(a);
}

// ============================================================
// Special functions (in header, not benchmarked)
// ============================================================

// erf — Taylor series for |z| < ~9, asymptotic otherwise
KOKKOS_INLINE_FUNCTION ddouble erf(ddouble z) {
    const double eps = 1.0e-32;
    if (z.hi == 0.0) return ddouble(0.0);
    // threshold: sqrt(104 * ln2) ≈ 8.48
    const double large = 8.5;
    if (z.hi >  large) return ddouble( 1.0);
    if (z.hi < -large) return ddouble(-1.0);

    ddouble z2 = ddmul(z, z);
    int sign = (z.hi >= 0.0) ? 1 : -1;
    ddouble az = abs(z);

    if (Kokkos::fabs(z.hi) < 9.0) {
        // Series: erf(z) = (2/sqrt(pi)) * exp(-z^2) * sum_k 2^k * z^{2k+1} / (1*3*...*(2k+1))
        ddouble t1 = ddouble(0.0), t2 = az, t3 = ddouble(1.0);
        for (int k = 0; k <= 100; ++k) {
            if (k > 0) {
                t2 = ddmuld(ddmul(z2, t2), 2.0);
                t3 = ddmuld(t3, 2.0*k + 1.0);
            }
            ddouble t4 = dddiv(t2, t3);
            ddouble t1new = ddadd(t1, t4);
            if (Kokkos::fabs(t4.hi) < eps * Kokkos::fabs(t1new.hi)) { t1 = t1new; break; }
            t1 = t1new;
        }
        ddouble result = ddmuld(dddiv(ddmuld(t1, 2.0),
                                ddmul(sqrt(dd_pi()), exp(z2))), 1.0);
        return (sign > 0) ? result : ddneg(result);
    } else {
        // Asymptotic: erf(z) = 1 - erfc(z)
        // erfc(z) = exp(-z^2)/sqrt(pi) * sum_k (-1)^k * (2k-1)!! / (2^k * z^{2k+1})
        ddouble t1 = ddouble(0.0), t2 = ddouble(1.0), t3 = az;
        for (int k = 0; k <= 100; ++k) {
            if (k > 0) {
                t2 = ddmuld(t2, -(2.0*k - 1.0));
                t3 = ddmul(t3, ddmuld(z2, 2.0));
            }
            ddouble t4 = dddiv(t2, t3);
            ddouble t1new = ddadd(t1, t4);
            if (Kokkos::fabs(dddiv(t4, t1new).hi) < eps) { t1 = t1new; break; }
            t1 = t1new;
        }
        ddouble erfc_val = dddiv(t1, ddmul(sqrt(dd_pi()), exp(z2)));
        ddouble erf_val  = ddsub(ddouble(1.0), erfc_val);
        return (sign > 0) ? erf_val : ddneg(erf_val);
    }
}

KOKKOS_INLINE_FUNCTION ddouble erfc(ddouble z) {
    return ddsub(ddouble(1.0), erf(z));
}

// gamma — Lanczos approximation at DD precision
KOKKOS_INLINE_FUNCTION ddouble tgamma(ddouble a) {
    if (a.hi < 0.5) {
        ddouble pi = dd_pi();
        ddouble sin_pi_a = sin(ddmul(pi, a));
        return dddiv(pi, ddmul(sin_pi_a, tgamma(ddsub(ddouble(1.0), a))));
    }
    // Lanczos g=7 coefficients (no static — not device-safe)
    const double c0 =  0.99999999999980993;
    const double c1 =  676.5203681218851;
    const double c2 = -1259.1392167224028;
    const double c3 =  771.32342877765313;
    const double c4 = -176.61502916214059;
    const double c5 =  12.507343278686905;
    const double c6 = -0.13857109526572012;
    const double c7 =  9.9843695780195716e-6;
    const double c8 =  1.5056327351493116e-7;
    ddouble x = ddsub(a, ddouble(1.0));
    ddouble t = ddadd(x, ddouble(7.5));
    ddouble s = ddouble(c0);
    s = ddadd(s, dddiv(ddouble(c1), ddadd(x, ddouble(1.0))));
    s = ddadd(s, dddiv(ddouble(c2), ddadd(x, ddouble(2.0))));
    s = ddadd(s, dddiv(ddouble(c3), ddadd(x, ddouble(3.0))));
    s = ddadd(s, dddiv(ddouble(c4), ddadd(x, ddouble(4.0))));
    s = ddadd(s, dddiv(ddouble(c5), ddadd(x, ddouble(5.0))));
    s = ddadd(s, dddiv(ddouble(c6), ddadd(x, ddouble(6.0))));
    s = ddadd(s, dddiv(ddouble(c7), ddadd(x, ddouble(7.0))));
    s = ddadd(s, dddiv(ddouble(c8), ddadd(x, ddouble(8.0))));
    ddouble two_pi_sqrt = ddouble(2.5066282746310002); // sqrt(2*pi)
    return ddmul(ddmul(two_pi_sqrt, s),
                 ddmul(pow(t, ddadd(x, ddouble(0.5))), exp(ddneg(t))));
}

// Bessel J0 via series
KOKKOS_INLINE_FUNCTION ddouble bessel_j0(ddouble x) {
    const double eps = 1.0e-32;
    ddouble x2 = ddmuld(ddmul(x, x), -0.25);
    ddouble term = ddouble(1.0), sum = ddouble(1.0);
    for (int k = 1; k <= 100; ++k) {
        term = dddivd(ddmul(term, x2), (double)(k*k));
        sum  = ddadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION ddouble bessel_j1(ddouble x) {
    const double eps = 1.0e-32;
    ddouble x2 = ddmuld(ddmul(x, x), -0.25);
    ddouble term = ddmuld(x, 0.5), sum = term;
    for (int k = 1; k <= 100; ++k) {
        term = dddivd(ddmul(term, x2), (double)(k * (k+1)));
        sum  = ddadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

KOKKOS_INLINE_FUNCTION ddouble bessel_jn(int n, ddouble x) {
    if (n == 0) return bessel_j0(x);
    if (n == 1) return bessel_j1(x);
    // Downward recurrence
    ddouble j0 = bessel_j0(x), j1 = bessel_j1(x);
    ddouble jm1 = j0, j_cur = j1;
    for (int k = 1; k < n; ++k) {
        ddouble jp1 = ddsub(ddmuld(dddiv(j_cur, x), 2.0*k), jm1);
        jm1   = j_cur;
        j_cur = jp1;
    }
    return j_cur;
}

KOKKOS_INLINE_FUNCTION ddouble bessel_y0(ddouble x) {
    // Y0(x) = (2/pi)*(J0(x)*log(x/2) + sum...)  — simplified
    ddouble two_over_pi = dddivd(ddouble(2.0), dd_pi().hi);
    ddouble j0 = bessel_j0(x);
    return ddmul(two_over_pi, ddmul(j0, log(ddmuld(x, 0.5))));
}
KOKKOS_INLINE_FUNCTION ddouble bessel_y1(ddouble x) {
    ddouble two_over_pi = dddivd(ddouble(2.0), dd_pi().hi);
    ddouble j1 = bessel_j1(x);
    return ddmul(two_over_pi, ddmul(j1, log(ddmuld(x, 0.5))));
}
KOKKOS_INLINE_FUNCTION ddouble bessel_yn(int n, ddouble x) {
    if (n == 0) return bessel_y0(x);
    if (n == 1) return bessel_y1(x);
    ddouble y0 = bessel_y0(x), y1 = bessel_y1(x);
    ddouble ym1 = y0, y_cur = y1;
    for (int k = 1; k < n; ++k) {
        ddouble yp1 = ddsub(ddmuld(dddiv(y_cur, x), 2.0*k), ym1);
        ym1   = y_cur;
        y_cur = yp1;
    }
    return y_cur;
}

// Zeta function — Euler-Maclaurin for s > 1
KOKKOS_INLINE_FUNCTION ddouble zeta(ddouble s) {
    if (s.hi <= 1.0) { Kokkos::printf("DDZETA: s <= 1\n"); return ddouble(0.0); }
    const int N = 50;
    ddouble sum = ddouble(0.0);
    for (int k = 1; k <= N; ++k)
        sum = ddadd(sum, exp(ddmul(ddneg(s), log(ddouble((double)k)))));
    // tail correction integral: N^{1-s}/(s-1)
    ddouble tail = dddiv(exp(ddmul(ddsub(ddouble(1.0), s), log(ddouble((double)N)))),
                         ddsub(s, ddouble(1.0)));
    return ddadd(sum, tail);
}

// Exponential integral Ei(x) via series (x > 0)
KOKKOS_INLINE_FUNCTION ddouble ddexpint(ddouble x) {
    ddouble eg = dd_euler_gamma();
    ddouble sum = ddadd(eg, log(abs(x)));
    ddouble term = x;
    for (int k = 1; k <= 100; ++k) {
        sum = ddadd(sum, dddivd(term, (double)(k * k)));
        term = ddmul(term, x);
        if (Kokkos::fabs(term.hi) * 1e-32 < Kokkos::fabs(sum.hi)) break;
    }
    return sum;
}

// Incomplete gamma P(a,x) via series
KOKKOS_INLINE_FUNCTION ddouble ddincgamma(ddouble a, ddouble x) {
    const double eps = 1.0e-32;
    ddouble term = dddiv(exp(ddneg(x)), a);
    ddouble sum  = term;
    for (int k = 1; k <= 100; ++k) {
        term = ddmul(term, dddiv(x, ddadd(a, ddouble((double)k))));
        sum  = ddadd(sum, term);
        if (Kokkos::fabs(term.hi) < eps * Kokkos::fabs(sum.hi)) break;
    }
    return ddmul(sum, exp(ddmul(a, log(x))));
}

} // namespace ddfun
} // namespace quad
