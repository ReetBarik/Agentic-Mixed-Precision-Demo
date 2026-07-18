#pragma once
// Float-float complex arithmetic — namespace quad::ffun
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on ff_math.hpp.

#include <ff_math.hpp>

#ifndef __CUDA_ARCH__
#  include <ostream>
#endif

namespace quad {
namespace ffun {

// ============================================================
// ffcomplex struct
// ============================================================
struct ffcomplex {
    ffloat re;
    ffloat im;

    KOKKOS_INLINE_FUNCTION ffcomplex() : re(0.0f), im(0.0f) {}
    KOKKOS_INLINE_FUNCTION ffcomplex(float r)               : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION ffcomplex(ffloat r)              : re(r),    im(0.0f) {}
    KOKKOS_INLINE_FUNCTION ffcomplex(float r, float i)      : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION ffcomplex(ffloat r, ffloat i)    : re(r),    im(i)    {}
    KOKKOS_INLINE_FUNCTION ffcomplex(const ffcomplex& o)    : re(o.re), im(o.im) {}
    KOKKOS_INLINE_FUNCTION ffcomplex& operator=(const ffcomplex& o) {
        re = o.re; im = o.im; return *this;
    }
    KOKKOS_INLINE_FUNCTION ffcomplex& operator=(ffloat r) {
        re = r; im = ffloat(0.0f); return *this;
    }

    KOKKOS_INLINE_FUNCTION ffcomplex operator+(ffcomplex b) const {
        return ffcomplex(ffadd(re, b.re), ffadd(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION ffcomplex operator-(ffcomplex b) const {
        return ffcomplex(ffsub(re, b.re), ffsub(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION ffcomplex operator*(ffcomplex b) const {
        return ffcomplex(ffsub(ffmul(re, b.re), ffmul(im, b.im)),
                         ffadd(ffmul(re, b.im), ffmul(im, b.re)));
    }
    KOKKOS_INLINE_FUNCTION ffcomplex operator/(ffcomplex b) const {
        if (b.re.hi == 0.0f && b.im.hi == 0.0f) {
            Kokkos::printf("FFCOMPLEX: division by zero\n");
            return ffcomplex();
        }
        ffloat denom = ffadd(ffmul(b.re, b.re), ffmul(b.im, b.im));
        ffloat inv   = ffdiv(ffloat(1.0f), denom);
        return ffcomplex(ffmul(ffadd(ffmul(re, b.re), ffmul(im, b.im)), inv),
                         ffmul(ffsub(ffmul(im, b.re), ffmul(re, b.im)), inv));
    }
    KOKKOS_INLINE_FUNCTION ffcomplex operator-() const {
        return ffcomplex(ffneg(re), ffneg(im));
    }

    KOKKOS_INLINE_FUNCTION ffcomplex& operator+=(ffcomplex b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ffcomplex& operator-=(ffcomplex b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ffcomplex& operator*=(ffcomplex b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION ffcomplex& operator/=(ffcomplex b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(ffcomplex b) const { return re==b.re && im==b.im; }
    KOKKOS_INLINE_FUNCTION bool operator!=(ffcomplex b) const { return !(*this == b); }

    KOKKOS_INLINE_FUNCTION ffloat real() const { return re; }
    KOKKOS_INLINE_FUNCTION ffloat imag() const { return im; }
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const ffcomplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed ffloat × ffcomplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex operator+(ffcomplex z, ffloat r) { return ffcomplex(ffadd(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION ffcomplex operator+(ffloat r, ffcomplex z) { return ffcomplex(ffadd(r, z.re), z.im); }
KOKKOS_INLINE_FUNCTION ffcomplex operator-(ffcomplex z, ffloat r) { return ffcomplex(ffsub(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION ffcomplex operator-(ffloat r, ffcomplex z) { return ffcomplex(ffsub(r, z.re), ffneg(z.im)); }
KOKKOS_INLINE_FUNCTION ffcomplex operator*(ffcomplex z, ffloat r) { return ffcomplex(ffmul(z.re, r), ffmul(z.im, r)); }
KOKKOS_INLINE_FUNCTION ffcomplex operator*(ffloat r, ffcomplex z) { return ffcomplex(ffmul(r, z.re), ffmul(r, z.im)); }
KOKKOS_INLINE_FUNCTION ffcomplex operator/(ffcomplex z, ffloat r) { return ffcomplex(ffdiv(z.re, r), ffdiv(z.im, r)); }
KOKKOS_INLINE_FUNCTION ffcomplex operator/(ffloat r, ffcomplex z) { return ffcomplex(r) / z; }

// ============================================================
// Mixed float × ffcomplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex operator+(ffcomplex z, float b) { return z + ffloat(b); }
KOKKOS_INLINE_FUNCTION ffcomplex operator+(float b, ffcomplex z) { return ffloat(b) + z; }
KOKKOS_INLINE_FUNCTION ffcomplex operator-(ffcomplex z, float b) { return z - ffloat(b); }
KOKKOS_INLINE_FUNCTION ffcomplex operator-(float b, ffcomplex z) { return ffloat(b) - z; }
KOKKOS_INLINE_FUNCTION ffcomplex operator*(ffcomplex z, float b) { return z * ffloat(b); }
KOKKOS_INLINE_FUNCTION ffcomplex operator*(float b, ffcomplex z) { return ffloat(b) * z; }
KOKKOS_INLINE_FUNCTION ffcomplex operator/(ffcomplex z, float b) { return z / ffloat(b); }
KOKKOS_INLINE_FUNCTION ffcomplex operator/(float b, ffcomplex z) { return ffloat(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

KOKKOS_INLINE_FUNCTION ffloat abs(ffcomplex z) {
    return sqrt(ffadd(ffmul(z.re, z.re), ffmul(z.im, z.im)));
}
KOKKOS_INLINE_FUNCTION ffcomplex conj(ffcomplex z) {
    return ffcomplex(z.re, ffneg(z.im));
}

// ============================================================
// Complex square root
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex sqrt(ffcomplex z) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return ffcomplex();
    ffloat r  = sqrt(ffadd(ffmul(z.re, z.re), ffmul(z.im, z.im)));
    ffloat a1 = abs(z.re);
    ffloat s2 = ffmulf(ffadd(r, a1), 0.5f);
    ffloat s0 = sqrt(s2);
    ffloat s1 = ffmulf(s0, 2.0f);
    ffcomplex b;
    if (z.re.hi >= 0.0f) {
        b.re = s0;
        b.im = ffdiv(z.im, s1);
    } else {
        b.re = ffdiv(z.im, s1);
        if (b.re.hi < 0.0f) b.re = ffneg(b.re);
        b.im = s0;
        if (z.im.hi < 0.0f) b.im = ffneg(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex exp(ffcomplex z) {
    ffloat er = exp(z.re);
    ffloat c, s;
    sincos(z.im, c, s);
    return ffcomplex(ffmul(er, c), ffmul(er, s));
}

KOKKOS_INLINE_FUNCTION ffcomplex log(ffcomplex z) {
    ffloat modulus = abs(z);
    ffloat arg     = atan2(z.im, z.re);
    return ffcomplex(log(modulus), arg);
}

KOKKOS_INLINE_FUNCTION ffcomplex log10(ffcomplex z) {
    ffcomplex lg = log(z);
    ffloat ln10 = ff_log10();
    return ffcomplex(ffdiv(lg.re, ln10), ffdiv(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex sin(ffcomplex z) {
    ffloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return ffcomplex(ffmul(sa, cb), ffmul(ca, sb));
}
KOKKOS_INLINE_FUNCTION ffcomplex cos(ffcomplex z) {
    ffloat ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return ffcomplex(ffmul(ca, cb), ffneg(ffmul(sa, sb)));
}
KOKKOS_INLINE_FUNCTION ffcomplex tan(ffcomplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex asin(ffcomplex z) {
    ffcomplex iz  = ffcomplex(ffneg(z.im), z.re);
    ffcomplex z2  = z * z;
    ffcomplex one_minus_z2 = ffcomplex(ffloat(1.0f)) - z2;
    ffcomplex sum = iz + sqrt(one_minus_z2);
    ffcomplex lg  = log(sum);
    return ffcomplex(lg.im, ffneg(lg.re));
}
KOKKOS_INLINE_FUNCTION ffcomplex acos(ffcomplex z) {
    ffloat pi_over_2 = ffmulf(ff_pi(), 0.5f);
    ffcomplex asin_z  = asin(z);
    return ffcomplex(ffsub(pi_over_2, asin_z.re), ffneg(asin_z.im));
}
KOKKOS_INLINE_FUNCTION ffcomplex atan(ffcomplex z) {
    ffcomplex iz    = ffcomplex(ffneg(z.im), z.re);
    ffcomplex num   = ffcomplex(ffloat(1.0f)) - iz;
    ffcomplex den   = ffcomplex(ffloat(1.0f)) + iz;
    ffcomplex ratio = num / den;
    ffcomplex lg    = log(ratio);
    return ffcomplex(ffmulf(ffneg(lg.im), 0.5f), ffmulf(lg.re, 0.5f));
}

// ============================================================
// Complex hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex sinh(ffcomplex z) {
    ffloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return ffcomplex(ffmul(sa, cb), ffmul(ca, sb));
}
KOKKOS_INLINE_FUNCTION ffcomplex cosh(ffcomplex z) {
    ffloat ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return ffcomplex(ffmul(ca, cb), ffmul(sa, sb));
}
KOKKOS_INLINE_FUNCTION ffcomplex tanh(ffcomplex z) {
    ffloat T = tanh(z.re);
    ffloat cb, sb;
    sincos(z.im, cb, sb);
    ffloat T2    = ffmul(T, T);
    ffloat denom = ffadd(ffmul(cb, cb), ffmul(T2, ffmul(sb, sb)));
    return ffcomplex(ffdiv(T, denom),
                     ffdiv(ffmul(ffmul(sb, cb), ffsub(ffloat(1.0f), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex asinh(ffcomplex z) {
    return log(z + sqrt(z*z + ffcomplex(ffloat(1.0f))));
}
KOKKOS_INLINE_FUNCTION ffcomplex acosh(ffcomplex z) {
    return log(z + sqrt(z*z - ffcomplex(ffloat(1.0f))));
}
KOKKOS_INLINE_FUNCTION ffcomplex atanh(ffcomplex z) {
    ffcomplex one = ffcomplex(ffloat(1.0f));
    ffcomplex lg  = log((one + z) / (one - z));
    return ffcomplex(ffmulf(lg.re, 0.5f), ffmulf(lg.im, 0.5f));
}

// ============================================================
// Complex power and polar
// ============================================================
KOKKOS_INLINE_FUNCTION ffcomplex pow(ffcomplex z, ffcomplex w) {
    if (z.re.hi == 0.0f && z.im.hi == 0.0f) return ffcomplex();
    return exp(w * log(z));
}

KOKKOS_INLINE_FUNCTION ffcomplex polar(ffloat r, ffloat theta) {
    ffloat c, s;
    sincos(theta, c, s);
    return ffcomplex(ffmul(r, c), ffmul(r, s));
}

} // namespace ffun
} // namespace quad
