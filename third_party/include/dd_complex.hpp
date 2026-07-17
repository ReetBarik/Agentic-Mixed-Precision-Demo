#pragma once
// Double-double complex arithmetic — namespace quad::ddfun
// All functions KOKKOS_INLINE_FUNCTION (host + device via Kokkos/CUDA).
// Depends on dd_math.hpp.

#include <dd_math.hpp>

#ifndef __CUDA_ARCH__
#  include <ostream>
#endif

namespace quad {
namespace ddfun {

// ============================================================
// ddcomplex struct
// ============================================================
struct ddcomplex {
    ddouble re;
    ddouble im;

    KOKKOS_INLINE_FUNCTION ddcomplex() : re(0.0), im(0.0) {}
    KOKKOS_INLINE_FUNCTION ddcomplex(double r)              : re(r),    im(0.0) {}
    KOKKOS_INLINE_FUNCTION ddcomplex(ddouble r)             : re(r),    im(0.0) {}
    KOKKOS_INLINE_FUNCTION ddcomplex(double r, double i)    : re(r),    im(i)   {}
    KOKKOS_INLINE_FUNCTION ddcomplex(ddouble r, ddouble i)  : re(r),    im(i)   {}
    KOKKOS_INLINE_FUNCTION ddcomplex(const ddcomplex& o)    : re(o.re), im(o.im){}
    KOKKOS_INLINE_FUNCTION ddcomplex& operator=(const ddcomplex& o) {
        re = o.re; im = o.im; return *this;
    }
    KOKKOS_INLINE_FUNCTION ddcomplex& operator=(ddouble r) {
        re = r; im = ddouble(0.0); return *this;
    }

    // Arithmetic
    KOKKOS_INLINE_FUNCTION ddcomplex operator+(ddcomplex b) const {
        return ddcomplex(ddadd(re, b.re), ddadd(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION ddcomplex operator-(ddcomplex b) const {
        return ddcomplex(ddsub(re, b.re), ddsub(im, b.im));
    }
    KOKKOS_INLINE_FUNCTION ddcomplex operator*(ddcomplex b) const {
        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        return ddcomplex(ddsub(ddmul(re, b.re), ddmul(im, b.im)),
                         ddadd(ddmul(re, b.im), ddmul(im, b.re)));
    }
    KOKKOS_INLINE_FUNCTION ddcomplex operator/(ddcomplex b) const {
        if (b.re.hi == 0.0 && b.im.hi == 0.0) {
            Kokkos::printf("DDCOMPLEX: division by zero\n");
            return ddcomplex();
        }
        // (a+bi)/(c+di) = [(ac+bd) + (bc-ad)i] / (c²+d²)
        ddouble denom = ddadd(ddmul(b.re, b.re), ddmul(b.im, b.im));
        ddouble inv   = dddiv(ddouble(1.0), denom);
        return ddcomplex(ddmul(ddadd(ddmul(re, b.re), ddmul(im, b.im)), inv),
                         ddmul(ddsub(ddmul(im, b.re), ddmul(re, b.im)), inv));
    }
    KOKKOS_INLINE_FUNCTION ddcomplex operator-() const {
        return ddcomplex(ddneg(re), ddneg(im));
    }

    KOKKOS_INLINE_FUNCTION ddcomplex& operator+=(ddcomplex b) { *this = *this + b; return *this; }
    KOKKOS_INLINE_FUNCTION ddcomplex& operator-=(ddcomplex b) { *this = *this - b; return *this; }
    KOKKOS_INLINE_FUNCTION ddcomplex& operator*=(ddcomplex b) { *this = *this * b; return *this; }
    KOKKOS_INLINE_FUNCTION ddcomplex& operator/=(ddcomplex b) { *this = *this / b; return *this; }

    KOKKOS_INLINE_FUNCTION bool operator==(ddcomplex b) const { return re==b.re && im==b.im; }
    KOKKOS_INLINE_FUNCTION bool operator!=(ddcomplex b) const { return !(*this == b); }

    KOKKOS_INLINE_FUNCTION ddouble real() const { return re; }
    KOKKOS_INLINE_FUNCTION ddouble imag() const { return im; }
};

#ifndef __CUDA_ARCH__
inline std::ostream& operator<<(std::ostream& os, const ddcomplex& z) {
    os << "(" << z.re << ") + (" << z.im << ")i";
    return os;
}
#endif

// ============================================================
// Mixed ddouble × ddcomplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex operator+(ddcomplex z, ddouble r) { return ddcomplex(ddadd(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION ddcomplex operator+(ddouble r, ddcomplex z) { return ddcomplex(ddadd(r, z.re), z.im); }
KOKKOS_INLINE_FUNCTION ddcomplex operator-(ddcomplex z, ddouble r) { return ddcomplex(ddsub(z.re, r), z.im); }
KOKKOS_INLINE_FUNCTION ddcomplex operator-(ddouble r, ddcomplex z) { return ddcomplex(ddsub(r, z.re), ddneg(z.im)); }
KOKKOS_INLINE_FUNCTION ddcomplex operator*(ddcomplex z, ddouble r) { return ddcomplex(ddmul(z.re, r), ddmul(z.im, r)); }
KOKKOS_INLINE_FUNCTION ddcomplex operator*(ddouble r, ddcomplex z) { return ddcomplex(ddmul(r, z.re), ddmul(r, z.im)); }
KOKKOS_INLINE_FUNCTION ddcomplex operator/(ddcomplex z, ddouble r) { return ddcomplex(dddiv(z.re, r), dddiv(z.im, r)); }
KOKKOS_INLINE_FUNCTION ddcomplex operator/(ddouble r, ddcomplex z) { return ddcomplex(r) / z; }

// ============================================================
// Mixed double × ddcomplex arithmetic
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex operator+(ddcomplex z, double b) { return z + ddouble(b); }
KOKKOS_INLINE_FUNCTION ddcomplex operator+(double b, ddcomplex z) { return ddouble(b) + z; }
KOKKOS_INLINE_FUNCTION ddcomplex operator-(ddcomplex z, double b) { return z - ddouble(b); }
KOKKOS_INLINE_FUNCTION ddcomplex operator-(double b, ddcomplex z) { return ddouble(b) - z; }
KOKKOS_INLINE_FUNCTION ddcomplex operator*(ddcomplex z, double b) { return z * ddouble(b); }
KOKKOS_INLINE_FUNCTION ddcomplex operator*(double b, ddcomplex z) { return ddouble(b) * z; }
KOKKOS_INLINE_FUNCTION ddcomplex operator/(ddcomplex z, double b) { return z / ddouble(b); }
KOKKOS_INLINE_FUNCTION ddcomplex operator/(double b, ddcomplex z) { return ddouble(b) / z; }

// ============================================================
// Basic complex operations
// ============================================================

KOKKOS_INLINE_FUNCTION ddouble abs(ddcomplex z) {
    return sqrt(ddadd(ddmul(z.re, z.re), ddmul(z.im, z.im)));
}
KOKKOS_INLINE_FUNCTION ddcomplex conj(ddcomplex z) {
    return ddcomplex(z.re, ddneg(z.im));
}

// ============================================================
// Complex square root
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex sqrt(ddcomplex z) {
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return ddcomplex();
    // B = sqrt((R+A1)/2) + i*sign(A2)*sqrt((R-A1)/2)  where R = |z|
    ddouble r  = sqrt(ddadd(ddmul(z.re, z.re), ddmul(z.im, z.im)));
    ddouble a1 = abs(z.re);
    ddouble s2 = ddmuld(ddadd(r, a1), 0.5);
    ddouble s0 = sqrt(s2);
    ddouble s1 = ddmuld(s0, 2.0);
    ddcomplex b;
    if (z.re.hi >= 0.0) {
        b.re = s0;
        b.im = dddiv(z.im, s1);
    } else {
        b.re = dddiv(z.im, s1);
        if (b.re.hi < 0.0) b.re = ddneg(b.re);
        b.im = s0;
        if (z.im.hi < 0.0) b.im = ddneg(b.im);
    }
    return b;
}

// ============================================================
// Complex exp / log
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex exp(ddcomplex z) {
    ddouble er = exp(z.re);
    ddouble c, s;
    sincos(z.im, c, s);
    return ddcomplex(ddmul(er, c), ddmul(er, s));
}

KOKKOS_INLINE_FUNCTION ddcomplex log(ddcomplex z) {
    ddouble modulus = abs(z);
    ddouble arg     = atan2(z.im, z.re); // atan2(im, re)
    return ddcomplex(log(modulus), arg);
}

KOKKOS_INLINE_FUNCTION ddcomplex log10(ddcomplex z) {
    ddcomplex lg = log(z);
    ddouble ln10 = dd_log10();
    return ddcomplex(dddiv(lg.re, ln10), dddiv(lg.im, ln10));
}

// ============================================================
// Complex trig
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex sin(ddcomplex z) {
    // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    ddouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return ddcomplex(ddmul(sa, cb), ddmul(ca, sb));
}
KOKKOS_INLINE_FUNCTION ddcomplex cos(ddcomplex z) {
    // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    ddouble ca, sa, cb, sb;
    sincos(z.re, ca, sa);
    sinhcosh(z.im, cb, sb);
    return ddcomplex(ddmul(ca, cb), ddneg(ddmul(sa, sb)));
}
KOKKOS_INLINE_FUNCTION ddcomplex tan(ddcomplex z) {
    return sin(z) / cos(z);
}

// ============================================================
// Complex inverse trig
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex asin(ddcomplex z) {
    // asin(z) = -i * log(iz + sqrt(1 - z^2))
    ddcomplex iz  = ddcomplex(ddneg(z.im), z.re);
    ddcomplex z2  = z * z;
    ddcomplex one_minus_z2 = ddcomplex(ddouble(1.0)) - z2;
    ddcomplex sum = iz + sqrt(one_minus_z2);
    ddcomplex lg  = log(sum);
    // multiply by -i: (a+bi)*(-i) = b - ai
    return ddcomplex(lg.im, ddneg(lg.re));
}
KOKKOS_INLINE_FUNCTION ddcomplex acos(ddcomplex z) {
    // acos(z) = pi/2 - asin(z)
    ddouble pi_over_2 = ddmuld(dd_pi(), 0.5);
    ddcomplex asin_z  = asin(z);
    return ddcomplex(ddsub(pi_over_2, asin_z.re), ddneg(asin_z.im));
}
KOKKOS_INLINE_FUNCTION ddcomplex atan(ddcomplex z) {
    // atan(z) = (i/2) * log((1-iz)/(1+iz))
    ddcomplex iz    = ddcomplex(ddneg(z.im), z.re);
    ddcomplex num   = ddcomplex(ddouble(1.0)) - iz;
    ddcomplex den   = ddcomplex(ddouble(1.0)) + iz;
    ddcomplex ratio = num / den;
    ddcomplex lg    = log(ratio);
    // multiply by i/2: (a+bi)*(i/2) = (-b/2) + (a/2)*i
    return ddcomplex(ddmuld(ddneg(lg.im), 0.5), ddmuld(lg.re, 0.5));
}

// ============================================================
// Complex hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex sinh(ddcomplex z) {
    // sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
    ddouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return ddcomplex(ddmul(sa, cb), ddmul(ca, sb));
}
KOKKOS_INLINE_FUNCTION ddcomplex cosh(ddcomplex z) {
    // cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
    ddouble ca, sa, cb, sb;
    sinhcosh(z.re, ca, sa);
    sincos(z.im, cb, sb);
    return ddcomplex(ddmul(ca, cb), ddmul(sa, sb));
}
KOKKOS_INLINE_FUNCTION ddcomplex tanh(ddcomplex z) {
    // tanh(a+bi): re = tanh(a) / (cos²(b) + tanh²(a)·sin²(b))
    //             im = sin(b)·cos(b)·(1 - tanh²(a)) / (cos²(b) + tanh²(a)·sin²(b))
    // Denominator ≥ 0 always; uses improved real tanh to avoid cancellation.
    ddouble T = tanh(z.re);
    ddouble cb, sb;
    sincos(z.im, cb, sb);
    ddouble T2    = ddmul(T, T);
    ddouble denom = ddadd(ddmul(cb, cb), ddmul(T2, ddmul(sb, sb)));
    return ddcomplex(dddiv(T, denom),
                     dddiv(ddmul(ddmul(sb, cb), ddsub(ddouble(1.0), T2)), denom));
}

// ============================================================
// Complex inverse hyperbolic
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex asinh(ddcomplex z) {
    // asinh(z) = log(z + sqrt(z^2 + 1))
    return log(z + sqrt(z*z + ddcomplex(ddouble(1.0))));
}
KOKKOS_INLINE_FUNCTION ddcomplex acosh(ddcomplex z) {
    // acosh(z) = log(z + sqrt(z^2 - 1))
    return log(z + sqrt(z*z - ddcomplex(ddouble(1.0))));
}
KOKKOS_INLINE_FUNCTION ddcomplex atanh(ddcomplex z) {
    // atanh(z) = (1/2)*log((1+z)/(1-z))
    ddcomplex one = ddcomplex(ddouble(1.0));
    ddcomplex lg  = log((one + z) / (one - z));
    return ddcomplex(ddmuld(lg.re, 0.5), ddmuld(lg.im, 0.5));
}

// ============================================================
// Complex power and polar
// ============================================================
KOKKOS_INLINE_FUNCTION ddcomplex pow(ddcomplex z, ddcomplex w) {
    // z^w = exp(w * log(z))
    if (z.re.hi == 0.0 && z.im.hi == 0.0) return ddcomplex();
    return exp(w * log(z));
}

// polar(r, theta) = r * (cos(theta) + i*sin(theta))
KOKKOS_INLINE_FUNCTION ddcomplex polar(ddouble r, ddouble theta) {
    ddouble c, s;
    sincos(theta, c, s);
    return ddcomplex(ddmul(r, c), ddmul(r, s));
}

} // namespace ddfun
} // namespace quad
