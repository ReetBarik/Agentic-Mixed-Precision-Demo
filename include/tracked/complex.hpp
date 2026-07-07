#pragma once
// tracked::Complex<T> — complex wrapper around two Tracked<T> reals.
//
// Every complex op decomposes into real tracked ops so each sub-op appears
// individually in the journal with its own condition number and source location.
// The journal schema is unchanged from v1 — Complex emits no records of its own.
//
// Usage:
//   #include <tracked/complex.hpp>
//
//   auto z = tracked::track("z", 1.0, 2.0);    // (1+2i) with provenance {"z"}
//   auto w = tracked::track("w", 3.0, 4.0);
//   auto r = z * w;                             // 4 muls, 1 sub, 1 add in journal
//   auto s = tracked::sqrt(z, TRACKED_HERE);    // all sub-ops attributed to this line

#include <tracked/ops.hpp>
#include <cmath>
#include <string>

namespace tracked {

// ---- Complex<T> -------------------------------------------------------------

template <class T>
class Complex {
public:
    Tracked<T> re_;
    Tracked<T> im_;

    Complex() = default;

    Complex(T re, T im = T(0))
        : re_(Tracked<T>(re)), im_(Tracked<T>(im)) {}

    Complex(Tracked<T> re, Tracked<T> im)
        : re_(std::move(re)), im_(std::move(im)) {}

    explicit Complex(Tracked<T> re)
        : re_(std::move(re)), im_(Tracked<T>(T(0))) {}

    const Tracked<T>& real() const { return re_; }
    const Tracked<T>& imag() const { return im_; }
    Tracked<T>&       real()       { return re_; }
    Tracked<T>&       imag()       { return im_; }

    // ---- Compound assignment -------------------------------------------------

    Complex& operator+=(const Complex& rhs) { *this = *this + rhs; return *this; }
    Complex& operator-=(const Complex& rhs) { *this = *this - rhs; return *this; }
    Complex& operator*=(const Complex& rhs) { *this = *this * rhs; return *this; }
    Complex& operator/=(const Complex& rhs) { *this = *this / rhs; return *this; }

    Complex& operator+=(const Tracked<T>& rhs) { *this = *this + Complex(rhs); return *this; }
    Complex& operator-=(const Tracked<T>& rhs) { *this = *this - Complex(rhs); return *this; }
    Complex& operator*=(const Tracked<T>& rhs) { *this = *this * Complex(rhs); return *this; }
    Complex& operator/=(const Tracked<T>& rhs) { *this = *this / Complex(rhs); return *this; }

    // ---- Comparisons (no ordering — complex is unordered) -------------------

    bool operator==(const Complex& rhs) const {
        return re_ == rhs.re_ && im_ == rhs.im_;
    }
    bool operator!=(const Complex& rhs) const { return !(*this == rhs); }
};

// ---- Factory ----------------------------------------------------------------

// Tag a fresh complex input. Both components share the same provenance id.
// For distinct ids per component, construct manually from two track() calls.
// Note: no default for im — avoids ambiguity with Tracked<T> track(id, value).
// For (re, 0i), call track("z", 1.0, 0.0) explicitly.
template <class T>
Complex<T> track(std::string id, T re, T im) {
    Tracked<T> re_t(re);  re_t.id_ = id;  re_t.prov_vars_ = {id};
    Tracked<T> im_t(im);  im_t.id_ = id;  im_t.prov_vars_ = {id};
    return Complex<T>(re_t, im_t);
}

// ---- Arithmetic (free functions) --------------------------------------------
//
// Use named locals, not temporaries, so every sub-op gets a stable source
// location and the journal can group them by call site.

template <class T>
Complex<T> operator+(const Complex<T>& a, const Complex<T>& b) {
    auto re = a.re_ + b.re_;
    auto im = a.im_ + b.im_;
    return Complex<T>(re, im);
}

template <class T>
Complex<T> operator-(const Complex<T>& a, const Complex<T>& b) {
    auto re = a.re_ - b.re_;
    auto im = a.im_ - b.im_;
    return Complex<T>(re, im);
}

template <class T>
Complex<T> operator*(const Complex<T>& a, const Complex<T>& b) {
    auto ac = a.re_ * b.re_;
    auto bd = a.im_ * b.im_;
    auto ad = a.re_ * b.im_;
    auto bc = a.im_ * b.re_;
    auto re = ac - bd;
    auto im = ad + bc;
    return Complex<T>(re, im);
}

// Division via the Smith (1962) algorithm — avoids overflow for large |b|.
//
// Naive (ac+bd, bc-ad)/(c²+d²) overflows when |c| or |d| ≈ 1e154 (double).
// Smith branches on |c| vs |d| so all intermediates stay bounded.
template <class T>
Complex<T> operator/(const Complex<T>& a, const Complex<T>& b) {
    using std::abs;
    if (abs(b.re_.value_) >= abs(b.im_.value_)) {
        // |c| >= |d|: r = d/c, den = c + r*d
        auto r   = b.im_ / b.re_;
        auto den = b.re_ + r * b.im_;
        auto re  = (a.re_ + a.im_ * r) / den;
        auto im  = (a.im_ - a.re_ * r) / den;
        return Complex<T>(re, im);
    } else {
        // |d| > |c|: r = c/d, den = d + r*c
        auto r   = b.re_ / b.im_;
        auto den = b.im_ + r * b.re_;
        auto re  = (a.re_ * r + a.im_) / den;
        auto im  = (a.im_ * r - a.re_) / den;
        return Complex<T>(re, im);
    }
}

template <class T>
Complex<T> operator-(const Complex<T>& a) {
    auto re = -a.re_;
    auto im = -a.im_;
    return Complex<T>(re, im);
}

// ---- Scalar overloads -------------------------------------------------------

template <class T>
Complex<T> operator+(const Complex<T>& a, const Tracked<T>& s) { return a + Complex<T>(s); }
template <class T>
Complex<T> operator+(const Tracked<T>& s, const Complex<T>& b) { return Complex<T>(s) + b; }
template <class T>
Complex<T> operator+(const Complex<T>& a, T s) { return a + Complex<T>(Tracked<T>(s)); }
template <class T>
Complex<T> operator+(T s, const Complex<T>& b) { return Complex<T>(Tracked<T>(s)) + b; }

template <class T>
Complex<T> operator-(const Complex<T>& a, const Tracked<T>& s) { return a - Complex<T>(s); }
template <class T>
Complex<T> operator-(const Tracked<T>& s, const Complex<T>& b) { return Complex<T>(s) - b; }
template <class T>
Complex<T> operator-(const Complex<T>& a, T s) { return a - Complex<T>(Tracked<T>(s)); }
template <class T>
Complex<T> operator-(T s, const Complex<T>& b) { return Complex<T>(Tracked<T>(s)) - b; }

template <class T>
Complex<T> operator*(const Complex<T>& a, const Tracked<T>& s) { return a * Complex<T>(s); }
template <class T>
Complex<T> operator*(const Tracked<T>& s, const Complex<T>& b) { return Complex<T>(s) * b; }
template <class T>
Complex<T> operator*(const Complex<T>& a, T s) { return a * Complex<T>(Tracked<T>(s)); }
template <class T>
Complex<T> operator*(T s, const Complex<T>& b) { return Complex<T>(Tracked<T>(s)) * b; }

template <class T>
Complex<T> operator/(const Complex<T>& a, const Tracked<T>& s) { return a / Complex<T>(s); }
template <class T>
Complex<T> operator/(const Tracked<T>& s, const Complex<T>& b) { return Complex<T>(s) / b; }
template <class T>
Complex<T> operator/(const Complex<T>& a, T s) { return a / Complex<T>(Tracked<T>(s)); }
template <class T>
Complex<T> operator/(T s, const Complex<T>& b) { return Complex<T>(Tracked<T>(s)) / b; }

// ---- Math functions ---------------------------------------------------------
//
// All decompose into named Tracked<T> ops. Free functions take an optional
// SourceLocation; pass TRACKED_HERE to attribute all sub-ops to that call site.

// norm(z) = re² + im²  (squared magnitude)
template <class T>
Tracked<T> norm(const Complex<T>& z, SourceLocation loc = {}) {
    auto re2 = mul(z.re_, z.re_, loc);
    auto im2 = mul(z.im_, z.im_, loc);
    return add(re2, im2, loc);
}

// abs(z) = sqrt(re² + im²).  Naive — may overflow for huge components.
template <class T>
Tracked<T> abs(const Complex<T>& z, SourceLocation loc = {}) {
    auto n = norm(z, loc);
    return tracked::sqrt(n, loc);
}

// arg(z) = atan2(im, re)
template <class T>
Tracked<T> arg(const Complex<T>& z, SourceLocation loc = {}) {
    return tracked::atan2(z.im_, z.re_, loc);
}

// conj(z) = (re, -im)
template <class T>
Complex<T> conj(const Complex<T>& z, SourceLocation loc = {}) {
    auto neg_im = neg(z.im_, loc);
    return Complex<T>(z.re_, neg_im);
}

// exp(z) = e^re · (cos(im) + i·sin(im))
// Decomposes into: 1 exp, 1 cos, 1 sin, 2 muls.
template <class T>
Complex<T> exp(const Complex<T>& z, SourceLocation loc = {}) {
    auto e    = tracked::exp(z.re_, loc);
    auto c    = tracked::cos(z.im_, loc);
    auto s    = tracked::sin(z.im_, loc);
    auto re   = mul(e, c, loc);
    auto im   = mul(e, s, loc);
    return Complex<T>(re, im);
}

// log(z) = (log|z|, arg(z))
// Branch cut on the negative real axis.
// Cond blows up near |z|=1 (log factor) and near the negative real axis (arg factor).
template <class T>
Complex<T> log(const Complex<T>& z, SourceLocation loc = {}) {
    auto r  = abs(z, loc);
    auto re = tracked::log(r, loc);
    auto im = arg(z, loc);
    return Complex<T>(re, im);
}

// sqrt(z) — cancellation-aware form that avoids the catastrophic case of
// sqrt(r)·(cos(arg/2) + i·sin(arg/2)) near the negative real axis.
//
// When re >= 0:   w = sqrt((r+re)/2),   result = (w, im/(2w))
// When re <  0:   w = sqrt((r-re)/2),   result = (|im|/(2w), sign(im)·w)
template <class T>
Complex<T> sqrt(const Complex<T>& z, SourceLocation loc = {}) {
    using std::abs; using std::sqrt;
    auto r  = abs(z, loc);

    if (z.re_.value_ >= T(0)) {
        // Stable branch for re >= 0
        auto sum  = add(r, z.re_, loc);
        auto half = Tracked<T>(T(0.5));
        auto hw   = mul(half, sum, loc);
        auto w    = tracked::sqrt(hw, loc);

        // im / (2w) — guard against w=0 (only at z=0 exactly)
        auto two  = Tracked<T>(T(2));
        auto tw   = mul(two, w, loc);
        auto im   = div(z.im_, tw, loc);
        return Complex<T>(w, im);
    } else {
        // Stable branch for re < 0
        auto diff  = sub(r, z.re_, loc);   // r - re = r + |re|, always positive
        auto half  = Tracked<T>(T(0.5));
        auto hd    = mul(half, diff, loc);
        auto w     = tracked::sqrt(hd, loc);

        // |im| / (2w)
        auto abs_im = tracked::abs(z.im_, loc);
        auto two    = Tracked<T>(T(2));
        auto tw     = mul(two, w, loc);
        auto re_out = div(abs_im, tw, loc);

        // sign(im) · w
        T sign_im = (z.im_.value_ >= T(0)) ? T(1) : T(-1);
        auto sign_t = Tracked<T>(sign_im);
        auto im_out = mul(sign_t, w, loc);

        return Complex<T>(re_out, im_out);
    }
}

// ---- Opaque barrier for Complex<T> -----------------------------------------
//
// Emits two real op="opaque" records sharing the same fn_name — one for the
// real component, one for the imaginary. Consumers identify a complex barrier
// by seeing two consecutive opaque records with the same fn_name.
//
// Both output components depend on both input components, so the optional
// tracked inputs are forwarded to both opaque calls.  This propagates input
// error bounds and provenance through the boundary.
template <class T, class... Inputs>
Complex<T> opaque(std::string_view fn_name, T re_val, T im_val,
                  const Inputs&... inputs) {
    auto re = tracked::opaque<T, Inputs...>(fn_name, re_val, inputs...);
    auto im = tracked::opaque<T, Inputs...>(fn_name, im_val, inputs...);
    return Complex<T>(re, im);
}

template <class T, class... Inputs>
Complex<T> opaque_at(std::string_view fn_name, T re_val, T im_val,
                     SourceLocation loc, const Inputs&... inputs) {
    auto re = tracked::opaque_at<T, Inputs...>(fn_name, re_val, loc, inputs...);
    auto im = tracked::opaque_at<T, Inputs...>(fn_name, im_val, loc, inputs...);
    return Complex<T>(re, im);
}

} // namespace tracked
