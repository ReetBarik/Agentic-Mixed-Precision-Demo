// ql_tracked_interop.hpp
// Tracked-datatype interop shim for QCDLoop+Kokkos box integrals.
//
// This header must be included BEFORE any qcdloop header so that:
//   (a) our forward declaration of ql::Constants<T> is in scope before
//       the primary template is defined in kokkosMaths.h, and
//   (b) our tracked-typed overloads of ql::Real / ql::Imag / ql::Sign /
//       ql::kAbs / ql::kLog / ql::kSqrt / ql::kConj / ql::kPow / ql::Max /
//       ql::Min / ql::iszero / ql::Htheta are candidates at every qualified
//       call site inside the qcdloop template bodies (C7 partial ordering).
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f

#pragma once

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cmath>
#include <type_traits>

// ---------------------------------------------------------------------------
// C5: Forward declaration of the primary template ql::Constants<T> BEFORE
// this shim's partial specialization on tracked::Tracked<T>. The primary
// definition lives in kokkosMaths.h which is included AFTER this shim.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
}

// ---------------------------------------------------------------------------
// Tracked-namespace additions (C3): the qcdloop templates apply certain
// operators / free-function names to tracked values that the Tracked API
// itself does not define. Add them in namespace tracked so ADL finds them,
// or as free functions in the global-visible location required by the call
// site. None of these introduce rounding — they are identity/dispatch glue.
// ---------------------------------------------------------------------------
namespace tracked {

// C3: unary operator+ on Tracked<T> — identity, no journal record.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// C3: unary operator+ on Complex<T> — identity, no journal record.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

namespace ql {

// ---------------------------------------------------------------------------
// C5 / Rule 5: Partial specialization of ql::Constants<T> keyed on the
// tracked scalar. Every named leaf constant the library primary exposes is
// routed through tracked::constant("<name>", T(value)) so the journal
// preserves the source identifier. Integer-count helpers (_num_C, _num_B)
// remain plain ints — they are discrete (Rule 1 / C6).
// ---------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tr = tracked::Tracked<T>;

    // Discrete counts — Rule 1: return raw int.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Chebyshev coefficient table — Rule 5: named constant per index.
    static Tr _C(int i) {
        static const double coeffs[19] = {
            0.4299669356081370,
            0.4097598753307711,
            -0.0185884366501460,
            0.0014575108406227,
            -0.0001430418444234,
            0.0000158841554188,
            -0.0000019078495939,
            0.0000002419518085,
            -0.0000000319334127,
            0.0000000043454506,
            -0.0000000006057848,
            0.0000000000861210,
            -0.0000000000124433,
            0.0000000000018226,
            -0.0000000000002701,
            0.0000000000000404,
            -0.0000000000000061,
            0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // Bernoulli coefficient table — Rule 5.
    static Tr _B(int i) {
        static const double coeffs[25] = {
            0.02777777777777777777777777777777777777777778774E0,
            -0.000277777777777777777777777777777777777777777778E0,
            4.72411186696900982615268329554043839758125472E-6,
            -9.18577307466196355085243974132863021751910641E-8,
            1.89788699889709990720091730192740293750394761E-9,
            -4.06476164514422552680590938629196667454705711E-11,
            8.92169102045645255521798731675274885151428361E-13,
            -1.993929586072107568723644347793789705630694749E-14,
            4.51898002961991819165047655285559322839681901E-16,
            -1.035651761218124701448341154221865666596091238E-17,
            2.39521862102618674574028374300098038167894899E-19,
            -5.58178587432500933628307450562541990556705462E-21,
            1.309150755418321285812307399186592301749849833E-22,
            -3.087419802426740293242279764866462431595565203E-24,
            7.31597565270220342035790560925214859103339899E-26,
            -1.740845657234000740989055147759702545340841422E-27,
            4.15763564461389971961789962077522667348825413E-29,
            -9.96214848828462210319400670245583884985485196E-31,
            2.394034424896165300521167987893749562934279156E-32,
            -5.76834735536739008429179316187765424407233225E-34,
            1.393179479647007977827886603911548331732410612E-35,
            -3.372121965485089470468473635254930958979742891E-37,
            8.17820877756210262176477721487283426787618937E-39,
            -1.987010831152385925564820669234786567541858996E-40,
            4.83577851804055089628705937311537820769430091E-42
        };
        return tracked::constant<T>(std::string("B[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // Rule 5: named constant "qlonshellcutoff".
    template <class TOutput, class TMass, class TScale>
    static Tr _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    // Rule 5: named-constant scalar leaves.
    static Tr _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }
    static Tr _pi2()  { auto p = _pi(); return p * p; }

    template <class TOutput, class TMass, class TScale>
    static Tr _pio3() { return _pi() / tracked::constant<T>("three", T(3)); }

    template <class TOutput, class TMass, class TScale>
    static Tr _pio6() { return _pi() / tracked::constant<T>("six", T(6)); }

    template <class TOutput, class TMass, class TScale>
    static Tr _pi2o3() { return _pi() * _pio3<TOutput, TMass, TScale>(); }

    template <class TOutput, class TMass, class TScale>
    static Tr _pi2o6() { return _pi() * _pio6<TOutput, TMass, TScale>(); }

    template <class TOutput, class TMass, class TScale>
    static Tr _pi2o12() { return _pi2() / tracked::constant<T>("twelve", T(12)); }

    static Tr _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
    static Tr _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static Tr _one()   { return tracked::constant<T>("one",   T(1.0)); }
    static Tr _two()   { return tracked::constant<T>("two",   T(2.0)); }
    static Tr _three() { return tracked::constant<T>("three", T(3.0)); }
    static Tr _four()  { return tracked::constant<T>("four",  T(4.0)); }
    static Tr _five()  { return tracked::constant<T>("five",  T(5.0)); }
    static Tr _six()   { return tracked::constant<T>("six",   T(6.0)); }
    static Tr _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

    static Tr _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
    static Tr _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
    static Tr _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
    static Tr _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static Tr _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static Tr _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static Tr _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static Tr _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static Tr _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Rule 3: complex-valued named constants produce Complex<Tracked<T>>.
    // TOutput here IS tracked::Complex<T> at instantiation.
    template <class TOutput, class TMass, class TScale>
    static TOutput _2ipi() {
        auto zero = tracked::constant<T>("zero", T(0));
        auto two_pi = tracked::constant<T>("two_pi", T(2.0 * M_PI));
        return TOutput(zero, two_pi);
    }

    template <class TOutput, class TMass, class TScale>
    static TOutput _ipio2() {
        auto zero = tracked::constant<T>("zero", T(0));
        auto pio2 = tracked::constant<T>("pi_over_2", T(M_PI * 0.5));
        return TOutput(zero, pio2);
    }

    template <class TOutput, class TMass, class TScale>
    static TOutput _ipi() {
        auto zero = tracked::constant<T>("zero", T(0));
        auto pi   = tracked::constant<T>("pi",   T(M_PI));
        return TOutput(zero, pi);
    }

    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps() {
        auto zero = tracked::constant<T>("zero", T(0));
        auto reps = tracked::constant<T>("reps", T(1e-16));
        return TOutput(zero, reps);
    }

    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps2() {
        auto zero = tracked::constant<T>("zero", T(0));
        auto reps2 = tracked::constant<T>("reps2", T(1e-16 * 1e-16));
        return TOutput(zero, reps2);
    }

    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps50() {
        auto zero  = tracked::constant<T>("zero",  T(0));
        auto e50   = tracked::constant<T>("eps50", T(1e-50));
        return TOutput(zero, e50);
    }
};

// ---------------------------------------------------------------------------
// C7 / Rule 2: Real / Imag / Sign — the library primary declares these for
// double and Kokkos::complex<double>. Our overloads must be strictly more
// specialized (concrete value parameter) than any deduced-template variant
// the library or Kokkos may inject.
// ---------------------------------------------------------------------------

// Rule 2: Real(Tracked) participates in downstream FP arithmetic — return tracked.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}

// C6 / Rule 2: Imag(Tracked) — scalar real has zero imaginary part, but the
// return flows into tracked FP arithmetic; return a named tracked zero so
// provenance is preserved.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::constant<T>("zero", T(0));
}

// Rule 3: Real / Imag on tracked complex return the tracked real component.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
    return z.real();
}

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
    return z.imag();
}

// C6 / Rule 2: Sign(Tracked<T>) is used numerically inside tracked arithmetic
// (e.g. cLn(..., Sign(...))) as +/-1 or 0, and that value flows into tracked
// FP expressions and predicates. Return the tracked-scalar +1 / 0 / -1 so
// provenance is preserved. The comparison against 0 uses raw .value() per
// Rule 7 (never lift comparisons into tracked booleans).
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    if (v > T(0)) return tracked::literal<T>(T(1));
    if (v < T(0)) return tracked::literal<T>(T(-1));
    return tracked::literal<T>(T(0));
}

// Rule 3: Sign on tracked complex returns z / |z| — a tracked complex.
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    auto n = tracked::abs(z);
    return z / tracked::Complex<T>(n);
}

// ---------------------------------------------------------------------------
// C7 / Rule 2: kAbs — the library declares this both as a generic template
// (kAbs<T>(T)) and as concrete overloads. Provide overloads strictly more
// specialized than the generic template so qualified calls ql::kAbs(tr)
// bind here.
// ---------------------------------------------------------------------------

// Rule 2: kAbs(Tracked<T>) returns tracked; delegate to tracked::abs.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// Rule 3: kAbs(Complex<Tracked<T>>) returns tracked real |z|.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// Rule 2: kLog(Tracked) — tracked in, tracked out.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Rule 3: kLog(Complex<Tracked<T>>) — tracked complex log.
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// Rule 2: kSqrt(Tracked).
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// Rule 3: kSqrt(Complex<Tracked<T>>).
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 2: kConj on tracked scalar is identity.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// Rule 3: kConj on tracked complex.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// ---------------------------------------------------------------------------
// C7 / Rule 2: kPow — library has generic template kPow<TOutput,TMass,TScale>
// with concrete (TOutput,int) and (TMass,int) overloads. Provide constrained
// overloads carrying the same three leading explicit template parameters so
// qualified calls `ql::kPow<TOutput,TMass,TScale>(x, n)` bind to us; the
// leading parameters are unused in the body. Integer power via multiply loop
// (C2: Tracked API provides no tracked pow).
// ---------------------------------------------------------------------------

// Rule 2 / C7: tracked scalar base, integer exponent.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> acc = tracked::constant<T>("one", T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        auto one = tracked::constant<T>("one", T(1));
        return one / acc;
    }
    return acc;
}

// Rule 3 / C7: tracked complex base, integer exponent.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Use "one" real constant so multiplicative identity has provenance.
    tracked::Complex<T> acc(tracked::constant<T>("one", T(1)),
                             tracked::constant<T>("zero", T(0)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(tracked::constant<T>("one", T(1)),
                                   tracked::constant<T>("zero", T(0)));
        return one_c / acc;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// C7 / Rule 2/3: Max and Min. Library defines concrete overloads only for
// (double,double) and (Kokkos::complex<double>, Kokkos::complex<double>).
// Provide tracked overloads. Rule 7: compare .value() only.
// ---------------------------------------------------------------------------

template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    // Rule 7: comparison on scalar magnitudes uses raw doubles.
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    auto na = tracked::abs(a);
    auto nb = tracked::abs(b);
    return (na.value() > nb.value()) ? a : b;
}

template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    auto na = tracked::abs(a);
    auto nb = tracked::abs(b);
    return (na.value() > nb.value()) ? b : a;
}

// ---------------------------------------------------------------------------
// C7 / Rule 1: iszero. Library primary is
//   template <TOutput,TMass,TScale> bool iszero(TScale const& x);
// The result is used only as a boolean selector in `if (...)` — Rule 1
// (discrete return -> raw bool).
// ---------------------------------------------------------------------------

template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    // Rule 7: unwrap for comparison; use plain doubles for the cutoff check.
    using std::abs;
    // Cutoff value matches ql::Constants::_qlonshellcutoff (1e-10).
    return abs(x.value()) < T(1e-10);
}

// ---------------------------------------------------------------------------
// C6 / Rule 2: Htheta — Heaviside-style helper whose 0/1 result flows into
// tracked FP arithmetic in eta2/eta5/Rint. Return tracked scalar.
// ---------------------------------------------------------------------------
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    // 0.5 * (1 + sign(x)) — express as a literal to preserve provenance
    // structure. Rule 7 for comparison.
    T v = x.value();
    if (v > T(0)) return tracked::literal<T>(T(1));
    if (v < T(0)) return tracked::literal<T>(T(0));
    return tracked::literal<T>(T(0.5));
}

} // namespace ql