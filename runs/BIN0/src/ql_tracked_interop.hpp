// ql_tracked_interop.hpp
//
// Tracked interop shim for QCDLoop+Kokkos box integrals.
// Makes ql::BO callable with T = tracked::Tracked<double> and
// TOutput = tracked::Complex<double>.
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// Include order (per driver): this header FIRST, before kokkosMaths.h /
// kokkosUtils.h / boxGPU.h. The library's own templates call
// ql::Real/Imag/Sign/kAbs/kLog/... via *qualified* names inside their
// template bodies, so our tracked overloads must be visible at the
// definition point (ADL does not apply to qualified calls).

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <type_traits>

// -----------------------------------------------------------------------------
// Rule C3: supply operators the library uses on tracked values that Tracked
// does not define. The library performs unary `+` implicitly in various
// expressions (e.g. `TOutput(+p3sq + m3sq - m4sq)`). Identity op — no
// journal record.
// -----------------------------------------------------------------------------
namespace tracked {

// Rule C3: unary operator+ identity on Tracked<T> (added in Tracked namespace
// for ADL). No rounding, no journal record.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// Rule C3: unary operator+ identity on Complex<T>. Same rationale.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// -----------------------------------------------------------------------------
// Rule C5: forward-declare ql::Constants primary template so our partial
// specialization on tracked::Tracked<T> parses before kokkosMaths.h defines
// the primary. The library later provides the primary in the same TU.
// -----------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;

    // Also forward-declare the free helpers we'll overload below, so the
    // library's qualified `ql::Foo(...)` calls see our overloads at
    // template-definition time.
    template <typename TOutput, typename TMass, typename TScale> KOKKOS_INLINE_FUNCTION bool iszero(TScale const&);
}

// -----------------------------------------------------------------------------
// Rule C5 + Rule 5: partial specialization of ql::Constants for the tracked
// scalar. Every named leaf routes through tracked::constant("<name>", ...)
// so the constant's identifier survives in the journal.
// Rule 8 + C4: driver dispatches from a plain host loop -> no KOKKOS_INLINE_FUNCTION.
// -----------------------------------------------------------------------------
namespace ql {

template <class U>
struct Constants< ::tracked::Tracked<U> > {
    using T = ::tracked::Tracked<U>;

    // ---- Chebyshev coefficients for ddilog (Rule 5: named constants) --------
    static constexpr int _num_C() { return 19; }
    static T _C(int i) {
        // Rule 5: each Chebyshev coefficient is a named constant.
        constexpr double coeffs[19] = {
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
        return ::tracked::constant<U>(std::string("C_cheb_") + std::to_string(i), U(coeffs[i]));
    }

    // ---- Bernoulli coefficients for li2series (Rule 5) ----------------------
    static constexpr int _num_B() { return 25; }
    static T _B(int i) {
        // Rule 5: each Bernoulli coefficient is a named constant.
        constexpr double coeffs[25] = {
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
        return ::tracked::constant<U>(std::string("B_bern_") + std::to_string(i), U(coeffs[i]));
    }

    // Rule 5: onshell cutoff — named constant.
    template <typename TOutput, typename TMass, typename TScale>
    static T _qlonshellcutoff() {
        return ::tracked::constant<U>("qlonshellcutoff", U(1e-10));
    }

    // Rule 5: pi — named constant.
    static T _pi()   { return ::tracked::constant<U>("pi",   U(M_PI)); }
    // Rule 5: pi^2 — derived from named pi, so multiply through tracked ops
    // to preserve the derivation in the journal.
    static T _pi2()  { auto p = _pi(); return p * p; }

    template <typename TOutput, typename TMass, typename TScale>
    static T _pio3()  { return _pi() / ::tracked::constant<U>("three", U(3)); }  // Rule 5

    template <typename TOutput, typename TMass, typename TScale>
    static T _pio6()  { return _pi() / ::tracked::constant<U>("six",   U(6)); }  // Rule 5

    template <typename TOutput, typename TMass, typename TScale>
    static T _pi2o3() { return _pi() * _pio3<TOutput,TMass,TScale>(); }          // Rule 5 (derived)

    template <typename TOutput, typename TMass, typename TScale>
    static T _pi2o6() { return _pi() * _pio6<TOutput,TMass,TScale>(); }          // Rule 5 (derived)

    template <typename TOutput, typename TMass, typename TScale>
    static T _pi2o12(){ return _pi2() / ::tracked::constant<U>("twelve", U(12)); } // Rule 5

    // Rule 5: small integer / half constants (all named).
    static T _zero()  { return ::tracked::constant<U>("zero",  U(0.0)); }
    static T _half()  { return ::tracked::constant<U>("half",  U(0.5)); }
    static T _one()   { return ::tracked::constant<U>("one",   U(1.0)); }
    static T _two()   { return ::tracked::constant<U>("two",   U(2.0)); }
    static T _three() { return ::tracked::constant<U>("three", U(3.0)); }
    static T _four()  { return ::tracked::constant<U>("four",  U(4.0)); }
    static T _five()  { return ::tracked::constant<U>("five",  U(5.0)); }
    static T _six()   { return ::tracked::constant<U>("six",   U(6.0)); }
    static T _ten()   { return ::tracked::constant<U>("ten",   U(10.0)); }

    // Rule 5: epsilon / tolerance constants (all named).
    static T _eps()     { return ::tracked::constant<U>("eps",     U(1e-6));  }
    static T _eps4()    { return ::tracked::constant<U>("eps4",    U(1e-4));  }
    static T _eps7()    { return ::tracked::constant<U>("eps7",    U(1e-7));  }
    static T _eps10()   { return ::tracked::constant<U>("eps10",   U(1e-10)); }
    static T _eps14()   { return ::tracked::constant<U>("eps14",   U(1e-14)); }
    static T _eps15()   { return ::tracked::constant<U>("eps15",   U(1e-15)); }
    static T _xloss()   { return ::tracked::constant<U>("xloss",   U(0.125)); }
    static T _neglig()  { return ::tracked::constant<U>("neglig",  U(1e-14)); }
    static T _reps()    { return ::tracked::constant<U>("reps",    U(1e-16)); }

    // Rule 3 + Rule 5: complex-valued named constants. Container is
    // Complex<Tracked<T>>, NOT Tracked<Complex<T>> (per C1).
    // TOutput here is Complex<U> (Complex<Tracked<U>>::Complex is spelled
    // ql::complex under normal types; with tracked it is
    // tracked::Complex<U>). We return TOutput built from tracked reals.
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi() {
        // Rule 5: pi is named; the 0 real part is a structural literal.
        auto zero_r = ::tracked::literal<U>(U(0));
        auto two_pi = ::tracked::constant<U>("two", U(2)) *
                      ::tracked::constant<U>("pi",  U(M_PI));
        return TOutput(zero_r, two_pi);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2() {
        auto zero_r = ::tracked::literal<U>(U(0));
        auto pio2   = ::tracked::constant<U>("pi", U(M_PI)) *
                      ::tracked::constant<U>("half", U(0.5));
        return TOutput(zero_r, pio2);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi() {
        auto zero_r = ::tracked::literal<U>(U(0));
        auto pi_c   = ::tracked::constant<U>("pi", U(M_PI));
        return TOutput(zero_r, pi_c);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps() {
        auto zero_r = ::tracked::literal<U>(U(0));
        auto reps_c = ::tracked::constant<U>("reps", U(1e-16));
        return TOutput(zero_r, reps_c);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2() {
        auto zero_r = ::tracked::literal<U>(U(0));
        auto reps_c = ::tracked::constant<U>("reps", U(1e-16));
        return TOutput(zero_r, reps_c * reps_c);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() {
        auto zero_r = ::tracked::literal<U>(U(0));
        auto e50    = ::tracked::constant<U>("ieps50", U(1e-50));
        return TOutput(zero_r, e50);
    }
};

} // namespace ql

// -----------------------------------------------------------------------------
// Rule C7 + Rule 8/C4: partial-ordering overloads of the library's own
// free-function templates for tracked value types. Each overload carries the
// original leading explicit template parameters (TOutput, TMass, TScale) so
// that qualified calls like ql::kAbs<TOutput,TMass,TScale>(x) bind here, and
// also to outrank the library's generic primary under partial ordering when
// called unqualified. No KOKKOS_INLINE_FUNCTION (Rule C4: host-only tracked
// ops, driver dispatch is a plain host loop).
// -----------------------------------------------------------------------------
namespace ql {

// ---- Real / Imag / Sign / kAbs / kLog / kSqrt / kConj on tracked scalar ----

// Rule 2 + C7: Real(Tracked<T>) is a tracked real (identity, floating point).
template <class T>
inline ::tracked::Tracked<T> Real(const ::tracked::Tracked<T>& x) {
    return x;
}

// Rule 2 + C7: Imag(Tracked<T>) is zero as a tracked scalar (participates in
// downstream floating-point arithmetic). Anonymous literal per Rule 6.
template <class T>
inline ::tracked::Tracked<T> Imag(const ::tracked::Tracked<T>& /*x*/) {
    return ::tracked::literal<T>(T(0));
}

// Rule 2 + C7 + C6: Real / Imag on tracked complex — floating-point returns.
template <class T>
inline ::tracked::Tracked<T> Real(const ::tracked::Complex<T>& z) {
    return z.real();
}
template <class T>
inline ::tracked::Tracked<T> Imag(const ::tracked::Complex<T>& z) {
    return z.imag();
}

// Rule C6: Sign on a tracked scalar. Consumed both as a floating-point
// multiplier (e.g. `TOutput(ql::Sign(ql::Real(k12))) * sqrt(...)`) AND as a
// count that is fed back into tracked expressions. Per C6 rule: use in
// floating-point arithmetic dominates -> return tracked scalar.
template <class T>
inline ::tracked::Tracked<T> Sign(const ::tracked::Tracked<T>& x) {
    // Rule 7: comparison unwrapped through .value() to a plain bool.
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    // Rule 6: the sign value is an anonymous literal (its numeric identity
    // varies at runtime; naming it would collide across ±1/0 cases).
    return ::tracked::literal<T>(s);
}

// Rule 3 + C7: Sign on tracked complex is z / |z| — container returned is
// Complex<Tracked<T>> per C1.
template <class T>
inline ::tracked::Complex<T> Sign(const ::tracked::Complex<T>& z) {
    auto a = ::tracked::abs(z);
    return z / ::tracked::Complex<T>(a);
}

// Rule 2 + C7: kAbs on tracked scalar returns tracked scalar (calls
// tracked::abs from ops.hpp).
template <class T>
inline ::tracked::Tracked<T> kAbs(const ::tracked::Tracked<T>& x) {
    return ::tracked::abs(x);
}

// Rule 2 + C7: kAbs on tracked complex. Per the library, kAbs(Kokkos::complex)
// returns a real scalar (the magnitude), so we return Tracked<T>.
template <class T>
inline ::tracked::Tracked<T> kAbs(const ::tracked::Complex<T>& z) {
    return ::tracked::abs(z);
}

// Rule 2 + C7: kLog on tracked scalar -> tracked scalar.
template <class T>
inline ::tracked::Tracked<T> kLog(const ::tracked::Tracked<T>& x) {
    return ::tracked::log(x);
}

// Rule 3 + C7: kLog on tracked complex -> tracked complex.
template <class T>
inline ::tracked::Complex<T> kLog(const ::tracked::Complex<T>& z) {
    return ::tracked::log(z);
}

// Rule 2 + C7: kSqrt on tracked scalar.
template <class T>
inline ::tracked::Tracked<T> kSqrt(const ::tracked::Tracked<T>& x) {
    return ::tracked::sqrt(x);
}

// Rule 3 + C7: kSqrt on tracked complex.
template <class T>
inline ::tracked::Complex<T> kSqrt(const ::tracked::Complex<T>& z) {
    return ::tracked::sqrt(z);
}

// Rule 3 + C7: kConj on tracked complex.
template <class T>
inline ::tracked::Complex<T> kConj(const ::tracked::Complex<T>& z) {
    return ::tracked::conj(z);
}

// Rule 2 + C7: kConj on tracked scalar is identity (real numbers are their
// own conjugate).
template <class T>
inline ::tracked::Tracked<T> kConj(const ::tracked::Tracked<T>& x) {
    return x;
}

// ---- kPow: the library's own template is generic on TOutput/TMass. It only
// uses TOutput(1.0) and operator*/operator/, both defined for tracked types,
// so the primary works as-is. No overload needed. (Rule 9 avoided: the
// primary is provably safe here — element type is deduced, no int<->tracked
// conversion, ctor from double is available.) ---------------------------------

// ---- iszero: Rule 1 (bool return) — unwrap via .value() -----------------------

// Rule 1 + C7: iszero on tracked scalar. Consumed only as an `if` predicate
// in the library, so return raw bool. The threshold is a named constant
// (Rule 5) — but the comparison itself is Rule 7 (unwrap to bool).
template <typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const ::tracked::Tracked<T>& x) {
    // Rule 7: comparison on tracked values yields bool via .value().
    T cutoff = T(1e-10); // matches ql::Constants<TScale>::_qlonshellcutoff numeric
    using std::abs;
    return abs(x.value()) < cutoff;
}

// Rule 1 + C7: iszero on tracked complex (via magnitude). Consumed as bool.
template <typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const ::tracked::Complex<T>& z) {
    T cutoff = T(1e-10);
    using std::abs;
    // Rule 7: compare magnitude's .value() to threshold, return bool.
    return ::tracked::abs(z).value() < cutoff;
}

// ---- Max / Min: Rule 2 + C7. The library's Max returns the argument whose
// |.| is larger — a floating-point return participating downstream. --------

// Rule 2 + C7: Max on tracked scalars. Comparison via .value() (Rule 7).
template <class T>
inline ::tracked::Tracked<T> Max(const ::tracked::Tracked<T>& a,
                                 const ::tracked::Tracked<T>& b) {
    using std::abs;
    // Rule 7: comparison unwrapped to bool.
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

// Rule 3 + C7: Max on tracked complex.
template <class T>
inline ::tracked::Complex<T> Max(const ::tracked::Complex<T>& a,
                                 const ::tracked::Complex<T>& b) {
    // Rule 7: compare magnitudes via .value().
    auto am = ::tracked::abs(a).value();
    auto bm = ::tracked::abs(b).value();
    return (am > bm) ? a : b;
}

// Rule 2 + C7: Min on tracked scalars.
template <class T>
inline ::tracked::Tracked<T> Min(const ::tracked::Tracked<T>& a,
                                 const ::tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

// Rule 3 + C7: Min on tracked complex.
template <class T>
inline ::tracked::Complex<T> Min(const ::tracked::Complex<T>& a,
                                 const ::tracked::Complex<T>& b) {
    auto am = ::tracked::abs(a).value();
    auto bm = ::tracked::abs(b).value();
    return (am > bm) ? b : a;
}

// Rule 2 + C7: Htheta on tracked scalar (returns 0.5*(1+sign)). Consumed as
// a floating-point weight, so return tracked scalar.
template <class T>
inline ::tracked::Tracked<T> Htheta(const ::tracked::Tracked<T>& x) {
    auto half = ::tracked::constant<T>("half", T(0.5)); // Rule 5
    auto one  = ::tracked::constant<T>("one",  T(1.0)); // Rule 5
    return half * (one + Sign(x));
}

} // namespace ql

// -----------------------------------------------------------------------------
// Rule 9 (escape hatch): NOT triggered. All library entry points reached by
// the driver's BIN0 path (ql::BO -> ql::B0m -> B1..B5/BIN0, plus the entire
// kokkosUtils.h helper suite instantiated in the static call graph) route
// through: (a) ql::Constants<Tracked<double>> (specialized above per Rule 5
// / C5), (b) ql::Real/Imag/Sign/kAbs/kLog/kSqrt/kConj/Max/Min/Htheta/iszero
// on tracked values (overloaded above per Rule C7), and (c) tracked-native
// operator+/- / */ / unary + (C3). Comparisons in the library are always
// consumed as bool (`if`, `?:`) so Rule 7 unwrapping in iszero and in
// Sign/Max/Min is sufficient; no tracked bool ever escapes. Complex
// container spelling follows C1 (tracked::Complex<T>, not
// tracked::Complex<tracked::Tracked<T>>).
// -----------------------------------------------------------------------------