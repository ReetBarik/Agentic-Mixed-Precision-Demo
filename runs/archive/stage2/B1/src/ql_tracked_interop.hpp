// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop+Kokkos box integrals (B1 spike).
// Included BEFORE any qcdloop header so tracked overloads are visible
// at every qualified ql::… call site inside qcdloop's template bodies.
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <type_traits>

// ---------------------------------------------------------------------------
// C5: Forward-declare the primary class template ql::Constants so our
// partial specialization on tracked::Tracked<T> parses before qcdloop's
// own kokkosMaths.h defines the primary. kokkosMaths.h will supply the
// full primary later in the same TU; our specialization then coexists.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;

    // Forward declarations of free functions used by qcdloop and by our own
    // shim overloads (defined either here or by kokkosMaths.h later).
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x);
}

// ---------------------------------------------------------------------------
// C3: Missing operators / free functions on tracked::Tracked<T> and
// tracked::Complex<T> that qcdloop's template bodies reference via ADL
// or as qualified ql::… names.
//
// These live in namespace tracked so ADL finds them for tracked arguments,
// and are ALSO re-exported into namespace ql via using-declarations below
// so qualified ql::kAbs(t) etc. resolve to the tracked overload rather
// than falling through to kokkosMaths.h's generic templates.
// ---------------------------------------------------------------------------
namespace tracked {

// Unary operator+ identity (C3). No rounding, no journal record.
// Rule 2 / C3: identity on a tracked scalar returns the tracked scalar.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// Unary operator+ identity on complex (C3).
// Rule 3 / C3.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// ql:: shim overloads for tracked scalar / tracked complex arguments.
// These are the fixed set of primitive helpers qcdloop invokes on floating-
// point values (kAbs, kLog, kSqrt, kConj, kPow, Real, Imag, Sign, Max, Min,
// Htheta, iszero). Every overload is constrained to a concrete tracked
// argument (C7) so it strictly outranks any generic ql::… template of the
// same name that kokkosMaths.h defines.
// ---------------------------------------------------------------------------
namespace ql {

// ---- kAbs ------------------------------------------------------------------
// Rule 2: |x| on a real tracked scalar → tracked scalar (feeds fp math).
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}
// Rule 2: |z| on a tracked complex returns a tracked real scalar.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// ---- kLog ------------------------------------------------------------------
// Rule 2 (C1): log on tracked real → tracked real.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}
// Rule 3 (C1): log on tracked complex → tracked complex.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// ---- kSqrt -----------------------------------------------------------------
// Rule 2: sqrt on tracked real → tracked real.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}
// Rule 3: sqrt on tracked complex → tracked complex.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// ---- kConj -----------------------------------------------------------------
// Rule 2: conj on a real is identity — return input (no op recorded).
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}
// Rule 3: conj on tracked complex → tracked complex.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// ---- kPow (integer exponent) ----------------------------------------------
// C2: Tracked API provides no pow(); implement as a multiply loop over the
// tracked operator*. Result feeds fp math → Rule 2 / Rule 3.
// C7: leading explicit template params (TOutput, TMass, TScale) carried on
// each constrained overload so qualified `ql::kPow<TOutput,TMass,TScale>(x,n)`
// call sites bind here and outrank the library's generic primary.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: anonymous literal 1 promoted via tracked::literal.
    tracked::Tracked<T> acc = tracked::literal(T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::literal(T(1));
        return one / acc;
    }
    return acc;
}
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> acc(tracked::literal(T(1)), tracked::literal(T(0)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal(T(1)), tracked::literal(T(0)));
        return one / acc;
    }
    return acc;
}

// ---- Real / Imag -----------------------------------------------------------
// C6: Real/Imag of a floating value flow into downstream fp math (as
// TScale operands to Sign, iszero-relative comparisons, ipi factors, …),
// so they MUST return the tracked scalar (Rule 2), not raw double.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { return x; }

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Rule 6: anonymous zero literal.
    return tracked::literal(T(0));
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

// ---- Sign ------------------------------------------------------------------
// C6: qcdloop's Sign result is consumed in floating expressions
// (multiplied by ipi, added to condition flags used as fp weights, fed
// into ql::eta/etatilde returning TOutput). So Sign on a tracked scalar
// returns a tracked scalar (Rule 2), NOT raw int.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    // Rule 6: the sign is a runtime-selected ±1/0, not a named constant.
    return tracked::literal(s);
}
// C6/Rule 3: Sign on a tracked complex is z/|z| — a tracked complex.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    tracked::Tracked<T> mag = tracked::abs(z);
    return z / mag;
}

// ---- Max / Min -------------------------------------------------------------
// qcdloop's Max/Min pick by |.| and return the underlying value. Rule 2:
// select branch with a plain-bool comparison (Rule 7) on .value(), then
// return the chosen tracked scalar (no new op).
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}
// Rule 3: complex variants.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    T am = std::hypot(a.real().value(), a.imag().value());
    T bm = std::hypot(b.real().value(), b.imag().value());
    return (am > bm) ? a : b;
}
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    T am = std::hypot(a.real().value(), a.imag().value());
    T bm = std::hypot(b.real().value(), b.imag().value());
    return (am > bm) ? b : a;
}

// ---- Htheta ----------------------------------------------------------------
// C6: Htheta feeds fp expressions (multiplied by 2ipi in eta2). Rule 2:
// return tracked scalar.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    // 0.5 * (1 + sign(x))
    // Rule 5: 0.5 is a named "half" and 1 is a named "one" per the
    // Complex/sqrt convention in the Tracked API.
    tracked::Tracked<T> half = tracked::constant("half", T(0.5));
    tracked::Tracked<T> one  = tracked::constant("one",  T(1));
    // Rule 6: the sign itself is anonymous ±1/0.
    tracked::Tracked<T> sgn  = tracked::literal(s);
    return half * (one + sgn);
}

// ---- iszero ----------------------------------------------------------------
// Rule 1: iszero is consumed ONLY as a branch condition inside if(...)
// (see box/*.h dispatchers). Return raw bool by unwrapping .value().
// C7: constrained overload on the tracked argument outranks the library's
// generic primary for qualified `ql::iszero<TOutput,TMass,TScale>(x)` calls.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) {
    using std::abs;
    // Match the library's cutoff of 1e-10 (see Constants::_qlonshellcutoff).
    return abs(x.value()) < T(1e-10);
}

} // namespace ql

// ---------------------------------------------------------------------------
// C5 partial specialization of ql::Constants on tracked::Tracked<T>.
// Every named leaf scalar is wrapped via tracked::constant so the journal
// preserves its name (Rule 5). Non-fp leaves (_num_C, _num_B) stay raw
// int (Rule 1). _C and _B mirror the primary's Chebyshev/Bernoulli tables.
// ---------------------------------------------------------------------------
namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {
    using U = tracked::Tracked<T>;

    // Rule 1: table sizes are counts — raw int.
    KOKKOS_INLINE_FUNCTION static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION static constexpr int _num_B() { return 25; }

    // Rule 5: named Chebyshev coefficients — one constant name per index.
    KOKKOS_INLINE_FUNCTION static U _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant("Constants::_C", T(coeffs[i]));
    }

    // Rule 5: named Bernoulli coefficients.
    KOKKOS_INLINE_FUNCTION static U _B(int i) {
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
        return tracked::constant("Constants::_B", T(coeffs[i]));
    }

    // Rule 5: onshell cutoff.
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _qlonshellcutoff() {
        return tracked::constant("qlonshellcutoff", T(1e-10));
    }

    // Rule 5: pi, pi^2 and their common ratios.
    KOKKOS_INLINE_FUNCTION static U _pi()  { return tracked::constant("pi",  T(M_PI)); }
    KOKKOS_INLINE_FUNCTION static U _pi2() {
        // pi^2 built from the named "pi" so its provenance chains through pi.
        U pi = _pi();
        return pi * pi;
    }

    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _pio3()   { return _pi() / tracked::constant("three", T(3)); }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _pio6()   { return _pi() / tracked::constant("six",   T(6)); }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _pi2o3()  { return _pi() * _pio3<TOutput, TMass, TScale>(); }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _pi2o6()  { return _pi() * _pio6<TOutput, TMass, TScale>(); }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static U _pi2o12() { return _pi2() / tracked::constant("twelve", T(12)); }

    // Rule 5: small integers and named tolerances used across the library.
    KOKKOS_INLINE_FUNCTION static U _zero()  { return tracked::constant("zero",  T(0)); }
    KOKKOS_INLINE_FUNCTION static U _half()  { return tracked::constant("half",  T(0.5)); }
    KOKKOS_INLINE_FUNCTION static U _one()   { return tracked::constant("one",   T(1)); }
    KOKKOS_INLINE_FUNCTION static U _two()   { return tracked::constant("two",   T(2)); }
    KOKKOS_INLINE_FUNCTION static U _three() { return tracked::constant("three", T(3)); }
    KOKKOS_INLINE_FUNCTION static U _four()  { return tracked::constant("four",  T(4)); }
    KOKKOS_INLINE_FUNCTION static U _five()  { return tracked::constant("five",  T(5)); }
    KOKKOS_INLINE_FUNCTION static U _six()   { return tracked::constant("six",   T(6)); }
    KOKKOS_INLINE_FUNCTION static U _ten()   { return tracked::constant("ten",   T(10)); }

    KOKKOS_INLINE_FUNCTION static U _eps()    { return tracked::constant("eps",    T(1e-6)); }
    KOKKOS_INLINE_FUNCTION static U _eps4()   { return tracked::constant("eps4",   T(1e-4)); }
    KOKKOS_INLINE_FUNCTION static U _eps7()   { return tracked::constant("eps7",   T(1e-7)); }
    KOKKOS_INLINE_FUNCTION static U _eps10()  { return tracked::constant("eps10",  T(1e-10)); }
    KOKKOS_INLINE_FUNCTION static U _eps14()  { return tracked::constant("eps14",  T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static U _eps15()  { return tracked::constant("eps15",  T(1e-15)); }
    KOKKOS_INLINE_FUNCTION static U _xloss()  { return tracked::constant("xloss",  T(0.125)); }
    KOKKOS_INLINE_FUNCTION static U _neglig() { return tracked::constant("neglig", T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static U _reps()   { return tracked::constant("reps",   T(1e-16)); }

    // Rule 3 + Rule 5: complex-valued named constants. TOutput is
    // tracked::Complex<T>. Build the imaginary unit factors from named
    // real constants so provenance is preserved.
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _2ipi() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_two() * Constants<tracked::Tracked<T>>::_pi());
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ipio2() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_pi() * Constants<tracked::Tracked<T>>::_half());
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ipi() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_pi());
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_reps());
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps2() {
        U r = Constants<tracked::Tracked<T>>::_reps();
        return TOutput(Constants<tracked::Tracked<T>>::_zero(), r * r);
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps50() {
        // Rule 5: named ieps50 constant, wrapped as the imaginary component.
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       tracked::constant("ieps50", T(1e-50)));
    }
};

// ---------------------------------------------------------------------------
// C5 partial specialization of ql::Constants on tracked::Complex<T>.
// Only used through the TOutput-parameterized accessors (_2ipi, _ipio2,
// _ipi, _ieps, _ieps50, _one, _two, _four, _half, _three) — mirror them.
// ---------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Complex<T>> {
    using C = tracked::Complex<T>;
    using R = tracked::Tracked<T>;

    // Rule 3: real-valued named constants promoted into a complex.
    KOKKOS_INLINE_FUNCTION static C _zero()  { return C(tracked::constant("zero",  T(0))); }
    KOKKOS_INLINE_FUNCTION static C _half()  { return C(tracked::constant("half",  T(0.5))); }
    KOKKOS_INLINE_FUNCTION static C _one()   { return C(tracked::constant("one",   T(1))); }
    KOKKOS_INLINE_FUNCTION static C _two()   { return C(tracked::constant("two",   T(2))); }
    KOKKOS_INLINE_FUNCTION static C _three() { return C(tracked::constant("three", T(3))); }
    KOKKOS_INLINE_FUNCTION static C _four()  { return C(tracked::constant("four",  T(4))); }

    // Complex-valued factories (Rule 3 + Rule 5): imaginary axis constants.
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static C _2ipi() {
        R pi   = tracked::constant("pi",  T(M_PI));
        R two  = tracked::constant("two", T(2));
        return C(tracked::constant("zero", T(0)), two * pi);
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static C _ipio2() {
        R pi   = tracked::constant("pi",   T(M_PI));
        R half = tracked::constant("half", T(0.5));
        return C(tracked::constant("zero", T(0)), pi * half);
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static C _ipi() {
        return C(tracked::constant("zero", T(0)), tracked::constant("pi", T(M_PI)));
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static C _ieps() {
        return C(tracked::constant("zero", T(0)), tracked::constant("reps", T(1e-16)));
    }
    template <class TOutput, class TMass, class TScale>
    KOKKOS_INLINE_FUNCTION static C _ieps50() {
        return C(tracked::constant("zero", T(0)), tracked::constant("ieps50", T(1e-50)));
    }
};

} // namespace ql