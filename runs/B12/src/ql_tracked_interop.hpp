// ql_tracked_interop.hpp
//
// Tracked<T> interop shim for QCDLoop+Kokkos box integrals (B12 spike).
//
// Makes ql::BO and its callees instantiable with:
//   TScale  = tracked::Tracked<double>
//   TMass   = tracked::Tracked<double>
//   TOutput = tracked::Complex<double>            (per C1: NOT Tracked<Complex<T>>)
//
// The driver includes this header BEFORE qcdloop's own headers so every
// tracked overload is visible at each qcdloop template's definition point
// (qcdloop uses qualified `ql::` calls that do not participate in ADL).
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
#include <cstdint>
#include <type_traits>

// ---------------------------------------------------------------------------
// Rule C3: Identity unary operator+ on tracked scalar / complex.
// Some qcdloop templates apply unary + in generic scalar contexts. Emit an
// identity that introduces no journal record (a genuine no-op).
// Placed in ::tracked so ADL finds it for tracked::Tracked / tracked::Complex.
// ---------------------------------------------------------------------------
namespace tracked {

// Rule C3: identity unary + on tracked scalar (no rounding, no journal record).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// Rule C3: identity unary + on tracked complex.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// Rule C5: forward-declare qcdloop's Constants primary template inside its
// own namespace, so our partial specialization (below) parses before
// kokkosMaths.h defines the full primary. The driver includes this shim
// BEFORE kokkosMaths.h.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
}

// ---------------------------------------------------------------------------
// Rule 5 / C5: partial specialization of ql::Constants for tracked scalar.
// Every named leaf constant is routed through tracked::constant("<name>", v)
// so the journal preserves the symbolic name. The interface mirrors the
// primary in kokkosMaths.h; each accessor returns the tracked scalar so it
// composes with tracked arithmetic without any implicit conversion.
//
// Chebyshev (_C) and Bernoulli (_B) tables enter arithmetic as tracked
// values but are anonymous table entries — Rule 6 → tracked::literal.
// ---------------------------------------------------------------------------
namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tk = tracked::Tracked<T>;

    // ---- Chebyshev coeff count (Rule 1: pure integer) -------------------
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Rule 6: table entries are anonymous inline literals.
    static Tk _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::literal<T>(T(coeffs[i]));
    }

    // ---- Bernoulli coeff count (Rule 1: pure integer) -------------------
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 6: table entries are anonymous inline literals.
    static Tk _B(int i) {
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
        return tracked::literal<T>(T(coeffs[i]));
    }

    // Rule 5 / C5: every named scalar constant routed through
    // tracked::constant so the journal preserves the source identifier.

    template <typename TOutput, typename TMass, typename TScale>
    static Tk _qlonshellcutoff() { return tracked::constant<T>("qlonshellcutoff", T(1e-10)); }

    static Tk _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }
    static Tk _pi2()  { return tracked::constant<T>("pi2",  T(M_PI) * T(M_PI)); }

    template <typename TOutput, typename TMass, typename TScale>
    static Tk _pio3()   { return tracked::constant<T>("pio3",   T(M_PI) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tk _pio6()   { return tracked::constant<T>("pio6",   T(M_PI) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tk _pi2o3()  { return tracked::constant<T>("pi2o3",  (T(M_PI) * T(M_PI)) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tk _pi2o6()  { return tracked::constant<T>("pi2o6",  (T(M_PI) * T(M_PI)) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Tk _pi2o12() { return tracked::constant<T>("pi2o12", (T(M_PI) * T(M_PI)) / T(12)); }

    static Tk _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
    static Tk _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static Tk _one()   { return tracked::constant<T>("one",   T(1.0)); }
    static Tk _two()   { return tracked::constant<T>("two",   T(2.0)); }
    static Tk _three() { return tracked::constant<T>("three", T(3.0)); }
    static Tk _four()  { return tracked::constant<T>("four",  T(4.0)); }
    static Tk _five()  { return tracked::constant<T>("five",  T(5.0)); }
    static Tk _six()   { return tracked::constant<T>("six",   T(6.0)); }
    static Tk _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

    static Tk _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
    static Tk _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
    static Tk _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
    static Tk _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static Tk _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static Tk _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static Tk _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static Tk _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static Tk _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Rule 3: complex-valued named constants return tracked::Complex<T>
    // (a container of tracked reals, per C1). Each component is a named
    // tracked constant so both real and imag legs are attributable.
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("2pi",  T(2.0) * T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",  T(0)),
                                   tracked::constant<T>("pio2",  T(M_PI) * T(0.5)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi",   T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("reps", T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",       T(0)),
                                   tracked::constant<T>("reps_sq",    T(1e-16) * T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(tracked::constant<T>("zero",   T(0)),
                                   tracked::constant<T>("eps50",  T(1e-50)));
    }
};

// ---------------------------------------------------------------------------
// Rule 5 / C5: partial specialization for TOutput = tracked::Complex<T>.
// The primary ql::Constants<TOutput> is used in kokkosMaths.h to synthesize
// _ieps50(), _2ipi(), etc. from the scalar constant tables via TOutput{re,im}.
// For our tracked TOutput, those factories must return tracked::Complex<T>
// values whose components are themselves named tracked constants.
// ---------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Complex<T>> {
    using C = tracked::Complex<T>;

    KOKKOS_INLINE_FUNCTION static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION static constexpr int _num_B() { return 25; }

    // Rule 3: complex-container returns; components are named constants.
    static C _zero()  { return C(tracked::constant<T>("zero",  T(0)),
                                 tracked::constant<T>("zero",  T(0))); }
    static C _half()  { return C(tracked::constant<T>("half",  T(0.5)),
                                 tracked::constant<T>("zero",  T(0))); }
    static C _one()   { return C(tracked::constant<T>("one",   T(1)),
                                 tracked::constant<T>("zero",  T(0))); }
    static C _two()   { return C(tracked::constant<T>("two",   T(2)),
                                 tracked::constant<T>("zero",  T(0))); }
    static C _three() { return C(tracked::constant<T>("three", T(3)),
                                 tracked::constant<T>("zero",  T(0))); }
    static C _four()  { return C(tracked::constant<T>("four",  T(4)),
                                 tracked::constant<T>("zero",  T(0))); }

    template <typename TOutput, typename TMass, typename TScale>
    static C _ieps50() {
        return C(tracked::constant<T>("zero",  T(0)),
                 tracked::constant<T>("eps50", T(1e-50)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static C _ieps() {
        return C(tracked::constant<T>("zero", T(0)),
                 tracked::constant<T>("reps", T(1e-16)));
    }
};

// ---------------------------------------------------------------------------
// Rule C7 / Rule 2: outrank qcdloop's own generic kAbs/kLog/kSqrt/kConj/kPow
// function templates by providing tracked-typed overloads. Each overload is
// strictly more specialized than the library primary under partial ordering,
// so qualified calls (ql::kAbs<T>(x)) bind to these when x is tracked.
// ---------------------------------------------------------------------------

// Rule 2 / C7: kAbs on tracked scalar returns tracked scalar (floating-point
// magnitude participating in downstream arithmetic).
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// Rule 2 / Rule 3 / C7: kAbs on tracked complex returns tracked scalar
// (|z| is a real value that flows into tracked arithmetic).
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// Rule 2 / C7: kLog on tracked scalar.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Rule 2 / Rule 3 / C7: kLog on tracked complex.
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// Rule 2 / C7: kSqrt on tracked scalar.
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// Rule 2 / Rule 3 / C7: kSqrt on tracked complex.
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 2 / C7: kConj on tracked scalar (real → itself; identity, no rounding).
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) { return x; }

// Rule 3 / C7: kConj on tracked complex.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// Rule 2 / C7: kPow on tracked scalar with integer exponent. No tracked::pow
// exists (per C2), so unroll to a multiply loop using tracked operator*.
// Mirrors the library primary's leading explicit-parameter arity
// <TOutput,TMass,TScale> so qualified calls
// ql::kPow<TOutput,TMass,TScale>(x, n) bind here.
template <typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: the identity "1" seeding the accumulator is an anonymous literal.
    tracked::Tracked<T> acc = tracked::literal<T>(T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::literal<T>(T(1));
        return one / acc;
    }
    return acc;
}

// Rule 3 / C7: kPow on tracked complex with integer exponent.
template <typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: seed accumulator with an anonymous complex "1".
    tracked::Complex<T> acc(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));
        return one / acc;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// Rule C7 / Rule 1: Real / Imag / Sign / Htheta on tracked types.
//
// Rule C6 applied per overload:
//   - Real(tracked scalar)  : the identity of a real scalar into scalar
//                             arithmetic — tracked scalar (Rule 2).
//   - Real(tracked complex) : the real component participates in tracked
//                             arithmetic — tracked scalar (Rule 2).
//   - Imag(tracked scalar)  : structurally zero, but flows into tracked
//                             arithmetic — tracked scalar (Rule 2), seeded
//                             from an anonymous literal.
//   - Imag(tracked complex) : imaginary component, tracked scalar (Rule 2).
//   - Sign(tracked scalar)  : a ±1/0 numeric selector that qcdloop mixes into
//                             floating-point arithmetic (see e.g. kokkosUtils.h
//                             solveabc / BIN4). Per C6, this is a
//                             floating-point return: tracked scalar (Rule 2).
//   - Sign(tracked complex) : normalized complex direction z/|z|; a Rule 3
//                             container return.
//   - Htheta(tracked)       : likewise fed into arithmetic; tracked scalar.
// ---------------------------------------------------------------------------

// Rule 2 / C6: Real of tracked scalar is the value itself (identity).
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { return x; }

// Rule 2 / C6: Real of tracked complex is its real component.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

// Rule 2 / C6: Imag of a tracked scalar is a structural zero literal — its
// value never entered the graph as a named quantity but must be a tracked
// scalar so it composes with downstream tracked ops.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::literal<T>(T(0));  // Rule 6: anonymous zero
}

// Rule 2 / C6: Imag of a tracked complex is its imaginary component.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

// Rule 2 / C6: Sign of a tracked scalar returns ±1 or 0 as a tracked scalar.
// The value is consumed by qcdloop as a multiplicative factor in tracked
// arithmetic (kfn, BIN0/1/2/3/4, R3int, ratreal, …) — a floating-point return
// per C6. The ±1/0 chosen is a runtime-selected literal (per complex-sqrt
// precedent in complex.hpp).
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
    return tracked::literal<T>(s);  // Rule 6: runtime-selected anonymous ±1/0
}

// Rule 3 / C6: Sign of a tracked complex is the unit-direction z/|z|, a
// container return.
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    tracked::Tracked<T> mag = tracked::abs(z);
    tracked::Complex<T> denom(mag, tracked::literal<T>(T(0)));  // Rule 6
    return z / denom;
}

// Rule 2 / C6: Htheta = 0.5 * (1 + sign(x)); flows into tracked arithmetic.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    tracked::Tracked<T> half = tracked::constant<T>("half", T(0.5));  // Rule 5
    tracked::Tracked<T> one  = tracked::constant<T>("one",  T(1));    // Rule 5
    return half * (one + Sign(x));
}

// ---------------------------------------------------------------------------
// Rule 1 (with C6 checked): iszero returns raw bool — used only as a branch
// selector inside qcdloop dispatchers (B0m..B4m, BO, ql::iszero calls in
// Ycalc, box_common.h, …). It never flows into floating-point arithmetic.
//
// Mirrors the library primary's leading explicit parameters so qualified
// calls ql::iszero<TOutput,TMass,TScale>(x) bind here (C7).
// ---------------------------------------------------------------------------

// Rule 1 / C7: iszero on tracked scalar → bool via .value().
template <typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    // Threshold matches kokkosMaths.h::_qlonshellcutoff (1e-10).
    using std::abs;
    return abs(x.value()) < T(1e-10);
}

// ---------------------------------------------------------------------------
// Rule 2 / C7: Max / Min on tracked scalar. qcdloop's Max/Min compare by
// magnitude and return one of the two inputs — a floating-point return that
// participates in downstream tracked arithmetic (Rule 2). Rule 7: the
// comparison itself is done on .value() (no lifted tracked booleans).
// ---------------------------------------------------------------------------

// Rule 2 / Rule 7 / C7: Max on tracked scalars (compare |value|, return whole
// tracked operand so its provenance is preserved).
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

// Rule 3 / Rule 7 / C7: Max on tracked complexes (compare |z|; return whole
// container).
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}

// Rule 2 / Rule 7 / C7: Min on tracked scalars.
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

// Rule 3 / Rule 7 / C7: Min on tracked complexes.
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}

} // namespace ql

// ---------------------------------------------------------------------------
// Mixed-type arithmetic between tracked::Tracked<T> / tracked::Complex<T>
// and plain-int operands.
//
// qcdloop performs constructions like `1 - x` and `x * 4` in generic code
// where the literal side is an int (see e.g. Constants<TScale>::_four() used
// with tracked arithmetic, and int * tracked in BIN4 / Ycalc paths that hit
// under partial instantiation — C3, scope by static instantiation). Tracked
// does not define these mixed operators, so provide them here in ::tracked
// (ADL). Each int is promoted with Rule 4 (Tracked(T v) ctor). No name is
// preserved because the operand is a bare literal.
//
// Rule 4: integer literal in a floating-point expression → promote through
// the scalar Tracked<T>(T v) ctor, which synthesizes a "_lit@?#N" id.
// ---------------------------------------------------------------------------
namespace tracked {

// Rule 4 / C3: tracked scalar op int.
template <class T> inline Tracked<T> operator+(const Tracked<T>& a, int b) { return a + Tracked<T>(T(b)); }
template <class T> inline Tracked<T> operator+(int a, const Tracked<T>& b) { return Tracked<T>(T(a)) + b; }
template <class T> inline Tracked<T> operator-(const Tracked<T>& a, int b) { return a - Tracked<T>(T(b)); }
template <class T> inline Tracked<T> operator-(int a, const Tracked<T>& b) { return Tracked<T>(T(a)) - b; }
template <class T> inline Tracked<T> operator*(const Tracked<T>& a, int b) { return a * Tracked<T>(T(b)); }
template <class T> inline Tracked<T> operator*(int a, const Tracked<T>& b) { return Tracked<T>(T(a)) * b; }
template <class T> inline Tracked<T> operator/(const Tracked<T>& a, int b) { return a / Tracked<T>(T(b)); }
template <class T> inline Tracked<T> operator/(int a, const Tracked<T>& b) { return Tracked<T>(T(a)) / b; }

// Rule 7: mixed comparisons produce plain bool via .value().
template <class T> inline bool operator==(const Tracked<T>& a, int b) { return a.value() == T(b); }
template <class T> inline bool operator!=(const Tracked<T>& a, int b) { return a.value() != T(b); }
template <class T> inline bool operator< (const Tracked<T>& a, int b) { return a.value() <  T(b); }
template <class T> inline bool operator> (const Tracked<T>& a, int b) { return a.value() >  T(b); }
template <class T> inline bool operator<=(const Tracked<T>& a, int b) { return a.value() <= T(b); }
template <class T> inline bool operator>=(const Tracked<T>& a, int b) { return a.value() >= T(b); }
template <class T> inline bool operator==(int a, const Tracked<T>& b) { return T(a) == b.value(); }
template <class T> inline bool operator!=(int a, const Tracked<T>& b) { return T(a) != b.value(); }
template <class T> inline bool operator< (int a, const Tracked<T>& b) { return T(a) <  b.value(); }
template <class T> inline bool operator> (int a, const Tracked<T>& b) { return T(a) >  b.value(); }
template <class T> inline bool operator<=(int a, const Tracked<T>& b) { return T(a) <= b.value(); }
template <class T> inline bool operator>=(int a, const Tracked<T>& b) { return T(a) >= b.value(); }

// Rule 4 / Rule 3 / C3: tracked complex op int (promote int through
// Complex<T>(T re, T im=0) which internally uses literal()).
template <class T> inline Complex<T> operator+(const Complex<T>& a, int b) { return a + Complex<T>(T(b)); }
template <class T> inline Complex<T> operator+(int a, const Complex<T>& b) { return Complex<T>(T(a)) + b; }
template <class T> inline Complex<T> operator-(const Complex<T>& a, int b) { return a - Complex<T>(T(b)); }
template <class T> inline Complex<T> operator-(int a, const Complex<T>& b) { return Complex<T>(T(a)) - b; }
template <class T> inline Complex<T> operator*(const Complex<T>& a, int b) { return a * Complex<T>(T(b)); }
template <class T> inline Complex<T> operator*(int a, const Complex<T>& b) { return Complex<T>(T(a)) * b; }
template <class T> inline Complex<T> operator/(const Complex<T>& a, int b) { return a / Complex<T>(T(b)); }
template <class T> inline Complex<T> operator/(int a, const Complex<T>& b) { return Complex<T>(T(a)) / b; }

} // namespace tracked

// ---------------------------------------------------------------------------
// std::abs / std::sqrt overloads on tracked scalar.
//
// tracked's complex.hpp uses `using std::abs; abs(b.re_.value_)` and
// `using std::sqrt; sqrt(...)` inside its templates; those calls are on
// underlying T values (double), so std overloads suffice. However, kokkosMaths
// std::max/std::abs calls inside the Tracked op bodies operate on T directly.
// No new std overloads are needed here — the existing std::abs(double) etc.
// serve tracked's internal ops.
// ---------------------------------------------------------------------------

// End of ql_tracked_interop.hpp