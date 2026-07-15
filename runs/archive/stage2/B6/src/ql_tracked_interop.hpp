// ql_tracked_interop.hpp
// Tracked interop shim for qcdloop (Kokkos box-integral library), B6 spike.
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This shim is included BEFORE any qcdloop headers, so qualified calls like
// ql::Foo(x) at qcdloop template definition sites can see the tracked
// overloads at definition time (qualified names bypass ADL — the overload
// must be visible at the point of definition of the calling template, not
// merely at instantiation).

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// C3: Missing operators on the tracked scalar / complex.
// qcdloop template bodies apply unary operator+ and mixed-scalar arithmetic
// on tracked values in various places. Add these as free functions in the
// tracked namespace so ADL (and qualified use inside the tracked namespace)
// picks them up. Identity ops introduce no rounding and emit no journal
// record.
// ---------------------------------------------------------------------------
namespace tracked {

// C3: unary operator+ on tracked scalar (identity).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// C3: unary operator+ on tracked complex (identity).
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// Forward-declare the qcdloop primary templates we specialize / overload,
// so this shim parses BEFORE the qcdloop headers are included (C5).
// ---------------------------------------------------------------------------
namespace ql {

// C5: forward declaration of the library's Constants<T> class template so
// the partial specialization below parses. The library supplies the full
// primary definition when kokkosMaths.h is included later in the TU.
template <typename T> struct Constants;

// Forward declarations of the qcdloop free-function primaries we override
// via constrained overloads (C7 partial-ordering trick).
template <typename TOutput, typename TMass, typename TScale> KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x);
template <typename TOutput, typename TMass, typename TScale> KOKKOS_INLINE_FUNCTION TOutput kPow(TOutput const& base, int const& exponent);
template <typename TOutput, typename TMass, typename TScale> KOKKOS_INLINE_FUNCTION TMass   kPow(TMass   const& base, int const& exponent);

} // namespace ql

// ===========================================================================
// Section 1 — Constants specialization (C5, Rule 5)
// ===========================================================================
//
// Rule 5 / C5: qcdloop dispatches all its named constants through the class
// template ql::Constants<T>. For T = tracked::Tracked<double> we specialize
// partially so every accessor mirrors the primary's interface but routes the
// leaf scalar through tracked::constant("<name>", ...), preserving the
// constant's name in the journal. Every literal used inside qcdloop as a
// designator (Constants<T>::_two(), ::_pi(), ::_eps(), ::_ieps50(), …) is
// wrapped here.
//
// TOutput/TMass/TScale template accessors return the tracked scalar too —
// the driver uses TOutput = tracked::Complex<double>, so any Constants
// accessor whose declared return type is TOutput must yield a
// tracked::Complex<double> built from tracked reals.
// ===========================================================================
namespace ql {

// C5: partial specialization keyed on the tracked scalar.
template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tk = tracked::Tracked<T>;
    using Ck = tracked::Complex<T>;

    // Rule 5: Chebyshev / Bernoulli tables — the LIBRARY names each entry
    // by its position and there is no per-entry symbolic name. Wrap each
    // returned scalar as a named tracked constant "C[i]" / "B[i]" so the
    // journal preserves the fact it came from the table.
    KOKKOS_INLINE_FUNCTION static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION static Tk _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    KOKKOS_INLINE_FUNCTION static constexpr int _num_B() { return 25; }
    KOKKOS_INLINE_FUNCTION static Tk _B(int i) {
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
        return tracked::constant<T>(std::string("B[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // Rule 5: each of these constants has a source-code identifier
    // (_qlonshellcutoff, _pi, _pi2, _pio3, …). Wrap via tracked::constant
    // preserving that identifier.
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    KOKKOS_INLINE_FUNCTION static Tk _pi()    { return tracked::constant<T>("pi",    T(M_PI)); }
    KOKKOS_INLINE_FUNCTION static Tk _pi2()   { return tracked::constant<T>("pi2",   T(M_PI * M_PI)); }

    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _pio3()   { return tracked::constant<T>("pio3",   T(M_PI / 3.0)); }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _pio6()   { return tracked::constant<T>("pio6",   T(M_PI / 6.0)); }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _pi2o3()  { return tracked::constant<T>("pi2o3",  T(M_PI * M_PI / 3.0)); }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _pi2o6()  { return tracked::constant<T>("pi2o6",  T(M_PI * M_PI / 6.0)); }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tk _pi2o12() { return tracked::constant<T>("pi2o12", T(M_PI * M_PI / 12.0)); }

    KOKKOS_INLINE_FUNCTION static Tk _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _half()  { return tracked::constant<T>("half",  T(0.5)); }
    KOKKOS_INLINE_FUNCTION static Tk _one()   { return tracked::constant<T>("one",   T(1.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _two()   { return tracked::constant<T>("two",   T(2.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _three() { return tracked::constant<T>("three", T(3.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _four()  { return tracked::constant<T>("four",  T(4.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _five()  { return tracked::constant<T>("five",  T(5.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _six()   { return tracked::constant<T>("six",   T(6.0)); }
    KOKKOS_INLINE_FUNCTION static Tk _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

    KOKKOS_INLINE_FUNCTION static Tk _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
    KOKKOS_INLINE_FUNCTION static Tk _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
    KOKKOS_INLINE_FUNCTION static Tk _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
    KOKKOS_INLINE_FUNCTION static Tk _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    KOKKOS_INLINE_FUNCTION static Tk _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static Tk _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    KOKKOS_INLINE_FUNCTION static Tk _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    KOKKOS_INLINE_FUNCTION static Tk _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static Tk _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Rule 3 / C1: these accessors return TOutput. The driver uses
    // TOutput = tracked::Complex<double>, so build the complex from tracked
    // reals — do NOT wrap the whole complex in Tracked<...>.
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _2ipi() {
        // Rule 5: named constant "2pi" for the imaginary component.
        return Ck(tracked::constant<T>("zero", T(0.0)),
                  tracked::constant<T>("2pi",  T(2.0 * M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _ipio2() {
        return Ck(tracked::constant<T>("zero",  T(0.0)),
                  tracked::constant<T>("pio2",  T(M_PI * 0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _ipi() {
        return Ck(tracked::constant<T>("zero", T(0.0)),
                  tracked::constant<T>("pi",   T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _ieps() {
        return Ck(tracked::constant<T>("zero", T(0.0)),
                  tracked::constant<T>("reps", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _ieps2() {
        return Ck(tracked::constant<T>("zero",   T(0.0)),
                  tracked::constant<T>("reps2",  T(1e-16 * 1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Ck _ieps50() {
        return Ck(tracked::constant<T>("zero",  T(0.0)),
                  tracked::constant<T>("eps50", T(1e-50)));
    }
};

} // namespace ql

// ===========================================================================
// Section 2 — Free-function overloads on tracked types (Rules 1/2/3, C7)
// ===========================================================================
//
// qcdloop calls these as qualified ql::Foo(x) inside its own templates. To
// win overload resolution over the library's own generic templates that get
// declared later, each shim overload is more specialized in the value
// parameter (const tracked::Tracked<T>& / const tracked::Complex<T>&) — C7.
// C4: driver invokes ql::BO from a plain host loop, so no execution-space
// annotation is required.
// ===========================================================================
namespace ql {

// ---- kAbs -----------------------------------------------------------------
// Rule 2 / C6: kAbs on a tracked scalar returns |x| as a tracked scalar
// (feeds arithmetic). Rule 3: kAbs on a tracked complex returns the real
// magnitude as a tracked scalar.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x, TRACKED_HERE);
}
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {
    return tracked::abs(x, TRACKED_HERE);
}

// ---- kLog -----------------------------------------------------------------
// Rule 2 / Rule 3: log of tracked scalar/complex.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x, TRACKED_HERE);
}
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& x) {
    return tracked::log(x, TRACKED_HERE);
}

// ---- kSqrt ----------------------------------------------------------------
// Rule 2 / Rule 3.
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x, TRACKED_HERE);
}
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {
    return tracked::sqrt(x, TRACKED_HERE);
}

// ---- kConj ----------------------------------------------------------------
// Rule 2: conj on a real is identity. Rule 3: conj on complex negates imag.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) { return x; }
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& x) {
    return tracked::conj(x, TRACKED_HERE);
}

// ---- Real / Imag ----------------------------------------------------------
// Rule 2: Real(scalar) is identity; Imag(scalar) is a literal zero (that
// zero feeds floating-point comparisons and arithmetic downstream, so it
// must be tracked, not a bare double).
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { return x; }
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Rule 6: anonymous inline zero.
    return tracked::literal<T>(T(0));
}
// Rule 3: Real/Imag on a tracked complex return the tracked scalar
// components.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& x) { return x.real(); }
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& x) { return x.imag(); }

// ---- Sign -----------------------------------------------------------------
// Rule 2 / C6: qcdloop uses Sign(x) inside floating-point expressions
// (multiplications, additions into cond/imag terms). The result feeds
// tracked arithmetic, so it must be a tracked scalar (+1 / 0 / -1) — NOT
// a raw int (Rule 1 would break provenance downstream).
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
    // Rule 6: the ±1/0 is an anonymous literal — the numeric selection is
    // runtime-dependent, so we do not name it as a constant.
    return tracked::literal<T>(s);
}
// Rule 3 / C6: Sign of a tracked complex returns the tracked complex
// z/|z| (used as a direction vector in expressions).
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& x) {
    auto mag = tracked::abs(x, TRACKED_HERE);
    return x / mag;
}

// ---- Max / Min ------------------------------------------------------------
// Rule 2 / C6: Max/Min return a tracked scalar (they select one of two
// arithmetic values that flows into subsequent tracked ops). Rule 7:
// comparison is on the underlying value only.
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    // Rule 7: bool comparison via .value().
    return (tracked::abs(a, TRACKED_HERE).value() > tracked::abs(b, TRACKED_HERE).value()) ? a : b;
}
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    return (tracked::abs(a, TRACKED_HERE).value() > tracked::abs(b, TRACKED_HERE).value()) ? b : a;
}
// Rule 3 / C6: Max/Min on tracked complex return tracked complex.
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a, TRACKED_HERE).value() > tracked::abs(b, TRACKED_HERE).value()) ? a : b;
}
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a, TRACKED_HERE).value() > tracked::abs(b, TRACKED_HERE).value()) ? b : a;
}

// ---- Htheta ---------------------------------------------------------------
// Rule 2 / C6: Heaviside 0/1 is used in expressions as a multiplicative
// factor (see eta2/eta5) — a floating-point contributor, must be tracked.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    T v = x.value();
    T h = (v > T(0)) ? T(1) : ((v < T(0)) ? T(0) : T(0.5));
    // Rule 6: anonymous literal 0 / 0.5 / 1 selected at runtime.
    return tracked::literal<T>(h);
}

// ---- iszero ---------------------------------------------------------------
// Rule 1 / C6: iszero result is consumed ONLY as a boolean predicate
// (branch conditions inside ql::B*/BIN*/Ycalc). Return raw bool.
// C7: leading explicit template parameters carried so ql::iszero<TOutput,
// TMass, TScale>(x) call sites bind to this constrained overload instead
// of the library's generic primary.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) {
    // Match the library predicate: |x| < qlonshellcutoff.
    T ax = std::abs(x.value());
    return ax < T(1e-10);
}

// ---- kPow -----------------------------------------------------------------
// Rule 2: base^n as a tracked scalar via a multiply loop (Tracked API has
// no pow — C2). Rule 3: same for tracked complex.
// C7: leading explicit template parameters carried; value parameter is the
// concrete tracked type so partial ordering picks this over the primary.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5: named constant "one" used as loop-accumulator seed (the
    // symbolic identity element, not an anonymous literal).
    tracked::Tracked<T> acc = tracked::constant<T>("one", T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Tracked<T> one_c = tracked::constant<T>("one", T(1));
        return one_c / acc;
    }
    return acc;
}
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5 / Rule 3: complex identity built from the tracked "one" real
    // and an anonymous zero imaginary component (Complex(re) ctor).
    tracked::Complex<T> acc(tracked::constant<T>("one", T(1)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(tracked::constant<T>("one", T(1)));
        return one_c / acc;
    }
    return acc;
}

} // namespace ql