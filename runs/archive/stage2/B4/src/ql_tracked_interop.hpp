// ql_tracked_interop.hpp
// Tracked<T> interop shim for qcdloop's ql:: namespace (box integrals).
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// Included BEFORE any qcdloop header in the driver. Every overload here
// must therefore forward-declare (or fully declare) whatever it references
// from ql:: so the specializations parse. The primary templates that
// qcdloop later defines (kokkosMaths.h, kokkosUtils.h, box/*.h) will find
// these tracked overloads at their template *definition* sites via the
// qualified name ql::Foo(...).
//
// Design notes:
//   * Tracked<T> is a HOST-ONLY type (uses std::string, journaling).
//     KOKKOS_INLINE_FUNCTION on shim overloads is unnecessary and would be
//     wrong (Rule 8 / C4): the driver invokes ql::BO from a plain host
//     loop, not a parallel_for, so no execution-space annotation is emitted.
//   * The tracked complex is spelled tracked::Complex<T> where T is the
//     underlying real (double), NOT tracked::Complex<tracked::Tracked<T>>
//     (C1). The driver's TOutput = tracked::Complex<double>.
//   * Every function-template overload that shadows a qcdloop primary
//     carries the leading explicit template parameters <TOutput, TMass,
//     TScale, ...> on the same overload so qualified calls of the form
//     ql::Foo<TOutput, TMass, TScale>(x) bind here (C7).

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

// -----------------------------------------------------------------------------
// C3: identity unary operator+ on tracked scalars / complex.
// Some qcdloop templates may apply unary + to values; Tracked/Complex do not
// define one. An identity introduces no rounding and emits no journal record.
// -----------------------------------------------------------------------------
namespace tracked {

// Rule C3: identity unary+ for Tracked<T>; ADL-found in namespace tracked.
template <class T>
inline const Tracked<T>& operator+(const Tracked<T>& x) { return x; }

// Rule C3: identity unary+ for Complex<T>; ADL-found in namespace tracked.
template <class T>
inline const Complex<T>& operator+(const Complex<T>& x) { return x; }

} // namespace tracked

// -----------------------------------------------------------------------------
// Forward-declare the ql:: Constants primary template so our specialization
// below parses before qcdloop's kokkosMaths.h supplies the full definition
// (Rule C5). Our specialization must mirror every accessor the driver's call
// graph can reach.
// -----------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
} // namespace ql

namespace ql {

// -----------------------------------------------------------------------------
// Rule 5 / C5: partial specialization of ql::Constants<T> for the tracked
// scalar. Every named leaf constant is routed through tracked::constant with
// its source-library name preserved in the journal. Anonymous scaling factors
// used inside derived constants (e.g. the "3" in _pio3) are literals (Rule 6).
// -----------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Tracked<T>> {
    using S = tracked::Tracked<T>;

    // ---- Chebyshev / Bernoulli coefficient tables (Rule 5: each is a named
    // constant with the same source-library name — we synthesize a per-index
    // name so distinct coefficients get distinct provenance ids).
    static constexpr int _num_C() { return 19; }
    static S _C(int i) {
        // Values verbatim from kokkosMaths.h::Constants<T>::_C.
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        // Rule 5: named constant "C[i]"; the coefficient's identity is its
        // library-visible name plus its index.
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    static constexpr int _num_B() { return 25; }
    static S _B(int i) {
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
        return tracked::constant<T>(std::string("B[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // ---- Named scalar constants (Rule 5: preserve library identifier). ------
    template <typename TOutput, typename TMass, typename TScale>
    static S _qlonshellcutoff() { return tracked::constant<T>("qlonshellcutoff", T(1e-10)); }

    static S _pi()  { return tracked::constant<T>("pi",  T(M_PI)); }
    static S _pi2() { return tracked::constant<T>("pi2", T(M_PI) * T(M_PI)); }

    template <typename TOutput, typename TMass, typename TScale>
    static S _pio3()   { return _pi() / tracked::constant<T>("three", T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    static S _pio6()   { return _pi() / tracked::constant<T>("six", T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    static S _pi2o3()  { return _pi() * _pio3<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static S _pi2o6()  { return _pi() * _pio6<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static S _pi2o12() { return _pi2() / tracked::constant<T>("twelve", T(12)); }

    static S _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
    static S _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static S _one()   { return tracked::constant<T>("one",   T(1.0)); }
    static S _two()   { return tracked::constant<T>("two",   T(2.0)); }
    static S _three() { return tracked::constant<T>("three", T(3.0)); }
    static S _four()  { return tracked::constant<T>("four",  T(4.0)); }
    static S _five()  { return tracked::constant<T>("five",  T(5.0)); }
    static S _six()   { return tracked::constant<T>("six",   T(6.0)); }
    static S _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

    static S _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
    static S _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
    static S _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
    static S _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static S _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static S _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static S _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static S _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static S _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // ---- Complex constants (Rule 3: return the tracked container). ---------
    // _2ipi, _ipio2, _ipi, _ieps, _ieps2, _ieps50 all produce a
    // tracked::Complex<T> because TOutput at the call site is a complex type.
    // These accept <TOutput, TMass, TScale> exactly as the primary does.
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi() {
        // Rule 3: complex-of-tracked constant "2ipi" = 0 + 2*pi*i.
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_two() *
                       Constants<tracked::Tracked<T>>::_pi());
    }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_pi() *
                       Constants<tracked::Tracked<T>>::_half());
    }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_pi());
    }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_reps());
    }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2() {
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       Constants<tracked::Tracked<T>>::_reps() *
                       Constants<tracked::Tracked<T>>::_reps());
    }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() {
        // Rule 5: named "ieps50" but library value is a bare literal 1e-50 in
        // the imaginary slot; preserve the identifier from the library source.
        return TOutput(Constants<tracked::Tracked<T>>::_zero(),
                       tracked::constant<T>("ieps50", T(1e-50)));
    }
};

// -----------------------------------------------------------------------------
// Rule 3 / C5: partial specialization of ql::Constants<tracked::Complex<T>>.
// A few call sites reference Constants<TOutput>::_half()/_one()/... where
// TOutput is the tracked complex; the returned value must be a tracked
// complex (with tracked-scalar real part, zero imaginary via literal(0)).
// -----------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Complex<T>> {
    using C = tracked::Complex<T>;
    using S = tracked::Tracked<T>;

    // Complex leaves (Rule 3): promote a real named constant into a complex.
    static C _zero()  { return C(Constants<S>::_zero()); }
    static C _half()  { return C(Constants<S>::_half()); }
    static C _one()   { return C(Constants<S>::_one()); }
    static C _two()   { return C(Constants<S>::_two()); }
    static C _three() { return C(Constants<S>::_three()); }
    static C _four()  { return C(Constants<S>::_four()); }

    // Rule 6: imaginary-unit-flavored complex constants (need explicit complex).
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi()   { return Constants<S>::template _2ipi<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2()  { return Constants<S>::template _ipio2<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi()    { return Constants<S>::template _ipi<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps()   { return Constants<S>::template _ieps<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2()  { return Constants<S>::template _ieps2<TOutput, TMass, TScale>(); }
    template <typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() { return Constants<S>::template _ieps50<TOutput, TMass, TScale>(); }
};

// =============================================================================
// Free-function shims (from kokkosMaths.h). Each is CONSTRAINED on the tracked
// concrete type (C7): partial ordering picks these overloads over qcdloop's
// generic template<T> versions.
// =============================================================================

// ---- kAbs (Rule 2: floating result; magnitude flows into tracked math) ------
// Overload for tracked scalar: |Tracked<T>| = abs(scalar) as a Tracked<T>.
template <class T>
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    // Rule 2: floating-point return, tracked arithmetic (abs op emits a
    // journal record with cond=1).
    return tracked::abs(x);
}

// Overload for tracked complex: |Complex<T>| = sqrt(re^2 + im^2), returned
// as Tracked<T> (Rule 2: real-valued but participates in downstream math).
template <class T>
tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    // Rule 2 / Rule 3: tracked complex magnitude via tracked::abs on Complex,
    // which decomposes into named tracked ops.
    return tracked::abs(z);
}

// ---- kLog (Rule 2: tracked log). --------------------------------------------
// Scalar overload.
template <class T>
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    // Rule 2: floating return; delegate to tracked::log which records cond.
    return tracked::log(x);
}
// Complex overload (Rule 3: complex-of-tracked).
template <class T>
tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// ---- kSqrt (Rule 2: tracked sqrt). ------------------------------------------
template <class T>
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}
template <class T>
tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// ---- kConj (Rule 3: complex conjugate). ------------------------------------
template <class T>
tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}
// Scalar kConj is a no-op mathematically; identity return preserves the
// tracked value with no journal record (Rule 2, cond=1 trivial).
template <class T>
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// ---- kPow (Rule 2: integer power via multiply loop; C2 — no tracked::pow). --
// Complex-tracked overload (C7: constrain the value parameter). The qcdloop
// primary is template<TOutput, TMass, TScale> kPow(TOutput const&, int const&).
// Carry the leading explicit template params so qualified calls bind here.
template <class TOutput, class TMass, class TScale, class T>
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    // Rule 2 / Rule 4: integer exponent stays raw int; base stays tracked.
    // C2: no tracked::pow exists — synthesize as repeated tracked multiply.
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: literal one (anonymous starting value) via constant("one"): the
    // multiplicative identity is a named library constant.
    tracked::Complex<T> tmp(ql::Constants<tracked::Tracked<T>>::_one());
    for (int i = 0; i < n; ++i) tmp = tmp * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(ql::Constants<tracked::Tracked<T>>::_one());
        return one_c / tmp;
    }
    return tmp;
}

// Scalar-tracked overload (C7).
template <class TOutput, class TMass, class TScale, class T>
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    // Rule 2 / C2: integer power as repeated tracked multiply.
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> tmp = ql::Constants<tracked::Tracked<T>>::_one();
    for (int i = 0; i < n; ++i) tmp = tmp * base;
    if (exponent < 0) {
        return ql::Constants<tracked::Tracked<T>>::_one() / tmp;
    }
    return tmp;
}

// ---- iszero (Rule 1: discrete bool result). ---------------------------------
// qcdloop primary: template<TOutput,TMass,TScale> bool iszero(TScale const&).
// Carry the three leading explicit template params (C7) so qualified calls
// ql::iszero<TOutput,TMass,TScale>(x) with x tracked bind here.
template <class TOutput, class TMass, class TScale, class T>
bool iszero(const tracked::Tracked<T>& x) {
    // Rule 1: discrete bool return. Unwrap via .value() and compare against
    // the library's onshell cutoff (also unwrapped to a bare T).
    using std::abs;
    return abs(x.value()) <
           ql::Constants<tracked::Tracked<T>>::template
               _qlonshellcutoff<TOutput, TMass, TScale>().value();
}

// Complex tracked overload (Rule 1). Handful of iszero call sites feed a
// complex through kAbs first; the direct-complex path is covered here too.
template <class TOutput, class TMass, class TScale, class T>
bool iszero(const tracked::Complex<T>& z) {
    // Rule 1: discrete. |z| unwrapped; note kAbs on tracked complex returns
    // a Tracked<T>, so pull out .value().
    using std::abs;
    T mag = kAbs(z).value();
    return abs(mag) <
           ql::Constants<tracked::Tracked<T>>::template
               _qlonshellcutoff<TOutput, TMass, TScale>().value();
}

// ---- Imag / Real (Rule 2: floating-point return, tracked). ------------------
// Real scalar: Imag(x) = 0.  Rule 2 in a floating-point context: the caller
// often multiplies this into tracked arithmetic (via TOutput(...)); return
// a tracked zero (named literal for provenance).
template <class T>
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Rule 2 / Rule 6: anonymous zero literal — no semantic name.
    return tracked::literal<T>(T(0));
}
// Complex tracked overload (Rule 2: real component of tracked complex).
template <class T>
const tracked::Tracked<T>& Imag(const tracked::Complex<T>& z) {
    // Rule 2: return the imaginary component as a tracked scalar (preserves
    // its provenance / journal id).
    return z.imag();
}

template <class T>
const tracked::Tracked<T>& Real(const tracked::Tracked<T>& x) {
    // Rule 2: identity for real scalar; keep provenance.
    return x;
}
template <class T>
const tracked::Tracked<T>& Real(const tracked::Complex<T>& z) {
    // Rule 2: real component of tracked complex.
    return z.real();
}

// ---- Sign. -----------------------------------------------------------------
// C6: Sign() results in this library flow directly into floating-point
// arithmetic (multiplied by tracked expressions, cast into TOutput, added
// to imaginary parts) — so this is a floating-point return (Rule 2), not a
// discrete int, and must preserve tracking.
template <class T>
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    // Rule 2 / C6: +1, -1, or 0 as a tracked scalar (anonymous literal — no
    // named identity, sign is data-dependent).
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return tracked::literal<T>(s);
}
// Complex overload: qcdloop returns z / |z|. Rule 3: container of tracked.
template <class T>
tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    // Rule 3 / C6: complex sign = z / |z|, preserves provenance through div.
    return z / kAbs(z);
}

// ---- Max / Min. -------------------------------------------------------------
// Return-by-value; Rule 2 / Rule 3 as appropriate. Comparison uses .value()
// via kAbs -> .value() to keep the branch discrete (Rule 7) while the
// returned value stays tracked.
template <class T>
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    // Rule 7: comparison on tracked values yields bool; Rule 2: selected
    // value returned as tracked (preserving provenance of the winner).
    return (kAbs(a).value() > kAbs(b).value()) ? a : b;
}
template <class T>
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    // Rule 7 + Rule 3.
    return (kAbs(a).value() > kAbs(b).value()) ? a : b;
}
template <class T>
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    return (kAbs(a).value() > kAbs(b).value()) ? b : a;
}
template <class T>
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    return (kAbs(a).value() > kAbs(b).value()) ? b : a;
}

// Mixed Max variants (kAbs(a) is a Tracked<T>, other operand may be a bare T
// or a Tracked<T>). The library writes `ql::Max(kAbs(a), TMass(one()))`.
// Provided constrained overloads handle both bare-T and Tracked<T> RHS.
// Rule 4: bare integer/floating literal promoted via Tracked<T>(T) ctor
// (synthesizes an anonymous _lit id).
template <class T>
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, T b) {
    // Rule 4 / Rule 7: promote bare literal via ctor, compare .value().
    tracked::Tracked<T> b_t(b);
    return (kAbs(a).value() > kAbs(b_t).value()) ? a : b_t;
}
template <class T>
tracked::Tracked<T> Max(T a, const tracked::Tracked<T>& b) {
    tracked::Tracked<T> a_t(a);
    return (kAbs(a_t).value() > kAbs(b).value()) ? a_t : b;
}

// ---- Htheta (Rule 2: 0.5 * (1 + sign(x))); flows into tracked complex). -----
template <class T>
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    // Rule 2 / C6: floating result, preserved through tracked ops.
    return ql::Constants<tracked::Tracked<T>>::_half() *
           (ql::Constants<tracked::Tracked<T>>::_one() + Sign(x));
}

} // namespace ql