// ql_tracked_interop.hpp — Tracked<T> interop shim for QCDLoop+Kokkos box integrals.
// SOURCE_HASH: cfad2410c3ddc32ab520cc03f18dd5e38f62b9fd0359678851e50da9f40a0ac8
//
// Purpose: make the ql::* templates in box/*.h + kokkosMaths.h + kokkosUtils.h
// instantiable with T = tracked::Tracked<double> / tracked::Complex<double>,
// so the driver's floating-point graph is journalled with condition numbers and
// provenance, while integer/discrete book-keeping (indices, signs used as
// selectors, iszero booleans) stays in native discrete types.
//
// Include order in the driver is:
//     #include "ql_tracked_interop.hpp"   // FIRST
//     #include "kokkosMaths.h"
//     #include "kokkosUtils.h"
//     #include "boxGPU.h"
//
// so this header must forward-declare any library class template it specializes
// (Rule C5) before defining the specialization.

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <cstdint>
#include <type_traits>

// ---------------------------------------------------------------------------
// C3. Missing operators on tracked scalars used by the library templates.
//
// The library writes expressions like `-x` where x is TScale/TMass; tracked
// already defines unary operator-. It also constructs Tracked from int
// literals through `TMass(1)`, `TScale(4)`, etc. The one-arg Tracked(T) ctor
// takes a T, so int -> Tracked<double> requires int -> double, which is a
// standard implicit conversion; that path works. No new operators needed on
// tracked::Tracked<T> for this driver's static call graph.
// ---------------------------------------------------------------------------

namespace tracked {

// C3: unary operator+ on Tracked<T>. The library never writes `+x` on a
// tracked scalar in the reachable branches, but Kokkos::complex arithmetic
// (through Kokkos::abs on our specialization below) can hit it; provide an
// identity that emits no journal record.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// C5. Forward-declare library primaries we specialize BEFORE they are defined.
// The library defines ql::Constants<T> as a class template in kokkosMaths.h,
// which this header is included before. We must forward-declare so our partial
// specialization on tracked::Tracked<T> parses.
// ---------------------------------------------------------------------------

namespace ql {
    template <typename T> struct Constants;      // primary in kokkosMaths.h
    using complex = Kokkos::complex<double>;     // matches library alias
} // namespace ql

// ---------------------------------------------------------------------------
// Kokkos support for tracked::Complex<double>.
//
// The library uses TOutput = Kokkos::complex<double> in its own tests, but the
// driver instantiates TOutput = tracked::Complex<double>. The library calls
// Kokkos::abs / Kokkos::log / Kokkos::sqrt / Kokkos::conj on TOutput values
// (via ql::kAbs / ql::kLog / ql::kSqrt / ql::kConj in kokkosMaths.h). Those
// templates dispatch by ADL into namespace Kokkos, so we provide overloads
// there. Rule 2: they return tracked (or bool for tests). Rule 3: containers
// stay as tracked::Complex.
// ---------------------------------------------------------------------------

namespace Kokkos {

// Rule 2: |z| for tracked complex returns tracked scalar.
template <class T>
inline tracked::Tracked<T> abs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// Rule 3: complex log stays as tracked complex.
template <class T>
inline tracked::Complex<T> log(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// Rule 3: complex sqrt stays as tracked complex.
template <class T>
inline tracked::Complex<T> sqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 3: conj stays as tracked complex.
template <class T>
inline tracked::Complex<T> conj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

} // namespace Kokkos

// ---------------------------------------------------------------------------
// ql::Constants specialization for tracked scalars (Rule C5).
//
// The library reads named constants through ql::Constants<T>::_zero(),
// _one(), _pi(), _pi2(), _pi2o6<...>(), _ieps50<...>(), etc. Under Rule 5
// (named constant) every leaf value must be routed through
// tracked::constant("<name>", T(value)) so the journal keeps the identifier.
//
// This is a PARTIAL specialization on Tracked<T> so the arithmetic type is
// carried through — every member returns tracked::Tracked<T> for the scalar
// accessors, or Kokkos::complex<tracked::Tracked<T>>-style pieces where the
// primary returns TOutput. We mirror the FULL primary interface: every
// accessor the driver's static call graph can reach.
// ---------------------------------------------------------------------------

namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {
    // Rule 5: named constants get named tracked constants.
    // Rule 1 exception (C6): the Chebyshev/Bernoulli coefficient count is an
    // integer used only as a loop bound — stays int.

    KOKKOS_INLINE_FUNCTION static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION static constexpr int _num_B() { return 25; }

    // Rule 5: each Chebyshev / Bernoulli coefficient is a named constant.
    // Index into the same literal tables as the primary, but wrap the result.
    static tracked::Tracked<T> _C(int i) {
        static const double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    static tracked::Tracked<T> _B(int i) {
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

    // Rule 5: onshell cutoff is a named constant.
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    // Rule 5: pi and its friends.
    static tracked::Tracked<T> _pi()   { return tracked::constant<T>("pi",  T(M_PI)); }
    static tracked::Tracked<T> _pi2()  { return tracked::constant<T>("pi2", T(M_PI) * T(M_PI)); }

    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pio3()   { return tracked::constant<T>("pi/3",   T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pio6()   { return tracked::constant<T>("pi/6",   T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o3()  { return tracked::constant<T>("pi2/3",  T(M_PI) * T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o6()  { return tracked::constant<T>("pi2/6",  T(M_PI) * T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o12() { return tracked::constant<T>("pi2/12", T(M_PI) * T(M_PI) / T(12)); }

    // Rule 5: small integer / half constants used by name in the library.
    static tracked::Tracked<T> _zero()  { return tracked::constant<T>("zero",  T(0)); }
    static tracked::Tracked<T> _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static tracked::Tracked<T> _one()   { return tracked::constant<T>("one",   T(1)); }
    static tracked::Tracked<T> _two()   { return tracked::constant<T>("two",   T(2)); }
    static tracked::Tracked<T> _three() { return tracked::constant<T>("three", T(3)); }
    static tracked::Tracked<T> _four()  { return tracked::constant<T>("four",  T(4)); }
    static tracked::Tracked<T> _five()  { return tracked::constant<T>("five",  T(5)); }
    static tracked::Tracked<T> _six()   { return tracked::constant<T>("six",   T(6)); }
    static tracked::Tracked<T> _ten()   { return tracked::constant<T>("ten",   T(10)); }

    // Rule 5: named tolerance / eps constants.
    static tracked::Tracked<T> _eps()    { return tracked::constant<T>("eps",    T(1e-6));  }
    static tracked::Tracked<T> _eps4()   { return tracked::constant<T>("eps4",   T(1e-4));  }
    static tracked::Tracked<T> _eps7()   { return tracked::constant<T>("eps7",   T(1e-7));  }
    static tracked::Tracked<T> _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static tracked::Tracked<T> _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static tracked::Tracked<T> _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static tracked::Tracked<T> _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static tracked::Tracked<T> _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static tracked::Tracked<T> _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Rule 3 + Rule 5: complex "constants" returning tracked complex containers.
    // These are TOutput-typed in the primary; TOutput here is
    // tracked::Complex<T> so we return that container built out of named
    // tracked scalars.
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",  T(0)),
                                   tracked::constant<T>("pi/2", T(M_PI) * T(0.5)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi",   T(M_PI)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("reps", T(1e-16)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",     T(0)),
                                   tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(tracked::constant<T>("zero",   T(0)),
                                   tracked::constant<T>("ieps50", T(1e-50)));
    }
};

} // namespace ql

// ---------------------------------------------------------------------------
// ql::kAbs / kLog / kSqrt / kConj / Imag / Real / Sign / Max / Min / iszero /
// Htheta overloads for tracked scalars and tracked complex.
//
// The library declares each of these; we provide OVERLOADS (not
// specializations) in namespace ql so that argument-dependent lookup picks
// them up when the argument's associated namespace is either ql (for tracked
// via ADL through friend injection... which is not the case here) or tracked
// (for our tracked types). ADL on tracked::Tracked<double> looks in namespace
// tracked, so we also need the overloads visible via qualified `ql::kAbs`
// calls — which the library does everywhere. Placing them in namespace ql
// makes qualified `ql::kAbs(x)` find them by ordinary lookup.
// ---------------------------------------------------------------------------

namespace ql {

// Rule 2: |x| returns tracked scalar for tracked scalar input.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// Rule 2: |z| returns tracked scalar (magnitude, not complex).
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// Rule 2: log for tracked scalar.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Rule 3: complex log stays complex.
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// Rule 2: sqrt for tracked scalar.
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// Rule 3: complex sqrt stays complex.
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 3: conj for tracked complex.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// Rule 2: conj of a real tracked scalar is itself.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// Rule 2 (C6): Imag/Real of tracked scalar — used only as a floating value
// downstream (multiplied into tracked expressions, compared numerically, or
// passed to Sign which itself feeds arithmetic). Return tracked so provenance
// survives.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Real tracked scalar has no imaginary part; return a named zero so the
    // downstream journal shows the origin.
    return tracked::constant<T>("zero", T(0));
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}

// Rule 3 -> Rule 2: real/imag of tracked complex return the tracked scalar
// component. These are member accessors, so they return references in the
// underlying type; we return by value.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
    return z.imag();
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
    return z.real();
}

// Rule 2 (C6): Sign of a tracked scalar. The library uses this both as a
// discrete selector (compared, used in eta) AND as a numerical multiplier
// injected into tracked arithmetic (e.g. `TOutput(ql::Sign(ql::Real(k12)))`).
// The latter is a floating-point participant, so return tracked. The value
// is +1 / 0 / -1 wrapped as a named tracked constant so its role is legible
// in the journal.
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    const T v = x.value();
    if (v > T(0)) return tracked::constant<T>("sign+1", T(1));
    if (v < T(0)) return tracked::constant<T>("sign-1", T(-1));
    return tracked::constant<T>("sign0", T(0));
}

// Rule 3 -> Rule 2: Sign of tracked complex returns a tracked scalar (the
// library uses this only after Real(...) — but we guard the exact-complex
// overload for completeness. sgn(z) = z/|z|; but the library only ever
// invokes Sign on real values via ql::Real(...) first, so a scalar return
// is the correct classification here.
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Complex<T>& z) {
    // Extract real part magnitude sign — matches the library's usage sites.
    return Sign(z.real());
}

// Rule 2: Max / Min by absolute value on tracked scalars. Returns whichever
// input by value, preserving that input's provenance (no arithmetic emitted).
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    return (std::abs(a.value()) > std::abs(b.value())) ? a : b;
}

template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    return (std::abs(a.value()) > std::abs(b.value())) ? b : a;
}

// Rule 3: Max/Min on tracked complex return tracked complex.
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}

template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}

// Rule 1 (C6): iszero is a predicate — its result feeds only into `if`
// branches, boolean combinators, and `bool` locals. Discrete return.
// Note: the library primary is `iszero<TOutput,TMass,TScale>(TScale const&)`,
// so we shadow it with a constrained overload. C7 partial ordering rules:
// our overload's value parameter is const tracked::Tracked<T>&, strictly more
// specialized than the primary's bare TScale template parameter, so qualified
// `ql::iszero<...>(x)` binds here.
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    // Match the primary's threshold: qlonshellcutoff = 1e-10 for double.
    return std::abs(x.value()) < T(1e-10);
}

// Rule 2: Htheta returns 0.5 * (1 + sign(x)) — feeds directly into tracked
// arithmetic (eta2 multiplies it into 2ipi). Floating-point return.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    const T v = x.value();
    if (v > T(0)) return tracked::constant<T>("Htheta_one",  T(1));
    if (v < T(0)) return tracked::constant<T>("Htheta_zero", T(0));
    return tracked::constant<T>("Htheta_half", T(0.5));
}

// ---------------------------------------------------------------------------
// C7. kPow overloads. The library primary is
//     template <class TOutput, class TMass, class TScale>
//     TOutput kPow(TOutput const&, int const&);
// and a sibling for TMass. We constrain the value parameter to concrete
// tracked types and carry the three leading explicit template parameters so
// qualified `ql::kPow<TOutput,TMass,TScale>(x, n)` binds here.
//
// tracked has no pow(); implement integer powers as a multiply loop over the
// tracked operator* (Rule C2 note about the missing pow).
// ---------------------------------------------------------------------------

// Rule 2 + Rule C7: tracked scalar base, integer exponent -> tracked scalar.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5: the multiplicative identity is a named constant.
    tracked::Tracked<T> acc = tracked::constant<T>("one", T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::constant<T>("one", T(1));
        acc = one / acc;
    }
    return acc;
}

// Rule 3 + Rule C7: tracked complex base, integer exponent -> tracked complex.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 5: named "one" constant lifted into a complex.
    tracked::Complex<T> acc(tracked::constant<T>("one",  T(1)),
                            tracked::constant<T>("zero", T(0)));
    for (int i = 0; i < n; ++i) acc = acc * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::constant<T>("one",  T(1)),
                                tracked::constant<T>("zero", T(0)));
        acc = one / acc;
    }
    return acc;
}

} // namespace ql

// ---------------------------------------------------------------------------
// End of shim.
//
// Notes on what is deliberately NOT emitted:
//
// * ql::BO / ql::B0m / ql::B1m / ql::B2m / ql::B3m / ql::B4m / ql::BIN*
//   / ql::B1..B16 dispatchers: these are the library's own generic templates
//   over <TOutput,TMass,TScale>. They compile as-is once (a) Constants,
//   (b) kAbs/kLog/kSqrt/kConj/Imag/Real/Sign/Max/Min/iszero/Htheta/kPow are
//   available for the tracked types, and (c) TOutput = tracked::Complex<T>
//   provides all arithmetic + Kokkos::{abs,log,sqrt,conj} — all satisfied
//   above. So the library templates re-use their existing definitions;
//   nothing to override under Rule C7 for the dispatch layer.
//
// * ql::cLn / Lnrat / ddilog / li2series / denspence / Li2omx / spencer /
//   ratgam / ratreal / kfn / cspence / xspence / eta / xeta / etatilde /
//   xetatilde / eta2 / eta3 / eta5 / cLi2omx2 / cLi2omx3 / Li2omx2 / Li2omrat
//   / L0 / L1 / solveabc / solveabcd / R / Rint / R2int / R3int / Zlogint /
//   fndd / ltli2series / ltspence: same reasoning — generic in T; compile
//   through once tracked provides the primitives above.
//
// * Ycalc / swap_b0m / swap_b1m / swap_b2m / swap_b3m / jsort_b0m: pure
//   integer/index computation on Kokkos::Array<int,...>. Rule 1: stays as-is.
//
// * Kokkos::parallel_for / KOKKOS_INLINE_FUNCTION annotations: the DRIVER
//   dispatches through a plain host `for` loop (comment in micro_driver.cpp
//   says "Host loop, NOT Kokkos::parallel_for: tracked ops are host-only").
//   Rule C4: no execution-space annotation needed on any shim overload above.