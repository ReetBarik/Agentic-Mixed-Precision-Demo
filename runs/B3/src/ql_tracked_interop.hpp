// ql_tracked_interop.hpp
// Tracked<T> interop shim for qcdloop (Kokkos edition), targeting the B3
// massless box branch exercised by micro_driver.cpp.
//
// Include order (from the driver):
//     #include "ql_tracked_interop.hpp"   // THIS header — MUST come first
//     #include "kokkosMaths.h"
//     #include "kokkosUtils.h"
//     #include "boxGPU.h"
// so that:
//   (C5) our forward-declaration of ql::Constants<T> and its partial
//        specializations for tracked::Tracked<T> / tracked::Complex<T> are
//        visible before kokkosMaths.h defines the primary template.
//   (C7) our overloads of ql::Real / ql::Imag / ql::Sign / ql::kAbs /
//        ql::kLog / ql::kSqrt / ql::kConj / ql::Max / ql::Min / ql::kPow /
//        ql::iszero / ql::Htheta are visible at every qualified `ql::foo(...)`
//        call site inside kokkosMaths.h / kokkosUtils.h / box/*.h templates.
//        (Qualified calls do NOT use ADL, so the overloads must be declared
//        in namespace `ql` before those templates are parsed.)
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
// Rule 8 / C4: execution-space annotation.
//
// The driver invokes ql::BO from a plain host `for` loop (tracked ops are
// host-only — they touch std::string and a thread-local journal buffer). So
// NO KOKKOS_INLINE_FUNCTION / __host__ __device__ annotation is emitted on
// any shim: adding a device annotation would either be wrong (device code
// cannot journal) or force uninstantiable code paths. Plain-host = no
// annotation.
// ---------------------------------------------------------------------------

// ===========================================================================
// Rule 5 / C5: forward-declare ql::Constants<T> so our partial
// specializations for tracked scalars parse before kokkosMaths.h defines
// the primary template.
// ===========================================================================
namespace ql {
    template <typename T> struct Constants;   // Rule 5 / C5: primary forward decl
}

// ===========================================================================
// Rule 3 / C1: tracked type aliases (the driver's own working types).
// ===========================================================================
// (No new aliases here — the driver declares `TScale = tracked::Tracked<double>`,
// `TMass = tracked::Tracked<double>`, `TOutput = tracked::Complex<double>`.
// The shim uses the tracked spellings verbatim per C1: tracked::Complex<T>
// is already a complex of two Tracked<T>, so we NEVER write
// tracked::Complex<tracked::Tracked<T>>.)

// ===========================================================================
// C3: identity operators the library instantiates on tracked values but the
// Tracked API doesn't define. Placed in namespace tracked so ADL finds them.
// unary operator+ is used by expressions like `+z` inside some qcdloop
// dispatch branches; supply it as identity so no journal record is emitted.
// ===========================================================================
namespace tracked {

// C3: unary operator+ on Tracked<T> (identity — no rounding, no journal).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) {
    return a;
}

// C3: unary operator+ on Complex<T> (identity — no journal).
template <class T>
inline Complex<T> operator+(const Complex<T>& a) {
    return a;
}

} // namespace tracked

// ===========================================================================
// ql:: overloads for tracked types.
//
// Every overload is placed in namespace `ql` so that qualified calls inside
// the qcdloop templates (e.g. ql::Real(x), ql::kAbs(x)) find them via
// ordinary lookup — ADL is not enough for qualified calls. (Rule C7.)
// ===========================================================================
namespace ql {

// ---------------------------------------------------------------------------
// Real / Imag on tracked scalar and tracked complex.
// Rule 2: floating-point return that participates in downstream error
// propagation, so we return the tracked scalar (never bare double).
// ---------------------------------------------------------------------------

// Rule 2: Real(Tracked<T>) — a real scalar is its own real part.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}

// Rule 2: Imag(Tracked<T>) — the imaginary part of a real is literal 0.
// Rule 6: anonymous inline literal via tracked::literal.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::literal(T(0));
}

// Rule 2 / Rule 3: Real(Complex<T>) — real part of a complex (already tracked).
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
    return z.real();
}

// Rule 2 / Rule 3: Imag(Complex<T>) — imag part of a complex (already tracked).
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
    return z.imag();
}

// ---------------------------------------------------------------------------
// Sign.
// C6: qcdloop consumes Sign as a numeric +/-1/0 factor that flows into
// tracked arithmetic (e.g. `TOutput(ql::Sign(...))` and `s * ...`). So this
// is a FLOATING-POINT return (Rule 2), NOT a discrete int (Rule 1).
// ---------------------------------------------------------------------------

// Rule 2 / C6: Sign(Tracked<T>) — returns +1 / 0 / -1 as tracked literal.
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    const T v = x.value();
    // Rule 6: anonymous literal — the sign result carries no user name.
    if (v > T(0)) return tracked::literal(T(1));
    if (v < T(0)) return tracked::literal(T(-1));
    return tracked::literal(T(0));
}

// Rule 3 / C6: Sign(Complex<T>) — complex sign is z / |z| (a complex).
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    auto a = tracked::abs(z);       // Rule 2: tracked real magnitude
    auto re = z.real() / a;
    auto im = z.imag() / a;
    return tracked::Complex<T>(re, im);
}

// ---------------------------------------------------------------------------
// kAbs.
// Rule 2: floating-point magnitude, participates in downstream tracked
// arithmetic — return tracked scalar (not bare double).
// C1: kAbs(Complex<T>) returns tracked::Tracked<T>, matching the library
// contract that Kokkos::abs(complex) returns a real, not a complex.
// ---------------------------------------------------------------------------

// Rule 2: kAbs(Tracked<T>).
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// Rule 2 / Rule 3 -> Rule 2: kAbs(Complex<T>) returns tracked real scalar.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z);
}

// ---------------------------------------------------------------------------
// kLog / kSqrt / kConj.
// Rule 2 for scalars, Rule 3 for complex.
// ---------------------------------------------------------------------------

// Rule 2: kLog(Tracked<T>) delegates to tracked::log.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Rule 3: kLog(Complex<T>) delegates to tracked::log (complex overload).
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// Rule 2: kSqrt(Tracked<T>).
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// Rule 3: kSqrt(Complex<T>).
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 2: kConj on a real is the value itself (identity, no rounding).
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// Rule 3: kConj(Complex<T>) via tracked::conj.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// ---------------------------------------------------------------------------
// Max / Min. Library semantics (see kokkosMaths.h): "returns whichever of
// a, b has the larger |.|". Rule 7: comparisons are done on .value() and
// yield plain bool; the RETURN, however, is a tracked scalar/complex
// (Rule 2 / Rule 3), because the picked value flows on into tracked math.
// ---------------------------------------------------------------------------

// Rule 2 / Rule 7: Max(Tracked<T>, Tracked<T>) — pick the one with larger |v|.
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                                const tracked::Tracked<T>& b) {
    using std::abs;
    // Rule 7: compare on the underlying value; result of the comparison is bool.
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

// Rule 3 / Rule 7: Max(Complex<T>, Complex<T>).
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                                const tracked::Complex<T>& b) {
    // Rule 7: compare magnitudes via .value(); no tracked bool.
    auto am = tracked::abs(a).value();
    auto bm = tracked::abs(b).value();
    return (am > bm) ? a : b;
}

// Rule 2 / Rule 7: Min(Tracked<T>, Tracked<T>).
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                                const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

// Rule 3 / Rule 7: Min(Complex<T>, Complex<T>).
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                                const tracked::Complex<T>& b) {
    auto am = tracked::abs(a).value();
    auto bm = tracked::abs(b).value();
    return (am > bm) ? b : a;
}

// ---------------------------------------------------------------------------
// kPow(base, int exponent).
//
// C7: outranks kokkosMaths.h's generic `template<TOutput,TMass,TScale>
// kPow(TOutput const&, int const&)` primary. Each concrete tracked-typed
// overload carries the leading explicit template parameters TOutput, TMass,
// TScale so that qualified calls like
//     ql::kPow<TOutput, TMass, TScale>(x, n)
// bind to it directly (partial ordering picks the concrete tracked overload
// over the bare-template primary).
//
// C2: no tracked::pow — implement integer powers as a multiply loop over
// tracked operator*.
// ---------------------------------------------------------------------------

// C7 / Rule 2: kPow on tracked real scalar.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base,
                                 const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 4 / Rule 6: promote the bare-1 seed as an anonymous literal.
    tracked::Tracked<T> acc = tracked::literal(T(1));
    for (int i = 0; i < n; ++i) acc = acc * base;   // tracked operator*
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::literal(T(1));
        acc = one / acc;
    }
    return acc;
}

// C7 / Rule 3: kPow on tracked complex.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base,
                                 const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // C2 / Rule 6: seed with a real-1 complex, both components literal(0/1).
    tracked::Complex<T> acc(tracked::literal(T(1)), tracked::literal(T(0)));
    for (int i = 0; i < n; ++i) acc = acc * base;   // tracked complex operator*
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal(T(1)), tracked::literal(T(0)));
        acc = one / acc;
    }
    return acc;
}

// ---------------------------------------------------------------------------
// iszero.
//
// Rule 1 / C6: iszero is consumed EXCLUSIVELY as a branch predicate (`if
// (ql::iszero(...))`), so this is a discrete return — plain bool, never a
// tracked bool. Unwrap through .value() at the boundary.
//
// C7: mirror the library's leading explicit-parameter arity
// <TOutput,TMass,TScale> so qualified calls bind to this overload.
//
// The library defines:
//   iszero<TOutput,TMass,TScale>(TScale const&)
// with one value-parameter overload. We provide constrained overloads on
// tracked scalar and tracked complex (both shapes reach the qualified call
// sites via statically instantiated branches — C3).
// ---------------------------------------------------------------------------

// Rule 1 / C6 / C7: iszero on tracked real scalar.
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    using std::abs;
    // Library threshold is ql::Constants<TScale>::_qlonshellcutoff() == 1e-10.
    // Compare on .value() (Rule 7).
    return abs(x.value()) < T(1e-10);
}

// Rule 1 / C6 / C7: iszero on tracked complex (via |z|.value()).
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Complex<T>& z) {
    using std::abs;
    // |z|^2 = re^2 + im^2; compare its square root against threshold — but
    // we do this in RAW T land because the result is only a branch predicate
    // (Rule 1), not an intermediate that flows into tracked math.
    const T re = z.real().value();
    const T im = z.imag().value();
    const T mag = std::sqrt(re * re + im * im);
    return mag < T(1e-10);
}

// ---------------------------------------------------------------------------
// Htheta (Heaviside step): 0.5 * (1 + sign(x)).
//
// C6: the result is used as a NUMERIC factor in tracked expressions inside
// kokkosUtils.h (see `eta2`, `Rint`). So this is Rule 2 (tracked return),
// NOT Rule 1 — even though the value is discrete-looking (0 / 0.5 / 1).
// ---------------------------------------------------------------------------

// Rule 2 / C6: Htheta on tracked real scalar.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    const T v = x.value();
    // Rule 6: anonymous literals — 0 / 0.5 / 1 have no user name here.
    if (v > T(0)) return tracked::literal(T(1));
    if (v < T(0)) return tracked::literal(T(0));
    return tracked::literal(T(0.5));
}

} // namespace ql

// ===========================================================================
// Rule 5 / C5: ql::Constants<T> partial specializations for tracked scalars.
//
// C5: partial specialization on the tracked scalar (not full explicit) so
// this specialization covers the whole tracked family generically. Every
// named leaf constant is routed through tracked::constant("<name>", T(v))
// so the journal preserves the source-level constant name. Every accessor
// the library's primary template exposes is mirrored here (see
// kokkosMaths.h::Constants<T>).
// ===========================================================================
namespace ql {

// Rule 5 / C5: partial specialization on tracked::Tracked<T> (T = double
// in the driver, but keep it generic).
template <class T>
struct Constants<tracked::Tracked<T>> {

    // -------- Chebyshev coefficient counts / accessors (Rule 5) ------------
    // These are consumed as loop bounds -> Rule 1 discrete int return.
    static constexpr int _num_C() { return 19; }
    static constexpr int _num_B() { return 25; }

    // Rule 5: Chebyshev coefficient — named leaf constant per-index.
    // We reuse the library's coefficient table by defining the primary in
    // terms of double, then wrap into a named tracked::constant. The name
    // includes the index so the journal records which coefficient was used.
    static tracked::Tracked<T> _C(int i) {
        // Mirror kokkosMaths.h::Constants<T>::_C exactly (double-precision table).
        static const double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>("Chebyshev_C[" + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // Rule 5: Bernoulli coefficient.
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
        return tracked::constant<T>("Bernoulli_B[" + std::to_string(i) + "]",
                                    T(coeffs[i]));
    }

    // -------- named leaf constants (Rule 5) ---------------------------------

    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    static tracked::Tracked<T> _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }
    static tracked::Tracked<T> _pi2()  {
        // Rule 5: named product — keep "pi2" as its own named constant.
        return tracked::constant<T>("pi2", T(M_PI) * T(M_PI));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pio3()   { return tracked::constant<T>("pi_over_3",  T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pio6()   { return tracked::constant<T>("pi_over_6",  T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o3()  { return tracked::constant<T>("pi2_over_3", T(M_PI) * T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o6()  { return tracked::constant<T>("pi2_over_6", T(M_PI) * T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Tracked<T> _pi2o12() { return tracked::constant<T>("pi2_over_12",T(M_PI) * T(M_PI) / T(12)); }

    static tracked::Tracked<T> _zero()  { return tracked::constant<T>("zero",  T(0)); }
    static tracked::Tracked<T> _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static tracked::Tracked<T> _one()   { return tracked::constant<T>("one",   T(1)); }
    static tracked::Tracked<T> _two()   { return tracked::constant<T>("two",   T(2)); }
    static tracked::Tracked<T> _three() { return tracked::constant<T>("three", T(3)); }
    static tracked::Tracked<T> _four()  { return tracked::constant<T>("four",  T(4)); }
    static tracked::Tracked<T> _five()  { return tracked::constant<T>("five",  T(5)); }
    static tracked::Tracked<T> _six()   { return tracked::constant<T>("six",   T(6)); }
    static tracked::Tracked<T> _ten()   { return tracked::constant<T>("ten",   T(10)); }

    static tracked::Tracked<T> _eps()   { return tracked::constant<T>("eps_1e-6",  T(1e-6));  }
    static tracked::Tracked<T> _eps4()  { return tracked::constant<T>("eps_1e-4",  T(1e-4));  }
    static tracked::Tracked<T> _eps7()  { return tracked::constant<T>("eps_1e-7",  T(1e-7));  }
    static tracked::Tracked<T> _eps10() { return tracked::constant<T>("eps_1e-10", T(1e-10)); }
    static tracked::Tracked<T> _eps14() { return tracked::constant<T>("eps_1e-14", T(1e-14)); }
    static tracked::Tracked<T> _eps15() { return tracked::constant<T>("eps_1e-15", T(1e-15)); }

    static tracked::Tracked<T> _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static tracked::Tracked<T> _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static tracked::Tracked<T> _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // -------- named complex constants (Rule 3 + Rule 5) --------------------
    // Return the container-of-tracked (Rule 3), with each named component
    // wired through tracked::constant (Rule 5) so both scalar factors show
    // up in the journal with their names.

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero",       T(0)),
            tracked::constant<T>("pi_over_2",  T(M_PI) * T(0.5)));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("pi",   T(M_PI)));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("reps", T(1e-16)));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero",       T(0)),
            tracked::constant<T>("reps_sq",    T(1e-16) * T(1e-16)));
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(
            tracked::constant<T>("zero",       T(0)),
            tracked::constant<T>("ieps50",     T(1e-50)));
    }
};

// Rule 5 / C5: partial specialization on tracked::Complex<T>.
//
// The library's box code sometimes writes `ql::Constants<TOutput>::_zero()`
// with TOutput = tracked::Complex<double>, expecting a TOutput back. So
// scalar accessors here return tracked::Complex<T> (a complex whose imag is
// literal(0)), and complex-named accessors also return tracked::Complex<T>.
// Each named leaf still routes through tracked::constant.
template <class T>
struct Constants<tracked::Complex<T>> {

    // Rule 1: sizes remain plain int (used as loop bounds only).
    static constexpr int _num_C() { return 19; }
    static constexpr int _num_B() { return 25; }

    // Rule 3 / Rule 5: promote real Chebyshev / Bernoulli coefficients into
    // complex — journal keeps the coefficient name via constant().
    static tracked::Complex<T> _C(int i) {
        auto s = Constants<tracked::Tracked<T>>::_C(i);
        return tracked::Complex<T>(s, tracked::literal(T(0)));  // Rule 6 for the zero imag
    }
    static tracked::Complex<T> _B(int i) {
        auto s = Constants<tracked::Tracked<T>>::_B(i);
        return tracked::Complex<T>(s, tracked::literal(T(0)));
    }

    // helper: wrap a real named constant as a complex (imag = literal 0).
    static tracked::Complex<T> _cwrap(const char* name, T v) {
        return tracked::Complex<T>(tracked::constant<T>(name, v),
                                   tracked::literal(T(0)));   // Rule 6
    }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _qlonshellcutoff() { return _cwrap("qlonshellcutoff", T(1e-10)); }

    static tracked::Complex<T> _pi()   { return _cwrap("pi",  T(M_PI)); }
    static tracked::Complex<T> _pi2()  { return _cwrap("pi2", T(M_PI) * T(M_PI)); }

    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _pio3()   { return _cwrap("pi_over_3",   T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _pio6()   { return _cwrap("pi_over_6",   T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _pi2o3()  { return _cwrap("pi2_over_3",  T(M_PI) * T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _pi2o6()  { return _cwrap("pi2_over_6",  T(M_PI) * T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _pi2o12() { return _cwrap("pi2_over_12", T(M_PI) * T(M_PI) / T(12)); }

    static tracked::Complex<T> _zero()  { return _cwrap("zero",  T(0));   }
    static tracked::Complex<T> _half()  { return _cwrap("half",  T(0.5)); }
    static tracked::Complex<T> _one()   { return _cwrap("one",   T(1));   }
    static tracked::Complex<T> _two()   { return _cwrap("two",   T(2));   }
    static tracked::Complex<T> _three() { return _cwrap("three", T(3));   }
    static tracked::Complex<T> _four()  { return _cwrap("four",  T(4));   }
    static tracked::Complex<T> _five()  { return _cwrap("five",  T(5));   }
    static tracked::Complex<T> _six()   { return _cwrap("six",   T(6));   }
    static tracked::Complex<T> _ten()   { return _cwrap("ten",   T(10));  }

    static tracked::Complex<T> _eps()   { return _cwrap("eps_1e-6",  T(1e-6));  }
    static tracked::Complex<T> _eps4()  { return _cwrap("eps_1e-4",  T(1e-4));  }
    static tracked::Complex<T> _eps7()  { return _cwrap("eps_1e-7",  T(1e-7));  }
    static tracked::Complex<T> _eps10() { return _cwrap("eps_1e-10", T(1e-10)); }
    static tracked::Complex<T> _eps14() { return _cwrap("eps_1e-14", T(1e-14)); }
    static tracked::Complex<T> _eps15() { return _cwrap("eps_1e-15", T(1e-15)); }

    static tracked::Complex<T> _xloss()  { return _cwrap("xloss",  T(0.125)); }
    static tracked::Complex<T> _neglig() { return _cwrap("neglig", T(1e-14)); }
    static tracked::Complex<T> _reps()   { return _cwrap("reps",   T(1e-16)); }

    // Rule 3 + Rule 5: named complex-imag constants.
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",      T(0)),
                                   tracked::constant<T>("pi_over_2", T(M_PI) * T(0.5)));
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
        return tracked::Complex<T>(tracked::constant<T>("zero",    T(0)),
                                   tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16)));
    }
    template <class TOutput, class TMass, class TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(tracked::constant<T>("zero",   T(0)),
                                   tracked::constant<T>("ieps50", T(1e-50)));
    }
};

} // namespace ql

// ===========================================================================
// End of shim. All qcdloop templates that follow (kokkosMaths.h,
// kokkosUtils.h, box/*.h, boxGPU.h) now see:
//   * tracked-typed ql::Real / Imag / Sign / kAbs / kLog / kSqrt / kConj /
//     Max / Min / kPow / iszero / Htheta      -> resolved via ordinary lookup
//   * ql::Constants<Tracked<T>> and ql::Constants<Complex<T>> specializations
//   * unary operator+ on Tracked<T> / Complex<T> (found via ADL in tracked)
// so every qualified call at the library's own definition sites binds to a
// tracked-aware overload, and every named constant enters the journal with
// its source-level name.
// ===========================================================================