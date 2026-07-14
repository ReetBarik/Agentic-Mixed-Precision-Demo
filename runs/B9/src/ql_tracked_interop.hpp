// ql_tracked_interop.hpp
// Tracked<T> interop shim for QCDLoop+Kokkos box integrals (B9 spike / B1m family).
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// Include order (see micro_driver.cpp): this header MUST precede kokkosMaths.h,
// kokkosUtils.h, and boxGPU.h so tracked overloads are visible at the point
// each ql:: template body is parsed — QCDLoop calls ql::Real / ql::Imag /
// ql::Sign / ql::kAbs / ql::kLog / ql::kSqrt / ql::kConj / ql::kPow / ql::Max /
// ql::Min / ql::iszero via QUALIFIED names (ADL does not apply).
//
// Design summary:
//   * TScale = TMass = tracked::Tracked<double>; TOutput = tracked::Complex<double>.
//     (C1: tracked::Complex<T> already wraps two Tracked<T> reals; the underlying
//     scalar T is `double`.)
//   * Every ql:: helper QCDLoop calls on a tracked scalar/complex is overloaded
//     here as a strictly-more-specialized function-template overload that carries
//     the same leading explicit template parameters QCDLoop's call sites name
//     (C7: outrank library primaries via partial ordering; carry <TOutput,TMass,
//     TScale> on the constrained overload so ql::foo<A,B,C>(x) binds directly).
//   * ql::Constants<T> is specialized on tracked::Tracked<T>; we forward-declare
//     the primary inside namespace ql first (C5) because this shim is included
//     before kokkosMaths.h defines it.
//   * Discrete-vs-floating classification follows USE (C6): ql::Sign on a
//     TRACKED scalar is consumed as a numeric ±1/0 folded back into tracked
//     expressions (e.g. `TOutput(ql::Sign(...))`), so it must return the tracked
//     scalar (Rule 2). ql::iszero returns bool (Rule 1 — used only in `if`).
//   * Execution-space annotation: KOKKOS_INLINE_FUNCTION is NOT emitted on the
//     tracked overloads (C4/Rule 8): the driver runs a plain host loop (no
//     parallel_for), and tracked ops journal to std::string — host-only. The
//     absence of KOKKOS_INLINE_FUNCTION on the OVERLOAD is fine: qcdloop templates
//     that carry KOKKOS_INLINE_FUNCTION themselves still compile when instantiated
//     on the host, and they simply forward through these overloads.

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// C3: identity operators the Tracked library does not define. QCDLoop writes
// bare `+x` in a couple of places; provide a no-op unary plus for tracked
// scalars and tracked complex. No journal record: unary + introduces no
// rounding. Found via ADL on tracked::Tracked / tracked::Complex.
// ---------------------------------------------------------------------------
namespace tracked {
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; }   // Rule C3

    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) { return a; }   // Rule C3
} // namespace tracked

// ---------------------------------------------------------------------------
// C5: forward-declare ql::Constants primary so our partial specialization on
// tracked::Tracked<T> parses. kokkosMaths.h supplies the full primary later
// in the same translation unit.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;                          // Rule C5
}

namespace ql {

// ===========================================================================
// Constants<tracked::Tracked<T>> — partial specialization keyed on the tracked
// scalar (Rule 5 / C5). Mirrors the FULL member interface of the library
// primary in kokkosMaths.h; every named leaf routes through tracked::constant
// so the source identifier survives in prov_consts. Chebyshev (_C) and
// Bernoulli (_B) coefficient tables are anonymous numeric literals — they are
// series-expansion coefficients, not user-facing constants — so they use
// tracked::literal (Rule 6).
// ===========================================================================

template <typename T>
struct Constants<tracked::Tracked<T>> {                              // Rule 5 / C5
    using Tr = tracked::Tracked<T>;

    // ---- Chebyshev table for ddilog --------------------------------------
    // Rule 6: numeric coefficients with no user-facing name -> literal().
    static int _num_C() { return 19; }                               // Rule 1: discrete count

    static Tr _C(int i) {                                            // Rule 6
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
        return tracked::literal<T>(T(coeffs[i]));                    // Rule 6
    }

    // ---- Bernoulli table for li2series -----------------------------------
    static int _num_B() { return 25; }                               // Rule 1

    static Tr _B(int i) {                                            // Rule 6
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
        return tracked::literal<T>(T(coeffs[i]));                    // Rule 6
    }

    // ---- Named scalar constants ------------------------------------------
    // Rule 5: every named leaf preserved by name via tracked::constant.
    template <typename TOutput, typename TMass, typename TScale>
    static Tr _qlonshellcutoff() { return tracked::constant<T>("qlonshellcutoff", T(1e-10)); }  // Rule 5

    static Tr _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }                         // Rule 5
    static Tr _pi2()  { auto p = _pi(); return p * p; }                                         // Rule 2 (derived)

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pio3()  { return _pi() / tracked::constant<T>("three", T(3)); }                  // Rule 5

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pio6()  { return _pi() / tracked::constant<T>("six",   T(6)); }                  // Rule 5

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o3() { return _pi() * _pio3<TOutput, TMass, TScale>(); }                      // Rule 2 (derived)

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o6() { return _pi() * _pio6<TOutput, TMass, TScale>(); }                      // Rule 2 (derived)

    template <typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o12(){ return _pi2() / tracked::constant<T>("twelve", T(12)); }               // Rule 5

    static Tr _zero()  { return tracked::constant<T>("zero",  T(0));   }                        // Rule 5
    static Tr _half()  { return tracked::constant<T>("half",  T(0.5)); }                        // Rule 5
    static Tr _one()   { return tracked::constant<T>("one",   T(1));   }                        // Rule 5
    static Tr _two()   { return tracked::constant<T>("two",   T(2));   }                        // Rule 5
    static Tr _three() { return tracked::constant<T>("three", T(3));   }                        // Rule 5
    static Tr _four()  { return tracked::constant<T>("four",  T(4));   }                        // Rule 5
    static Tr _five()  { return tracked::constant<T>("five",  T(5));   }                        // Rule 5
    static Tr _six()   { return tracked::constant<T>("six",   T(6));   }                        // Rule 5
    static Tr _ten()   { return tracked::constant<T>("ten",   T(10));  }                        // Rule 5

    static Tr _eps()   { return tracked::constant<T>("eps",   T(1e-6));  }                      // Rule 5
    static Tr _eps4()  { return tracked::constant<T>("eps4",  T(1e-4));  }                      // Rule 5
    static Tr _eps7()  { return tracked::constant<T>("eps7",  T(1e-7));  }                      // Rule 5
    static Tr _eps10() { return tracked::constant<T>("eps10", T(1e-10)); }                      // Rule 5
    static Tr _eps14() { return tracked::constant<T>("eps14", T(1e-14)); }                      // Rule 5
    static Tr _eps15() { return tracked::constant<T>("eps15", T(1e-15)); }                      // Rule 5
    static Tr _xloss() { return tracked::constant<T>("xloss", T(0.125));  }                     // Rule 5
    static Tr _neglig(){ return tracked::constant<T>("neglig",T(1e-14)); }                      // Rule 5
    static Tr _reps()  { return tracked::constant<T>("reps",  T(1e-16)); }                      // Rule 5

    // ---- Complex-valued constants ----------------------------------------
    // Rule 3: container of tracked -> tracked::Complex<T>, not
    // tracked::Tracked<Complex<T>>. Real/imag parts route through named
    // constant() calls preserving the source identifier.
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {                                                       // Rule 3 / Rule 5
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("two",  T(2)) * tracked::constant<T>("pi", T(M_PI)));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {                                                      // Rule 3 / Rule 5
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi",   T(M_PI)) * tracked::constant<T>("half", T(0.5)));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {                                                        // Rule 3 / Rule 5
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi",   T(M_PI)));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {                                                       // Rule 3 / Rule 5
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("reps", T(1e-16)));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {                                                      // Rule 3 / Rule 5
        auto r = tracked::constant<T>("reps", T(1e-16));
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)), r * r);
    }

    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {                                                     // Rule 3 / Rule 5
        return tracked::Complex<T>(tracked::constant<T>("zero",   T(0)),
                                   tracked::constant<T>("ieps50", T(1e-50)));
    }
};

// ===========================================================================
// ql::kPow — integer power via multiply loop.
// C2: no tracked::pow exists, so implement as a fold of tracked operator*.
// C7: constrained on the concrete tracked value type; carries the full
// <TOutput,TMass,TScale> leading explicit-parameter list so qualified calls
// ql::kPow<TOutput,TMass,TScale>(x, n) bind here in preference to the
// library primary. Rule 2 (returns tracked).
// ===========================================================================

// Overload for tracked scalar operand.
template <typename TOutput, typename TMass, typename TScale, typename T>
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {  // Rule 2 / C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> temp = tracked::literal<T>(T(1));                          // Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;                                // C2: fold *
    if (exponent < 0) return tracked::literal<T>(T(1)) / temp;                     // Rule 6
    return temp;
}

// Overload for tracked complex operand.
template <typename TOutput, typename TMass, typename TScale, typename T>
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {   // Rule 3 / C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> temp(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));// Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;                                // C2
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));
        return one / temp;
    }
    return temp;
}

// ===========================================================================
// Elementary math dispatchers. QCDLoop uses BOTH generic ql::kAbs(x) (deduced)
// and, in tracked::-typed positions, expects the tracked overloads to win by
// partial ordering (C7). We provide constrained overloads for scalar and
// complex tracked types. No leading explicit template parameters here — these
// helpers are always called with a single deduced argument.
// ===========================================================================

// ---- kAbs -----------------------------------------------------------------
template <class T>
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {                           // Rule 2 / C7
    return tracked::abs(x);
}

template <class T>
tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {                           // Rule 2 (result is real magnitude) / C7
    return tracked::abs(x);                                                        // container -> real Tracked<T>
}

// ---- kLog -----------------------------------------------------------------
template <class T>
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {                           // Rule 2 / C7
    return tracked::log(x);
}

template <class T>
tracked::Complex<T> kLog(const tracked::Complex<T>& x) {                           // Rule 3 / C7
    return tracked::log(x);
}

// ---- kSqrt ----------------------------------------------------------------
template <class T>
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {                          // Rule 2 / C7
    return tracked::sqrt(x);
}

template <class T>
tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {                          // Rule 3 / C7
    return tracked::sqrt(x);
}

// ---- kConj ----------------------------------------------------------------
// A real scalar is its own conjugate — pass through (no journal record; the
// identity is an implementation-level no-op, mirroring kokkosMaths.h's
// double overload of kConj which returns the argument unchanged).
template <class T>
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {                          // Rule 2 / C7
    return x;
}

template <class T>
tracked::Complex<T> kConj(const tracked::Complex<T>& x) {                          // Rule 3 / C7
    return tracked::conj(x);
}

// ===========================================================================
// Real / Imag / Sign / Max / Min / Htheta — projections and selectors.
//
// C6: classify by USE, not by name. QCDLoop feeds Real/Imag/Sign results back
// into tracked arithmetic (`TOutput(ql::Sign(...))`, `ql::Real(x) * ieps`,
// etc.), so these MUST return the tracked scalar to preserve provenance
// (Rule 2). Reserving raw double would sever the graph.
// ===========================================================================

// ---- Real -----------------------------------------------------------------
template <class T>
tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {                           // Rule 2 / C6 / C7
    return x;                                                                      // identity for real scalar
}

template <class T>
tracked::Tracked<T> Real(const tracked::Complex<T>& x) {                           // Rule 2 / C6 / C7
    return x.real();                                                               // real component of tracked complex
}

// ---- Imag -----------------------------------------------------------------
// Real scalar's imaginary part: structural zero. Use literal() (Rule 6) —
// this padding zero was never named by the user, mirroring Complex(T re)'s
// treatment of its zero imag component in complex.hpp.
template <class T>
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {                       // Rule 2 / C6 / C7
    return tracked::literal<T>(T(0));                                              // Rule 6
}

template <class T>
tracked::Tracked<T> Imag(const tracked::Complex<T>& x) {                           // Rule 2 / C6 / C7
    return x.imag();
}

// ---- Sign -----------------------------------------------------------------
// C6: Sign(x) is consumed as a numeric ±1 / 0 folded into tracked arithmetic
// (e.g. `TOutput(ql::Sign(ql::Real(k12)))` inside sqrt-branching, `ir13 *
// ql::Sign(ql::Real(r24))`), so it MUST return the tracked scalar
// (Rule 2) — not raw int. The ±1/0 selector is a runtime choice, so it enters
// the graph as an anonymous literal (Rule 6), matching complex.hpp::sqrt's
// treatment of sign(im).
template <class T>
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {                           // Rule 2 / C6 / C7
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return tracked::literal<T>(s);                                                 // Rule 6
}

template <class T>
tracked::Complex<T> Sign(const tracked::Complex<T>& x) {                           // Rule 3 / C6 / C7 (z / |z|)
    return x / tracked::Complex<T>(tracked::abs(x));
}

// ---- Max / Min ------------------------------------------------------------
// QCDLoop's Max/Min compare by |·| and return one of the operands unchanged.
// Comparison uses raw values (Rule 7); returned tracked value keeps its
// provenance intact.
template <class T>
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {   // Rule 2 / Rule 7 / C7
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

template <class T>
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {   // Rule 3 / Rule 7 / C7
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}

template <class T>
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {   // Rule 2 / Rule 7 / C7
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

template <class T>
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {   // Rule 3 / Rule 7 / C7
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}

// ---- Htheta ---------------------------------------------------------------
// Heaviside — 0/1 output fed back into tracked arithmetic (multiplied into
// 2ipi expressions in eta2). C6 -> Rule 2, anonymous literal (Rule 6).
template <class T>
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {                          // Rule 2 / C6 / C7
    T s = (T(0) < x.value()) - (x.value() < T(0));
    T h = T(0.5) * (T(1) + s);
    return tracked::literal<T>(h);                                                  // Rule 6
}

// ===========================================================================
// iszero — pure discrete predicate (used only as an `if` condition), so
// Rule 1 applies: return raw bool. Never Tracked<bool>. Full explicit
// specialization on the tracked scalar so qualified
// ql::iszero<TOutput,TMass,TScale>(x) selects it.
// ===========================================================================

template <typename TOutput, typename TMass, typename TScale, typename T>
bool iszero(const tracked::Tracked<T>& x) {                                         // Rule 1 / C7
    using std::abs;
    // Match kokkosMaths.h's cutoff (1e-10).
    return abs(x.value()) < T(1e-10);
}

// Convenience overload: ql::iszero on a bare double literal appears
// nowhere in the driver's tracked path, but a couple of qcdloop call
// sites pass expressions that already unwrap to T. Provide only when T
// is exactly the tracked scalar — no risk of catch-all ambiguity.

// ===========================================================================
// printDoubleBits — QCDLoop's template overload for masses; only ever
// called from a debug-print path never exercised in the driver. Provide a
// no-op so instantiation succeeds. Rule 1 (returns void, discrete side
// effect).
// ===========================================================================
template <class T>
void printDoubleBits(const tracked::Tracked<T>& /*x*/) {                            // Rule 1 / C7
    // Intentional no-op: journaling captures values through the id, not
    // through this ad-hoc printer.
}

} // namespace ql