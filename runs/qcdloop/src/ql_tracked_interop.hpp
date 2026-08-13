// ql_tracked_interop.hpp
// SOURCE_HASH: 25f2b895f28aa7fe2d953e3184974ba497a1a3dc934b31f64c048867df6b43ec
//
// Tracked interop shim for qcdloop's box-integral headers.
// Every overload/specialization carries a rule tag naming the classification
// rule (Rules 1-9, C1-C7) that justified it.
//
// Include order (per the driver): this header is included BEFORE
// kokkosMaths.h / kokkosUtils.h / boxGPU.h so that qualified calls in the
// library templates (ql::Real, ql::Imag, ql::Sign, ql::kAbs, ql::kLog,
// ql::kSqrt, ql::kConj, ql::Max, ql::Min, ql::iszero, ql::Htheta,
// ql::Constants<T>::..., ql::kPow) resolve to the tracked overloads at each
// template's definition point. Non-ADL qualified lookup requires it.
//
// The library defines its scalar/complex/kPow/etc. as function TEMPLATES
// inside namespace ql, so we outrank them by partial ordering (C7): each
// tracked overload constrains its value parameter to the concrete tracked
// type and carries the leading explicit template parameters the call sites
// name. This is not a catch-all Base forwarder.

#pragma once

// ---- Kokkos (needed for Kokkos::Array / Kokkos::View / Kokkos::complex) ----
#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

// ---- Tracked API -----------------------------------------------------------
#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Rule C5 forward declaration: we specialize ql::Constants<T> for the tracked
// scalar and tracked complex BEFORE the library's primary template is defined
// (this header is included before kokkosMaths.h). Forward-declare the primary
// so the specializations parse.
// ---------------------------------------------------------------------------
namespace ql {
    template<typename T> struct Constants;                              // Rule C5: forward decl of library primary
    // KOKKOS_INLINE_FUNCTION is a macro provided by Kokkos_Core.hpp.
}

// ---------------------------------------------------------------------------
// Rule C3: unary operator+ on tracked scalar (identity). Not defined by the
// Tracked API, and the library's macro-heavy code may instantiate it. Added
// as a free function in namespace tracked for ADL, emits no journal record.
// ---------------------------------------------------------------------------
namespace tracked {

    template<class T>
    inline Tracked<T> operator+(const Tracked<T>& a) {   // Rule C3: identity, no journal record
        return a;
    }

    template<class T>
    inline Complex<T> operator+(const Complex<T>& a) {   // Rule C3: identity on tracked complex
        return a;
    }

} // namespace tracked

// ===========================================================================
// ql:: shims
// ===========================================================================
namespace ql {

    // ---- Type aliases for readability ----
    // Rule C1: tracked complex is tracked::Complex<T> (components are already
    // Tracked<T> reals) — NOT tracked::Complex<Tracked<T>>.
    template<class T> using _TS = tracked::Tracked<T>;      // Tracked scalar
    template<class T> using _TC = tracked::Complex<T>;      // Tracked complex

    // =======================================================================
    // Constants<T> — Rule C5 specializations on the tracked scalar and tracked
    // complex. Mirror the library primary's full accessor set so any call in
    // the statically-instantiated call graph finds a member. Each named leaf
    // scalar routes through tracked::constant("<name>", T(value)) so the
    // journal keeps every constant name (Rule 5). The Chebyshev/Bernoulli
    // coefficient tables (_C, _B) are anonymous inline literal tables —
    // Rule 6 (tracked::literal).
    // =======================================================================

    // ---- Specialization on the tracked SCALAR (Rule C5, partial spec) ------
    template<class T>
    struct Constants< tracked::Tracked<T> > {

        using Trk = tracked::Tracked<T>;

        // ---- Chebyshev coefficients (double table, retagged as literals) ---
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_C() { return 19; }                    // Rule 1: discrete count

        static Trk _C(int i) {                                          // Rule 6: anonymous literal table
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

        // ---- Bernoulli coefficients ---------------------------------------
        KOKKOS_INLINE_FUNCTION
        static constexpr int _num_B() { return 25; }                    // Rule 1

        static Trk _B(int i) {                                          // Rule 6
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

        // ---- Named numeric constants (Rule 5: preserve source identifier) --
        template<typename TOutput, typename TMass, typename TScale>
        static Trk _qlonshellcutoff() {                                 // Rule 5
            return tracked::constant<T>("qlonshellcutoff", T(1e-10));
        }

        static Trk _pi()       { return tracked::constant<T>("pi",       T(M_PI)); }              // Rule 5
        static Trk _pi2()      { return tracked::constant<T>("pi2",      T(M_PI) * T(M_PI)); }    // Rule 5

        template<typename TOutput, typename TMass, typename TScale>
        static Trk _pio3()     { return tracked::constant<T>("pio3",     T(M_PI) / T(3)); }       // Rule 5
        template<typename TOutput, typename TMass, typename TScale>
        static Trk _pio6()     { return tracked::constant<T>("pio6",     T(M_PI) / T(6)); }       // Rule 5
        template<typename TOutput, typename TMass, typename TScale>
        static Trk _pi2o3()    { return tracked::constant<T>("pi2o3",    T(M_PI) * T(M_PI) / T(3)); }  // Rule 5
        template<typename TOutput, typename TMass, typename TScale>
        static Trk _pi2o6()    { return tracked::constant<T>("pi2o6",    T(M_PI) * T(M_PI) / T(6)); }  // Rule 5
        template<typename TOutput, typename TMass, typename TScale>
        static Trk _pi2o12()   { return tracked::constant<T>("pi2o12",   T(M_PI) * T(M_PI) / T(12)); } // Rule 5

        static Trk _zero()  { return tracked::constant<T>("zero",  T(0.0)); }   // Rule 5
        static Trk _half()  { return tracked::constant<T>("half",  T(0.5)); }   // Rule 5
        static Trk _one()   { return tracked::constant<T>("one",   T(1.0)); }   // Rule 5
        static Trk _two()   { return tracked::constant<T>("two",   T(2.0)); }   // Rule 5
        static Trk _three() { return tracked::constant<T>("three", T(3.0)); }   // Rule 5
        static Trk _four()  { return tracked::constant<T>("four",  T(4.0)); }   // Rule 5
        static Trk _five()  { return tracked::constant<T>("five",  T(5.0)); }   // Rule 5
        static Trk _six()   { return tracked::constant<T>("six",   T(6.0)); }   // Rule 5
        static Trk _ten()   { return tracked::constant<T>("ten",   T(10.0)); }  // Rule 5

        static Trk _eps()     { return tracked::constant<T>("eps",     T(1e-6));  }  // Rule 5
        static Trk _eps4()    { return tracked::constant<T>("eps4",    T(1e-4));  }  // Rule 5
        static Trk _eps7()    { return tracked::constant<T>("eps7",    T(1e-7));  }  // Rule 5
        static Trk _eps10()   { return tracked::constant<T>("eps10",   T(1e-10)); }  // Rule 5
        static Trk _eps14()   { return tracked::constant<T>("eps14",   T(1e-14)); }  // Rule 5
        static Trk _eps15()   { return tracked::constant<T>("eps15",   T(1e-15)); }  // Rule 5
        static Trk _xloss()   { return tracked::constant<T>("xloss",   T(0.125)); }  // Rule 5
        static Trk _neglig()  { return tracked::constant<T>("neglig",  T(1e-14)); }  // Rule 5
        static Trk _reps()    { return tracked::constant<T>("reps",    T(1e-16)); }  // Rule 5

        // ---- Complex-valued named constants (Rule 3 + Rule 5). Return
        // tracked::Complex<T> because these are consumed as complex values in
        // the library (multiplied with TOutput etc.).
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _2ipi() {                            // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("two_pi", T(2) * T(M_PI));
            return tracked::Complex<T>(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _ipio2() {                           // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("pio2", T(M_PI) * T(0.5));
            return tracked::Complex<T>(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _ipi() {                             // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("pi",   T(M_PI));
            return tracked::Complex<T>(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _ieps() {                            // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("reps", T(1e-16));
            return tracked::Complex<T>(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _ieps2() {                           // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16));
            return tracked::Complex<T>(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static tracked::Complex<T> _ieps50() {                          // Rule 5 + Rule 3
            auto re = tracked::constant<T>("zero",   T(0));
            auto im = tracked::constant<T>("ieps50", T(1e-50));
            return tracked::Complex<T>(re, im);
        }
    };

    // ---- Specialization on the tracked COMPLEX (Rule C5, partial spec) ----
    //
    // When TOutput = tracked::Complex<T> the library asks for Constants<TOutput>
    // and expects TOutput-valued members like _half(), _one(), _two(), _zero(),
    // _ieps50(), etc. Route each through the scalar specialization and lift.
    template<class T>
    struct Constants< tracked::Complex<T> > {

        using Cpx = tracked::Complex<T>;

        static Cpx _zero()  { return Cpx(Constants<tracked::Tracked<T>>::_zero()); }   // Rule 5 + Rule 3
        static Cpx _half()  { return Cpx(Constants<tracked::Tracked<T>>::_half()); }   // Rule 5 + Rule 3
        static Cpx _one()   { return Cpx(Constants<tracked::Tracked<T>>::_one());  }   // Rule 5 + Rule 3
        static Cpx _two()   { return Cpx(Constants<tracked::Tracked<T>>::_two());  }   // Rule 5 + Rule 3
        static Cpx _three() { return Cpx(Constants<tracked::Tracked<T>>::_three());}   // Rule 5 + Rule 3
        static Cpx _four()  { return Cpx(Constants<tracked::Tracked<T>>::_four()); }   // Rule 5 + Rule 3
        static Cpx _five()  { return Cpx(Constants<tracked::Tracked<T>>::_five()); }   // Rule 5 + Rule 3

        // Complex-valued imaginary constants (Rule 5 + Rule 3)
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _2ipi() {
            auto re = tracked::constant<T>("zero",   T(0));
            auto im = tracked::constant<T>("two_pi", T(2) * T(M_PI));
            return Cpx(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _ipio2() {
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("pio2", T(M_PI) * T(0.5));
            return Cpx(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _ipi() {
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("pi",   T(M_PI));
            return Cpx(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _ieps() {
            auto re = tracked::constant<T>("zero", T(0));
            auto im = tracked::constant<T>("reps", T(1e-16));
            return Cpx(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _ieps2() {
            auto re = tracked::constant<T>("zero",    T(0));
            auto im = tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16));
            return Cpx(re, im);
        }
        template<typename TOutput, typename TMass, typename TScale>
        static Cpx _ieps50() {
            auto re = tracked::constant<T>("zero",   T(0));
            auto im = tracked::constant<T>("ieps50", T(1e-50));
            return Cpx(re, im);
        }
    };

    // =======================================================================
    // kPow — the library declares two overloads (TOutput-only, TMass-only) as
    // function templates over three type parameters <TOutput,TMass,TScale>.
    // We must outrank both with constrained overloads on the concrete tracked
    // scalar and tracked complex. Each carries the full leading explicit
    // template parameters (Rule C7) so qualified calls like
    // ql::kPow<TOutput,TMass,TScale>(x, n) bind here.
    //
    // Integer-power body: multiply loop over tracked operator* — Tracked has
    // no pow, per C2 we implement it manually. `exponent` stays a raw int
    // (Rule 1: it's a discrete count).
    // =======================================================================

    template<typename TOutput, typename TMass, typename TScale, class T>    // Rule C7 + Rule 2
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kPow(tracked::Tracked<T> const& base, int const& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;                  // Rule 1: discrete count
        tracked::Tracked<T> temp = tracked::constant<T>("one", T(1));        // Rule 5
        for (int i = 0; i < n; ++i) temp = temp * base;                     // C2: implement pow via *
        if (exponent < 0) {
            tracked::Tracked<T> one_t = tracked::constant<T>("one", T(1));   // Rule 5
            return one_t / temp;
        }
        return temp;
    }

    template<typename TOutput, typename TMass, typename TScale, class T>    // Rule C7 + Rule 3
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kPow(tracked::Complex<T> const& base, int const& exponent) {
        const int n = exponent < 0 ? -exponent : exponent;                  // Rule 1
        auto one_re = tracked::constant<T>("one", T(1));                    // Rule 5
        tracked::Complex<T> temp(one_re);
        for (int i = 0; i < n; ++i) temp = temp * base;                     // C2
        if (exponent < 0) {
            auto one_re2 = tracked::constant<T>("one", T(1));                // Rule 5
            tracked::Complex<T> one_c(one_re2);
            return one_c / temp;
        }
        return temp;
    }

    // =======================================================================
    // Math dispatch functions (kAbs, kLog, kSqrt, kConj) on tracked types.
    // The library's kAbs is a function template; kLog/kSqrt/kConj are too.
    // Each shim constrains to the concrete tracked type (Rule C7).
    //
    // kAbs of a complex returns a REAL magnitude (tracked scalar) — Rule 6
    // for the type domain: the library's kAbs(complex) explicitly returns
    // double. We mirror that: kAbs(tracked complex) -> tracked scalar.
    // =======================================================================

    template<class T>                                                       // Rule 2
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kAbs(tracked::Tracked<T> const& x) {
        return tracked::abs(x);                                             // uses tracked::abs from ops.hpp
    }

    template<class T>                                                       // Rule 2 (real magnitude of complex)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kAbs(tracked::Complex<T> const& x) {
        return tracked::abs(x);                                             // tracked::abs on complex -> real
    }

    template<class T>                                                       // Rule 2
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kLog(tracked::Tracked<T> const& x) {
        return tracked::log(x);
    }

    template<class T>                                                       // Rule 3
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kLog(tracked::Complex<T> const& x) {
        return tracked::log(x);
    }

    template<class T>                                                       // Rule 2
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kSqrt(tracked::Tracked<T> const& x) {
        return tracked::sqrt(x);
    }

    template<class T>                                                       // Rule 3
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kSqrt(tracked::Complex<T> const& x) {
        return tracked::sqrt(x);
    }

    template<class T>                                                       // Rule 2 (real conj == identity)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> kConj(tracked::Tracked<T> const& x) {
        return x;
    }

    template<class T>                                                       // Rule 3
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> kConj(tracked::Complex<T> const& x) {
        return tracked::conj(x);
    }

    // =======================================================================
    // Real / Imag / Sign — component extractors and sign-of-value.
    //
    // Real(tracked scalar) is identity — Rule 2, tracked scalar in, tracked
    // scalar out (feeds arithmetic downstream, per C6).
    //
    // Imag(tracked scalar) must return a tracked scalar zero (a literal, not
    // the named "zero" constant — the imaginary part of a real is a padding
    // artifact, cf. tracked::Complex's own choice for this case). Rule 6.
    //
    // Real/Imag on tracked complex — component accessors, Rule 2.
    //
    // Sign — Rule C6: the library's own overload set is (double -> int) and
    // (complex -> complex). Distinguish by USE:
    //   * Sign(scalar): result is a numeric ±1/0 that gets MULTIPLIED into
    //     tracked expressions (e.g. `TOutput(ql::Sign(ql::Real(k12))) * ...`),
    //     so it MUST return a tracked scalar preserving provenance (Rule 2 +
    //     C6). We reproduce the ±1/0 as a fresh anonymous literal (Rule 6)
    //     — the sign is a runtime-selected value, not a named constant.
    //   * Sign(complex): library returns x / |x| (a complex direction). Rule 3
    //     — return tracked::Complex<T>.
    // =======================================================================

    template<class T>                                                       // Rule 2 (identity)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Real(tracked::Tracked<T> const& x) {
        return x;
    }

    template<class T>                                                       // Rule 6
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Imag(tracked::Tracked<T> const& /*x*/) {
        return tracked::literal<T>(T(0));
    }

    template<class T>                                                       // Rule 2 (component of complex)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Real(tracked::Complex<T> const& x) {
        return x.real();
    }

    template<class T>                                                       // Rule 2 (component of complex)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Imag(tracked::Complex<T> const& x) {
        return x.imag();
    }

    template<class T>                                                       // Rule 2 + Rule C6 + Rule 7
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Sign(tracked::Tracked<T> const& x) {
        // Rule 7: comparison on tracked values goes through .value().
        // Rule C6: result flows into tracked arithmetic (mul into TOutput),
        // so return the tracked type with a fresh literal for the ±1/0 value.
        const T v = x.value();
        const T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
        return tracked::literal<T>(s);                                      // Rule 6: anonymous ±1/0
    }

    template<class T>                                                       // Rule 3 + Rule C6
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Sign(tracked::Complex<T> const& x) {
        // Library defines Sign(complex) = x / |x|. |x| is real (tracked scalar).
        auto mag = tracked::abs(x);                                         // Rule 2
        return x / mag;                                                     // Rule 3 (complex / scalar)
    }

    // =======================================================================
    // Max / Min / Htheta / iszero
    //
    // Max/Min on tracked types: the library compares by magnitude and returns
    // one of the two operands. Rule 7: comparison via .value(); Rule 2/3:
    // returns tracked type (preserves provenance of the winning operand).
    //
    // Htheta: 0.5 * (1 + Sign(x)) — result flows into tracked math (Rule 2).
    //
    // iszero: purely a discrete branch selector (used inside `if`), returns
    // bool — Rule 1 (and Rule 7 for the comparison via .value()).
    // =======================================================================

    template<class T>                                                       // Rule 2 + Rule 7
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Max(tracked::Tracked<T> const& a, tracked::Tracked<T> const& b) {
        using std::abs;
        return (abs(a.value()) > abs(b.value())) ? a : b;                   // Rule 7: compare via .value()
    }

    template<class T>                                                       // Rule 3 + Rule 7
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Max(tracked::Complex<T> const& a, tracked::Complex<T> const& b) {
        auto amag = tracked::abs(a);                                        // Rule 2
        auto bmag = tracked::abs(b);
        return (amag.value() > bmag.value()) ? a : b;                       // Rule 7
    }

    template<class T>                                                       // Rule 2 + Rule 7
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Min(tracked::Tracked<T> const& a, tracked::Tracked<T> const& b) {
        using std::abs;
        return (abs(a.value()) > abs(b.value())) ? b : a;                   // Rule 7
    }

    template<class T>                                                       // Rule 3 + Rule 7
    KOKKOS_INLINE_FUNCTION
    tracked::Complex<T> Min(tracked::Complex<T> const& a, tracked::Complex<T> const& b) {
        auto amag = tracked::abs(a);
        auto bmag = tracked::abs(b);
        return (amag.value() > bmag.value()) ? b : a;                       // Rule 7
    }

    template<class T>                                                       // Rule 2 (feeds arithmetic)
    KOKKOS_INLINE_FUNCTION
    tracked::Tracked<T> Htheta(tracked::Tracked<T> const& x) {
        // 0.5 * (1 + Sign(x)) — build in tracked arithmetic.
        auto s    = Sign(x);                                                // Rule 2
        auto one  = tracked::constant<T>("one",  T(1));                     // Rule 5
        auto half = tracked::constant<T>("half", T(0.5));                   // Rule 5
        return half * (one + s);
    }

    // iszero (Rule C7): the library's own iszero is a function template over
    // <TOutput,TMass,TScale> taking a TScale — used purely as a discrete
    // branch selector (Rule 1) inside `if`. Provide overloads for both the
    // tracked scalar and the tracked complex (the library instantiates it
    // with TScale = tracked::Tracked<T> and, for some call sites, with
    // TScale = tracked::Complex<T> when the tracked complex is the scale
    // type; the safe move is to shim both). Rule 7 governs the comparison.
    template<typename TOutput, typename TMass, typename TScale, class T>    // Rule C7 + Rule 1
    KOKKOS_INLINE_FUNCTION
    bool iszero(tracked::Tracked<T> const& x) {
        using std::abs;
        // Match the library's cutoff constant (_qlonshellcutoff = 1e-10).
        return abs(x.value()) < T(1e-10);                                   // Rule 7 + Rule 1
    }

    template<typename TOutput, typename TMass, typename TScale, class T>    // Rule C7 + Rule 1
    KOKKOS_INLINE_FUNCTION
    bool iszero(tracked::Complex<T> const& x) {
        using std::abs;
        // |x| < cutoff via component magnitudes.
        auto mag = tracked::abs(x);                                         // Rule 2 (produces tracked scalar)
        return abs(mag.value()) < T(1e-10);                                 // Rule 7 + Rule 1
    }

} // namespace ql