// ql_tracked_interop.hpp
// Tracked interop shim for the qcdloop (ql) box-integral kernels, so the
// driver can instantiate ql::BO with T = Tracked<double> and
// Complex<Tracked<double>>.
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
#include <string>

// -----------------------------------------------------------------------------
// Rule 5 / C5: forward-declare ql::Constants so our partial specialization on
// tracked::Tracked<T> parses before the primary template definition in
// kokkosMaths.h (this shim is included FIRST per the driver's include order).
// -----------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;   // Rule 5 / C5: forward decl of library primary.
}

namespace ql {

// =============================================================================
// Rule 5 / C5: Partial specialization of Constants<T> for tracked scalars.
// Mirrors the FULL member interface of the library primary (kokkosMaths.h),
// routing every named leaf constant through tracked::constant("<name>", T(v))
// so no library symbol is lost and each constant keeps its name in the journal.
// Chebyshev/Bernoulli coefficient tables and _qlonshellcutoff/_eps* thresholds
// are also emitted as named constants; unnamed helper composites (e.g. _pi2)
// go through tracked arithmetic on already-named constants.
// =============================================================================
template <class T>
struct Constants<tracked::Tracked<T>> {

    using Trk = tracked::Tracked<T>;

    // ---- Chebyshev coefficient table for ddilog ---------------------------
    // Rule 5: every element is a named constant (schematic name "_C[i]").
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }  // Rule 1: pure integer count.

    static Trk _C(int i) {
        // Rule 5: named constants preserve identity in the journal.
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
        return tracked::constant<T>(std::string("_C[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // ---- Bernoulli coefficient table for li2series ------------------------
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }  // Rule 1: pure integer count.

    static Trk _B(int i) {
        // Rule 5: named constants.
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
        return tracked::constant<T>(std::string("_B[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // ---- Onshell / eps thresholds (all named) -----------------------------
    // Rule 5: every threshold is a designator in the library and keeps its name.
    template <typename TOutput, typename TMass, typename TScale>
    static Trk _qlonshellcutoff() { return tracked::constant<T>("_qlonshellcutoff", T(1e-10)); }

    static Trk _pi()   { return tracked::constant<T>("_pi",   T(M_PI)); }
    static Trk _pi2()  {
        // Named composite: preserve "_pi2" identity even though it derives from _pi.
        return tracked::constant<T>("_pi2", T(M_PI) * T(M_PI));
    }

    template <typename TOutput, typename TMass, typename TScale>
    static Trk _pio3()  { return tracked::constant<T>("_pio3",  T(M_PI) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Trk _pio6()  { return tracked::constant<T>("_pio6",  T(M_PI) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Trk _pi2o3() { return tracked::constant<T>("_pi2o3", (T(M_PI) * T(M_PI)) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Trk _pi2o6() { return tracked::constant<T>("_pi2o6", (T(M_PI) * T(M_PI)) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    static Trk _pi2o12(){ return tracked::constant<T>("_pi2o12",(T(M_PI) * T(M_PI)) / T(12)); }

    // ---- Small integer constants (named per library) ----------------------
    // Rule 5: library refers to these by named accessors, so they must be named.
    static Trk _zero()  { return tracked::constant<T>("_zero",  T(0));   }
    static Trk _half()  { return tracked::constant<T>("_half",  T(0.5)); }
    static Trk _one()   { return tracked::constant<T>("_one",   T(1));   }
    static Trk _two()   { return tracked::constant<T>("_two",   T(2));   }
    static Trk _three() { return tracked::constant<T>("_three", T(3));   }
    static Trk _four()  { return tracked::constant<T>("_four",  T(4));   }
    static Trk _five()  { return tracked::constant<T>("_five",  T(5));   }
    static Trk _six()   { return tracked::constant<T>("_six",   T(6));   }
    static Trk _ten()   { return tracked::constant<T>("_ten",   T(10));  }

    // ---- Small floating-point tolerances (named) --------------------------
    static Trk _eps()    { return tracked::constant<T>("_eps",    T(1e-6));  }
    static Trk _eps4()   { return tracked::constant<T>("_eps4",   T(1e-4));  }
    static Trk _eps7()   { return tracked::constant<T>("_eps7",   T(1e-7));  }
    static Trk _eps10()  { return tracked::constant<T>("_eps10",  T(1e-10)); }
    static Trk _eps14()  { return tracked::constant<T>("_eps14",  T(1e-14)); }
    static Trk _eps15()  { return tracked::constant<T>("_eps15",  T(1e-15)); }
    static Trk _xloss()  { return tracked::constant<T>("_xloss",  T(0.125)); }
    static Trk _neglig() { return tracked::constant<T>("_neglig", T(1e-14)); }
    static Trk _reps()   { return tracked::constant<T>("_reps",   T(1e-16)); }

    // ---- Complex-valued named constants -----------------------------------
    // Rule 3: containers of tracked -> Complex<Tracked<T>>, not Tracked<Complex<T>>.
    // Rule 5: each keeps its named-constant identity via tracked::constant.
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {
        // Rule 3 + Rule 5: complex (0, 2*pi) built from named real components.
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0)),
            tracked::constant<T>("_2pi",  T(2) * T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",  T(0)),
            tracked::constant<T>("_pio2",  T(M_PI) * T(0.5)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0)),
            tracked::constant<T>("_pi",   T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0)),
            tracked::constant<T>("_reps", T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",     T(0)),
            tracked::constant<T>("_reps_sq",  T(1e-16) * T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {
        // Rule 5: preserve "_ieps50" name in the journal.
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",   T(0)),
            tracked::constant<T>("_ieps50", T(1e-50)));
    }
};

// =============================================================================
// Rule 5 / C5: Partial specialization of Constants<Complex<T>> for tracked
// complex outputs. The library uses Constants<TOutput>::_one() etc. inside the
// box templates when TOutput is complex, so the complex specialization is
// required. Each accessor returns a tracked::Complex<T> whose real component
// is a named constant.
// =============================================================================
template <class T>
struct Constants<tracked::Complex<T>> {

    using Cx  = tracked::Complex<T>;
    using Trk = tracked::Tracked<T>;

    // Integer-count / no-value accessors — Rule 1.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Scalar-valued factories return tracked complex (Rule 3 container rule).
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _qlonshellcutoff() {
        return Cx(tracked::constant<T>("_qlonshellcutoff", T(1e-10)),
                  tracked::constant<T>("_zero", T(0)));
    }

    static Cx _pi()   { return Cx(tracked::constant<T>("_pi",  T(M_PI)),           tracked::constant<T>("_zero", T(0))); }
    static Cx _pi2()  { return Cx(tracked::constant<T>("_pi2", T(M_PI)*T(M_PI)),   tracked::constant<T>("_zero", T(0))); }

    template <typename TOutput, typename TMass, typename TScale>
    static Cx _pio3()  { return Cx(tracked::constant<T>("_pio3",  T(M_PI)/T(3)), tracked::constant<T>("_zero", T(0))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _pio6()  { return Cx(tracked::constant<T>("_pio6",  T(M_PI)/T(6)), tracked::constant<T>("_zero", T(0))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _pi2o3() { return Cx(tracked::constant<T>("_pi2o3", (T(M_PI)*T(M_PI))/T(3)),  tracked::constant<T>("_zero", T(0))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _pi2o6() { return Cx(tracked::constant<T>("_pi2o6", (T(M_PI)*T(M_PI))/T(6)),  tracked::constant<T>("_zero", T(0))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _pi2o12(){ return Cx(tracked::constant<T>("_pi2o12",(T(M_PI)*T(M_PI))/T(12)), tracked::constant<T>("_zero", T(0))); }

    static Cx _zero()  { return Cx(tracked::constant<T>("_zero",  T(0)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _half()  { return Cx(tracked::constant<T>("_half",  T(0.5)), tracked::constant<T>("_zero", T(0))); }
    static Cx _one()   { return Cx(tracked::constant<T>("_one",   T(1)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _two()   { return Cx(tracked::constant<T>("_two",   T(2)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _three() { return Cx(tracked::constant<T>("_three", T(3)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _four()  { return Cx(tracked::constant<T>("_four",  T(4)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _five()  { return Cx(tracked::constant<T>("_five",  T(5)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _six()   { return Cx(tracked::constant<T>("_six",   T(6)),   tracked::constant<T>("_zero", T(0))); }
    static Cx _ten()   { return Cx(tracked::constant<T>("_ten",   T(10)),  tracked::constant<T>("_zero", T(0))); }

    static Cx _eps()    { return Cx(tracked::constant<T>("_eps",    T(1e-6)),  tracked::constant<T>("_zero", T(0))); }
    static Cx _eps4()   { return Cx(tracked::constant<T>("_eps4",   T(1e-4)),  tracked::constant<T>("_zero", T(0))); }
    static Cx _eps7()   { return Cx(tracked::constant<T>("_eps7",   T(1e-7)),  tracked::constant<T>("_zero", T(0))); }
    static Cx _eps10()  { return Cx(tracked::constant<T>("_eps10",  T(1e-10)), tracked::constant<T>("_zero", T(0))); }
    static Cx _eps14()  { return Cx(tracked::constant<T>("_eps14",  T(1e-14)), tracked::constant<T>("_zero", T(0))); }
    static Cx _eps15()  { return Cx(tracked::constant<T>("_eps15",  T(1e-15)), tracked::constant<T>("_zero", T(0))); }
    static Cx _xloss()  { return Cx(tracked::constant<T>("_xloss",  T(0.125)), tracked::constant<T>("_zero", T(0))); }
    static Cx _neglig() { return Cx(tracked::constant<T>("_neglig", T(1e-14)), tracked::constant<T>("_zero", T(0))); }
    static Cx _reps()   { return Cx(tracked::constant<T>("_reps",   T(1e-16)), tracked::constant<T>("_zero", T(0))); }

    // Complex named constants: same shape as Tracked<T> specialization above.
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _2ipi()  { return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_2pi",  T(2)*T(M_PI))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ipio2() { return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_pio2", T(M_PI)*T(0.5))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ipi()   { return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_pi",   T(M_PI))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ieps()  { return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_reps", T(1e-16))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ieps2() { return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_reps_sq", T(1e-16)*T(1e-16))); }
    template <typename TOutput, typename TMass, typename TScale>
    static Cx _ieps50(){ return Cx(tracked::constant<T>("_zero", T(0)), tracked::constant<T>("_ieps50", T(1e-50))); }
};

// =============================================================================
// Scalar helpers: Real, Imag, Sign, kAbs, kLog, kSqrt, kConj, Max, Min, Htheta
// All are called via QUALIFIED names (ql::Foo) inside library templates, so
// these overloads must live in namespace ql. Rule 8 / C4: driver dispatches
// tracked calls from a plain host loop, no execution-space annotation needed.
// =============================================================================

// ---- Real / Imag ------------------------------------------------------------
// Rule 2: real part of a scalar carries error provenance -> return tracked.
template <class T>
tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    // Rule 2: identity on the real component; no journal record (C3-style identity).
    return x;
}
// Rule 2 + Rule 3: real part of tracked complex is a tracked scalar.
template <class T>
tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
    return z.real();
}

// Rule 2: imaginary part of tracked scalar is defined as zero, and the library
// consumes it in floating-point expressions -> return tracked literal(0).
template <class T>
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Rule 6: anonymous inline zero -> literal.
    return tracked::literal<T>(T(0));
}
template <class T>
tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
    // Rule 3: component of a tracked complex is a tracked scalar.
    return z.imag();
}

// ---- Sign -------------------------------------------------------------------
// Rule 6 (C6): Sign(x) yields ±1 or 0 and the library then multiplies/adds it
// into floating-point expressions (e.g. `TOutput(ql::Sign(...)) * ...`), so
// per C6 this is a FLOATING-POINT return and must be tracked, not raw int.
template <class T>
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    // Rule 7: comparisons on tracked -> plain bool via .value().
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    // Rule 6: anonymous literal (runtime-selected ±1/0, not a named constant).
    return tracked::literal<T>(s);
}
// Sign for tracked complex: z / |z|. Rule 3: container of tracked.
template <class T>
tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    // Rule 2: |z| is tracked; result Complex<Tracked<T>> per Rule 3.
    auto mag = tracked::abs(z);
    return tracked::Complex<T>(z.real() / mag, z.imag() / mag);
}

// ---- kAbs -------------------------------------------------------------------
// Rule 2: absolute value of tracked scalar returns tracked (kept in fp domain).
template <class T>
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    // Rule 2: routes through tracked::abs so a journal record is emitted.
    return tracked::abs(x);
}
template <class T>
tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    // Rule 2 + Rule 3: |z| is a tracked scalar.
    return tracked::abs(z);
}

// ---- kLog / kSqrt / kConj ---------------------------------------------------
// Rule 2: transcendental returns tracked.
template <class T>
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}
// Rule 3: log of complex returns tracked complex.
template <class T>
tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

template <class T>
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}
// Rule 3: sqrt of complex returns tracked complex.
template <class T>
tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// Rule 3: complex conjugate stays complex; real conj is identity.
template <class T>
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    // Rule 2: identity on real scalar (no rounding); no journal record.
    return x;
}
template <class T>
tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// ---- Max / Min --------------------------------------------------------------
// Rule 2: kinematic |a| vs |b| selectors returning a scalar that feeds fp
// arithmetic -> return tracked. Rule 7: comparison uses .value().
template <class T>
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    // Rule 7: bool via .value(); Rule 2: returned value stays tracked.
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}
template <class T>
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    // Rule 3: container form; comparison on |·|.value().
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}
template <class T>
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}
template <class T>
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}

// ---- Htheta -----------------------------------------------------------------
// Rule 6 (C6): Heaviside result flows into fp arithmetic (multiplied into
// tracked expressions), so return tracked. Value is 0/0.5/1 selected at runtime.
template <class T>
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    // Rule 7: compare .value() to steer branch; Rule 6: anonymous literal.
    T s = (T(0) < x.value()) - (x.value() < T(0));
    return tracked::literal<T>(T(0.5) * (T(1) + s));
}

// ---- iszero -----------------------------------------------------------------
// Rule 1 (C6): iszero is a discrete predicate consumed by `if (...)` branches
// and boolean composition throughout box_common.h and B*m.h. Return raw bool.
template <typename TOutput, typename TMass, typename TScale, class T>
bool iszero(const tracked::Tracked<T>& x) {
    // Rule 1: unwrap to .value() for a plain-bool discrete result.
    // Cutoff mirrors the library primary: _qlonshellcutoff = 1e-10.
    return std::abs(x.value()) < T(1e-10);
}

// ---- kPow -------------------------------------------------------------------
// C2: no tracked::pow exists — implement integer power as a multiply loop over
// the tracked operator*. Two overloads mirror the library primary (which
// declares one per numeric category), and both must be MORE SPECIALIZED than
// those primaries so qualified calls ql::kPow<...>(x,n) bind to the tracked
// overload (C7).
// C7: the library primary is
//   template<typename TOutput,typename TMass,typename TScale>
//   TOutput kPow(TOutput const&, int const&);
// so we carry the same leading explicit params on each constrained overload.
template <typename TOutput, typename TMass, typename TScale, class T>
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, int const& exponent) {
    // C2: multiply loop; Rule 2 result is tracked.
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: anonymous literal 1 as the accumulator seed.
    tracked::Tracked<T> temp = tracked::literal<T>(T(1));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        // Rule 6: anonymous literal 1 for the reciprocal numerator.
        return tracked::literal<T>(T(1)) / temp;
    }
    return temp;
}

template <typename TOutput, typename TMass, typename TScale, class T>
tracked::Complex<T> kPow(const tracked::Complex<T>& base, int const& exponent) {
    // C2 + Rule 3: multiply loop; result is tracked complex.
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 6: anonymous complex literal 1 + 0i.
    tracked::Complex<T> temp(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal<T>(T(1)), tracked::literal<T>(T(0)));
        return one / temp;
    }
    return temp;
}

} // namespace ql

// =============================================================================
// C3: Missing operators the library's templates statically instantiate but the
// tracked API doesn't define. Placed in namespace tracked so ADL finds them.
// Rule 7-style identity: no journal record (they are identity ops).
// =============================================================================
namespace tracked {

// C3: unary operator+ on tracked scalar (identity; no rounding, no record).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// C3: unary operator+ on tracked complex (identity).
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked