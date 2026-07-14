// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop+Kokkos box integrals (B10 spike).
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// Purpose: make ql::BO / ql::B0m / ql::B1m / ql::B2m instantiable with
//   TOutput = tracked::Complex<double>
//   TMass   = tracked::Tracked<double>
//   TScale  = tracked::Tracked<double>
//
// Include order in the driver: this header FIRST, then kokkosMaths.h /
// kokkosUtils.h / boxGPU.h. qcdloop's own templates call ql::Real,
// ql::Imag, ql::Sign, ql::kAbs, ql::kLog, ql::kSqrt, ql::kConj,
// ql::iszero, ql::Max, ql::Min, ql::Htheta, ql::kPow via *qualified*
// names — no ADL — so the tracked overloads must be visible in namespace
// ql at the definition point of every template that uses them. Emitting
// them before the qcdloop headers accomplishes that. Overload resolution
// prefers the concrete tracked overloads over kokkosMaths.h's generic
// primaries (C7 partial ordering on the value parameter).

#pragma once

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cmath>
#include <type_traits>

// ---------------------------------------------------------------------------
// C3: missing operators on tracked scalar.
//
// qcdloop templates apply unary operator+ / operator- to tracked scalars via
// expressions like `-x` where x is a Tracked<T>. Unary operator- is already
// provided by tracked.hpp; unary operator+ is not. Add it as a free function
// in namespace tracked so ADL finds it. Identity: no rounding, no journal
// record.
// ---------------------------------------------------------------------------
namespace tracked {

// C3: unary operator+ identity for Tracked<T> (no rounding, no record).
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

// C3: unary operator+ identity for Complex<T> (no rounding, no record).
template <class T>
inline Complex<T> operator+(const Complex<T>& a) { return a; }

} // namespace tracked

// ---------------------------------------------------------------------------
// C5: forward-declare the primary Constants<T> template that qcdloop owns,
// so our partial specialization on tracked::Tracked<T> parses before the
// primary is defined (this header is included BEFORE kokkosMaths.h).
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
} // namespace ql

namespace ql {

// ===========================================================================
// C5 + Rule 5: partial specialization of ql::Constants for tracked scalars.
//
// Mirror the FULL member interface of the primary (kokkosMaths.h::Constants).
// Every named leaf scalar is routed through tracked::constant("<name>", ...)
// so the journal preserves the source-level constant name. Chebyshev _C(i)
// and Bernoulli _B(i) tables are also promoted through constant() with a
// per-index name so provenance is uniquely tagged.
//
// The template parameter T here is the underlying real scalar (double for
// this integration). Members return TrackedT = tracked::Tracked<T>.
// ===========================================================================
template <class T>
struct Constants<tracked::Tracked<T>> {

    using TrackedT = tracked::Tracked<T>;

    // Rule 5: named integer size, kept as raw int (Rule 1 — array bound).
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Rule 5: Chebyshev coefficient table, promoted through constant().
    // Not KOKKOS_INLINE_FUNCTION — tracked::constant() allocates strings
    // (host-only). C4: shim used from a host loop, no device annotation.
    static TrackedT _C(int i) {
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
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // Rule 1: table size is a discrete count.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 5: Bernoulli coefficient table, promoted through constant().
    static TrackedT _B(int i) {
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
        return tracked::constant<T>(std::string("B[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // Rule 5: named onshell cutoff.
    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    // Rule 5: named pi and derivatives. Each accessor names its own constant.
    static TrackedT _pi()    { return tracked::constant<T>("pi",    T(M_PI)); }
    static TrackedT _pi2()   { return tracked::constant<T>("pi2",   T(M_PI) * T(M_PI)); }

    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _pio3()  { return tracked::constant<T>("pio3",  T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _pio6()  { return tracked::constant<T>("pio6",  T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _pi2o3() { return tracked::constant<T>("pi2o3", T(M_PI) * T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _pi2o6() { return tracked::constant<T>("pi2o6", T(M_PI) * T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static TrackedT _pi2o12(){ return tracked::constant<T>("pi2o12",T(M_PI) * T(M_PI) / T(12)); }

    // Rule 5: named numeric constants.
    static TrackedT _zero()  { return tracked::constant<T>("zero",  T(0)); }
    static TrackedT _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static TrackedT _one()   { return tracked::constant<T>("one",   T(1)); }
    static TrackedT _two()   { return tracked::constant<T>("two",   T(2)); }
    static TrackedT _three() { return tracked::constant<T>("three", T(3)); }
    static TrackedT _four()  { return tracked::constant<T>("four",  T(4)); }
    static TrackedT _five()  { return tracked::constant<T>("five",  T(5)); }
    static TrackedT _six()   { return tracked::constant<T>("six",   T(6)); }
    static TrackedT _ten()   { return tracked::constant<T>("ten",   T(10)); }
    static TrackedT _eps()   { return tracked::constant<T>("eps",   T(1e-6)); }
    static TrackedT _eps4()  { return tracked::constant<T>("eps4",  T(1e-4)); }
    static TrackedT _eps7()  { return tracked::constant<T>("eps7",  T(1e-7)); }
    static TrackedT _eps10() { return tracked::constant<T>("eps10", T(1e-10)); }
    static TrackedT _eps14() { return tracked::constant<T>("eps14", T(1e-14)); }
    static TrackedT _eps15() { return tracked::constant<T>("eps15", T(1e-15)); }
    static TrackedT _xloss() { return tracked::constant<T>("xloss", T(0.125)); }
    static TrackedT _neglig(){ return tracked::constant<T>("neglig",T(1e-14)); }
    static TrackedT _reps()  { return tracked::constant<T>("reps",  T(1e-16)); }

    // Rule 5 + Rule 3: complex-valued named constants. TOutput here is
    // tracked::Complex<T>; construct from named real/imag tracked parts.
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2() {
        return TOutput(tracked::constant<T>("zero",  T(0)),
                       tracked::constant<T>("pio2",  T(M_PI) * T(0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("pi",   T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("reps", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2() {
        return TOutput(tracked::constant<T>("zero",  T(0)),
                       tracked::constant<T>("reps2", T(1e-16) * T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() {
        return TOutput(tracked::constant<T>("zero",   T(0)),
                       tracked::constant<T>("eps50",  T(1e-50)));
    }
};

// ===========================================================================
// C5 + Rule 5: partial specialization for tracked::Complex<T>.
//
// The primary is instantiated with TOutput = Complex<T> to yield complex
// constants (e.g. TOutput(1.0), TOutput(0.5)). Provide the members qcdloop
// reaches on the complex branch — the scalar family (_zero, _half, _one,
// _two, _three, _four) and the complex-valued named constants (_ieps,
// _ieps50, _2ipi, ...). Every leaf routes through tracked::constant() to
// preserve name in the journal.
// ===========================================================================
template <class T>
struct Constants<tracked::Complex<T>> {

    using CT = tracked::Complex<T>;

    // Rule 5: scalar constants promoted into the complex container (Rule 3:
    // container-of-tracked, not tracked-of-container).
    static CT _zero()  { return CT(tracked::constant<T>("zero",  T(0)),   tracked::constant<T>("zero", T(0))); }
    static CT _half()  { return CT(tracked::constant<T>("half",  T(0.5)), tracked::constant<T>("zero", T(0))); }
    static CT _one()   { return CT(tracked::constant<T>("one",   T(1)),   tracked::constant<T>("zero", T(0))); }
    static CT _two()   { return CT(tracked::constant<T>("two",   T(2)),   tracked::constant<T>("zero", T(0))); }
    static CT _three() { return CT(tracked::constant<T>("three", T(3)),   tracked::constant<T>("zero", T(0))); }
    static CT _four()  { return CT(tracked::constant<T>("four",  T(4)),   tracked::constant<T>("zero", T(0))); }

    // Rule 5 + Rule 3: complex-valued named constants. These are the "i*eps"
    // forms qcdloop calls with the leading Constants<TOutput>:: qualification.
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ieps() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("reps", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ieps2() {
        return CT(tracked::constant<T>("zero",  T(0)),
                  tracked::constant<T>("reps2", T(1e-16) * T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ieps50() {
        return CT(tracked::constant<T>("zero",  T(0)),
                  tracked::constant<T>("eps50", T(1e-50)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _2ipi() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ipio2() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("pio2", T(M_PI) * T(0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ipi() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("pi",   T(M_PI)));
    }
};

// ===========================================================================
// C7 + Rule 2/3: shim overloads that outrank the qcdloop primaries.
//
// qcdloop declares its own generic templates ql::kAbs<T>, ql::kLog<T>,
// ql::kSqrt<T>, ql::kConj<T>, and explicit overloads ql::Real / ql::Imag /
// ql::Sign / ql::Max / ql::Min / ql::Htheta for double and
// Kokkos::complex<double>. Templates call these via ql::foo(x) at
// definition sites. For each, provide a concrete-typed tracked overload
// (Tracked<T> or Complex<T>) so partial ordering selects ours.
// ===========================================================================

// -------- ql::Real (Rule 2: FP scalar) -------------------------------------

// C7 + Rule 2: real part of a tracked scalar IS the scalar itself.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { return x; }

// C7 + Rule 2: real part of a tracked complex is its .real() component.
template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

// -------- ql::Imag (Rule 2: FP scalar) -------------------------------------

// C7 + Rule 2: imag of a tracked scalar is the tracked literal 0 (promoted
// so downstream ops see a valid operand id). Rule 6: anonymous literal.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::literal(T(0)); // Rule 6: anonymous literal
}

// C7 + Rule 2: imag of a tracked complex is its .imag() component.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

// -------- ql::Sign (C6: sign feeds FP arithmetic — return tracked) --------

// C6: Sign(x) in qcdloop is multiplied into tracked expressions
// (e.g. `TOutput(ql::Sign(ql::Real(k12)))`), so it is a FLOATING-POINT
// return (Rule 2), NOT a discrete int (Rule 1). Preserve provenance.
// Rule 6: the ±1 selector is an anonymous literal (there is no user-named
// constant "one_signed"; using tracked::constant("one",...) would collide
// with the named "one" constant elsewhere).
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    // Compare on the raw value (Rule 7: comparisons unwrap).
    T v = x.value();
    if (v > T(0)) return tracked::literal(T( 1)); // Rule 6
    if (v < T(0)) return tracked::literal(T(-1)); // Rule 6
    return tracked::literal(T(0));                // Rule 6
}

// C7 + C6 + Rule 3: complex Sign returns z / |z| as a tracked complex.
template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    // Compute |z| in tracked land then divide (Rule 3: container-of-tracked).
    tracked::Tracked<T> m = tracked::abs(z); // uses tracked::abs(Complex)
    // Guard against zero magnitude: return zero complex (Rule 6 literal).
    if (m.value() == T(0)) {
        return tracked::Complex<T>(tracked::literal(T(0)),
                                   tracked::literal(T(0)));
    }
    return z / m; // tracked complex / tracked scalar
}

// -------- ql::kAbs (Rule 2/3) ---------------------------------------------

// C7 + Rule 2: |Tracked<T>| via tracked::abs (in ops.hpp).
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// C7 + Rule 2: |Complex<T>| = sqrt(re^2 + im^2), returned as tracked scalar.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    return tracked::abs(z); // decomposes into named tracked sub-ops
}

// -------- ql::kLog (Rule 2/3) ---------------------------------------------

// C7 + Rule 2: log of a tracked scalar via tracked::log (in ops.hpp).
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// C7 + Rule 3: log of a tracked complex.
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    return tracked::log(z);
}

// -------- ql::kSqrt (Rule 2/3) --------------------------------------------

// C7 + Rule 2: sqrt of a tracked scalar via tracked::sqrt (in ops.hpp).
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// C7 + Rule 3: sqrt of a tracked complex.
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

// -------- ql::kConj (Rule 2/3) --------------------------------------------

// C7 + Rule 2: conjugate of a real tracked scalar is itself.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) { return x; }

// C7 + Rule 3: conjugate of a tracked complex.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// -------- ql::kPow (Rule 2/3) ---------------------------------------------
//
// C7 + C2: qcdloop declares kPow<TOutput,TMass,TScale>(T const&, int const&)
// as three-explicit-parameter function templates keyed on the value type
// (double / Kokkos::complex<double>). C7 says: emit one constrained overload
// per concrete tracked argument shape, each carrying the SAME leading
// explicit template parameters, so the qualified call sites
//   ql::kPow<TOutput,TMass,TScale>(x, n)
// bind directly to ours and outrank the qcdloop primaries. Implement as an
// integer power loop (C2: no tracked::pow exists), all through the tracked
// operator*.

// C7 + C2 + Rule 2: integer power of a tracked scalar.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, int const& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> tmp = tracked::constant<T>("one", T(1)); // Rule 5
    for (int i = 0; i < n; ++i) tmp = tmp * base;
    if (exponent < 0) {
        return tracked::constant<T>("one", T(1)) / tmp; // Rule 5
    }
    return tmp;
}

// C7 + C2 + Rule 3: integer power of a tracked complex.
template <class TOutput, class TMass, class TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, int const& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> tmp(tracked::constant<T>("one", T(1)),  // Rule 5
                            tracked::constant<T>("zero", T(0)));
    for (int i = 0; i < n; ++i) tmp = tmp * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(tracked::constant<T>("one", T(1)),
                                  tracked::constant<T>("zero", T(0)));
        return one_c / tmp;
    }
    return tmp;
}

// -------- ql::iszero (Rule 1: discrete predicate) -------------------------
//
// qcdloop declares iszero as
//   template<TOutput,TMass,TScale> bool iszero(TScale const& x)
// with body `kAbs(x) < qlonshellcutoff`. It is consumed ONLY as a branch
// condition (if / boolean combinators) — never fed into arithmetic — so it
// stays a raw bool under Rule 1 (C6 check: purely discrete use).

// C7 + Rule 1: iszero of a tracked scalar — compare unwrapped |x| to cutoff.
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    // Rule 7: comparisons on tracked values unwrap.
    return std::abs(x.value()) < T(1e-10);
}

// C7 + Rule 1: iszero of a tracked complex — compare |z| to cutoff.
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const tracked::Complex<T>& z) {
    T re = z.real().value();
    T im = z.imag().value();
    return std::sqrt(re*re + im*im) < T(1e-10);
}

// -------- ql::Max / ql::Min (Rule 2/3: FP-returning selectors) ------------
//
// qcdloop's Max/Min compare by |a| vs |b| and return the winner. Because the
// result flows into subsequent floating-point arithmetic (e.g. scalefac used
// in divisions), this is a FP return (Rule 2/3), not a discrete selector
// (Rule 1). Compare on unwrapped values (Rule 7) but return the tracked
// argument itself so provenance is preserved.

// C7 + Rule 2: Max on tracked scalars.
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    return (std::abs(a.value()) > std::abs(b.value())) ? a : b; // Rule 7
}

// C7 + Rule 3: Max on tracked complex.
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    T ma = std::sqrt(a.real().value()*a.real().value() + a.imag().value()*a.imag().value());
    T mb = std::sqrt(b.real().value()*b.real().value() + b.imag().value()*b.imag().value());
    return (ma > mb) ? a : b; // Rule 7
}

// C7 + Rule 2: Min on tracked scalars.
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    return (std::abs(a.value()) > std::abs(b.value())) ? b : a; // Rule 7
}

// C7 + Rule 3: Min on tracked complex.
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    T ma = std::sqrt(a.real().value()*a.real().value() + a.imag().value()*a.imag().value());
    T mb = std::sqrt(b.real().value()*b.real().value() + b.imag().value()*b.imag().value());
    return (ma > mb) ? b : a; // Rule 7
}

// -------- ql::Htheta (C6: step function feeds FP arithmetic) --------------
//
// C6: Htheta = 0.5 * (1 + sign(x)) is multiplied into tracked expressions
// (e.g. inside cspence / eta2), so its result is FLOATING-POINT (Rule 2),
// not a discrete selector. Return a tracked literal 0 or 1 (Rule 6: the
// step selector is not a user-named constant, so literal(), not constant()).
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    // Rule 7: unwrap the comparison; Rule 6: literal for the step value.
    return (x.value() > T(0)) ? tracked::literal(T(1)) : tracked::literal(T(0));
}

} // namespace ql