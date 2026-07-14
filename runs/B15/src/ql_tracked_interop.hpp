// ql_tracked_interop.hpp
// Tracked interop shim for the qcdloop (ql::) box-integral library.
//
// Purpose: make ql::BO<TOutput, TMass, TScale> callable with
//   TScale = tracked::Tracked<double>
//   TMass  = tracked::Tracked<double>
//   TOutput = tracked::Complex<double>
// while preserving condition-number and error-propagation fidelity.
//
// Include order (see driver): this header must be included BEFORE
// kokkosMaths.h / kokkosUtils.h / boxGPU.h so that:
//   (a) our ql::Constants<tracked::Tracked<T>> partial specialization (C5)
//       is visible before the library's primary template is instantiated;
//   (b) our qualified ql::Real / ql::Imag / ql::Sign / ql::kAbs / ql::kLog /
//       ql::kSqrt / ql::kConj / ql::iszero / ql::Max / ql::Min / ql::Htheta
//       tracked overloads are candidates at every qualified call site inside
//       the library's own template bodies (qualified calls disable ADL, so
//       these must be declared, not merely findable);
//   (c) our ql::kPow overload strictly-more-specialized than the library's
//       template<TOutput,TMass,TScale>(TOutput,int) primary wins under
//       partial ordering (C7).
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
#include <cstddef>
#include <type_traits>

// ---------------------------------------------------------------------------
// C5: forward-declare the library's primary Constants template inside its own
// namespace so our partial specialization on tracked::Tracked<T> can parse
// before the library header is included. The library supplies the full
// primary later in the same TU.
// ---------------------------------------------------------------------------
namespace ql {
    template<typename T> struct Constants;
}

// ---------------------------------------------------------------------------
// Rule C3: identity operator+ for tracked scalars.
//
// The library never applies unary + to tracked scalars in the visible source,
// but we add it defensively (ADL-visible in namespace tracked) since some
// arithmetic patterns in the DAG could otherwise fail to compile. Identity =
// no journal record.
// ---------------------------------------------------------------------------
namespace tracked {
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; }  // Rule C3
    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) { return a; }  // Rule C3
} // namespace tracked

// ---------------------------------------------------------------------------
// All library-visible shims live in namespace ql.
// ---------------------------------------------------------------------------
namespace ql {

// ===========================================================================
// C5 + Rule 5: named-constants specialization.
//
// Every leaf accessor mirrors the primary's interface and routes through
// tracked::constant("<name>", T(value)) so the semantic name is preserved
// in prov_consts. Integer-count accessors (_num_C, _num_B) stay as plain
// ints — Rule 1 (discrete return / index).
//
// The template parameters of _ieps50 / _2ipi / _ipio2 / _ipi / _ieps /
// _ieps2 / _pio3 / _pio6 / _pi2o3 / _pi2o6 / _pi2o12 / _qlonshellcutoff
// mirror the library primary exactly so qualified calls
// `Constants<TScale>::template _foo<TOutput,TMass,TScale>()` continue to bind.
// ===========================================================================

template <class T>
struct Constants<tracked::Tracked<T>> {   // C5: partial specialization on tracked scalar

    using U = tracked::Tracked<T>;

    // Rule 1: integer count, not a tracked value.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Rule 5 + Rule 2: named Chebyshev coefficients — one named constant per index.
    // The primary uses a compile-time coeffs[] table; we mirror it and name each
    // entry "C[i]" so the journal preserves attribution to the source constant.
    static U _C(int i) {
        static const double coeffs[19] = {
            0.4299669356081370,   0.4097598753307711,  -0.0185884366501460,
            0.0014575108406227,  -0.0001430418444234,  0.0000158841554188,
           -0.0000019078495939,   0.0000002419518085, -0.0000000319334127,
            0.0000000043454506,  -0.0000000006057848,  0.0000000000861210,
           -0.0000000000124433,   0.0000000000018226, -0.0000000000002701,
            0.0000000000000404,  -0.0000000000000061,  0.0000000000000009,
           -0.0000000000000001
        };
        // Rule 5: named constant preserves the identifier "C[i]" in the journal.
        return tracked::constant<T>(std::string("C[") + std::to_string(i) + "]", T(coeffs[i]));
    }

    // Rule 1: integer count.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 5: named Bernoulli coefficients.
    static U _B(int i) {
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

    // Rule 5: on-shell cutoff.
    template<typename TOutput, typename TMass, typename TScale>
    static U _qlonshellcutoff() { return tracked::constant<T>("qlonshellcutoff", T(1e-10)); }

    // Rule 5: named mathematical constants. Each must be a tracked scalar with
    // its source-code identifier preserved.
    static U _pi()  { return tracked::constant<T>("pi",  T(M_PI)); }
    static U _pi2() { return tracked::constant<T>("pi2", T(M_PI) * T(M_PI)); }

    template<typename TOutput, typename TMass, typename TScale>
    static U _pio3()  { return tracked::constant<T>("pi/3",   T(M_PI) / T(3)); }

    template<typename TOutput, typename TMass, typename TScale>
    static U _pio6()  { return tracked::constant<T>("pi/6",   T(M_PI) / T(6)); }

    template<typename TOutput, typename TMass, typename TScale>
    static U _pi2o3() { return tracked::constant<T>("pi2/3",  T(M_PI) * T(M_PI) / T(3)); }

    template<typename TOutput, typename TMass, typename TScale>
    static U _pi2o6() { return tracked::constant<T>("pi2/6",  T(M_PI) * T(M_PI) / T(6)); }

    template<typename TOutput, typename TMass, typename TScale>
    static U _pi2o12(){ return tracked::constant<T>("pi2/12", T(M_PI) * T(M_PI) / T(12)); }

    // Rule 5: small numeric constants named as they appear in library sources.
    static U _zero()   { return tracked::constant<T>("zero",   T(0.0)); }
    static U _half()   { return tracked::constant<T>("half",   T(0.5)); }
    static U _one()    { return tracked::constant<T>("one",    T(1.0)); }
    static U _two()    { return tracked::constant<T>("two",    T(2.0)); }
    static U _three()  { return tracked::constant<T>("three",  T(3.0)); }
    static U _four()   { return tracked::constant<T>("four",   T(4.0)); }
    static U _five()   { return tracked::constant<T>("five",   T(5.0)); }
    static U _six()    { return tracked::constant<T>("six",    T(6.0)); }
    static U _ten()    { return tracked::constant<T>("ten",    T(10.0)); }

    static U _eps()    { return tracked::constant<T>("eps",    T(1e-6));  }
    static U _eps4()   { return tracked::constant<T>("eps4",   T(1e-4));  }
    static U _eps7()   { return tracked::constant<T>("eps7",   T(1e-7));  }
    static U _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static U _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static U _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static U _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static U _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static U _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Rule 3 + Rule 5: container of tracked reals, with the imag component
    // seeded by named constants preserving the semantic identifier.
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("2*pi", T(2) * T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi/2", T(M_PI) * T(0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("pi",   T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(tracked::constant<T>("zero", T(0)),
                                   tracked::constant<T>("reps", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(tracked::constant<T>("zero",      T(0)),
                                   tracked::constant<T>("reps*reps", T(1e-16) * T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(tracked::constant<T>("zero",  T(0)),
                                   tracked::constant<T>("ieps50", T(1e-50)));
    }
};

// ===========================================================================
// C5 + Rule 5: named-constants specialization for tracked::Complex<T>.
//
// Some library sites call `Constants<TOutput>::_foo()` where TOutput is the
// tracked complex type (e.g. `Constants<TOutput>::_half()`, `_zero()`, `_one()`,
// `_two()`, `_ieps50<...>()`, `_2ipi<...>()`). Each must return the tracked
// complex container (Rule 3) built from named tracked reals (Rule 5).
// ===========================================================================

template <class T>
struct Constants<tracked::Complex<T>> {   // C5: partial specialization on tracked complex

    using CT = tracked::Complex<T>;

    // Rule 3 + Rule 5: named-constant real, zero imag padded via constant("zero").
    static CT _zero() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("zero", T(0)));
    }
    static CT _half() {
        return CT(tracked::constant<T>("half", T(0.5)),
                  tracked::constant<T>("zero", T(0)));
    }
    static CT _one() {
        return CT(tracked::constant<T>("one", T(1)),
                  tracked::constant<T>("zero", T(0)));
    }
    static CT _two() {
        return CT(tracked::constant<T>("two", T(2)),
                  tracked::constant<T>("zero", T(0)));
    }
    static CT _three() {
        return CT(tracked::constant<T>("three", T(3)),
                  tracked::constant<T>("zero", T(0)));
    }
    static CT _four() {
        return CT(tracked::constant<T>("four", T(4)),
                  tracked::constant<T>("zero", T(0)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static CT _ieps50() {
        return CT(tracked::constant<T>("zero",   T(0)),
                  tracked::constant<T>("ieps50", T(1e-50)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ieps() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("reps", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _2ipi() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("2*pi", T(2) * T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ipio2() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("pi/2", T(M_PI) * T(0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static CT _ipi() {
        return CT(tracked::constant<T>("zero", T(0)),
                  tracked::constant<T>("pi",   T(M_PI)));
    }
};

// ===========================================================================
// C7: kPow overloads strictly more specialized than the library's
//   template<typename TOutput, typename TMass, typename TScale>
//   TOutput kPow(TOutput const& base, int const& exponent);
//   template<typename TOutput, typename TMass, typename TScale>
//   TMass   kPow(TMass   const& base, int const& exponent);
//
// The library primaries have three explicit template parameters; every
// qualified call site writes `ql::kPow<TOutput, TMass, TScale>(base, n)`.
// Each of our overloads therefore carries those three leading explicit
// template parameters (unused in the body) plus one deduced parameter
// constrained to a concrete tracked value type. Partial ordering picks our
// overload.
//
// C2: Tracked has no free `pow`; we implement integer powers as a loop of
// tracked multiplications, mirroring the library's own algorithm.
// Rule 6: bare integer literal 1 in a tracked expression -> literal(T(1)).
// ===========================================================================

template<typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, int const& exponent) {  // Rule C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> temp = tracked::literal(T(1));  // Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;    // C2: multiply-loop
    return exponent < 0 ? (tracked::literal(T(1)) / temp) : temp;
}

template<typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, int const& exponent) {  // Rule C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> temp(tracked::literal(T(1)), tracked::literal(T(0)));  // Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(tracked::literal(T(1)), tracked::literal(T(0)));
        return one_c / temp;
    }
    return temp;
}

// ===========================================================================
// Rule 2 / Rule 8 (no annotation, host-only per C4): tracked math dispatch
// functions. The library defines these as generic templates (kAbs, kLog,
// kSqrt, kConj) plus explicit overloads for double and Kokkos::complex<double>.
// Our tracked overloads are non-template and take the concrete tracked type,
// so they are the best match at qualified call sites.
// ===========================================================================

// Rule 2: |Tracked<T>| — floating-point return, participates in propagation.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {   // Rule 2
    return tracked::abs(x);
}

// Rule 2: |Complex<T>| collapses to a tracked REAL scalar (that's what
// tracked::abs on a Complex returns). This mirrors kokkosMaths.h's
// `double kAbs(Kokkos::complex<double>)` behavior.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {   // Rule 2
    return tracked::abs(x);
}

// Rule 2: log / sqrt / conj on tracked scalars and complex.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {   // Rule 2
    return tracked::log(x);
}
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& x) {   // Rule 2/3
    return tracked::log(x);
}
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {  // Rule 2
    return tracked::sqrt(x);
}
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {  // Rule 2/3
    return tracked::sqrt(x);
}
// Rule 3: conj is (re, -im) — the negation is a tracked op on the imag part.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& x) {  // Rule 3
    return tracked::conj(x);
}
// Rule C3: conj of a real is identity; supply for completeness.
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {  // Rule C3
    return x;
}

// ===========================================================================
// Rule 2 / Rule 1 (per C6): Imag / Real / Sign.
//
// Imag / Real appear in floating-point expressions throughout the library
// (assigned to TScale, multiplied into tracked values, compared to zero AND
// fed into further tracked arithmetic). Per C6, "used in an expression that
// flows into floating-point" -> return tracked. Real of a tracked scalar is
// the scalar itself; Imag of a tracked scalar is a tracked zero.
//
// Sign returns +1 / -1 / 0. It is used BOTH as a discrete selector in `if`
// tests AND multiplied into tracked expressions. Per C6, since the value
// flows into tracked arithmetic (e.g. `TOutput(Sign(Real(k12)))` or
// `Sign(...) * ir13`), it must return a tracked scalar so provenance is
// preserved and int<->tracked conversions never happen.
// ===========================================================================

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {  // Rule 2 + C6
    // Padding zero from a real scalar — anonymous literal, not a named "zero"
    // (same rationale as tracked::Complex(re)'s explicit ctor).
    return tracked::literal(T(0));
}

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& x) {      // Rule 2 + C6
    return x.imag();
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {      // Rule 2 + C6
    return x;
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& x) {      // Rule 2 + C6
    return x.real();
}

// Sign: +1 / 0 / -1 packaged as a tracked scalar (C6). We inspect .value()
// (Rule 7-style peek — no comparison lifted into tracked) and return a
// literal, since the ±1 varies at runtime and constant() would dedupe by name.
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {      // Rule 2 + C6
    T v = x.value();
    T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    return tracked::literal(s);                                       // Rule 6
}

template <class T>
inline tracked::Tracked<T> Sign(const tracked::Complex<T>& x) {      // Rule 2 + C6
    // Library's complex Sign is `x / |x|`. Return the tracked real magnitude
    // sign only if that made sense; but the visible use is on real args and
    // the complex overload appears unused in the pruned call graph. Emit a
    // tracked scalar computed as x.real().value() sign for safety — matches
    // the double overload's semantics on the real axis.
    T v = x.real().value();
    T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    return tracked::literal(s);
}

// ===========================================================================
// Rule 1: iszero returns a raw bool. The library only ever consumes iszero()
// as a branch condition (an `if` guard, a boolean operand of &&/||) — never
// mixes it into floating-point arithmetic — so C6 keeps this Rule 1.
//
// Comparison uses .value() only (Rule 7), never lifts the compare into a
// tracked bool.
// ===========================================================================

// Overload for tracked-scalar TScale. Mirrors the library's own
// iszero<TOutput,TMass,TScale>(TScale) shape.
template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) {                          // Rule 1 + Rule 7
    using std::abs;
    // Cutoff matches Constants<TScale>::_qlonshellcutoff<...>() = 1e-10.
    return abs(x.value()) < T(1e-10);
}

// Overload for tracked-complex arguments (used e.g. when kAbs of a complex
// is folded through iszero in the library). Uses .value() of the magnitude.
template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Complex<T>& x) {                          // Rule 1 + Rule 7
    using std::abs;
    // |x| via tracked::abs would emit records; peek at components directly.
    T re = x.real().value();
    T im = x.imag().value();
    return (re * re + im * im) < T(1e-10) * T(1e-10);
}

// ===========================================================================
// Rule 2 / Rule 3: Max / Min. The library defines these as picking the
// argument with larger/smaller |value|. We mirror it in tracked land: compare
// magnitudes via .value() (Rule 7), return the selected tracked value verbatim
// so its full provenance flows onward (Rule 2).
// ===========================================================================

template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {  // Rule 2 + Rule 7
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}
template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {  // Rule 3 + Rule 7
    using std::abs;
    T ma = std::sqrt(a.real().value() * a.real().value() + a.imag().value() * a.imag().value());
    T mb = std::sqrt(b.real().value() * b.real().value() + b.imag().value() * b.imag().value());
    return (ma > mb) ? a : b;
}
template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {  // Rule 2 + Rule 7
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}
template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {  // Rule 3 + Rule 7
    using std::abs;
    T ma = std::sqrt(a.real().value() * a.real().value() + a.imag().value() * a.imag().value());
    T mb = std::sqrt(b.real().value() * b.real().value() + b.imag().value() * b.imag().value());
    return (ma > mb) ? b : a;
}

// ===========================================================================
// Rule 2 + C6: Htheta returns a tracked scalar (0 or 1), because the library
// multiplies it directly into tracked expressions (see eta2 / eta5). Keeping
// it discrete would force an int -> tracked conversion Tracked lacks.
// ===========================================================================

template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {    // Rule 2 + C6
    T v = x.value();
    // 0.5 * (1 + sign(v))  — matches library formula, evaluated on .value().
    T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    T h = T(0.5) * (T(1) + s);
    return tracked::literal(h);                                       // Rule 6
}

} // namespace ql

// ---------------------------------------------------------------------------
// Rule C3: mixed-type arithmetic that the library's templates instantiate
// but that the Tracked API does not natively provide.
//
// The library writes things like `T(k12 - Max(kAbs(k12), TMass(One)) * ieps50)`
// where the outer subtract mixes a tracked scalar (k12) with a tracked complex
// (ieps50 comes from Constants<TScale>::_ieps50<...>() which we made return a
// tracked complex). That expression should not occur because ieps50 is scaled
// by a tracked real; the result is a tracked complex fed to the outer
// TOutput(...) constructor. tracked::Complex already provides scalar mixed
// operators, so nothing further is needed here. This comment documents that
// we relied on Complex<T>'s built-in Tracked<T>-mixed arithmetic — no extra
// overloads required.
//
// However, one thing IS missing from the Tracked API surface: mixed operators
// between tracked::Tracked<T> and a Complex expression that yields a Complex
// on the left (e.g. `TMass * TOutput(Sign(...))`). Those are all handled by
// promoting the Tracked to a Complex via tracked's `T`+Complex overloads,
// which take a bare T not a Tracked<T>. Provide adapters:
// ---------------------------------------------------------------------------

namespace tracked {

// Rule C3: Complex ⊕ Tracked scalar (scalar on right) — the API defines
// Complex ⊕ Tracked already; nothing to add. Same for scalar on left.

// Rule C3: Complex(Tracked) explicit promotion helper is provided by the
// Tracked API constructor `Complex(Tracked<T> re)`. No extra overload needed.

} // namespace tracked

// End of ql_tracked_interop.hpp