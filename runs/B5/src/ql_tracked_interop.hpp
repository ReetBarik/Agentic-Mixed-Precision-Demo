// ql_tracked_interop.hpp
// Tracked<T> interop shim for QCDLoop + Kokkos (B5 spike).
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This shim makes the qcdloop headers callable with
//   TMass   = tracked::Tracked<double>
//   TScale  = tracked::Tracked<double>
//   TOutput = tracked::Complex<double>
//
// Include order (see driver comment): this header MUST be included BEFORE
// kokkosMaths.h / kokkosUtils.h / boxGPU.h so the tracked overloads are
// visible at every qualified ql::* call site inside qcdloop's templates.

#pragma once

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <cmath>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// Forward-declare the ql namespace + the Constants primary template so our
// partial specialization below parses cleanly even though this header is
// included BEFORE kokkosMaths.h supplies the primary definition.
// [Rule C5] Specializing a class template the target library owns.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;   // [Rule C5] forward decl
}

// ---------------------------------------------------------------------------
// [Rule C3] Missing unary operator+ on tracked scalar/complex.
// qcdloop template bodies never actually apply unary '+' to tracked values in
// the paths we exercise, but if a future path did, ADL would need this. Add
// it defensively as a no-op identity in the tracked namespace so ADL finds
// it. No journal record (identity — introduces no rounding).
// ---------------------------------------------------------------------------
namespace tracked {
    // [Rule C3] identity operator+; no rounding, no record
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; }

    // [Rule C3] identity operator+; no rounding, no record
    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) { return a; }
} // namespace tracked

// ---------------------------------------------------------------------------
// ql:: overloads and specializations.
// The driver's underlying real scalar is `double` (see using T = double). All
// tracked overloads bind to Tracked<double> / Complex<double>.
// ---------------------------------------------------------------------------
namespace ql {

// =============================================================================
// [Rule C5] Constants<Tracked<T>> partial specialization.
// The library's primary Constants<T> exposes named scalar leaves via static
// member functions (_pi(), _two(), _half(), _eps(), ...). Every named leaf
// MUST route through tracked::constant("<name>", T(v)) so the constant keeps
// its identity in the journal. Mirror the FULL interface qcdloop's templates
// reach in the B0m/B1m/B2m dispatch paths (which is essentially the whole
// primary — the dispatchers touch nearly every named constant).
// =============================================================================
template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tr = tracked::Tracked<T>;

    // ---- integer-index Chebyshev/Bernoulli tables ----------------------------
    // [Rule 1] Discrete return: table size is an int, not tracked.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // [Rule 5] Named constant per Chebyshev coefficient.
    KOKKOS_INLINE_FUNCTION
    static Tr _C(int i) {
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
        return tracked::constant<T>("C[" + std::to_string(i) + "]", T(coeffs[i]));
    }

    // [Rule 1] Discrete return.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // [Rule 5] Named constant per Bernoulli coefficient.
    KOKKOS_INLINE_FUNCTION
    static Tr _B(int i) {
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
        return tracked::constant<T>("B[" + std::to_string(i) + "]", T(coeffs[i]));
    }

    // ---- onshell cutoff (named) ---------------------------------------------
    // [Rule 5] Named constant "qlonshellcutoff".
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    // ---- π and derivatives (named) ------------------------------------------
    // [Rule 5] Named constants; each carries its source designator's name.
    KOKKOS_INLINE_FUNCTION static Tr _pi()    { return tracked::constant<T>("pi",   T(M_PI)); }
    KOKKOS_INLINE_FUNCTION static Tr _pi2()   { return tracked::constant<T>("pi2",  T(M_PI) * T(M_PI)); }

    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _pio3()  { return tracked::constant<T>("pio3",  T(M_PI) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _pio6()  { return tracked::constant<T>("pio6",  T(M_PI) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _pi2o3() { return tracked::constant<T>("pi2o3", T(M_PI) * T(M_PI) / T(3)); }
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _pi2o6() { return tracked::constant<T>("pi2o6", T(M_PI) * T(M_PI) / T(6)); }
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static Tr _pi2o12(){ return tracked::constant<T>("pi2o12",T(M_PI) * T(M_PI) / T(12)); }

    // ---- small named integers / fractions -----------------------------------
    // [Rule 5] Named constants (identifiers in source: _zero, _half, _one, ...).
    KOKKOS_INLINE_FUNCTION static Tr _zero()  { return tracked::constant<T>("zero",  T(0)); }
    KOKKOS_INLINE_FUNCTION static Tr _half()  { return tracked::constant<T>("half",  T(0.5)); }
    KOKKOS_INLINE_FUNCTION static Tr _one()   { return tracked::constant<T>("one",   T(1)); }
    KOKKOS_INLINE_FUNCTION static Tr _two()   { return tracked::constant<T>("two",   T(2)); }
    KOKKOS_INLINE_FUNCTION static Tr _three() { return tracked::constant<T>("three", T(3)); }
    KOKKOS_INLINE_FUNCTION static Tr _four()  { return tracked::constant<T>("four",  T(4)); }
    KOKKOS_INLINE_FUNCTION static Tr _five()  { return tracked::constant<T>("five",  T(5)); }
    KOKKOS_INLINE_FUNCTION static Tr _six()   { return tracked::constant<T>("six",   T(6)); }
    KOKKOS_INLINE_FUNCTION static Tr _ten()   { return tracked::constant<T>("ten",   T(10)); }

    // ---- tolerance / epsilon named constants --------------------------------
    // [Rule 5] Named constants.
    KOKKOS_INLINE_FUNCTION static Tr _eps()    { return tracked::constant<T>("eps",    T(1e-6)); }
    KOKKOS_INLINE_FUNCTION static Tr _eps4()   { return tracked::constant<T>("eps4",   T(1e-4)); }
    KOKKOS_INLINE_FUNCTION static Tr _eps7()   { return tracked::constant<T>("eps7",   T(1e-7)); }
    KOKKOS_INLINE_FUNCTION static Tr _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    KOKKOS_INLINE_FUNCTION static Tr _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static Tr _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    KOKKOS_INLINE_FUNCTION static Tr _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    KOKKOS_INLINE_FUNCTION static Tr _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    KOKKOS_INLINE_FUNCTION static Tr _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // ---- complex-valued named constants (used with TOutput=Complex<T>) -----
    // Constants<TScale>::template _2ipi<TOutput,...>() is invoked from qcdloop
    // with TOutput = tracked::Complex<T>. Route the imaginary component
    // through named constants; the real part is a named "zero".
    // [Rule 5] Named constant + [Rule 3] container of tracked -> Complex<Tracked<T>>.
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _2ipi() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("2pi",  T(2) * T(M_PI)));
    }
    // [Rule 5] Named constant, [Rule 3] container return.
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ipio2() {
        return TOutput(tracked::constant<T>("zero",  T(0)),
                       tracked::constant<T>("pio2",  T(M_PI) * T(0.5)));
    }
    // [Rule 5] / [Rule 3].
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ipi() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("pi",   T(M_PI)));
    }
    // [Rule 5] / [Rule 3].
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps() {
        return TOutput(tracked::constant<T>("zero", T(0)),
                       tracked::constant<T>("reps", T(1e-16)));
    }
    // [Rule 5] / [Rule 3].
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps2() {
        return TOutput(tracked::constant<T>("zero",  T(0)),
                       tracked::constant<T>("reps2", T(1e-16) * T(1e-16)));
    }
    // [Rule 5] / [Rule 3]. Named "ieps50" — matches source designator.
    template <typename TOutput, typename TMass, typename TScale>
    KOKKOS_INLINE_FUNCTION static TOutput _ieps50() {
        return TOutput(tracked::constant<T>("zero",  T(0)),
                       tracked::constant<T>("ieps50",T(1e-50)));
    }
};

// =============================================================================
// [Rule C7] Constrained overloads that outrank the library's own templates.
// Each carries the leading explicit template parameters the qualified call
// sites name, so a call like ql::kPow<TOutput,TMass,TScale>(base, n) still
// binds here (more specialized than the primary) — no forwarders.
// =============================================================================

// ---- kPow ---------------------------------------------------------------
// [Rule C7] Outrank library's `template<class TOutput,class TMass,class TScale>
//           kPow(TOutput const&, int const&)`. Tracked scalar overload.
// [Rule 2] Floating-point return participating in downstream error prop.
// [Rule C2] Integer power by multiply loop over tracked operator*.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // [Rule 5] "one" is a named constant seed for the running product.
    tracked::Tracked<T> temp = tracked::constant<T>("one", T(1));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::constant<T>("one", T(1));
        return one / temp;
    }
    return temp;
}

// [Rule C7] Tracked complex overload for kPow. [Rule 3] Container return.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // [Rule 5] "one" named constant seed, wrapped as complex.
    tracked::Complex<T> temp(tracked::constant<T>("one", T(1)),
                             tracked::constant<T>("zero", T(0)));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::constant<T>("one", T(1)),
                                tracked::constant<T>("zero", T(0)));
        return one / temp;
    }
    return temp;
}

// ---- kAbs ---------------------------------------------------------------
// [Rule C7] Constrained overload for tracked scalar.
// [Rule 2] |x| participates in downstream error propagation.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// [Rule C7] Constrained overload for tracked complex.
// [Rule 2] Real-valued magnitude of a complex participates in downstream fp math.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {
    return tracked::abs(x);
}

// ---- kLog ---------------------------------------------------------------
// [Rule C7] Constrained overload; [Rule 2] fp return.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}
// [Rule C7] / [Rule 3].
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kLog(const tracked::Complex<T>& x) {
    return tracked::log(x);
}

// ---- kSqrt --------------------------------------------------------------
// [Rule C7] / [Rule 2].
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}
// [Rule C7] / [Rule 3].
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {
    return tracked::sqrt(x);
}

// ---- kConj --------------------------------------------------------------
// [Rule C7] Constrained overload; [Rule 3] container of tracked returned.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kConj(const tracked::Complex<T>& x) {
    return tracked::conj(x);
}
// [Rule C7] real "conjugate" is identity; [Rule 2] fp return.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// ---- iszero -------------------------------------------------------------
// [Rule C7] Outrank library `iszero<TOutput,TMass,TScale>(TScale const&)`.
// [Rule 1] Discrete return: iszero is a boolean predicate consumed by if-tests.
// [Rule C6] Confirmed: every use site consumes the result inside an `if`
//           (never multiplied into fp), so bool is correct here.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) {
    // Compare underlying values directly — no journal record for a predicate.
    using std::abs;
    return abs(x.value()) < T(1e-10);
}

// [Rule C7] Also cover tracked complex; iszero on complex is called via
// kAbs(complex) which returns tracked scalar (handled above), so this
// overload is defensive for direct-on-complex uses in the instantiated graph.
// [Rule 1] Discrete bool return.
template <class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Complex<T>& x) {
    using std::abs;
    return (abs(x.real().value()) < T(1e-10)) &&
           (abs(x.imag().value()) < T(1e-10));
}

// ---- Imag / Real --------------------------------------------------------
// [Rule C7] Outrank library's Imag(double)/Imag(Kokkos::complex<double>).
// [Rule 2] Component of a tracked complex participates in downstream fp math.
// [Rule 3] Complex container decomposed to its tracked-scalar component.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Complex<T>& x) {
    return x.imag();
}
// [Rule C7] For a tracked *real* scalar, Imag is a fresh zero literal.
// [Rule 6] Anonymous zero — bare literal, no name.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::literal<T>(T(0));
}

// [Rule C7] Real component of tracked complex. [Rule 2] fp return.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Complex<T>& x) {
    return x.real();
}
// [Rule C7] Real of a real tracked scalar is identity. [Rule 2] fp return.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}

// ---- Sign ---------------------------------------------------------------
// [Rule C7] Constrained overloads that outrank library's Sign(double) /
// Sign(Kokkos::complex<double>).
// [Rule C6] Sign is consumed BOTH as a discrete branch selector AND as a
// numeric ±1/0 multiplier feeding tracked expressions (e.g. in xspence /
// xetatilde / etatilde and the Lnrat imaginary-cut correction). Therefore
// Sign MUST return the tracked scalar to preserve provenance — this is the
// [Rule 2] classification per C6, not [Rule 1].
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    const T v = x.value();
    const T s = (T(0) < v) ? T(1) : (v < T(0) ? T(-1) : T(0));
    // [Rule 6] Anonymous ±1/0 literal — the sign is runtime-selected;
    // literals dedup by generated id, not by value, so distinct ±1 samples
    // remain distinct in the journal.
    return tracked::literal<T>(s);
}

// [Rule C7] Sign of a complex is z/|z| (used e.g. in kfn dispatch).
// [Rule 3] Container return: Complex<Tracked<T>>.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Sign(const tracked::Complex<T>& x) {
    // sign(z) = z / |z| — routes through tracked ops naturally.
    tracked::Tracked<T> mag = tracked::abs(x);
    return x / mag;
}

// ---- Max / Min ----------------------------------------------------------
// [Rule C7] Constrained overloads for tracked scalars.
// [Rule 2] fp return; picks by magnitude but forwards the *tracked* operand.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    // [Rule 7] Comparison on tracked values -> plain bool via .value().
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

// [Rule C7] / [Rule 3] tracked complex overload.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    // |z| = sqrt(re²+im²); use underlying values for the branch (no record needed
    // to decide the branch — the returned operand itself is tracked).
    const T amag = std::hypot(a.real().value(), a.imag().value());
    const T bmag = std::hypot(b.real().value(), b.imag().value());
    return (amag > bmag) ? a : b;   // [Rule 7] discrete branch on doubles.
}

// [Rule C7] Tracked-scalar Min.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;   // [Rule 7]
}
// [Rule C7] / [Rule 3] tracked-complex Min.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) {
    const T amag = std::hypot(a.real().value(), a.imag().value());
    const T bmag = std::hypot(b.real().value(), b.imag().value());
    return (amag > bmag) ? b : a;   // [Rule 7]
}

// ---- Htheta -------------------------------------------------------------
// [Rule C7] Outrank library's Htheta(double).
// [Rule C6] Htheta feeds fp multiplications inside eta2 -> return tracked.
// [Rule 2] fp return participating in downstream error propagation.
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    // 0.5*(1 + sign(x)); use named "half" + "one" plus tracked Sign above.
    // [Rule 5] "half" and "one" named constants.
    auto half = tracked::constant<T>("half", T(0.5));
    auto one  = tracked::constant<T>("one",  T(1));
    return half * (one + Sign(x));
}

} // namespace ql

// ---------------------------------------------------------------------------
// [Rule 8] / [Rule C4] Execution-space annotation follows the DRIVER.
// The driver invokes ql::BO from a plain host `for` loop (tracked ops are
// host-only — they call std::string / journaling). No parallel dispatch is
// used, so NO execution-space annotations beyond KOKKOS_INLINE_FUNCTION are
// required on the shim overloads — and KOKKOS_INLINE_FUNCTION expands to a
// host-callable inline on non-device builds, which is exactly what we need.
// The KOKKOS_INLINE_FUNCTION tags above are provided only to match the
// signature style of the library primaries (helps overload resolution when
// the library's own KOKKOS_INLINE_FUNCTION-tagged templates are considered).
// ---------------------------------------------------------------------------