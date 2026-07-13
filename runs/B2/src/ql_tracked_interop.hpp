// ql_tracked_interop.hpp
// Tracked interop shim for qcdloop (Kokkos port), B2 spike.
//
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This header must be included BEFORE any qcdloop headers (see driver
// include order) so that:
//   (a) our ql::Constants<Tracked<T>> partial specialization is visible
//       when qcdloop's own templates are instantiated, and
//   (b) our tracked overloads of ql::Real/Imag/Sign/kAbs/kLog/kSqrt/
//       kConj/kPow/Max/Min/Htheta/iszero/cLn/Lnrat are candidates at the
//       qualified call sites inside qcdloop's templates (qualified calls
//       do NOT go through ADL, so visibility at the definition point is
//       what matters).
//
// Rule 8 / C4: the driver invokes ql::BO from a plain host for-loop
// (tracked ops are host-only). No KOKKOS_INLINE_FUNCTION / __host__
// __device__ annotations are needed on our shims.
//
// C1: the tracked complex is spelled tracked::Complex<T> with T=double
// (NOT tracked::Complex<tracked::Tracked<double>>).
//
// C7: we do NOT introduce any catch-all Base forwarders. Each shim is a
// concrete-typed overload; qualified qcdloop calls resolve to the qcdloop
// primary for the native (double / Kokkos::complex<double>) types and to
// our tracked overloads for tracked arguments.

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <type_traits>

// ---- Forward-declare qcdloop's primary Constants template (C5) --------------
// Our partial specialization below must parse before ql::Constants is
// otherwise defined; forward-declaring inside qcdloop's own namespace
// gives us that. kokkosMaths.h will supply the primary definition later
// in the same TU.
namespace ql {
    template <typename T> struct Constants;
}

namespace tracked {

// ---- C3: missing operators the qcdloop templates apply to Tracked ----------
//
// qcdloop uses unary operator+ on tracked scalars in a few spots (identity
// promotion in generic expressions). The Tracked API doesn't define it, so
// add it as a free function found by ADL. Identity introduces no rounding
// and emits no journal record.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) {
    // Rule C3: identity operator, no journal record.
    return a;
}

} // namespace tracked

// =============================================================================
// ql namespace: overloads / specializations for tracked types
// =============================================================================
namespace ql {

// Convenience aliases used only inside this shim.
using _TrD  = ::tracked::Tracked<double>;
using _TCxD = ::tracked::Complex<double>;

// -----------------------------------------------------------------------------
// Rule 5 / C5: Constants<Tracked<T>> partial specialization.
//
// qcdloop's primary Constants<T> exposes ~30 named leaf scalars used all
// through the library. Route every leaf through tracked::constant(...) so
// each keeps its original library name in the journal (prov_consts).
//
// This is a PARTIAL specialization on Tracked<T>, so it covers the tracked
// scalar generically. Members return Tracked<T> (the tracked scalar), NOT
// tracked::Complex — the complex-valued Constants members (_2ipi, _ipio2,
// _ipi, _ieps, _ieps2, _ieps50) are templated on <TOutput,TMass,TScale> in
// the primary and return TOutput. We provide the matching templated members
// dispatched on TOutput below.
// -----------------------------------------------------------------------------
template <class T>
struct Constants< ::tracked::Tracked<T> > {

    // ---- Chebyshev / Bernoulli coefficient tables ---------------------------
    // Rule 1: array *sizes* are discrete counts, not floats.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 2 / Rule 6: named library constants — but these are numeric-table
    // *entries*, not semantic constants that appear by name in user math.
    // Emit them as literals so the journal doesn't get polluted with 44
    // "_C[i]" / "_B[i]" named constants.
    static ::tracked::Tracked<T> _C(int i) {
        // Mirror the primary's table exactly.
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
        return ::tracked::literal<T>(T(coeffs[i]));  // Rule 6
    }

    static ::tracked::Tracked<T> _B(int i) {
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
        return ::tracked::literal<T>(T(coeffs[i]));  // Rule 6
    }

    // ---- Named scalar constants (Rule 5) ------------------------------------
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _qlonshellcutoff() {
        return ::tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    static ::tracked::Tracked<T> _pi()   { return ::tracked::constant<T>("pi",   T(M_PI)); }
    static ::tracked::Tracked<T> _pi2()  { return ::tracked::constant<T>("pi2",  T(M_PI) * T(M_PI)); }

    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _pio3()   { return ::tracked::constant<T>("pio3",   T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _pio6()   { return ::tracked::constant<T>("pio6",   T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _pi2o3()  { return ::tracked::constant<T>("pi2o3",  T(M_PI) * T(M_PI) / T(3)); }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _pi2o6()  { return ::tracked::constant<T>("pi2o6",  T(M_PI) * T(M_PI) / T(6)); }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Tracked<T> _pi2o12() { return ::tracked::constant<T>("pi2o12", T(M_PI) * T(M_PI) / T(12)); }

    static ::tracked::Tracked<T> _zero()  { return ::tracked::constant<T>("zero",  T(0)); }
    static ::tracked::Tracked<T> _half()  { return ::tracked::constant<T>("half",  T(0.5)); }
    static ::tracked::Tracked<T> _one()   { return ::tracked::constant<T>("one",   T(1)); }
    static ::tracked::Tracked<T> _two()   { return ::tracked::constant<T>("two",   T(2)); }
    static ::tracked::Tracked<T> _three() { return ::tracked::constant<T>("three", T(3)); }
    static ::tracked::Tracked<T> _four()  { return ::tracked::constant<T>("four",  T(4)); }
    static ::tracked::Tracked<T> _five()  { return ::tracked::constant<T>("five",  T(5)); }
    static ::tracked::Tracked<T> _six()   { return ::tracked::constant<T>("six",   T(6)); }
    static ::tracked::Tracked<T> _ten()   { return ::tracked::constant<T>("ten",   T(10)); }

    static ::tracked::Tracked<T> _eps()    { return ::tracked::constant<T>("eps",    T(1e-6)); }
    static ::tracked::Tracked<T> _eps4()   { return ::tracked::constant<T>("eps4",   T(1e-4)); }
    static ::tracked::Tracked<T> _eps7()   { return ::tracked::constant<T>("eps7",   T(1e-7)); }
    static ::tracked::Tracked<T> _eps10()  { return ::tracked::constant<T>("eps10",  T(1e-10)); }
    static ::tracked::Tracked<T> _eps14()  { return ::tracked::constant<T>("eps14",  T(1e-14)); }
    static ::tracked::Tracked<T> _eps15()  { return ::tracked::constant<T>("eps15",  T(1e-15)); }
    static ::tracked::Tracked<T> _xloss()  { return ::tracked::constant<T>("xloss",  T(0.125)); }
    static ::tracked::Tracked<T> _neglig() { return ::tracked::constant<T>("neglig", T(1e-14)); }
    static ::tracked::Tracked<T> _reps()   { return ::tracked::constant<T>("reps",   T(1e-16)); }

    // ---- Complex-valued named constants (Rule 3 + Rule 5) -------------------
    // These members are templated on <TOutput,TMass,TScale> in the primary and
    // return TOutput. When TOutput = tracked::Complex<T>, we must return a
    // tracked complex whose components are named-tracked reals.
    template <class TOutput, class TMass, class TScale>
    static TOutput _2ipi() {
        // 0 + i * 2*pi. Real part is anonymous 0 (structural), imag is 2*pi.
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("2pi", T(2) * T(M_PI));        // Rule 5
        return TOutput(zero, im);                                         // C1
    }
    template <class TOutput, class TMass, class TScale>
    static TOutput _ipio2() {
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("pio2", T(M_PI) * T(0.5));     // Rule 5
        return TOutput(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static TOutput _ipi() {
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("pi", T(M_PI));                // Rule 5
        return TOutput(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps() {
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("reps", T(1e-16));             // Rule 5
        return TOutput(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps2() {
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16)); // Rule 5
        return TOutput(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static TOutput _ieps50() {
        auto zero = ::tracked::literal<T>(T(0));                          // Rule 6
        auto im   = ::tracked::constant<T>("ieps50", T(1e-50));           // Rule 5
        return TOutput(zero, im);
    }
};

// -----------------------------------------------------------------------------
// Rule 5 / C5: Constants<Complex<T>> partial specialization.
//
// qcdloop calls Constants<TOutput>::_zero(), _half(), _one(), _two(),
// _three(), _four() with TOutput = Complex. Provide those leaf accessors
// returning tracked complex; other members aren't reached with TOutput=complex.
// -----------------------------------------------------------------------------
template <class T>
struct Constants< ::tracked::Complex<T> > {
    static ::tracked::Complex<T> _zero() {
        // Rule 5 real "zero" + structural imag literal (Rule 6, C1).
        auto re = ::tracked::constant<T>("zero", T(0));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }
    static ::tracked::Complex<T> _half() {
        auto re = ::tracked::constant<T>("half", T(0.5));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }
    static ::tracked::Complex<T> _one() {
        auto re = ::tracked::constant<T>("one", T(1));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }
    static ::tracked::Complex<T> _two() {
        auto re = ::tracked::constant<T>("two", T(2));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }
    static ::tracked::Complex<T> _three() {
        auto re = ::tracked::constant<T>("three", T(3));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }
    static ::tracked::Complex<T> _four() {
        auto re = ::tracked::constant<T>("four", T(4));
        auto im = ::tracked::literal<T>(T(0));
        return ::tracked::Complex<T>(re, im);
    }

    // Rule 3: complex-valued i*<name> constants also appear with
    // TOutput=Complex when Constants<TOutput>::template _2ipi<...>() is
    // called (some qcdloop templates specialize this way). Forward to
    // the same shape as Constants<Tracked<T>>.
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _2ipi() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("2pi", T(2) * T(M_PI));
        return ::tracked::Complex<T>(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _ipio2() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("pio2", T(M_PI) * T(0.5));
        return ::tracked::Complex<T>(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _ipi() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("pi", T(M_PI));
        return ::tracked::Complex<T>(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _ieps() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("reps", T(1e-16));
        return ::tracked::Complex<T>(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _ieps2() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16));
        return ::tracked::Complex<T>(zero, im);
    }
    template <class TOutput, class TMass, class TScale>
    static ::tracked::Complex<T> _ieps50() {
        auto zero = ::tracked::literal<T>(T(0));
        auto im   = ::tracked::constant<T>("ieps50", T(1e-50));
        return ::tracked::Complex<T>(zero, im);
    }
};

// =============================================================================
// Scalar accessors: Real, Imag, Sign
// =============================================================================

// Rule 2 (C6): Real(Tracked<T>) is a floating-point value that flows into
// downstream tracked arithmetic (kAbs, cLn arguments, etc). Return tracked.
template <class T>
inline ::tracked::Tracked<T> Real(const ::tracked::Tracked<T>& x) {
    return x;   // real scalar's real part is itself; no rounding, no record
}

// Rule 2: Real(Complex<Tracked>) returns the real component (already tracked).
template <class T>
inline ::tracked::Tracked<T> Real(const ::tracked::Complex<T>& z) {
    return z.real();
}

// Rule 2 (C6): Imag(Tracked<T>) — for a real scalar, imag is a structural
// zero that then participates in tracked arithmetic. Return a tracked literal
// so downstream ops keep valid operand ids.
template <class T>
inline ::tracked::Tracked<T> Imag(const ::tracked::Tracked<T>& /*x*/) {
    return ::tracked::literal<T>(T(0));  // Rule 6: structural anonymous 0
}

template <class T>
inline ::tracked::Tracked<T> Imag(const ::tracked::Complex<T>& z) {
    return z.imag();
}

// Rule 1 vs Rule 2 (C6): Sign in qcdloop is consumed both as a discrete
// selector AND multiplied into tracked expressions. The overwhelming use is
// numeric (multiplied into TOutput via TOutput(ql::Sign(...)) or into tracked
// operands), so return a Tracked<T> per C6 — never lose provenance on a sign
// that flows into the result. Callers that need a raw int (comparisons only)
// take .value() explicitly.
template <class T>
inline ::tracked::Tracked<T> Sign(const ::tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return ::tracked::literal<T>(s);  // Rule 6: sign is a runtime-selected ±1/0
}

// Sign of a tracked complex — matches the qcdloop primary Sign(complex) = z/|z|.
template <class T>
inline ::tracked::Complex<T> Sign(const ::tracked::Complex<T>& z) {
    using ::tracked::abs;
    auto mag = abs(z);
    return z / mag;
}

// =============================================================================
// kAbs
// =============================================================================

// Rule 2: absolute value of a tracked scalar is a tracked floating-point op.
template <class T>
inline ::tracked::Tracked<T> kAbs(const ::tracked::Tracked<T>& x) {
    return ::tracked::abs(x);
}

// Rule 2 / Rule 3: |Complex<Tracked>| is a tracked SCALAR (sqrt(re^2+im^2)),
// mirroring the primary kAbs(Kokkos::complex<double>) returning double.
template <class T>
inline ::tracked::Tracked<T> kAbs(const ::tracked::Complex<T>& z) {
    return ::tracked::abs(z);
}

// =============================================================================
// kLog / kSqrt / kConj
// =============================================================================

// Rule 2
template <class T>
inline ::tracked::Tracked<T> kLog(const ::tracked::Tracked<T>& x) {
    return ::tracked::log(x);
}
// Rule 3
template <class T>
inline ::tracked::Complex<T> kLog(const ::tracked::Complex<T>& z) {
    return ::tracked::log(z);
}

// Rule 2
template <class T>
inline ::tracked::Tracked<T> kSqrt(const ::tracked::Tracked<T>& x) {
    return ::tracked::sqrt(x);
}
// Rule 3
template <class T>
inline ::tracked::Complex<T> kSqrt(const ::tracked::Complex<T>& z) {
    return ::tracked::sqrt(z);
}

// Rule 2: conj of a real scalar is itself, no record.
template <class T>
inline ::tracked::Tracked<T> kConj(const ::tracked::Tracked<T>& x) {
    return x;   // C3-style identity for the real branch
}
// Rule 3: conj of a tracked complex.
template <class T>
inline ::tracked::Complex<T> kConj(const ::tracked::Complex<T>& z) {
    return ::tracked::conj(z);
}

// =============================================================================
// kPow (integer exponent) — C2: no tracked::pow exists; implement as a
// multiply loop over tracked operator*.
// =============================================================================

// Rule 2
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Tracked<T> kPow(const ::tracked::Tracked<T>& base, int const& exponent) {
    // C7: this overload carries the same leading explicit template
    // parameters as the qcdloop primary so `ql::kPow<TOutput,TMass,TScale>(x,n)`
    // qualified calls bind here for tracked scalar inputs.
    const int n = exponent < 0 ? -exponent : exponent;
    ::tracked::Tracked<T> temp = ::tracked::constant<T>("one", T(1));  // Rule 5
    for (int i = 0; i < n; ++i)
        temp = temp * base;                                            // Rule 2 (mul)
    if (exponent < 0) {
        auto one = ::tracked::constant<T>("one", T(1));                // Rule 5
        return one / temp;                                             // Rule 2 (div)
    }
    return temp;
}

// Rule 3
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Complex<T> kPow(const ::tracked::Complex<T>& base, int const& exponent) {
    // C7 (complex-argument overload).
    const int n = exponent < 0 ? -exponent : exponent;
    // Build "1 + 0i" as a tracked complex whose real is the named "one".
    auto one_re = ::tracked::constant<T>("one", T(1));                 // Rule 5
    auto zero_im = ::tracked::literal<T>(T(0));                        // Rule 6
    ::tracked::Complex<T> temp(one_re, zero_im);
    for (int i = 0; i < n; ++i)
        temp = temp * base;                                            // Rule 3 (complex mul)
    if (exponent < 0) {
        auto one_re2 = ::tracked::constant<T>("one", T(1));            // Rule 5
        auto zero_im2 = ::tracked::literal<T>(T(0));                   // Rule 6
        ::tracked::Complex<T> one_c(one_re2, zero_im2);
        return one_c / temp;                                           // Rule 3
    }
    return temp;
}

// =============================================================================
// Max / Min — Rule 2 / Rule 3
// =============================================================================

// Rule 2: |a| vs |b| is a comparison (Rule 7) on tracked scalars; SELECT the
// tracked operand — do NOT rebuild via literal — so provenance is preserved.
template <class T>
inline ::tracked::Tracked<T> Max(const ::tracked::Tracked<T>& a,
                                 const ::tracked::Tracked<T>& b) {
    using std::abs;
    // Rule 7: comparison lowers to bool via .value().
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

// Rule 3
template <class T>
inline ::tracked::Complex<T> Max(const ::tracked::Complex<T>& a,
                                 const ::tracked::Complex<T>& b) {
    // Rule 7: compare magnitudes' .value(). Do NOT lift into tracked bool.
    auto mag_a = ::tracked::abs(a);
    auto mag_b = ::tracked::abs(b);
    return (mag_a.value() > mag_b.value()) ? a : b;
}

// Rule 2
template <class T>
inline ::tracked::Tracked<T> Min(const ::tracked::Tracked<T>& a,
                                 const ::tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;  // matches primary
}

// Rule 3
template <class T>
inline ::tracked::Complex<T> Min(const ::tracked::Complex<T>& a,
                                 const ::tracked::Complex<T>& b) {
    auto mag_a = ::tracked::abs(a);
    auto mag_b = ::tracked::abs(b);
    return (mag_a.value() > mag_b.value()) ? b : a;
}

// =============================================================================
// Htheta — Rule 2 (C6): Heaviside used in tracked arithmetic; keep tracked.
// =============================================================================
template <class T>
inline ::tracked::Tracked<T> Htheta(const ::tracked::Tracked<T>& x) {
    // 0.5 * (1 + sign(x)). Sign is discrete; wrap as literal (Rule 6).
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    T h = T(0.5) * (T(1) + s);
    return ::tracked::literal<T>(h);
}

// =============================================================================
// iszero — Rule 1: consumed strictly as a boolean selector in `if (...)`.
//
// C7: qcdloop declares `template<class TOutput,class TMass,class TScale>
// bool iszero(TScale const&)`. Our shim carries the same three leading
// explicit template parameters so qualified calls
// `ql::iszero<TOutput,TMass,TScale>(x)` bind here for tracked arguments and
// outrank the primary by partial ordering (concrete Tracked<T> value
// parameter beats the primary's bare TScale). Same shape for complex.
// =============================================================================
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const ::tracked::Tracked<T>& x) {
    // Rule 1: bool selector — unwrap via .value(); do NOT return Tracked<bool>.
    using std::abs;
    return abs(x.value()) < T(1e-10);   // matches Constants::_qlonshellcutoff
}

template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(const ::tracked::Complex<T>& z) {
    // Rule 1: bool selector — compare tracked magnitude's .value().
    auto mag = ::tracked::abs(z);
    return mag.value() < T(1e-10);
}

// =============================================================================
// cLn — Rule 3: qcdloop's cLn is a bridge from (complex or real) into a
// complex logarithm result. It's templated on <TOutput,TMass,TScale> in the
// primary and takes a scalar imaginary-sign argument (TScale). Our tracked
// overload constrains the *value* parameter to the concrete tracked type
// (per C7); the leading explicit template params are carried so qualified
// calls `ql::cLn<TOutput,TMass,TScale>(...)` bind here.
// =============================================================================

// Complex-argument form: cLn(z, isig) where z is tracked complex, isig is a
// tracked scalar (or a raw T when the caller passes a bare _zero() from our
// Constants<Complex> specialization -- but Sign()/isig callers always pass a
// tracked scalar in this driver).
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Complex<T> cLn(const ::tracked::Complex<T>& z,
                                 const ::tracked::Tracked<T>& isig) {
    // Rule 7: branch on .value() of imag/real parts and isig.
    if (z.imag().value() == T(0) && z.real().value() <= T(0)) {
        // Emit ipi * sign(isig) then log(-z).
        T s = (T(0) < isig.value()) - (isig.value() < T(0));
        auto pi_t = ::tracked::constant<T>("pi", T(M_PI));            // Rule 5
        auto sign_lit = ::tracked::literal<T>(s);                     // Rule 6
        auto im = pi_t * sign_lit;                                    // Rule 2
        auto zero_re = ::tracked::literal<T>(T(0));                   // Rule 6
        ::tracked::Complex<T> ipi_term(zero_re, im);                  // Rule 3
        return ::tracked::log(-z) + ipi_term;                         // Rule 3
    }
    return ::tracked::log(z);                                         // Rule 3
}

// Real-argument overload: cLn(x, isig) where x is a tracked real scalar,
// returns tracked complex. Mirrors the primary's real-x branch.
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Complex<T> cLn(const ::tracked::Tracked<T>& x,
                                 const ::tracked::Tracked<T>& isig) {
    if (x.value() > T(0)) {
        auto lnx = ::tracked::log(x);                                 // Rule 2
        auto zero_im = ::tracked::literal<T>(T(0));                   // Rule 6
        return ::tracked::Complex<T>(lnx, zero_im);                   // Rule 3
    } else {
        // log(-x) + i*pi*sign(isig)
        auto neg_x = -x;                                              // Rule 2 (neg)
        auto lnnegx = ::tracked::log(neg_x);                          // Rule 2
        T s = (T(0) < isig.value()) - (isig.value() < T(0));
        auto pi_t = ::tracked::constant<T>("pi", T(M_PI));            // Rule 5
        auto sign_lit = ::tracked::literal<T>(s);                     // Rule 6
        auto im = pi_t * sign_lit;                                    // Rule 2
        return ::tracked::Complex<T>(lnnegx, im);                     // Rule 3
    }
}

// =============================================================================
// Lnrat — Rule 3: log ratio, complex-valued result. Two overloads (both
// carrying the leading explicit template parameters per C7): complex/complex
// and real/real, matching the primaries.
// =============================================================================

// Complex/complex Lnrat (tracked complex operands).
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Complex<T> Lnrat(const ::tracked::Complex<T>& x,
                                   const ::tracked::Complex<T>& y) {
    // r = x / y
    auto r = x / y;                                                   // Rule 3
    // If imag(r) is zero, emit log(|r|) - ipio2 * (sign(-Re x) - sign(-Re y))
    if (r.imag().value() == T(0)) {
        auto mag = ::tracked::abs(r);                                 // Rule 2
        auto lnmag = ::tracked::log(mag);                             // Rule 2
        // sign(-Re x) - sign(-Re y) as discrete int -> literal Rule 6
        T sx = T(0) < (-x.real().value()) ? T(1)
             : ((-x.real().value()) < T(0) ? T(-1) : T(0));
        T sy = T(0) < (-y.real().value()) ? T(1)
             : ((-y.real().value()) < T(0) ? T(-1) : T(0));
        T d = sx - sy;
        auto d_lit = ::tracked::literal<T>(d);                        // Rule 6
        auto pio2 = ::tracked::constant<T>("pio2", T(M_PI) * T(0.5)); // Rule 5
        auto imag_term = pio2 * d_lit;                                // Rule 2
        // Result: (lnmag, 0) - (0, imag_term) = (lnmag, -imag_term)
        auto zero_lit = ::tracked::literal<T>(T(0));                  // Rule 6
        auto neg_im = -imag_term;                                     // Rule 2
        return ::tracked::Complex<T>(lnmag, zero_lit) +
               ::tracked::Complex<T>(zero_lit, neg_im);               // Rule 3
    }
    return ::tracked::log(r);                                         // Rule 3
}

// Real/real Lnrat: both operands are tracked scalars, returns tracked complex.
template <class TOutput, class TMass, class TScale, class T>
inline ::tracked::Complex<T> Lnrat(const ::tracked::Tracked<T>& x,
                                   const ::tracked::Tracked<T>& y) {
    // log(|x/y|) - ipio2 * (sign(-x) - sign(-y))
    auto ratio = x / y;                                               // Rule 2
    auto absr  = ::tracked::abs(ratio);                               // Rule 2
    auto lnr   = ::tracked::log(absr);                                // Rule 2

    T sx = T(0) < (-x.value()) ? T(1)
         : ((-x.value()) < T(0) ? T(-1) : T(0));
    T sy = T(0) < (-y.value()) ? T(1)
         : ((-y.value()) < T(0) ? T(-1) : T(0));
    T d = sx - sy;
    auto d_lit = ::tracked::literal<T>(d);                            // Rule 6
    auto pio2  = ::tracked::constant<T>("pio2", T(M_PI) * T(0.5));    // Rule 5
    auto imag_term = pio2 * d_lit;                                    // Rule 2
    auto neg_im = -imag_term;                                         // Rule 2
    auto zero_lit = ::tracked::literal<T>(T(0));                      // Rule 6

    return ::tracked::Complex<T>(lnr, neg_im);                        // Rule 3
    (void)zero_lit;
}

// =============================================================================
// printDoubleBits — Rule 1: purely diagnostic sink. qcdloop's overload does
// a bit-cast on a plain arithmetic scalar; for a Tracked<T> we unwrap to
// .value() before dumping. No journal record (identity, no rounding).
// =============================================================================
template <class T>
inline void printDoubleBits(const ::tracked::Tracked<T>& x) {
    // Rule 1: discrete side effect (I/O); unwrap.
    printDoubleBits(x.value());  // dispatches to the primary double overload
}

} // namespace ql

// UNCLASSIFIED SECTION -------------------------------------------------------
//
// The following symbols COULD be reached by qcdloop's template bodies but are
// NOT exercised on the driver's B2 branch (m1=m2=m3=m4=0, so BO() dispatches
// only into B0m -> B1/B2/B3/B4/B5 and their helpers). If a future kinematic
// configuration is added to this driver, expand the shim as needed:
//
//   - ddilog(Tracked<T>)             (used by Li2omrat real-arg branch;
//                                     Rule 2, would return Tracked<T>)
//   - li2series / ltli2series /      (used by cLi2omx2 / denspence /
//     denspence / cspence / xspence   cspence chains; Rule 3, return
//                                      Complex<Tracked>)
//   - fndd, Zlogint, R2int/R3int,    (higher-order helpers; Rule 3)
//     Rint, cLi2omx2, cLi2omx3,
//     Li2omx, Li2omx2, Li2omrat
//   - solveabc / solveabcd           (Rule 3: fill Kokkos::Array<Complex<T>,2>)
//   - kfn, ratgam, ratreal, R,       (Rule 3 / Rule 2 mixed)
//     eta / eta2 / eta3 / eta5,
//     etatilde, xeta, xetatilde
//
// These are inside qcdloop's B*.h template bodies. Because the templates are
// instantiated as a whole for T = Tracked<double>, they DO get parsed and
// name-lookup'd -- but ADL through Kokkos::Array<Tracked<...>,N> plus the
// qualified `ql::foo<...>(...)` call sites means the compiler resolves them
// against our shim only when actually instantiated on the exercised path.
// The B2 driver path exercises B1 (massless one-nonzero-invariant box) and
// its dependencies (Lnrat scalar+complex, kSqrt, kLog, kAbs, kPow, Constants,
// cLn scalar+complex, Sign, Real, Imag, Max, iszero) -- all provided above.
//
// If the build surface expands and one of the above is instantiated on a
// tracked type, add the missing overload following the same Rule
// annotations. Do NOT add speculative overloads now: emitting a wrong
// signature is worse than a clean template-instantiation error naming the
// exact missing symbol.