// ql_tracked_interop.hpp
// Tracked<T> interop shim for QCDLoop+Kokkos box integrals (B13 spike).
//
// This header must be included BEFORE any qcdloop headers so that the
// tracked overloads/specializations are visible at template definition
// points for qualified `ql::foo(...)` calls in kokkosMaths.h / kokkosUtils.h
// / box/*.h. See micro_driver.cpp's include-order comment.
//
// SOURCE_HASH: 23ab5b943f77d62b226f709b899613a5283e28d1b32baf58228b7d9de543d075

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <type_traits>

// -----------------------------------------------------------------------------
// C3: unary operator+ on Tracked<T>.
// QCDLoop templates occasionally use unary `+` on scalars (e.g. `+p3sq + m3sq`
// in B11/B12/B13). Tracked<T> does not define operator+ unary, so add it as a
// free function in namespace tracked (found via ADL). Identity — no rounding,
// no journal record.
// -----------------------------------------------------------------------------
namespace tracked {
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; }
} // namespace tracked

// -----------------------------------------------------------------------------
// Rule 5 / C5: forward-declare the primary Constants template inside namespace
// ql so our partial specialization on tracked::Tracked<T> parses before
// kokkosMaths.h supplies the primary definition (the interop header is
// included FIRST in the driver).
// -----------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
} // namespace ql

// -----------------------------------------------------------------------------
// Rule 5 / C5: partial specialization of ql::Constants for tracked scalars.
// Every leaf that returns a floating-point value is routed through
// tracked::constant("<name>", ...) so it participates in error/provenance
// tracking under its source-level name. Integer-like literal helpers
// (_num_C, _num_B) return raw int (Rule 1: discrete counts).
//
// Uses tracked::Complex<T> for TOutput-shaped constants (C1: never
// double-wrap as tracked::Complex<tracked::Tracked<T>>).
// -----------------------------------------------------------------------------
namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {
    using Tr = tracked::Tracked<T>;

    // Rule 1: discrete count of Chebyshev coefficients.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Rule 5: named-constant table entry. Each coefficient name is
    // "_C[i]" so the journal preserves per-index identity.
    // Rule 2: floating-point return participating in downstream error prop.
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
        std::string name = "_C[" + std::to_string(i) + "]";
        return tracked::constant<T>(name, T(coeffs[i]));
    }

    // Rule 1: discrete count.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 5 / Rule 2: named Bernoulli series coefficients.
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
        std::string name = "_B[" + std::to_string(i) + "]";
        return tracked::constant<T>(name, T(coeffs[i]));
    }

    // Rule 5 / Rule 2: onshell cutoff constant.
    template<typename TOutput, typename TMass, typename TScale>
    static Tr _qlonshellcutoff() { return tracked::constant<T>("_qlonshellcutoff", T(1e-10)); }

    // Rule 5 / Rule 2: named math constants.
    static Tr _pi()   { return tracked::constant<T>("_pi",   T(M_PI)); }
    static Tr _pi2()  { return tracked::constant<T>("_pi2",  T(M_PI) * T(M_PI)); }

    template<typename TOutput, typename TMass, typename TScale>
    static Tr _pio3() { return tracked::constant<T>("_pio3", T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static Tr _pio6() { return tracked::constant<T>("_pio6", T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o3(){ return tracked::constant<T>("_pi2o3", T(M_PI) * T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o6(){ return tracked::constant<T>("_pi2o6", T(M_PI) * T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static Tr _pi2o12(){ return tracked::constant<T>("_pi2o12", T(M_PI) * T(M_PI) / T(12)); }

    // Rule 5: small integer / bookkeeping constants used by qcdloop code.
    static Tr _zero()  { return tracked::constant<T>("_zero",  T(0.0)); }
    static Tr _half()  { return tracked::constant<T>("_half",  T(0.5)); }
    static Tr _one()   { return tracked::constant<T>("_one",   T(1.0)); }
    static Tr _two()   { return tracked::constant<T>("_two",   T(2.0)); }
    static Tr _three() { return tracked::constant<T>("_three", T(3.0)); }
    static Tr _four()  { return tracked::constant<T>("_four",  T(4.0)); }
    static Tr _five()  { return tracked::constant<T>("_five",  T(5.0)); }
    static Tr _six()   { return tracked::constant<T>("_six",   T(6.0)); }
    static Tr _ten()   { return tracked::constant<T>("_ten",   T(10.0)); }

    // Rule 5: named tolerance constants.
    static Tr _eps()    { return tracked::constant<T>("_eps",    T(1e-6)); }
    static Tr _eps4()   { return tracked::constant<T>("_eps4",   T(1e-4)); }
    static Tr _eps7()   { return tracked::constant<T>("_eps7",   T(1e-7)); }
    static Tr _eps10()  { return tracked::constant<T>("_eps10",  T(1e-10)); }
    static Tr _eps14()  { return tracked::constant<T>("_eps14",  T(1e-14)); }
    static Tr _eps15()  { return tracked::constant<T>("_eps15",  T(1e-15)); }
    static Tr _xloss()  { return tracked::constant<T>("_xloss",  T(0.125)); }
    static Tr _neglig() { return tracked::constant<T>("_neglig", T(1e-14)); }
    static Tr _reps()   { return tracked::constant<T>("_reps",   T(1e-16)); }

    // Rule 3 / C1: containers of tracked reals — tracked::Complex<T>, not
    // tracked::Complex<tracked::Tracked<T>>. Each imaginary component uses the
    // Tracked<T>(T) ctor which synthesizes a "_lit@?#N" id (Rule 4).
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _2ipi() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0.0)),
            tracked::constant<T>("_2pi",  T(2) * T(M_PI))
        );
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipio2() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",  T(0.0)),
            tracked::constant<T>("_pio2",  T(M_PI) * T(0.5))
        );
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ipi() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0.0)),
            tracked::constant<T>("_pi",   T(M_PI))
        );
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero", T(0.0)),
            tracked::constant<T>("_reps", T(1e-16))
        );
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps2() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",   T(0.0)),
            tracked::constant<T>("_reps2",  T(1e-16) * T(1e-16))
        );
    }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Complex<T> _ieps50() {
        return tracked::Complex<T>(
            tracked::constant<T>("_zero",   T(0.0)),
            tracked::constant<T>("_ieps50", T(1e-50))
        );
    }
};

} // namespace ql

// -----------------------------------------------------------------------------
// ql:: free-function overloads for tracked types.
//
// The DRIVER (micro_driver.cpp) runs the tracked computation from a plain host
// loop (see the "Host loop, NOT Kokkos::parallel_for" comment). Per C4, that
// means NO execution-space annotation on the shim overloads (KOKKOS_INLINE_FUNCTION
// would restrict them to host+device, which is unnecessary and incompatible with
// journaling).
// -----------------------------------------------------------------------------
namespace ql {

// ---- kAbs -------------------------------------------------------------------
// Rule 2: |x| — real-valued, participates in downstream fp arithmetic.
template <class T>
inline tracked::Tracked<T> kAbs(tracked::Tracked<T> const& x) {
    return tracked::abs(x);
}
// Rule 2 / Rule 3: |z| for tracked complex — returns tracked real scalar.
template <class T>
inline tracked::Tracked<T> kAbs(tracked::Complex<T> const& z) {
    return tracked::abs(z);
}

// ---- kLog / kSqrt / kConj ---------------------------------------------------
// Rule 2: log(x), sqrt(x) for tracked real.
template <class T>
inline tracked::Tracked<T> kLog(tracked::Tracked<T> const& x)  { return tracked::log(x); }
template <class T>
inline tracked::Tracked<T> kSqrt(tracked::Tracked<T> const& x) { return tracked::sqrt(x); }

// Rule 3 / Rule 2: log(z) / sqrt(z) for tracked complex.
template <class T>
inline tracked::Complex<T> kLog(tracked::Complex<T> const& z)  { return tracked::log(z); }
template <class T>
inline tracked::Complex<T> kSqrt(tracked::Complex<T> const& z) { return tracked::sqrt(z); }

// Rule 3: conj(z) — real component unchanged, imaginary negated.
template <class T>
inline tracked::Complex<T> kConj(tracked::Complex<T> const& z) { return tracked::conj(z); }

// Rule 2: kConj on real is identity.
template <class T>
inline tracked::Tracked<T> kConj(tracked::Tracked<T> const& x) { return x; }

// ---- Real / Imag ------------------------------------------------------------
// Rule 2: real and imag parts flow directly into tracked arithmetic (feed
// discriminant expressions, cond checks, etc.), so return Tracked<T>.
// C6: even though callers sometimes threshold Real(...) with `>`, the *values*
// are consumed as floating-point elsewhere, so keep provenance intact.
template <class T>
inline tracked::Tracked<T> Real(tracked::Tracked<T> const& x) { return x; }
template <class T>
inline tracked::Tracked<T> Real(tracked::Complex<T> const& z) { return z.real(); }

// Rule 2: Imag of a real tracked scalar is anonymous zero (Rule 6).
template <class T>
inline tracked::Tracked<T> Imag(tracked::Tracked<T> const& /*x*/) {
    return tracked::literal(T(0));
}
template <class T>
inline tracked::Tracked<T> Imag(tracked::Complex<T> const& z) { return z.imag(); }

// ---- Sign -------------------------------------------------------------------
// C6: qcdloop's Sign(...) yields ±1 / 0 that is then multiplied INTO
// floating-point expressions (e.g. `TOutput(ql::Sign(ql::Real(k12))) * ...`),
// so this is Rule 2 (floating-point), not Rule 1 (discrete). Preserve
// provenance by returning a tracked scalar.
template <class T>
inline tracked::Tracked<T> Sign(tracked::Tracked<T> const& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return tracked::literal(s);
}
// Rule 3: sign of a complex is z/|z| — a tracked complex container.
template <class T>
inline tracked::Complex<T> Sign(tracked::Complex<T> const& z) {
    auto a = tracked::abs(z);
    return z / a;
}

// ---- Max / Min --------------------------------------------------------------
// Rule 2: qcdloop's Max/Min select by |a| vs |b| and return the *original*
// value (with sign), which then flows into fp arithmetic. Keep provenance
// intact by returning one of the tracked inputs directly.
// Rule 7: the comparison of |a| vs |b| uses `.value()` implicitly through
// Tracked<T>::operator>, which is fine (already returns bool).
template <class T>
inline tracked::Tracked<T> Max(tracked::Tracked<T> const& a,
                               tracked::Tracked<T> const& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}
template <class T>
inline tracked::Tracked<T> Min(tracked::Tracked<T> const& a,
                               tracked::Tracked<T> const& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}
// Rule 3: same for tracked complex.
template <class T>
inline tracked::Complex<T> Max(tracked::Complex<T> const& a,
                               tracked::Complex<T> const& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? a : b;
}
template <class T>
inline tracked::Complex<T> Min(tracked::Complex<T> const& a,
                               tracked::Complex<T> const& b) {
    return (tracked::abs(a).value() > tracked::abs(b).value()) ? b : a;
}

// ---- iszero -----------------------------------------------------------------
// Rule 1: iszero returns bool; only consumed as a branch condition inside
// qcdloop control flow (never multiplied into fp expressions). Compare via
// .value() (Rule 7).
template <class TOutput, class TMass, class TScale, class T>
inline bool iszero(tracked::Tracked<T> const& x) {
    // Match kokkosMaths.h::iszero: cutoff = 1e-10 for double.
    return tracked::abs(x).value() < T(1e-10);
}

// ---- Htheta -----------------------------------------------------------------
// C6: Htheta appears inside multiplicative fp expressions (eta2, etc.),
// so Rule 2: return a tracked scalar (0 or 1).
template <class T>
inline tracked::Tracked<T> Htheta(tracked::Tracked<T> const& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return tracked::literal(T(0.5) * (T(1) + s));
}

// ---- kPow -------------------------------------------------------------------
// Rule 2 / C2: no tracked::pow exists; implement integer power via a
// multiply loop over tracked operator*, which keeps every multiplication
// in the journal.
template<typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Tracked<T> kPow(tracked::Tracked<T> const& base, int const& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 4: integer literal 1 promoted via the Tracked<T>(T) ctor.
    tracked::Tracked<T> temp(T(1));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Tracked<T> one(T(1));
        return one / temp;
    }
    return temp;
}

// Rule 3 / Rule 2: same for tracked complex base.
template<typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Complex<T> kPow(tracked::Complex<T> const& base, int const& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    // Rule 4: literal 1 for the complex identity (real=1, imag=0).
    tracked::Complex<T> temp(T(1), T(0));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one(T(1), T(0));
        return one / temp;
    }
    return temp;
}

} // namespace ql