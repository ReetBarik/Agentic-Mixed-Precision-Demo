// ql_tracked_interop.hpp
// Tracked interop shim for qcdloop (B8 spike / massive box family)
// SOURCE_HASH: 551c835fad3d9551d32c5ef4332d393770cb8055772c6a865bd4be0d3a0dd06f
//
// This shim makes ql::BO callable with T = tracked::Tracked<double>
// (TMass, TScale) and TOutput = tracked::Complex<double>. It is included
// BEFORE qcdloop's own headers so:
//  * Rule C5: forward-declare ql::Constants primary before specializing.
//  * Rule C7: constrained overloads outrank library primaries via partial
//    ordering at qualified-call sites.
//  * Rule 5 : named constants like ql::Constants<T>::_two() are routed
//    through tracked::constant("<name>", ...) so the journal preserves the
//    symbolic identity.
//
// Execution model (Rule 8 / C4): the driver calls ql::BO from a plain host
// loop (NOT inside a Kokkos::parallel_for). Tracked ops are host-only.
// We therefore emit NO KOKKOS_INLINE_FUNCTION / __host__ __device__
// annotations on shim overloads.

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

// -----------------------------------------------------------------------------
// Rule C5: forward-declare qcdloop's Constants primary template inside its own
// namespace so our partial specialization on tracked::Tracked<T> parses. The
// library supplies the full primary later in the same translation unit
// (kokkosMaths.h is included after this shim).
// -----------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
} // namespace ql

// -----------------------------------------------------------------------------
// Rule C3: supply operators the qcdloop templates apply to tracked values but
// the Tracked API does not itself define. These live in namespace tracked so
// ADL finds them at the library's call sites.
//
// * unary operator+ : an identity, no rounding, no journal record.
// * scalar arithmetic between Tracked<T> and raw T : the library writes
//   expressions like `TMass(ql::Constants<TMass>::_four()) * a` where one
//   operand is a raw T and the other is Tracked<T>. Promote the raw scalar
//   via literal() so it enters the graph with a valid id.
// * mixed Complex<T> * T and Complex<T> * Tracked<T> when T is the underlying
//   real (double): the tracked/complex.hpp header already defines these for
//   its own T parameter — no extra shims required.
// -----------------------------------------------------------------------------
namespace tracked {

// Rule C3 : identity operator+ on a tracked scalar, ADL-visible.
template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) {
    return a;
}

// Rule C3 : identity operator+ on a tracked complex, ADL-visible.
template <class T>
inline Complex<T> operator+(const Complex<T>& a) {
    return a;
}

} // namespace tracked

// -----------------------------------------------------------------------------
// Rule C5 : partial specialization of ql::Constants for tracked::Tracked<T>.
// Mirrors the FULL member interface of the primary in kokkosMaths.h so every
// leaf constant the driver's call graph can reach is routed through
// tracked::constant("<name>", T(v)) — the journal keeps each constant's name.
//
// Return types are the tracked scalar (Rule 2 / Rule 3): the library uses
// these values in floating-point arithmetic that must propagate error.
// _2ipi/_ipio2/_ipi/_ieps/_ieps50 return the container type (Rule 3), which
// for TOutput = tracked::Complex<T> is tracked::Complex<T>.
//
// Rule 1 : integer-valued members (_num_C, _num_B) stay `int` — they're
// discrete loop bounds/indices, never enter floating-point math.
// -----------------------------------------------------------------------------
namespace ql {

template <class T>
struct Constants<tracked::Tracked<T>> {

    // Rule 1 : discrete return (loop bound); NOT tracked.
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Rule 2 / Rule 5 : Chebyshev coefficient, named constant "_C<i>".
    static tracked::Tracked<T> _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>(std::string("_C") + std::to_string(i),
                                    T(coeffs[i]));
    }

    // Rule 1 : discrete return (loop bound).
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Rule 2 / Rule 5 : Bernoulli coefficient, named "_B<i>".
    static tracked::Tracked<T> _B(int i) {
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
        return tracked::constant<T>(std::string("_B") + std::to_string(i),
                                    T(coeffs[i]));
    }

    // Rule 2 / Rule 5 : onshell cutoff constant.
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _qlonshellcutoff() {
        return tracked::constant<T>("_qlonshellcutoff", T(1e-10));
    }

    // Rule 2 / Rule 5 : named scalar constants.
    static tracked::Tracked<T> _pi()   { return tracked::constant<T>("_pi",   T(M_PI)); }
    static tracked::Tracked<T> _pi2()  { return tracked::constant<T>("_pi2",  T(M_PI) * T(M_PI)); }

    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _pio3()   { return tracked::constant<T>("_pio3",   T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _pio6()   { return tracked::constant<T>("_pio6",   T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _pi2o3()  { return tracked::constant<T>("_pi2o3",  T(M_PI) * T(M_PI) / T(3)); }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _pi2o6()  { return tracked::constant<T>("_pi2o6",  T(M_PI) * T(M_PI) / T(6)); }
    template<typename TOutput, typename TMass, typename TScale>
    static tracked::Tracked<T> _pi2o12() { return tracked::constant<T>("_pi2o12", T(M_PI) * T(M_PI) / T(12)); }

    static tracked::Tracked<T> _zero()  { return tracked::constant<T>("_zero",  T(0)); }
    static tracked::Tracked<T> _half()  { return tracked::constant<T>("_half",  T(0.5)); }
    static tracked::Tracked<T> _one()   { return tracked::constant<T>("_one",   T(1)); }
    static tracked::Tracked<T> _two()   { return tracked::constant<T>("_two",   T(2)); }
    static tracked::Tracked<T> _three() { return tracked::constant<T>("_three", T(3)); }
    static tracked::Tracked<T> _four()  { return tracked::constant<T>("_four",  T(4)); }
    static tracked::Tracked<T> _five()  { return tracked::constant<T>("_five",  T(5)); }
    static tracked::Tracked<T> _six()   { return tracked::constant<T>("_six",   T(6)); }
    static tracked::Tracked<T> _ten()   { return tracked::constant<T>("_ten",   T(10)); }

    static tracked::Tracked<T> _eps()    { return tracked::constant<T>("_eps",    T(1e-6)); }
    static tracked::Tracked<T> _eps4()   { return tracked::constant<T>("_eps4",   T(1e-4)); }
    static tracked::Tracked<T> _eps7()   { return tracked::constant<T>("_eps7",   T(1e-7)); }
    static tracked::Tracked<T> _eps10()  { return tracked::constant<T>("_eps10",  T(1e-10)); }
    static tracked::Tracked<T> _eps14()  { return tracked::constant<T>("_eps14",  T(1e-14)); }
    static tracked::Tracked<T> _eps15()  { return tracked::constant<T>("_eps15",  T(1e-15)); }
    static tracked::Tracked<T> _xloss()  { return tracked::constant<T>("_xloss",  T(0.125)); }
    static tracked::Tracked<T> _neglig() { return tracked::constant<T>("_neglig", T(1e-14)); }
    static tracked::Tracked<T> _reps()   { return tracked::constant<T>("_reps",   T(1e-16)); }

    // Rule 3 / Rule 5 : complex-valued constants. TOutput here is the tracked
    // complex container (tracked::Complex<T>); construct it from two tracked
    // reals so components keep their names.
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi() {
        return TOutput(tracked::constant<T>("_2ipi_re", T(0)),
                       tracked::constant<T>("_2ipi_im", T(2) * T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2() {
        return TOutput(tracked::constant<T>("_ipio2_re", T(0)),
                       tracked::constant<T>("_ipio2_im", T(M_PI) * T(0.5)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi() {
        return TOutput(tracked::constant<T>("_ipi_re", T(0)),
                       tracked::constant<T>("_ipi_im", T(M_PI)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps() {
        return TOutput(tracked::constant<T>("_ieps_re", T(0)),
                       tracked::constant<T>("_ieps_im", T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2() {
        return TOutput(tracked::constant<T>("_ieps2_re", T(0)),
                       tracked::constant<T>("_ieps2_im", T(1e-16) * T(1e-16)));
    }
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() {
        // Matches primary: Constants<TScale>::_ieps50 returns TOutput with im=1e-50.
        return TOutput(tracked::constant<T>("_ieps50_re", T(0)),
                       tracked::constant<T>("_ieps50_im", T(1e-50)));
    }
};

// -----------------------------------------------------------------------------
// Rule C7 : outrank the library's own function templates via partial ordering.
// Each shim overload constrains its VALUE parameter to a concrete tracked type
// so a qualified `ql::foo<...>(x)` call from qcdloop's own headers binds to
// the shim, not the library primary. Every shim also carries the same leading
// explicit-template-parameter shape as the library primary so qualified calls
// like `ql::kAbs(x)` and `ql::iszero<TOutput,TMass,TScale>(x)` both work.
// -----------------------------------------------------------------------------

// ---- kPow ----
// Primary:
//   template<typename TOutput,typename TMass,typename TScale>
//   TOutput kPow(TOutput const&, int const&);
//   template<typename TOutput,typename TMass,typename TScale>
//   TMass   kPow(TMass   const&, int const&);
// Rule 2: floating-point return that participates in propagation.
// Rule 4/6: exponent is a literal integer -> stays raw int (Rule 1: index).
// Rule C7: constrained on tracked value + same 3 leading explicit params.
template<typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> temp = tracked::constant<T>("kPow_one", T(1));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0)
        return tracked::constant<T>("kPow_one", T(1)) / temp;
    return temp;
}

// Rule C7 : same overload for tracked::Complex<T>.
template<typename TOutput, typename TMass, typename TScale, class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> temp(tracked::constant<T>("kPow_one", T(1)));
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::constant<T>("kPow_one", T(1)));
        return one / temp;
    }
    return temp;
}

// ---- kAbs ----
// Rule 2: magnitude used in FP arithmetic downstream -> tracked scalar.
// Rule C7: constrained on tracked scalar / tracked complex.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

// Rule 2 / Rule 3 : |Complex<T>| is a tracked real scalar.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) {
    return tracked::abs(x);
}

// ---- kLog ----
// Rule 2 : natural log participates in propagation.
template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Rule 3 : log of a tracked complex returns a tracked complex.
template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& x) {
    return tracked::log(x);
}

// ---- kSqrt ----
// Rule 2 : sqrt propagates.
template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

// Rule 3 : sqrt of tracked complex returns tracked complex.
template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) {
    return tracked::sqrt(x);
}

// ---- kConj ----
// Rule 3 : conjugate of a tracked complex is a tracked complex.
template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& x) {
    return tracked::conj(x);
}

// Rule 2 : conj of a real is itself (identity, no rounding).
template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;
}

// ---- iszero ----
// Rule 1 : discrete boolean predicate used only in `if` branches / dispatch.
// Rule 7 : compare .value() against the library's own onshell cutoff (1e-10),
// return raw bool.
// Rule C7 : constrained overload with the same 3 leading explicit params as
// the library primary `template<TOutput,TMass,TScale> bool iszero(TScale const&)`.
template<typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    using std::abs;
    return abs(x.value()) < T(1e-10);
}

// Rule 1 : |Complex|<cutoff predicate. iszero is called on tracked::Complex
// only through kAbs(...) results which are already tracked scalars, but we
// provide it for safety (kAbs on complex returns Tracked<T>, not Complex<T>,
// so this overload is defensive).
template<typename TOutput, typename TMass, typename TScale, class T>
inline bool iszero(const tracked::Complex<T>& x) {
    using std::abs; using std::sqrt;
    T r = x.real().value();
    T i = x.imag().value();
    return sqrt(r*r + i*i) < T(1e-10);
}

// ---- Imag / Real ----
// Rule 2 (via C6): Real/Imag results feed FP arithmetic in the library
// (e.g. `Real(ki) * ...`, complex builds, signed ratios). Return the tracked
// scalar so provenance propagates. The library-primary returns for double are
// `double` — our constrained overload is picked at qualified call sites via
// partial ordering.
template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Imaginary part of a real is anonymous 0 (literal, not named).
    return tracked::literal<T>(T(0));
}

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& x) {
    return x.imag();
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& x) {
    return x.real();
}

// ---- Sign ----
// C6 disambiguation: qcdloop uses Sign(...) both as a discrete +/-/0 selector
// (in `if (ir == ik)` style predicates) AND in floating-point expressions
// (e.g. `TOutput(ql::Sign(ql::Real(k12)))` -> the ±1 is multiplied into
// tracked arithmetic). Therefore Sign returning a tracked scalar (Rule 2)
// preserves provenance, and it's still comparable via operator< / operator>
// (Rule 7) because Tracked<T>'s comparisons unwrap .value(). Returning raw
// int would strip the sign's provenance where it feeds FP math.
template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    // Anonymous literal (Rule 6): the sign value has no user-facing name.
    return tracked::literal<T>(s);
}

template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& x) {
    // Matches library primary: x / |x|.
    return x / tracked::abs(x);
}

// ---- Max / Min ----
// Rule 2 : chosen value is a tracked scalar; no new op emitted since we
// return one of the operands verbatim (its id/provenance are preserved).
template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? a : b;
}

template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    T am = tracked::abs(a).value();
    T bm = tracked::abs(b).value();
    return (am > bm) ? a : b;
}

template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    using std::abs;
    return (abs(a.value()) > abs(b.value())) ? b : a;
}

template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    T am = tracked::abs(a).value();
    T bm = tracked::abs(b).value();
    return (am > bm) ? b : a;
}

// ---- Htheta ----
// C6 : the Heaviside result is multiplied into complex expressions
// (see eta2/eta5 in kokkosUtils.h), so it participates in FP arithmetic.
// Rule 2 : return a tracked scalar.
template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    T h = T(0.5) * (T(1) + s);
    return tracked::literal<T>(h);
}

} // namespace ql