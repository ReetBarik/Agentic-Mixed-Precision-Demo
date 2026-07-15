// ql_tracked_interop.hpp
// Tracked interop shim for QCDLoop + Kokkos (BIN1 spike).
//
// This header MUST be included BEFORE any qcdloop headers (kokkosMaths.h,
// kokkosUtils.h, boxGPU.h, box/*.h). The tracked overloads of ql::Real,
// ql::Imag, ql::Sign, ql::kAbs, ql::kLog, ql::kSqrt, ql::kConj, ql::Max,
// ql::Min, ql::iszero, and the ql::Constants specialization must be
// visible at each template's definition point (qualified ql:: calls
// bypass ADL).
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

// ---- Forward-declare ql::Constants primary (Rule C5) ------------------------
// The qcdloop primary definition lives in kokkosMaths.h (included after this
// shim). Forward-declare so our partial specializations parse.
namespace ql {
    template<typename T> struct Constants;
}

// ---- Unary operator+ for tracked scalar (Rule C3) --------------------------
// qcdloop templates sometimes apply unary + to tracked values (identity).
// Provide via ADL in the tracked namespace. Introduces no rounding, emits no
// journal record.
namespace tracked {

template <class T>
inline Tracked<T> operator+(const Tracked<T>& a) {
    return a;
}

template <class T>
inline Complex<T> operator+(const Complex<T>& a) {
    return a;
}

} // namespace tracked

namespace ql {

// ============================================================================
// Constants<tracked::Tracked<T>> specialization (Rule C5, Rule 5)
// ============================================================================
// Partial specialization keyed on the tracked scalar. Every named leaf is
// routed through tracked::constant("<name>", T(value)) so its identity
// survives into the journal (Rule 5). Chebyshev / Bernoulli tables are
// promoted per-element via constant() with per-index names.
//
// Mirrors the full public interface of the primary ql::Constants<T> template
// as used by the transitive call graph of ql::BO for BIN1 (kokkosMaths.h,
// kokkosUtils.h, box/box_common.h, box/B0m.h, box/B1m.h, box/B2m.h).

template<typename T>
struct Constants<tracked::Tracked<T>> {
    using TT = tracked::Tracked<T>;

    // Chebyshev coefficient count (discrete — Rule 1)
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_C() { return 19; }

    // Chebyshev coefficient i (Rule 5: named constant per index)
    static TT _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::constant<T>(std::string("C_") + std::to_string(i), T(coeffs[i]));
    }

    // Bernoulli coefficient count (discrete — Rule 1)
    KOKKOS_INLINE_FUNCTION
    static constexpr int _num_B() { return 25; }

    // Bernoulli coefficient i (Rule 5: named constant per index)
    static TT _B(int i) {
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
        return tracked::constant<T>(std::string("B_") + std::to_string(i), T(coeffs[i]));
    }

    // Onshell cutoff (Rule 5)
    template<typename TOutput, typename TMass, typename TScale>
    static TT _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10));
    }

    // pi and derived (Rule 5)
    static TT _pi()   { return tracked::constant<T>("pi",   T(M_PI)); }
    static TT _pi2()  { return _pi() * _pi(); }

    template<typename TOutput, typename TMass, typename TScale>
    static TT _pio3() { return _pi() / tracked::constant<T>("three", T(3)); }

    template<typename TOutput, typename TMass, typename TScale>
    static TT _pio6() { return _pi() / tracked::constant<T>("six", T(6)); }

    template<typename TOutput, typename TMass, typename TScale>
    static TT _pi2o3() { return _pi() * _pio3<TOutput, TMass, TScale>(); }

    template<typename TOutput, typename TMass, typename TScale>
    static TT _pi2o6() { return _pi() * _pio6<TOutput, TMass, TScale>(); }

    template<typename TOutput, typename TMass, typename TScale>
    static TT _pi2o12() { return _pi2() / tracked::constant<T>("twelve", T(12)); }

    // Small integer / literal named constants (Rule 5)
    static TT _zero()  { return tracked::constant<T>("zero",  T(0.0)); }
    static TT _half()  { return tracked::constant<T>("half",  T(0.5)); }
    static TT _one()   { return tracked::constant<T>("one",   T(1.0)); }
    static TT _two()   { return tracked::constant<T>("two",   T(2.0)); }
    static TT _three() { return tracked::constant<T>("three", T(3.0)); }
    static TT _four()  { return tracked::constant<T>("four",  T(4.0)); }
    static TT _five()  { return tracked::constant<T>("five",  T(5.0)); }
    static TT _six()   { return tracked::constant<T>("six",   T(6.0)); }
    static TT _ten()   { return tracked::constant<T>("ten",   T(10.0)); }

    static TT _eps()    { return tracked::constant<T>("eps",    T(1e-6));  }
    static TT _eps4()   { return tracked::constant<T>("eps4",   T(1e-4));  }
    static TT _eps7()   { return tracked::constant<T>("eps7",   T(1e-7));  }
    static TT _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); }
    static TT _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); }
    static TT _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); }
    static TT _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); }
    static TT _neglig() { return tracked::constant<T>("neglig", T(1e-14)); }
    static TT _reps()   { return tracked::constant<T>("reps",   T(1e-16)); }

    // Complex named constants (Rule 3: container of tracked; Rule 5)
    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _2ipi() {
        // Complex constant (0, 2*pi) — real is zero literal, imag is a named
        // constant "2pi" so its identity is preserved in the journal.
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("2pi", T(2.0) * T(M_PI))
        );
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipio2() {
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("pi/2", T(M_PI) * T(0.5))
        );
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ipi() {
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("pi", T(M_PI))
        );
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps() {
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("reps", T(1e-16))
        );
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps2() {
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16))
        );
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TOutput _ieps50() {
        return TOutput(
            tracked::literal(T(0.0)),
            tracked::constant<T>("ieps50", T(1e-50))
        );
    }
};

// ============================================================================
// Constants<tracked::Complex<T>> specialization (Rule C5, Rule 5)
// ============================================================================
// Some qcdloop code paths (e.g. ql::Constants<TOutput>::_half()) request a
// scalar named constant expressed in TOutput = Complex<T>. Provide the
// full-complex mirror so those call sites resolve.

template<typename T>
struct Constants<tracked::Complex<T>> {
    using TC = tracked::Complex<T>;
    using TT = tracked::Tracked<T>;

    // Real-valued named constants promoted into a complex with zero imag
    // (Rule 3 + Rule 5). The imag component is an anonymous literal (padding).
    static TC _zero()  { return TC(tracked::constant<T>("zero",  T(0.0))); }
    static TC _half()  { return TC(tracked::constant<T>("half",  T(0.5))); }
    static TC _one()   { return TC(tracked::constant<T>("one",   T(1.0))); }
    static TC _two()   { return TC(tracked::constant<T>("two",   T(2.0))); }
    static TC _three() { return TC(tracked::constant<T>("three", T(3.0))); }
    static TC _four()  { return TC(tracked::constant<T>("four",  T(4.0))); }

    // Imaginary-unit-scaled constants (Rule 3 + Rule 5): pure imag values.
    template<typename TOutput, typename TMass, typename TScale>
    static TC _2ipi() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("2pi", T(2.0) * T(M_PI)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TC _ipio2() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("pi/2", T(M_PI) * T(0.5)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TC _ipi() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("pi", T(M_PI)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TC _ieps() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("reps", T(1e-16)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TC _ieps2() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("reps_sq", T(1e-16) * T(1e-16)));
    }

    template<typename TOutput, typename TMass, typename TScale>
    static TC _ieps50() {
        return TC(tracked::literal(T(0.0)),
                  tracked::constant<T>("ieps50", T(1e-50)));
    }
};

// ============================================================================
// Real / Imag / Sign — discrete-vs-tracked classified per use (Rule C6)
// ============================================================================
//
// Real(z) and Imag(z) flow into floating-point arithmetic throughout qcdloop
// (multiplied into products, added into sums, compared to zero, threaded into
// complex constructors). By Rule C6 they are FLOATING-POINT returns and must
// preserve tracking (Rule 2 / Rule 3). Real of a tracked scalar is identity;
// Imag of a tracked scalar is a zero literal.
//
// Sign(x) is consumed BOTH ways in qcdloop:
//   - As a numeric ±1 fed into complex constructors, multiplied into tracked
//     expressions (e.g. `TOutput(ql::Sign(ql::Real(k12))) * ql::kSqrt(...)`) —
//     Rule C6 says this is a floating-point return, so it stays tracked.
//   - As a plain integer in a few discrete contexts (irij scaling by ±1).
//     Kept as tracked too, since the tracked ±1 can be multiplied into a
//     tracked TScale without loss.

// --- Real / Imag on Tracked<T> (Rule 2 + Rule C6) ---------------------------

template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) {
    return x;  // identity — real part of a real is itself
}

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    // Imag of a real scalar is exactly 0; use a literal so downstream ops
    // have a non-empty operand id (no provenance role — Rule 6).
    return tracked::literal(T(0.0));
}

// --- Real / Imag on Complex<T> (Rule 2 + Rule 3 + Rule C6) ------------------

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) {
    return z.real();
}

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) {
    return z.imag();
}

// --- Sign (Rule 2 / Rule C6) ------------------------------------------------
// Sign returns a tracked ±1 or 0; downstream code multiplies this into
// tracked expressions, so keep provenance. The ±1 itself is a runtime-selected
// literal (see the identical rationale in tracked::sqrt for Complex).

template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    return tracked::literal(s);
}

template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z) {
    // Sign of a complex = z / |z| (Rule 3 — result is a complex container).
    // Delegate to tracked's own abs/complex division so per-op provenance is
    // recorded.
    auto mag = tracked::abs(z);
    return z / mag;
}

// ============================================================================
// kAbs — Rule 2 (floating-point return participating in downstream math)
// ============================================================================

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    // Complex magnitude — returns tracked scalar (Rule 2).
    return tracked::abs(z);
}

// ============================================================================
// kLog / kSqrt / kConj — Rule 2 / Rule 3
// ============================================================================

template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    // Complex log — returns complex (Rule 3).
    return tracked::log(z);
}

template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) {
    return tracked::sqrt(x);
}

template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z) {
    return tracked::sqrt(z);
}

template <class T>
inline tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) {
    return x;  // conj of a real is itself
}

template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z) {
    return tracked::conj(z);
}

// ============================================================================
// Max / Min (Rule 2 / Rule 3, Rule 7)
// ============================================================================
// qcdloop's Max/Min select by |a| vs |b|. Comparison is via .value() (Rule 7),
// and the return is the tracked value itself (preserving journal identity of
// the selected operand).

template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    // Compare magnitudes on the raw values (Rule 7).
    T av = std::abs(a.value());
    T bv = std::abs(b.value());
    return (av > bv) ? a : b;
}

template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    // Compare complex magnitudes — |z|² comparison via .value() (Rule 7).
    T av = std::abs(a.real().value()) * std::abs(a.real().value())
         + std::abs(a.imag().value()) * std::abs(a.imag().value());
    T bv = std::abs(b.real().value()) * std::abs(b.real().value())
         + std::abs(b.imag().value()) * std::abs(b.imag().value());
    return (av > bv) ? a : b;
}

template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    T av = std::abs(a.value());
    T bv = std::abs(b.value());
    return (av > bv) ? b : a;
}

template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    T av = std::abs(a.real().value()) * std::abs(a.real().value())
         + std::abs(a.imag().value()) * std::abs(a.imag().value());
    T bv = std::abs(b.real().value()) * std::abs(b.real().value())
         + std::abs(b.imag().value()) * std::abs(b.imag().value());
    return (av > bv) ? b : a;
}

// ============================================================================
// iszero — Rule 1 (discrete bool, used ONLY as an if-condition selector)
// ============================================================================
// The primary ql::iszero returns bool. Every use in the call graph is a
// discrete branch selector (if (iszero(...)), || / && chains inside if).
// Return raw bool by unwrapping through .value() (Rule 7).
//
// The primary is a function template with 3 explicit type parameters
// <TOutput, TMass, TScale> and one value parameter of type TScale. Per Rule C7,
// each concrete tracked overload carries the leading explicit template
// parameters directly so qualified calls ql::iszero<A,B,C>(x) bind to it and
// outrank the primary by partial ordering.

template<class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) {
    // Cutoff is a named constant (Rule 5) — preserve name in journal by using
    // its raw value here (Rule 7: comparison on .value()).
    T cutoff = T(1e-10);
    T v = x.value();
    return (v < T(0) ? -v : v) < cutoff;
}

template<class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Complex<T>& z) {
    T cutoff = T(1e-10);
    T re = z.real().value();
    T im = z.imag().value();
    T mag = std::sqrt(re * re + im * im);
    return mag < cutoff;
}

// ============================================================================
// kPow — Rule 2 / Rule 3 (integer exponent, tracked base)
// ============================================================================
// Primary takes (base, int exponent) with 3 leading explicit template params.
// Per Rule C7, mirror the leading explicit arity on each concrete overload.
// Tracked API does not define pow, so implement as a multiply loop over
// operator* (Rule C2).

template<class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> temp = tracked::literal(T(1.0));  // Rule 6 (anonymous 1)
    for (int i = 0; i < n; ++i) {
        temp = temp * base;
    }
    if (exponent < 0) {
        tracked::Tracked<T> one = tracked::literal(T(1.0));
        return one / temp;
    }
    return temp;
}

template<class TOutput, class TMass, class TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> temp(tracked::literal(T(1.0)), tracked::literal(T(0.0)));  // Rule 6
    for (int i = 0; i < n; ++i) {
        temp = temp * base;
    }
    if (exponent < 0) {
        tracked::Complex<T> one(tracked::literal(T(1.0)), tracked::literal(T(0.0)));
        return one / temp;
    }
    return temp;
}

// ============================================================================
// Htheta — Rule 2 (Heaviside step, feeds tracked arithmetic per Rule C6)
// ============================================================================
// Htheta(x) = 0.5 * (1 + Sign(x)) — yields 0 or 1, but the result is
// multiplied into tracked expressions (see eta2 in kokkosUtils.h), so return
// tracked (Rule 2 + Rule C6).

template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    T v = x.value();
    T s = (T(0) < v) - (v < T(0));
    // Result is a numeric 0 or 1 — literal (Rule 6). No provenance role.
    T h = T(0.5) * (T(1) + s);
    return tracked::literal(h);
}

} // namespace ql