// ql_tracked_interop.hpp
// Tracked<T> / Complex<T> interop shim for the QCDLoop + Kokkos box integrals.
// Generated for BIN3 micro_driver (3-internal-mass box branch, full dispatch tree).
//
// SOURCE_HASH: cfad2410c3ddc32ab520cc03f18dd5e38f62b9fd0359678851e50da9f40a0ac8

#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <type_traits>

// ---------------------------------------------------------------------------
// C5: forward-declare the primary Constants template in ql:: BEFORE our
// specialization parses. The library's own primary is defined later in the
// same TU (kokkosMaths.h), which is fine.
// ---------------------------------------------------------------------------
namespace ql {
    template <typename T> struct Constants;
}

// ---------------------------------------------------------------------------
// C3: identity unary operator+ on Tracked<T> / Complex<T>. Some library
// templates rely on unary + being available generically; Tracked doesn't
// define it. Emit as a free function in the tracked namespace for ADL. No
// journal record (identity, no rounding).
// Rule justification: Rule C3.
// ---------------------------------------------------------------------------
namespace tracked {
    template <class T>
    inline Tracked<T> operator+(const Tracked<T>& a) { return a; } // Rule C3

    template <class T>
    inline Complex<T> operator+(const Complex<T>& a) { return a; } // Rule C3
} // namespace tracked

// ===========================================================================
// ql:: namespace shims
// ===========================================================================
namespace ql {

// ---------------------------------------------------------------------------
// Rule 5 / C5: partial specialization of ql::Constants<T> on the tracked
// scalar. Each named leaf is routed through tracked::constant("<name>", ...)
// so the journal preserves the constant's identity. Members that are
// container/output types (e.g. _ieps50, _2ipi, _ipi, _ipio2, _ieps, _ieps2)
// route through tracked::Complex<T> when the primary returns TOutput; the
// primary here is generic on T so we mirror by returning tracked scalars for
// the T=Tracked<T> instantiation.
//
// The library also invokes _ieps50<TOutput,TMass,TScale>() etc. on
// Constants<TScale> — those template members forward TOutput. We provide
// them mirroring the primary's shape.
//
// Rule justification: Rule 5 (named constants) + Rule C5 (class-template
// partial specialization on the tracked scalar). Chebyshev / Bernoulli
// arrays are anonymous inline literals (Rule 6) — driven by index, they
// are not user-named constants.
// ---------------------------------------------------------------------------
template <class T>
struct Constants<tracked::Tracked<T>> {
    using S = tracked::Tracked<T>;

    // ---- table sizes (Rule 1: discrete return) ------------------------
    KOKKOS_INLINE_FUNCTION static constexpr int _num_C() { return 19; } // Rule 1
    KOKKOS_INLINE_FUNCTION static constexpr int _num_B() { return 25; } // Rule 1

    // ---- Chebyshev coefficients (Rule 6: anonymous inline literals) ---
    static inline S _C(int i) {
        constexpr double coeffs[19] = {
            0.4299669356081370, 0.4097598753307711, -0.0185884366501460,
            0.0014575108406227, -0.0001430418444234, 0.0000158841554188,
            -0.0000019078495939, 0.0000002419518085, -0.0000000319334127,
            0.0000000043454506, -0.0000000006057848, 0.0000000000861210,
            -0.0000000000124433, 0.0000000000018226, -0.0000000000002701,
            0.0000000000000404, -0.0000000000000061, 0.0000000000000009,
            -0.0000000000000001
        };
        return tracked::literal<T>(T(coeffs[i])); // Rule 6
    }

    // ---- Bernoulli-like coefficients (Rule 6) -------------------------
    static inline S _B(int i) {
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
        return tracked::literal<T>(T(coeffs[i])); // Rule 6
    }

    // ---- named onshell cutoff (Rule 5) --------------------------------
    template <typename TOutput, typename TMass, typename TScale>
    static inline S _qlonshellcutoff() {
        return tracked::constant<T>("qlonshellcutoff", T(1e-10)); // Rule 5
    }

    // ---- named mathematical constants (Rule 5) ------------------------
    static inline S _pi()    { return tracked::constant<T>("pi",  T(M_PI));         } // Rule 5
    static inline S _pi2()   { return tracked::constant<T>("pi2", T(M_PI) * T(M_PI)); } // Rule 5

    template <typename TOutput, typename TMass, typename TScale>
    static inline S _pio3()   { return tracked::constant<T>("pio3",   T(M_PI) / T(3));  } // Rule 5
    template <typename TOutput, typename TMass, typename TScale>
    static inline S _pio6()   { return tracked::constant<T>("pio6",   T(M_PI) / T(6));  } // Rule 5
    template <typename TOutput, typename TMass, typename TScale>
    static inline S _pi2o3()  { return tracked::constant<T>("pi2o3",  T(M_PI) * T(M_PI) / T(3));  } // Rule 5
    template <typename TOutput, typename TMass, typename TScale>
    static inline S _pi2o6()  { return tracked::constant<T>("pi2o6",  T(M_PI) * T(M_PI) / T(6));  } // Rule 5
    template <typename TOutput, typename TMass, typename TScale>
    static inline S _pi2o12() { return tracked::constant<T>("pi2o12", T(M_PI) * T(M_PI) / T(12)); } // Rule 5

    // ---- named small integers (Rule 5) --------------------------------
    static inline S _zero()  { return tracked::constant<T>("zero",  T(0));    } // Rule 5
    static inline S _half()  { return tracked::constant<T>("half",  T(0.5));  } // Rule 5
    static inline S _one()   { return tracked::constant<T>("one",   T(1));    } // Rule 5
    static inline S _two()   { return tracked::constant<T>("two",   T(2));    } // Rule 5
    static inline S _three() { return tracked::constant<T>("three", T(3));    } // Rule 5
    static inline S _four()  { return tracked::constant<T>("four",  T(4));    } // Rule 5
    static inline S _five()  { return tracked::constant<T>("five",  T(5));    } // Rule 5
    static inline S _six()   { return tracked::constant<T>("six",   T(6));    } // Rule 5
    static inline S _ten()   { return tracked::constant<T>("ten",   T(10));   } // Rule 5

    // ---- named epsilons (Rule 5) --------------------------------------
    static inline S _eps()    { return tracked::constant<T>("eps",    T(1e-6));  } // Rule 5
    static inline S _eps4()   { return tracked::constant<T>("eps4",   T(1e-4));  } // Rule 5
    static inline S _eps7()   { return tracked::constant<T>("eps7",   T(1e-7));  } // Rule 5
    static inline S _eps10()  { return tracked::constant<T>("eps10",  T(1e-10)); } // Rule 5
    static inline S _eps14()  { return tracked::constant<T>("eps14",  T(1e-14)); } // Rule 5
    static inline S _eps15()  { return tracked::constant<T>("eps15",  T(1e-15)); } // Rule 5
    static inline S _xloss()  { return tracked::constant<T>("xloss",  T(0.125)); } // Rule 5
    static inline S _neglig() { return tracked::constant<T>("neglig", T(1e-14)); } // Rule 5
    static inline S _reps()   { return tracked::constant<T>("reps",   T(1e-16)); } // Rule 5

    // ---- complex-shaped named constants (Rule 3 + Rule 5) -------------
    // These return TOutput in the primary. For the tracked case TOutput is
    // tracked::Complex<T>, so we return that container (Rule 3) built out
    // of named tracked scalar leaves.
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _2ipi() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("two_pi", T(2) * T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _ipio2() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("pio2", T(M_PI) * T(0.5)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _ipi() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("pi",   T(M_PI)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _ieps() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero", T(0)),
            tracked::constant<T>("reps", T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _ieps2() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero",  T(0)),
            tracked::constant<T>("reps2", T(1e-16) * T(1e-16)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Complex<T> _ieps50() { // Rule 3 + Rule 5
        return tracked::Complex<T>(
            tracked::constant<T>("zero",   T(0)),
            tracked::constant<T>("ieps50", T(1e-50)));
    }
};

// ---------------------------------------------------------------------------
// Utility: unwrap Tracked<T> back to T for calls where we need to build a
// raw scalar (e.g. std::mt19937 seeds, printing). Not used by shims; the
// shims stay tracked.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// kPow overloads. The library's kPow<TOutput,TMass,TScale>(base, int) is
// declared on both TOutput and TMass. Under partial ordering (Rule C7), we
// must supply constrained overloads keyed on the tracked value types,
// carrying the same leading explicit template parameters.
//
// Rule 2 (floating-point return) + Rule C7 (outrank the library primary).
// ---------------------------------------------------------------------------
template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, const int& exponent) { // Rule 2 + Rule C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> temp = tracked::literal<T>(T(1)); // Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) return tracked::literal<T>(T(1)) / temp; // Rule 6
    return temp;
}

template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kPow(const tracked::Complex<T>& base, const int& exponent) { // Rule 3 + Rule C7
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> temp(tracked::literal<T>(T(1)), tracked::literal<T>(T(0))); // Rule 6
    for (int i = 0; i < n; ++i) temp = temp * base;
    if (exponent < 0) {
        tracked::Complex<T> one_c(tracked::literal<T>(T(1)), tracked::literal<T>(T(0))); // Rule 6
        return one_c / temp;
    }
    return temp;
}

// ---------------------------------------------------------------------------
// Math dispatch: kAbs / kLog / kSqrt / kConj on tracked scalar and complex.
// kAbs returns REAL in both scalar and complex cases (see kokkosMaths.h).
// Rule 2 (scalar) / Rule 3 (complex operations that return container).
// ---------------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) { // Rule 2
    return tracked::abs(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kAbs(const tracked::Complex<T>& x) { // Rule 2 (complex |z| is real)
    return tracked::abs(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) { // Rule 2
    return tracked::log(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kLog(const tracked::Complex<T>& x) { // Rule 3
    return tracked::log(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x) { // Rule 2
    return tracked::sqrt(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kSqrt(const tracked::Complex<T>& x) { // Rule 3
    return tracked::sqrt(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> kConj(const tracked::Complex<T>& x) { // Rule 3
    return tracked::conj(x);
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> kConj(const tracked::Tracked<T>& x) { // Rule 2 (real conj is identity)
    return x;
}

// ---------------------------------------------------------------------------
// iszero: Rule 1 (discrete bool return). Compares underlying values.
// The library primary is template<TOutput,TMass,TScale> bool iszero(TScale&).
// Under Rule C7 we carry the same leading explicit template parameters.
// ---------------------------------------------------------------------------
template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Tracked<T>& x) { // Rule 1 + Rule C7
    return std::abs(x.value()) < T(1e-10);
}

template <typename TOutput, typename TMass, typename TScale, class T>
KOKKOS_INLINE_FUNCTION
bool iszero(const tracked::Complex<T>& x) { // Rule 1 + Rule C7
    T mag2 = x.real().value() * x.real().value()
           + x.imag().value() * x.imag().value();
    return std::sqrt(mag2) < T(1e-10);
}

// ---------------------------------------------------------------------------
// Imag / Real: Rule 2 (floating-point return participating in downstream
// arithmetic). Real part of a real is the value; imag is a named-zero.
// ---------------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) { // Rule 2
    return tracked::constant<T>("zero", T(0)); // Rule 5
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Imag(const tracked::Complex<T>& x) { // Rule 2
    return x.imag();
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { // Rule 2
    return x;
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Real(const tracked::Complex<T>& x) { // Rule 2
    return x.real();
}

// ---------------------------------------------------------------------------
// Sign: C6 — the library uses Sign both as a numeric ±1 multiplier that
// flows into tracked arithmetic AND as a discrete int in a handful of
// places (comparisons like `ir == ik`). Its dominant use is as a numeric
// multiplier (`TOutput(ql::Sign(...))`), so per C6 we return the tracked
// scalar for tracked inputs (Rule 2). Comparisons against these tracked
// ±1/0 values still work through Tracked's operator== on values.
// ---------------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) { // Rule 2 + Rule C6
    T v = x.value();
    T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
    return tracked::literal<T>(s); // Rule 6 (runtime-selected ±1/0)
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Sign(const tracked::Complex<T>& x) { // Rule 3 + Rule C6
    return x / tracked::abs(x);
}

// ---------------------------------------------------------------------------
// Max / Min: choose by |value|, return the chosen tracked value unmodified.
// Rule 2 / Rule 3 (floating-point / complex return). No new op emitted;
// this is a select, not an arithmetic op.
// ---------------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Max(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) { // Rule 2
    return (std::abs(a.value()) > std::abs(b.value())) ? a : b;
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Max(const tracked::Complex<T>& a, const tracked::Complex<T>& b) { // Rule 3
    T am = std::sqrt(a.real().value()*a.real().value() + a.imag().value()*a.imag().value());
    T bm = std::sqrt(b.real().value()*b.real().value() + b.imag().value()*b.imag().value());
    return (am > bm) ? a : b;
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Min(const tracked::Tracked<T>& a, const tracked::Tracked<T>& b) { // Rule 2
    return (std::abs(a.value()) > std::abs(b.value())) ? b : a;
}

template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Complex<T> Min(const tracked::Complex<T>& a, const tracked::Complex<T>& b) { // Rule 3
    T am = std::sqrt(a.real().value()*a.real().value() + a.imag().value()*a.imag().value());
    T bm = std::sqrt(b.real().value()*b.real().value() + b.imag().value()*b.imag().value());
    return (am > bm) ? b : a;
}

// ---------------------------------------------------------------------------
// Htheta: Heaviside step. C6: result flows into arithmetic expressions
// (multiplies into TOutput factors), so must return the tracked scalar
// (Rule 2), not a raw double.
// ---------------------------------------------------------------------------
template <class T>
KOKKOS_INLINE_FUNCTION
tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) { // Rule 2 + Rule C6
    T v = x.value();
    T s = (T(0) < v) ? T(1) : ((v < T(0)) ? T(-1) : T(0));
    // 0.5 * (1 + sign(x)); step value is a runtime-selected literal.
    return tracked::literal<T>(T(0.5) * (T(1) + s)); // Rule 6
}

} // namespace ql

// Note: cLn, Lnrat, ddilog, li2series, denspence, Li2omx*, spencer, eta*,
// xspence, xeta, xetatilde, etatilde, kfn, ratreal, ratgam, solveabc(d),
// R2int, R3int, Rint, R, Zlogint, ltli2series, ltspence, cspence, L0, L1,
// fndd, Ycalc — these library templates are themselves generic on
// TOutput/TMass/TScale and are compiled directly against the tracked types
// via the primitive operators, kAbs/kLog/kSqrt/kConj, iszero, Imag/Real,
// Sign, Max/Min, Htheta, kPow, and Constants above. They therefore need no
// separate shim overloads; providing the primitives is sufficient for
// partial-ordering to keep the library's own definitions selected and for
// them to compile. (Rule C7 does not apply because these are function
// templates the library owns and we are not redefining them.)

// ---------------------------------------------------------------------------
// Execution-space annotations:
// Rule 8 + Rule C4 — the driver invokes ql::BO from a plain host loop
// (NOT inside Kokkos::parallel_for). Tracked ops use std::string and
// journaling, which are host-only. Even though the library's own functions
// carry KOKKOS_INLINE_FUNCTION (which expands to `inline` when only the
// host backend is enabled), we omit any device annotation on our shim
// overloads that would otherwise be flagged for device execution. The
// KOKKOS_INLINE_FUNCTION tags above are safe: with host-only Kokkos they
// reduce to `inline`, and the driver never launches a device kernel.
// ---------------------------------------------------------------------------