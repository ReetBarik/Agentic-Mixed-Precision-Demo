// ql_tracked_interop.hpp
//
// All ql:: overloads needed to instantiate ql::B13() (and ql::BO()'s
// full box-integral dispatcher) with tracked types.
//
// Kept out of qcdloop's own headers (kokkosMaths.h, kokkosUtils.h) so
// the surface changes required for Tracked instrumentation are visible
// in one place and easy to audit against future qcdloop revisions.
//
// Reference: ReetBarik/qcdloop@ddfun_enabled src/qcdloop/kokkosMaths_dd.h
// — a functional, production-tested precision swap for double-double.
// Every overload here mirrors the DD overload set in name, signature
// shape, and semantics. Deviations are documented per-overload.
//
// Two idioms (from runs/cln/, runs/lnrat/ prior art):
//
//   interop_shim : delegate to an already-instrumented tracked:: op.
//                  Every arithmetic step of the underlying computation
//                  gets a journal record.
//
//   opaque_wrap  : call the raw (non-tracked) implementation and re-enter
//                  the tracked domain via tracked::opaque_at, preserving
//                  provenance across the boundary. Used when the raw op
//                  is opaque enough (or not overloaded for tracked types)
//                  that interior instrumentation isn't useful.
//
// Overloads provided (grouped by category):
//
//   A. Math dispatch (mirrors kokkosMaths_dd.h):
//        kAbs, kLog, kSqrt, kPow      (scalar + complex)
//        Real, Imag                   (scalar + complex)
//        Sign                         (scalar + complex)
//        iszero                       (scalar)
//        kConj                        (complex)
//        Max, Min                     (scalar + complex)
//        Htheta                       (scalar)
//
//   B13 body itself uses only: Constants, Real, Imag, iszero, kSqrt,
//   kPow, kLog, Lnrat, Li2omrat, Li2omx2, cLn, spencer, ratreal, ratgam.
//   kAbs/kLog/kSqrt/kPow/Real/Imag/Sign/iszero cover B13's call graph
//   through the dilog family. kConj/Max/Min/Htheta are unused by B13
//   but included so this header stays a complete audit surface for
//   future kernels (kConj shows up in xspence; Htheta in eta/spencer
//   deep paths).
//
// Include order in the driver:
//   Kokkos_Core.hpp -> tracked/* -> qcdloop_headers/kokkosMaths.h
//   -> qcdloop_headers/kokkosUtils.h -> ql_tracked_interop.hpp
//   -> qcdloop_headers/boxGPU.h
//
// Rationale: define the overloads after qcdloop's base ql:: dispatchers
// (so their more-generic templates are already visible for ADL) and
// before boxGPU.h / the Bnm headers (so B13's template body sees the
// tracked overloads at the point of instantiation, not just via ADL).

#pragma once

#include <cmath>

#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>
#include <tracked/ops.hpp>

// =============================================================================
// tracked:: numeric-affordance shims
// =============================================================================
//
// qcdloop's box bodies (e.g. B13 in box/B2m.h:299-306) write expressions like
//   TOutput(+p3sq + m3sq - m4sq)
// i.e. a *unary plus* on a TMass. This works for the double-double reference
// type (ddouble has unary+) but tracked::Tracked deliberately ships only the
// binary arithmetic operators. Unary plus is the numeric identity, so we add
// it here as a free function in namespace tracked (found by ADL for `+x`),
// keeping the shim in this single audit-surface header rather than editing the
// vendored tracked library. No journal record: identity introduces no rounding.
namespace tracked {

template <class T>
inline Tracked<T> operator+(const Tracked<T>& x) { return x; }

} // namespace tracked

namespace ql {

// Forward declaration of the primary template. In this build the driver
// includes ql_tracked_interop.hpp *before* qcdloop_headers/kokkosMaths.h
// (see micro_driver.cpp), so the primary ql::Constants<T> definition is
// not yet visible here. A partial specialization only needs the primary
// template *declared* at parse time; kokkosMaths.h supplies the full
// definition later in the same TU, before any instantiation.
template <typename T>
struct Constants;

// ============================================================
// Constants<Tracked<T>> specialization
//
// Local B13-spike workaround for the tracked-library constant-
// naming gap. Routes every constant through tracked::track so
// provenance is preserved into arithmetic that touches
// ql::Constants::_foo(). See MEMORY.md follow-up (a) for scope
// and a link to any upstream Tracked issue that supersedes this.
//
// Without this, the generic ql::Constants<T> (kokkosMaths.h)
// returns T(literal_double); for T = tracked::Tracked<double>
// that mints an *anonymous* Tracked (empty provenance). Paths
// that subtract two such constants produce the useless
//   {"op":"sub","in":["_","_"],"prov":[],"cond":9e15,...}
// records (2397 / 0.73% of the 256-sample journal at 7a5e231).
//
// Fix: mirror the FULL generic Constants<T> interface so no
// ql:: symbol is lost, but mint each leaf scalar constant via
// tracked::track("_name", value). Numeric values are copied
// verbatim from qcdloop_headers/kokkosMaths.h's generic
// template (NOT the DD reference, whose eps/coeff values differ)
// so the tracked build reproduces the untracked spike's numerics
// bit-for-bit; only the provenance labels are added.
//
// track() uses std::string / the journal, so these members are
// host-only plain-inline (not KOKKOS_INLINE_FUNCTION / constexpr)
// — fine for the Serial-only spike. _num_C()/_num_B() stay
// constexpr int: they are loop bounds, carry no value to track.
// ============================================================

template <class T>
struct Constants<tracked::Tracked<T>> {

    // ---- coefficient-count accessors (loop bounds, untracked) ----
    static constexpr int _num_C() { return 19; }
    static constexpr int _num_B() { return 25; }

    // ---- Chebyshev coefficients for ddilog (19 terms) ----
    static inline tracked::Tracked<T> _C(int i) {
        const double coeffs[19] = {
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
        return tracked::track<T>("_C", T(coeffs[i]));
    }

    // ---- Bernoulli coefficients for li2series (25 terms) ----
    static inline tracked::Tracked<T> _B(int i) {
        const double coeffs[25] = {
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
        return tracked::track<T>("_B", T(coeffs[i]));
    }

    // ---- on-shell cutoff (templated to match generic signature) ----
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _qlonshellcutoff() {
        return tracked::track<T>("_qlonshellcutoff", T(1e-10));
    }

    // ---- pi family ----
    static inline tracked::Tracked<T> _pi() {
        return tracked::track<T>("_pi", T(M_PI));
    }
    static inline tracked::Tracked<T> _pi2() {
        return tracked::track<T>("_pi2", T(M_PI * M_PI));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _pio3() {
        return tracked::track<T>("_pio3", T(M_PI / 3.0));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _pio6() {
        return tracked::track<T>("_pio6", T(M_PI / 6.0));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _pi2o3() {
        return tracked::track<T>("_pi2o3", T(M_PI * (M_PI / 3.0)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _pi2o6() {
        return tracked::track<T>("_pi2o6", T(M_PI * (M_PI / 6.0)));
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline tracked::Tracked<T> _pi2o12() {
        return tracked::track<T>("_pi2o12", T((M_PI * M_PI) / 12.0));
    }

    // ---- integer / simple-fraction constants ----
    static inline tracked::Tracked<T> _zero()  { return tracked::track<T>("_zero",  T(0.0));  }
    static inline tracked::Tracked<T> _half()  { return tracked::track<T>("_half",  T(0.5));  }
    static inline tracked::Tracked<T> _one()   { return tracked::track<T>("_one",   T(1.0));  }
    static inline tracked::Tracked<T> _two()   { return tracked::track<T>("_two",   T(2.0));  }
    static inline tracked::Tracked<T> _three() { return tracked::track<T>("_three", T(3.0));  }
    static inline tracked::Tracked<T> _four()  { return tracked::track<T>("_four",  T(4.0));  }
    static inline tracked::Tracked<T> _five()  { return tracked::track<T>("_five",  T(5.0));  }
    static inline tracked::Tracked<T> _six()   { return tracked::track<T>("_six",   T(6.0));  }
    static inline tracked::Tracked<T> _ten()   { return tracked::track<T>("_ten",   T(10.0)); }

    // ---- epsilon / tolerance constants ----
    static inline tracked::Tracked<T> _eps()    { return tracked::track<T>("_eps",    T(1e-6));  }
    static inline tracked::Tracked<T> _eps4()   { return tracked::track<T>("_eps4",   T(1e-4));  }
    static inline tracked::Tracked<T> _eps7()   { return tracked::track<T>("_eps7",   T(1e-7));  }
    static inline tracked::Tracked<T> _eps10()  { return tracked::track<T>("_eps10",  T(1e-10)); }
    static inline tracked::Tracked<T> _eps14()  { return tracked::track<T>("_eps14",  T(1e-14)); }
    static inline tracked::Tracked<T> _eps15()  { return tracked::track<T>("_eps15",  T(1e-15)); }
    static inline tracked::Tracked<T> _xloss()  { return tracked::track<T>("_xloss",  T(0.125)); }
    static inline tracked::Tracked<T> _neglig() { return tracked::track<T>("_neglig", T(1e-14)); }
    static inline tracked::Tracked<T> _reps()   { return tracked::track<T>("_reps",   T(1e-16)); }

    // ---- complex composites (verbatim from generic; the scalar
    //      constants they build from are now named via the above) ----
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _2ipi() {
        return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_two() * Constants<TScale>::_pi()};
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _ipio2() {
        return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi() * Constants<TScale>::_half()};
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _ipi() {
        return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_pi()};
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _ieps() {
        return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps()};
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _ieps2() {
        return TOutput{Constants<TScale>::_zero(), Constants<TScale>::_reps() * Constants<TScale>::_reps()};
    }
    template <typename TOutput, typename TMass, typename TScale>
    static inline TOutput _ieps50() {
        return TOutput{Constants<TScale>::_zero(), TScale(1e-50)};
    }
};

// =============================================================================
// kAbs
// =============================================================================

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x,
                                tracked::SourceLocation loc = {}) {
    return tracked::abs(x, loc);
}

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z,
                                tracked::SourceLocation loc = {}) {
    // |z| = sqrt(re*re + im*im), fully instrumented (interop_shim style).
    return tracked::sqrt(z.real() * z.real() + z.imag() * z.imag(), loc);
}

// =============================================================================
// kLog
// =============================================================================

template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x,
                                tracked::SourceLocation loc = {}) {
    return tracked::log(x, loc);
}

template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z,
                                tracked::SourceLocation loc = {}) {
    // opaque_wrap: compute the raw Kokkos::complex log and re-wrap the
    // scalar components via tracked::opaque_at, pulling provenance from
    // both input components.
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    Kokkos::complex<T> r = Kokkos::log(raw);
    auto out_re = tracked::opaque_at<T>("Kokkos::log.re", r.real(),
                                        loc, z.real(), z.imag());
    auto out_im = tracked::opaque_at<T>("Kokkos::log.im", r.imag(),
                                        loc, z.real(), z.imag());
    return tracked::Complex<T>(out_re, out_im);
}

// =============================================================================
// kSqrt
// =============================================================================

template <class T>
inline tracked::Tracked<T> kSqrt(const tracked::Tracked<T>& x,
                                 tracked::SourceLocation loc = {}) {
    return tracked::sqrt(x, loc);
}

template <class T>
inline tracked::Complex<T> kSqrt(const tracked::Complex<T>& z,
                                 tracked::SourceLocation loc = {}) {
    return tracked::sqrt(z, loc);
}

// =============================================================================
// kPow
// =============================================================================
//
// Matches qcdloop@master's semantics (kokkosMaths_dd.h drops the
// negative-exponent guard; we keep it since B13 could grow one via
// future work). Multiplicative loop — every step already tracked via
// operator*.

template <class T>
inline tracked::Tracked<T> kPow(const tracked::Tracked<T>& base, int exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Tracked<T> t(T(1));
    for (int i = 0; i < n; ++i) t = t * base;
    return exponent < 0 ? tracked::Tracked<T>(T(1)) / t : t;
}

template <class T>
inline tracked::Complex<T> kPow(const tracked::Complex<T>& base, int exponent) {
    const int n = exponent < 0 ? -exponent : exponent;
    tracked::Complex<T> t(T(1));
    for (int i = 0; i < n; ++i) t = t * base;
    return exponent < 0 ? tracked::Complex<T>(T(1)) / t : t;
}

// =============================================================================
// Real / Imag
// =============================================================================
//
// Value-preserving projections. Returning a Tracked (not a bare T) keeps
// downstream ops on the tracked path and preserves provenance.

template <class T>
inline tracked::Tracked<T> Real(const tracked::Tracked<T>& x) { return x; }

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Tracked<T>& /*x*/) {
    return tracked::Tracked<T>(T(0));
}

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

// =============================================================================
// Sign
// =============================================================================
//
// Mirrors DD's Sign(ddouble)/Sign(ddcomplex) — returns the numeric type,
// not int. Preserves provenance through comparisons.
//
// B13 uses Sign in expressions like TOutput(ql::Sign(-ql::Real(x))), so
// the return type must be convertible to TOutput (tracked::Complex<T>)
// while preserving the tracked provenance graph.

template <class T>
inline tracked::Tracked<T> Sign(const tracked::Tracked<T>& x) {
    const T v = x.value();
    if (v > T(0)) return tracked::Tracked<T>(T( 1));
    if (v < T(0)) return tracked::Tracked<T>(T(-1));
    return                tracked::Tracked<T>(T( 0));
}

template <class T>
inline tracked::Complex<T> Sign(const tracked::Complex<T>& z,
                                tracked::SourceLocation loc = {}) {
    // DD's Sign(ddcomplex) = z / |z|. We build it from instrumented ops.
    return z / kAbs(z, loc);
}

// =============================================================================
// iszero
// =============================================================================
//
// bool return, no tracking wrapper needed. Threshold matches
// Constants::_qlonshellcutoff (1e-10) used elsewhere in qcdloop.

template <class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    const T v = x.value();
    return (v < T(0) ? -v : v) < T(1e-10);
}

// =============================================================================
// kConj
// =============================================================================
//
// Used by xspence (not on B13's call path, but included for completeness).

template <class T>
inline tracked::Complex<T> kConj(const tracked::Complex<T>& z,
                                 tracked::SourceLocation loc = {}) {
    return tracked::conj(z, loc);
}

// =============================================================================
// Max / Min
// =============================================================================
//
// DD's semantics: compare by |a| vs |b|. Not on B13's call path but
// included for completeness with the DD overload set.

template <class T>
inline tracked::Tracked<T> Max(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    return kAbs(a).value() > kAbs(b).value() ? a : b;
}

template <class T>
inline tracked::Complex<T> Max(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return kAbs(a).value() > kAbs(b).value() ? a : b;
}

template <class T>
inline tracked::Tracked<T> Min(const tracked::Tracked<T>& a,
                               const tracked::Tracked<T>& b) {
    return kAbs(a).value() > kAbs(b).value() ? b : a;
}

template <class T>
inline tracked::Complex<T> Min(const tracked::Complex<T>& a,
                               const tracked::Complex<T>& b) {
    return kAbs(a).value() > kAbs(b).value() ? b : a;
}

// =============================================================================
// Htheta
// =============================================================================
//
// Heaviside step: 0.5 * (1 + Sign(x)). Used in eta functions / xspence
// deep paths — not on B13's direct call graph but part of the DD surface.

template <class T>
inline tracked::Tracked<T> Htheta(const tracked::Tracked<T>& x) {
    return tracked::Tracked<T>(T(0.5)) *
           (tracked::Tracked<T>(T(1.0)) + Sign(x));
}

} // namespace ql
