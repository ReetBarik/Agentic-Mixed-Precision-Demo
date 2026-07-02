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

namespace ql {

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
