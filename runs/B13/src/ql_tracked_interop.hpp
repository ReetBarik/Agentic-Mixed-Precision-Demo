// ql_tracked_interop.hpp
//
// All ql:: overloads needed to instantiate ql::B13() with tracked types.
//
// Kept out of qcdloop's own headers (kokkosMaths.h, kokkosUtils.h) so that
// the surface changes required for Tracked instrumentation are visible in
// one place and easy to audit against future qcdloop revisions.
//
// Idioms follow the cln/lnrat prior art (see runs/cln/src/micro_driver.cpp):
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
//   A. Ported from cln/lnrat drivers (identical semantics):
//        ql::kAbs(Tracked<T>)     -> tracked::abs           (interop_shim)
//        ql::kAbs(Complex<T>)     -> sqrt(re*re + im*im)    (interop_shim)
//        ql::kLog(Tracked<T>)     -> tracked::log           (interop_shim)
//        ql::kLog(Complex<T>)     -> Kokkos::log wrap       (opaque_wrap)
//
//   B. New for B13's math surface:
//        ql::Real(Tracked<T>)     -> identity
//        ql::Real(Complex<T>)     -> .real()
//        ql::Imag(Tracked<T>)     -> zero
//        ql::Imag(Complex<T>)     -> .imag()
//        ql::Sign(Tracked<T>)     -> underlying sign, int
//        ql::iszero(Tracked<T>)   -> |value| < 1e-10
//        ql::kSqrt(Tracked<T>)    -> tracked::sqrt          (interop_shim)
//        ql::kSqrt(Complex<T>)    -> tracked::sqrt          (interop_shim)
//        ql::kPow(Tracked<T>,int) -> repeated *=            (already tracked)
//        ql::kPow(Complex<T>,int) -> repeated *=            (already tracked)
//
// Include order in the driver:
//   Kokkos_Core.hpp -> tracked/* headers -> qcdloop_headers/kokkosMaths.h
//   -> qcdloop_headers/kokkosUtils.h -> ql_tracked_interop.hpp
//   -> qcdloop_headers/box/B2m.h
//
// Rationale: define the overloads after qcdloop's base ql:: dispatchers
// (so their more-generic templates are already visible) and before B2m.h
// (so B13's template body sees tracked overloads at the point of
// instantiation, not just via ADL).

#pragma once

#include <cmath>

#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>
#include <tracked/ops.hpp>

namespace ql {

// -----------------------------------------------------------------------------
// A. Ported from cln/lnrat drivers.
// -----------------------------------------------------------------------------

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x,
                                tracked::SourceLocation loc = {}) {
    return tracked::abs(x, loc);
}

template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z,
                                tracked::SourceLocation loc = {}) {
    // |z| = sqrt(re*re + im*im) — built from instrumented tracked ops so
    // every arithmetic step gets a journal record.
    return tracked::sqrt(z.real() * z.real() + z.imag() * z.imag(), loc);
}

template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x,
                                tracked::SourceLocation loc = {}) {
    return tracked::log(x, loc);
}

template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z,
                                tracked::SourceLocation loc = {}) {
    // opaque_wrap: compute the raw Kokkos::complex log and re-wrap the two
    // scalar components as a tracked Complex with provenance pulled from
    // both input components.
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    Kokkos::complex<T> r = Kokkos::log(raw);
    auto out_re = tracked::opaque_at<T>("Kokkos::log.re", r.real(),
                                        loc, z.real(), z.imag());
    auto out_im = tracked::opaque_at<T>("Kokkos::log.im", r.imag(),
                                        loc, z.real(), z.imag());
    return tracked::Complex<T>(out_re, out_im);
}

// -----------------------------------------------------------------------------
// B. New for B13.
// -----------------------------------------------------------------------------

// Real / Imag: value-preserving projections. No arithmetic, so no journal
// entry needed. Returning a Tracked keeps downstream ops on the tracked path.
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

// Sign: integer result. Not tracked (bool/int-typed control flow).
template <class T>
inline int Sign(const tracked::Tracked<T>& x) {
    return x.value() < T(0) ? -1 : 1;
}

// iszero: threshold on underlying value. qcdloop's kokkosMaths.h uses the
// same 1e-10 cutoff (see Constants::_qlonshellcutoff).
template <class T>
inline bool iszero(const tracked::Tracked<T>& x) {
    T v = x.value();
    return (v < T(0) ? -v : v) < T(1e-10);
}

// kSqrt: interop_shim.
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

// kPow: multiplicative loop, every step already tracked via operator*.
// Matches qcdloop@master semantics (handles negative exponents), unlike
// the drifted copy in Agentic-Mixed-Precision-Demo/src/kokkosMaths.h.
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

} // namespace ql
