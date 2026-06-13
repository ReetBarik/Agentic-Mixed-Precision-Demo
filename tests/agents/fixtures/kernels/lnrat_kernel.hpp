#pragma once
// Lnrat fixture for the characterizer agent.
//
// Self-contained: includes the minimal ql:: surface this kernel needs.
// ql::kLog and ql::kAbs are DECLARED here but NOT defined — the generated
// micro-driver provides those definitions (opaque_wrap or interop_shim).
//
// Lnrat(x, y) = log(x - i*eps) - log(y - i*eps), with a branch-cut
// correction when x/y lands on the negative real axis.

#include <Kokkos_Core.hpp>
#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>

namespace ql {

template <class T>
struct Constants {
    static T _zero() { return T(0); }
    static T _pi()   { return T(3.14159265358979323846); }

    // (0, pi/2) as a complex TOut — used for the branch-cut correction.
    template <class TOut, class TMass, class TScale>
    static TOut _ipio2() {
        return TOut(T(0), T(3.14159265358979323846 / 2.0));
    }
};

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

// Sign on a scalar — returns int per QCDLoop convention.
template <class T>
inline int Sign(const T& x) { return (T(0) < x) - (x < T(0)); }

// Zero-check: TScale is already tracked::Tracked<double>, so take it directly.
template <class TOutput, class TMass, class TScale>
inline bool iszero(const TScale& x) {
    return x.value() == 0.0;
}

// Forward declarations — driver provides definitions.
template <class T> tracked::Tracked<T> kAbs(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x);
template <class T> tracked::Complex<T> kLog(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kLog(const tracked::Tracked<T>& x);

} // namespace ql

// Complex overload: x and y are TOutput (complex).
// Note: the std::is_same branch in the original kokkosUtils.h is omitted —
// it assigned an unused variable that doesn't compile with tracked types.
template <class TOutput, class TMass, class TScale>
KOKKOS_INLINE_FUNCTION TOutput Lnrat(TOutput const& x, TOutput const& y) {
    const TOutput r = x / y;
    if (ql::iszero<TOutput, TMass, TScale>(ql::Imag(r))) {
        return TOutput(ql::kLog(ql::kAbs(r)))
               - ql::Constants<TScale>::template _ipio2<TOutput, TMass, TScale>()
               * TOutput(ql::Sign(-ql::Real(x)) - ql::Sign(-ql::Real(y)));
    } else {
        return ql::kLog(r);
    }
}

// Real scalar overload: x and y are TScale.
template <class TOutput, class TMass, class TScale>
KOKKOS_INLINE_FUNCTION TOutput Lnrat(TScale const& x, TScale const& y) {
    return TOutput(ql::kLog(ql::kAbs(x / y)))
           - (ql::Constants<TScale>::template _ipio2<TOutput, TMass, TScale>()
              * TOutput(ql::Sign(-x) - ql::Sign(-y)));
}
