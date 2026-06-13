#pragma once
// Shared ql:: surface for QCDLoop-family fixture kernels.
// Each kernel fixture includes this header instead of defining its own ql:: namespace.
//
// ql::kLog and ql::kAbs are DECLARED here but NOT defined.  The generated
// micro-driver provides those definitions (opaque_wrap or interop_shim).

#include <Kokkos_Core.hpp>
#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>

namespace ql {

template <class T>
struct Constants {
    static T _zero() { return T(0); }
    static T _pi()   { return T(3.14159265358979323846); }

    // (0, pi/2) as a complex TOut — used by Lnrat's branch-cut correction.
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

// Zero-check on a tracked scalar.
template <class TOutput, class TMass, class TScale>
inline bool iszero(const tracked::Tracked<TScale>& x) {
    return x.value() == 0.0;
}

// Forward declarations — driver provides definitions.
template <class T> tracked::Tracked<T> kAbs(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x);
template <class T> tracked::Complex<T> kLog(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kLog(const tracked::Tracked<T>& x);

} // namespace ql
