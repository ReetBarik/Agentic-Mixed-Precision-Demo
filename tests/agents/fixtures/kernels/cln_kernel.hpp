#pragma once
// cLn fixture for the characterizer agent.
//
// Self-contained: includes the minimal ql:: surface this kernel needs.
// ql::kLog and ql::kAbs are DECLARED here but NOT defined — the generated
// micro-driver provides those definitions (opaque_wrap or interop_shim).

#include <Kokkos_Core.hpp>
#include <tracked/tracked.hpp>
#include <tracked/complex.hpp>

namespace ql {

template <class T>
struct Constants {
    static T _zero() { return T(0); }
    static T _pi()   { return T(3.14159265358979323846); }
};

template <class T>
inline tracked::Tracked<T> Imag(const tracked::Complex<T>& z) { return z.imag(); }

template <class T>
inline tracked::Tracked<T> Real(const tracked::Complex<T>& z) { return z.real(); }

// Sign on a scalar — returns int per QCDLoop convention.
template <class T>
inline int Sign(const T& x) { return (T(0) < x) - (x < T(0)); }

// Forward declarations — driver provides definitions.
template <class T> tracked::Tracked<T> kAbs(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x);
template <class T> tracked::Complex<T> kLog(const tracked::Complex<T>& z);
template <class T> tracked::Tracked<T> kLog(const tracked::Tracked<T>& x);

} // namespace ql

// cLn — complex logarithm from kokkosUtils.h.
// If imag(z)==0 and real(z)<=0, adds the branch-cut term +i*pi*sign(isig).
// TOutput is the complex output type; TScale is the real scalar type for isig.
template <class TOutput, class TMass, class TScale>
KOKKOS_INLINE_FUNCTION TOutput cLn(TOutput const& z, TScale const& isig) {
    TOutput cln;
    if (ql::Imag(z) == ql::Constants<TScale>::_zero() &&
        ql::Real(z) <= ql::Constants<TScale>::_zero()) {
        TOutput temp(ql::Constants<TScale>::_zero(),
                     ql::Constants<TScale>::_pi() * TScale(ql::Sign(isig)));
        cln = ql::kLog(-z) + temp;
    } else {
        cln = ql::kLog(z);
    }
    return cln;
}
