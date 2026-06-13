// Auto-generated micro-driver for Lnrat (complex overload)
// Framework: kokkos-serial
// Characterizes numerical sensitivity via the Tracked library.

#include <Kokkos_Core.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <random>
#include <iostream>

// ---------------------------------------------------------------------------
// ql::kLog / ql::kAbs definitions.
//
// ql:: declares these as forward-decls; the kernel header expects the driver
// to provide bodies.  We choose:
//   * kAbs(Tracked<T>)        -> tracked::abs        (interop_shim)
//   * kLog(Tracked<T>)        -> tracked::log        (interop_shim)
//   * kAbs(Complex<T>)        -> tracked::opaque     (opaque_wrap)
//   * kLog(Complex<T>)        -> tracked::opaque     (opaque_wrap)
//
// The complex variants are wrapped opaquely because Kokkos::log / Kokkos::abs
// are not overloaded for tracked::Complex<T>; we compute the raw result on
// Kokkos::complex<T> and re-enter the tracked domain via tracked::opaque,
// preserving provenance through both real and imaginary parts.
// ---------------------------------------------------------------------------

// Pull the kernel header in first so the forward declarations exist.
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/lnrat_kernel.hpp"

namespace ql {

// Scalar Tracked overloads — direct delegation to tracked:: math.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    return tracked::abs(x);
}

template <class T>
inline tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    return tracked::log(x);
}

// Complex overloads — opaque wrap around Kokkos::abs / Kokkos::log.
template <class T>
inline tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    T r = Kokkos::abs(raw);
    return tracked::opaque<T>("Kokkos::abs", r, z.real(), z.imag());
}

template <class T>
inline tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    Kokkos::complex<T> r = Kokkos::log(raw);
    auto re = tracked::opaque<T>("Kokkos::log.re", r.real(),
                                 z.real(), z.imag());
    auto im = tracked::opaque<T>("Kokkos::log.im", r.imag(),
                                 z.real(), z.imag());
    return tracked::Complex<T>(re, im);
}

} // namespace ql

int main() {
    Kokkos::initialize();
    {
        using Tracked = tracked::Tracked<double>;
        using Complex = tracked::Complex<double>;

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> x_re_dist(0.5, 3.0);
        std::uniform_real_distribution<double> x_im_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> y_re_dist(0.5, 3.0);
        std::uniform_real_distribution<double> y_im_dist(-1.0, 1.0);

        constexpr int sample_count = 256;

        for (int i = 0; i < sample_count; ++i) {
            double x_re_s = x_re_dist(rng);
            double x_im_s = x_im_dist(rng);
            double y_re_s = y_re_dist(rng);
            double y_im_s = y_im_dist(rng);

            auto x = tracked::track("x", x_re_s, x_im_s);
            auto y = tracked::track("y", y_re_s, y_im_s);

            Complex result =
                Lnrat<Complex, Tracked, Tracked>(x, y);

            if (i == 0) {
                std::cout << "Lnrat(x, y) sample 0: ("
                          << result.real().value() << ", "
                          << result.imag().value() << ")\n";
            }
        }

        tracked::journal::flush("journal.jsonl");
    }
    Kokkos::finalize();
    return 0;
}
