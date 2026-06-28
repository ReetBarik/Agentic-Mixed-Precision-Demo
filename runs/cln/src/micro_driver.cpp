// Auto-generated micro-driver for kernel `cLn`.
// Framework: kokkos-serial
// Instruments cLn with the Tracked library to characterize numerical
// sensitivity (condition numbers + relative forward error) of every
// arithmetic op via JSONL records.

#include <Kokkos_Core.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <Kokkos_Complex.hpp>

#include <iostream>
#include <random>
#include <string>

// Include the kernel under test.  This header forward-declares
// ql::kLog / ql::kAbs; we provide the definitions below before any use.
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/cln_kernel.hpp"

// -----------------------------------------------------------------------------
// Interop definitions for ql::kLog / ql::kAbs on tracked types.
//
// The kernel header forward-declares these inside namespace ql; we provide
// the definitions here.  For the complex log we use `opaque_wrap` because
// Kokkos::log(Kokkos::complex<T>) is not overloaded for tracked types and
// the interior decomposition is not the focus of this characterization.
// We still forward both tracked components so provenance is preserved.
//
// The real-scalar overloads of kLog/kAbs and the real-scalar abs of a
// complex value delegate to the templated tracked::log / tracked::abs
// (interop_shim style), which are first-class instrumented ops.
// -----------------------------------------------------------------------------
namespace ql {

// TRACKED_HERE attributes each instrumented op to this driver line so the
// journal carries per-line source locations (the kernel uses operator
// overloads + these ql:: wrappers, neither of which can carry TRACKED_HERE).
template <class T>
tracked::Tracked<T> kAbs(const tracked::Tracked<T>& x) {
    // interop_shim: delegate directly to tracked::abs (instrumented).
    return tracked::abs(x, TRACKED_HERE);
}

template <class T>
tracked::Tracked<T> kAbs(const tracked::Complex<T>& z) {
    // |z| = sqrt(re*re + im*im) — built from instrumented tracked ops so
    // every arithmetic step gets a journal record.
    auto re = z.real();
    auto im = z.imag();
    return tracked::sqrt(re * re + im * im, TRACKED_HERE);
}

template <class T>
tracked::Tracked<T> kLog(const tracked::Tracked<T>& x) {
    // interop_shim: tracked::log is the instrumented scalar logarithm.
    return tracked::log(x, TRACKED_HERE);
}

template <class T>
tracked::Complex<T> kLog(const tracked::Complex<T>& z) {
    // opaque_wrap: compute the raw Kokkos::complex log and re-wrap the
    // two scalar components as a tracked Complex with provenance pulled
    // from both input components.
    Kokkos::complex<T> raw{z.real().value(), z.imag().value()};
    Kokkos::complex<T> r = Kokkos::log(raw);
    auto out_re = tracked::opaque_at<T>("Kokkos::log.re", r.real(),
                                        TRACKED_HERE, z.real(), z.imag());
    auto out_im = tracked::opaque_at<T>("Kokkos::log.im", r.imag(),
                                        TRACKED_HERE, z.real(), z.imag());
    return tracked::Complex<T>(out_re, out_im);
}

} // namespace ql

// -----------------------------------------------------------------------------
// main: 256-sample characterization loop.
// -----------------------------------------------------------------------------
int main() {
    Kokkos::initialize();
    {
        constexpr int sample_count = 256;

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> z_re_dist(-2.0, 2.0);
        std::uniform_real_distribution<double> z_im_dist(-1.0, 1.0);
        std::uniform_real_distribution<double> isig_dist(-1.0, 1.0);

        for (int i = 0; i < sample_count; ++i) {
            double z_re_sample = z_re_dist(rng);
            double z_im_sample = z_im_dist(rng);
            double isig_sample = isig_dist(rng);

            auto z    = tracked::track("z",    z_re_sample, z_im_sample);
            auto isig = tracked::track("isig", isig_sample);

            auto result =
                cLn<tracked::Complex<double>,
                    tracked::Tracked<double>,
                    tracked::Tracked<double>>(z, isig);

            if (i == 0) {
                std::cout << "cLn(z=(" << z_re_sample << "," << z_im_sample
                          << "), isig=" << isig_sample << ") = ("
                          << result.real().value() << ", "
                          << result.imag().value() << ")\n";
            }
        }

        tracked::journal::flush("journal.jsonl");
    }
    Kokkos::finalize();
    return 0;
}
