// runs/B16/src/micro_driver.cpp
//
// Stage 2 leaf validation (post-C8): instantiate ql::BO routing by kinematics
// to the 3-internal-mass box branch (the B3m family). B16 is the named
// three-mass finite box (Beenakker et al.); with these kinematics B3m
// dispatches to ql::B16(). Both B16() and BIN3() are statically instantiated,
// so this target exercises the C8 int<->Tracked boundary annotations that the
// integrator emits for box/B3m.h (and box/B4m.h) via the <app>.patch.
//
// Recipe adapted from examples/boxGPU_test.cc's "// B16" block (three-mass):
//   m1=0, m2=m22, m3=m32, m4=m42;
//   p1=m22, p2=rs(low,up), p3=rs(low,up), p4=m42, p5=r(low,up), p6=r(low,up).
// As with the rest of the sweep, the upstream srand/rand draws are replaced by
// the shared mt19937(12345) + r_signed/r_uniform helpers so op-count
// distributions cross-compare with the B13 reference; each array slot is tracked
// as an independent input carrying its numeric value (preserving per-slot
// provenance).
//
// Structure mirrors runs/B13/src/micro_driver.cpp: same Views, same 256-sample
// batch, same host-loop execution model (tracked ops are host-only, so no
// Kokkos::parallel_for), same per-sample scope.
//
// The qcdloop headers come from the UN-PRUNED shared tree
// runs/qcdloop_headers_full/ (B3m/B4m restored), not B13's pruned tree.

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

// ql_tracked_interop.hpp must come FIRST (see B13/BIN2 drivers): qcdloop's own
// template bodies call ql::Real/Imag/Sign/kAbs/kLog on tracked types via
// *qualified* names, so the tracked overloads must be visible at each template's
// definition point.
#include "ql_tracked_interop.hpp"
#include "kokkosMaths.h"
#include "kokkosUtils.h"
#include "boxGPU.h"

using T       = double;
using TScale  = tracked::Tracked<T>;
using TMass   = tracked::Tracked<T>;
using TOutput = tracked::Complex<T>;

namespace {

// Kinematic ranges from boxGPU_test.cc (low=100, up=1000000).
constexpr double kLow = 100.0;
constexpr double kUp  = 1'000'000.0;
constexpr double kMu2 = 91.2 * 91.2;

// Three-mass constants from boxGPU_test.cc (m22=4.9*4.9, m32=10, m42=50.*50.).
constexpr double kM22 = 4.9 * 4.9;
constexpr double kM32 = 10.0;
constexpr double kM42 = 50.0 * 50.0;

double r_uniform(std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Signed uniform, matching boxGPU_test.cc rs(low,up). B16 uses it for the two
// signed invariants p2, p3.
double r_signed(std::mt19937& rng, double lo, double hi) {
    double v = r_uniform(rng, lo, hi);
    std::uniform_real_distribution<double> s(0.0, 1.0);
    return s(rng) < 0.5 ? -v : v;
}

// Wrap a bare double as a tracked scalar with a stable id so the journal
// preserves per-input provenance across the batch.
TMass make_mass(const char* stem, int i, double v) {
    std::string id = std::string(stem) + "[" + std::to_string(i) + "]";
    return tracked::track<T>(id, v);
}

} // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Small default; can override on the command line.
        int sample_count = 256;
        if (argc > 1) {
            try {
                sample_count = std::stoi(argv[1]);
                if (sample_count <= 0) sample_count = 256;
            } catch (...) {
                sample_count = 256;
            }
        }
        std::cout << "B16 spike: sample_count = " << sample_count << "\n";

        using HostSpace = Kokkos::HostSpace;
        Kokkos::View<TScale*,         HostSpace> mu2("mu2", sample_count);
        Kokkos::View<TMass*   [4],    HostSpace> m  ("m",   sample_count);
        Kokkos::View<TScale*  [6],    HostSpace> p  ("p",   sample_count);
        Kokkos::View<TOutput* [3],    HostSpace> res("res", sample_count);

        std::mt19937 rng(12345);

        // B16 configuration (three-mass, boxGPU_test.cc "// B16"):
        //   m1=0, m2=m22, m3=m32, m4=m42;
        //   p1=m22, p2=rs, p3=rs, p4=m42, p5=r, p6=r.
        for (int i = 0; i < sample_count; ++i) {
            mu2(i) = tracked::track<T>("mu2[" + std::to_string(i) + "]", kMu2);

            m(i, 0) = make_mass("m1", i, 0.0);
            m(i, 1) = make_mass("m2", i, kM22);
            m(i, 2) = make_mass("m3", i, kM32);
            m(i, 3) = make_mass("m4", i, kM42);

            p(i, 0) = tracked::track<T>("p1[" + std::to_string(i) + "]", kM22);
            p(i, 1) = tracked::track<T>("p2[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 2) = tracked::track<T>("p3[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 3) = tracked::track<T>("p4[" + std::to_string(i) + "]", kM42);
            p(i, 4) = tracked::track<T>("p5[" + std::to_string(i) + "]", r_uniform(rng, kLow, kUp));
            p(i, 5) = tracked::track<T>("p6[" + std::to_string(i) + "]", r_uniform(rng, kLow, kUp));
        }

        // Host loop, NOT Kokkos::parallel_for: tracked ops are host-only.
        int printed = 0;
        for (int i = 0; i < sample_count; ++i) {
            tracked::scope sample_scope("sample=" + std::to_string(i));
            ql::BO<TOutput, TMass, TScale>(res, mu2, m, p, i);

            if (printed < 3) {
                std::cout << "  B16[" << i << "] coeff0 = ("
                          << res(i, 0).real().value() << ", "
                          << res(i, 0).imag().value() << ")\n";
                ++printed;
            }
        }

        tracked::journal::flush("journal.jsonl");
        std::cout << "wrote journal.jsonl\n";
    }
    Kokkos::finalize();
    return 0;
}
