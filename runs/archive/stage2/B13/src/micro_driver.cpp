// runs/B13/src/micro_driver.cpp
//
// Feasibility spike: instantiate ql::BO (routing to B13 by kinematics)
// from qcdloop@master with tracked types and dump a journal.
//
// Reference driver: ReetBarik/qcdloop@ddfun_enabled
// examples/box/boxGPU_test_dd_B13.cc. That branch is a functional example
// of qcdloop with a non-double type (double-double), so we mirror its
// call pattern:
//
//   * fill Views for (mu2, m, p) with the same B13 kinematics that
//     boxGPU_test.cc uses on master;
//   * call ql::BO<TOutput,TMass,TScale>() per element — BO()'s dispatcher
//     selects B13 based on the mass configuration (m1=m2=0, m3^2=m32,
//     m4^2=m42);
//   * dump the tracked journal.
//
// Key deviation from the DD driver: we execute on the host directly,
// not through Kokkos::parallel_for. The tracked library's ops are
// host-only (no KOKKOS_INLINE_FUNCTION), so a KOKKOS_LAMBDA closure
// containing them would fail the device-function check even on Serial.
// The cln/lnrat prior art (runs/cln/src/micro_driver.cpp) uses the same
// host-loop pattern.

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

// qcdloop headers vendored at runs/B13/qcdloop_headers/. Include order
// matters, and ql_tracked_interop.hpp must come FIRST: qcdloop's own
// template bodies (kokkosMaths.h::iszero, all of kokkosUtils.h, box/*.h)
// call ql::Real/Imag/Sign/kAbs/kLog on tracked types via *qualified*
// names (ql::Foo). For a qualified call, ADL does not apply, so the
// tracked overloads must be visible at each template's *definition*
// point, not merely before instantiation. Putting the interop first
// makes every tracked overload a candidate at every qcdloop call site;
// overload resolution then prefers the more-specialized tracked overload
// over kokkosMaths.h's generic kAbs<T>(T)/kLog<T>(T) templates.
#include "ql_tracked_interop.hpp"
#include "kokkosMaths.h"
#include "kokkosUtils.h"
#include "boxGPU.h"

using T       = double;
using TScale  = tracked::Tracked<T>;
using TMass   = tracked::Tracked<T>;
using TOutput = tracked::Complex<T>;

namespace {

// Kinematic ranges from boxGPU_test.cc / boxGPU_test_dd_B13.cc.
constexpr double kLow = 100.0;
constexpr double kUp  = 1'000'000.0;
constexpr double kM32 = 10.0;
constexpr double kM42 = 50.0 * 50.0;
constexpr double kMu2 = 91.2 * 91.2;

double r_uniform(std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Signed uniform, matching boxGPU_test.cc rs(low,up).
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
        std::cout << "B13 spike: sample_count = " << sample_count << "\n";

        // Host-space Views. TMass = tracked::Tracked<double> is a non-trivial
        // type, so we rely on Kokkos::HostSpace's default allocator (which
        // uses new/delete, invoking the tracked constructor).
        using HostSpace = Kokkos::HostSpace;
        Kokkos::View<TScale*,         HostSpace> mu2("mu2", sample_count);
        Kokkos::View<TMass*   [4],    HostSpace> m  ("m",   sample_count);
        Kokkos::View<TScale*  [6],    HostSpace> p  ("p",   sample_count);
        Kokkos::View<TOutput* [3],    HostSpace> res("res", sample_count);

        std::mt19937 rng(12345);

        // B13 mass configuration: m1=m2=0, m3^2=m32, m4^2=m42.
        // Momenta: p1=0, p2/p3/p4=rs, p5/p6=r (matches boxGPU_test.cc B13
        // subsection).
        for (int i = 0; i < sample_count; ++i) {
            mu2(i) = tracked::track<T>("mu2[" + std::to_string(i) + "]", kMu2);

            m(i, 0) = make_mass("m1", i, 0.0);
            m(i, 1) = make_mass("m2", i, 0.0);
            m(i, 2) = make_mass("m3", i, kM32);
            m(i, 3) = make_mass("m4", i, kM42);

            p(i, 0) = tracked::track<T>("p1[" + std::to_string(i) + "]", 0.0);
            p(i, 1) = tracked::track<T>("p2[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 2) = tracked::track<T>("p3[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 3) = tracked::track<T>("p4[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 4) = tracked::track<T>("p5[" + std::to_string(i) + "]", r_uniform(rng, kLow, kUp));
            p(i, 5) = tracked::track<T>("p6[" + std::to_string(i) + "]", r_uniform(rng, kLow, kUp));
        }

        // Host loop, NOT Kokkos::parallel_for: tracked ops are host-only.
        int printed = 0;
        for (int i = 0; i < sample_count; ++i) {
            // v0.3 scope: every derived id produced inside ql::BO for this
            // sample gets an "@sample=<i>" suffix, so hot records self-document
            // which sample produced them and downstream queries can filter by
            // sample without cross-referencing. Input track() ids (m1[i], p2[i],
            // …) keep their bare names — scope only affects generated ids.
            tracked::scope sample_scope("sample=" + std::to_string(i));
            ql::BO<TOutput, TMass, TScale>(res, mu2, m, p, i);

            if (printed < 3) {
                std::cout << "  B13[" << i << "] coeff0 = ("
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
