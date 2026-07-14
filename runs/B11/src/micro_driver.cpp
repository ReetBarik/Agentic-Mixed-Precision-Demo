// runs/B11/src/micro_driver.cpp
//
// Stage 2 leaf validation: instantiate ql::BO (routing by kinematics to a
// MASSIVE box branch — TWO nonzero internal masses, the B2m family shared by
// B11..B15 and the B13 reference) from qcdloop@master with tracked types and
// dump a journal.
//
// Recipe copied verbatim from examples/boxGPU_test.cc's "// B11" block:
// internal masses m1=m2=0, m3=m32, m4=m42 — the "two mass integrals"
// constants declared at boxGPU_test.cc:555-557 (m22=4.9*4.9, m32=10,
// m42=50*50=2500; B11 uses only m32 and m42). External legs p1=0, p2=m32,
// p3=rs(low,up) (SIGNED uniform), p4=m42, p5=r(low,up), p6=r(low,up). This
// is the first B2m-branch leaf (nonzero-internal-mass count == 2). Each
// array slot is tracked as an independent input carrying its numeric value
// (10.0 for the m32-valued slots, 2500.0 for the m42-valued slots),
// preserving per-slot provenance — the same convention B1..B10 use for
// random draws and constant masses.
//
// Structure mirrors runs/B13/src/micro_driver.cpp (the locked Stage 1
// reference) so op-count distributions cross-compare: same Views, same
// 256-sample batch, same host-loop execution model (tracked ops are
// host-only, so no Kokkos::parallel_for), same per-sample scope, same
// mt19937(12345) draw helpers.

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

// qcdloop headers vendored at runs/B13/qcdloop_headers/ (shared header
// tree; see the Stage 2 spec). Include order matters, and
// ql_tracked_interop.hpp must come FIRST: qcdloop's own template bodies
// (kokkosMaths.h::iszero, all of kokkosUtils.h, box/*.h) call
// ql::Real/Imag/Sign/kAbs/kLog on tracked types via *qualified* names
// (ql::Foo). For a qualified call, ADL does not apply, so the tracked
// overloads must be visible at each template's *definition* point, not
// merely before instantiation. Putting the interop first makes every
// tracked overload a candidate at every qcdloop call site; overload
// resolution then prefers the more-specialized tracked overload over
// kokkosMaths.h's generic kAbs<T>(T)/kLog<T>(T) templates.
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

// Two-mass constants from boxGPU_test.cc:556-557
// (`double m32 = 10;`, `double m42 = 50.*50.;`). B11 uses m32 and m42.
constexpr double kM32 = 10.0;
constexpr double kM42 = 50.0 * 50.0;

double r_uniform(std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Signed uniform, matching boxGPU_test.cc rs(low,up). Used by B11 for p3,
// which can therefore be negative and steer the box dispatcher into a
// discriminant branch.
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
        std::cout << "B11 spike: sample_count = " << sample_count << "\n";

        // Host-space Views. TMass = tracked::Tracked<double> is a non-trivial
        // type, so we rely on Kokkos::HostSpace's default allocator (which
        // uses new/delete, invoking the tracked constructor).
        using HostSpace = Kokkos::HostSpace;
        Kokkos::View<TScale*,         HostSpace> mu2("mu2", sample_count);
        Kokkos::View<TMass*   [4],    HostSpace> m  ("m",   sample_count);
        Kokkos::View<TScale*  [6],    HostSpace> p  ("p",   sample_count);
        Kokkos::View<TOutput* [3],    HostSpace> res("res", sample_count);

        std::mt19937 rng(12345);

        // B11 configuration (verbatim from boxGPU_test.cc "// B11", two-mass):
        //   m1=m2=0, m3=m32, m4=m42; p1=0, p2=m32; p3=rs(low,up) [signed];
        //   p4=m42; p5=r(low,up), p6=r(low,up).
        for (int i = 0; i < sample_count; ++i) {
            mu2(i) = tracked::track<T>("mu2[" + std::to_string(i) + "]", kMu2);

            m(i, 0) = make_mass("m1", i, 0.0);
            m(i, 1) = make_mass("m2", i, 0.0);
            m(i, 2) = make_mass("m3", i, kM32);
            m(i, 3) = make_mass("m4", i, kM42);

            p(i, 0) = tracked::track<T>("p1[" + std::to_string(i) + "]", 0.0);
            p(i, 1) = tracked::track<T>("p2[" + std::to_string(i) + "]", kM32);
            p(i, 2) = tracked::track<T>("p3[" + std::to_string(i) + "]", r_signed(rng, kLow, kUp));
            p(i, 3) = tracked::track<T>("p4[" + std::to_string(i) + "]", kM42);
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
                std::cout << "  B11[" << i << "] coeff0 = ("
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
