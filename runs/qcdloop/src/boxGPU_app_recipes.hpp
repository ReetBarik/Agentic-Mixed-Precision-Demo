// runs/qcdloop/src/boxGPU_app_recipes.hpp
//
// Shared recipe + dispatch for the Validator's two *application* drivers
// (vanilla and double-double).  This is the de-tracked twin of
// boxGPU_tracked.cpp: same 21-integral recipes, same mt19937(12345) input
// generation, same --sample-offset chunking contract — but NO Tracked type, no
// scopes, no journal.  Instead it prints each dispatched sample's three Laurent
// coefficients (coeff0/1/2, real+imag) as exact hex so the Validator can compute
// precise-digits by comparing a candidate build against a double-double
// ground-truth build on bit-identical inputs.
//
// Templated on <TOutput, TMass, TScale, Printer> so ONE recipe source serves
// both builds; each thin shim (boxGPU_vanilla.cpp / boxGPU_dd.cpp) fixes the
// concrete types + a Printer and includes the target library's boxGPU.h FIRST
// (so ql::BO<...> is visible when this template is parsed):
//
//   * vanilla : TOutput=Kokkos::complex<double>, TMass=TScale=double,
//               built against runs/qcdloop_headers_full (master).
//   * dd      : TOutput=ql::ddfun::ddcomplex, TMass=TScale=ql::ddfun::ddouble,
//               built with -DUSE_DD_COMPLEX against qcdloop@ddfun_enabled.
//
// Input-identity is by construction: both drivers run the SAME mt19937(12345)
// recipe below, so vanilla/candidate/DD evaluate the same physical points.  The
// recipes are copied verbatim from boxGPU_tracked.cpp (which reproduced the
// Stage-2 per-target draws bit-for-bit); see runs/qcdloop/VALIDATION.md.
//
// Output line (one per dispatched sample, tag-prefixed so Kokkos banners are
// ignored by the parser):
//   RES,<integral>,<global_sample_idx>,c0re,c0im,c1re,c1im,c2re,c2im
// where each component is a hex token emitted by Printer:
//   * vanilla : 0x<16-hex>            (IEEE-754 bit pattern of the double)
//   * dd      : 0x<16-hex>|0x<16-hex> (hi|lo bit patterns of the two doubles)

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace ql_app {

// ---- exact hex encoding of a double (IEEE-754 bit pattern) ------------------
inline std::string dhex(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof(u));
    char buf[19];  // "0x" + 16 hex + NUL
    std::snprintf(buf, sizeof(buf), "0x%016llx", (unsigned long long)u);
    return std::string(buf);
}

// ---- kinematic constants (from boxGPU_test.cc / boxGPU_tracked.cpp) ---------
namespace consts {
constexpr double kLow = 100.0;
constexpr double kUp  = 1'000'000.0;
constexpr double kMu2 = 91.2 * 91.2;
constexpr double kM2      = 10.0;
constexpr double kM32     = 10.0;
constexpr double kM42     = 50.0 * 50.0;
constexpr double kM22     = 4.9 * 4.9;
constexpr double kMassVal = 10.0;
constexpr int    kSeedBase = 12345;
}  // namespace consts

inline double r_uniform(std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Signed uniform, matching boxGPU_test.cc rs(low,up): draw magnitude, then a
// second draw for the sign.  Consumes two rng values (vs one for r_uniform), so
// draw order is load-bearing for bit-identical reproduction.
inline double r_signed(std::mt19937& rng, double lo, double hi) {
    double v = r_uniform(rng, lo, hi);
    std::uniform_real_distribution<double> s(0.0, 1.0);
    return s(rng) < 0.5 ? -v : v;
}

// The 21 in-scope integral names, in dispatch order (B1..B16 then BIN0..BIN4).
inline const std::vector<std::string>& integral_names() {
    static const std::vector<std::string> names = {
        "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10",
        "B11","B12","B13","B14","B15","B16",
        "BIN0","BIN1","BIN2","BIN3","BIN4",
    };
    return names;
}

// run_app<TOutput,TMass,TScale,Printer>: parse --sample-count/--sample-offset,
// then run every integral through ql::BO on the SAME mt19937 draws, printing the
// three coeffs of each dispatched sample via Printer.  Mirrors boxGPU_tracked
// main() minus tracking.
template <class TOutput, class TMass, class TScale, class Printer>
int run_app(int argc, char* argv[]) {
    using namespace consts;
    Kokkos::initialize(argc, argv);
    int rc = 0;
    {
        int sample_count = 256;
        int sample_offset = 0;
        for (int a = 1; a < argc; ++a) {
            std::string arg = argv[a];
            if (arg == "--sample-count" && a + 1 < argc) {
                try { int n = std::stoi(argv[++a]); if (n > 0) sample_count = n; }
                catch (...) {}
            } else if (arg == "--sample-offset" && a + 1 < argc) {
                try { int n = std::stoi(argv[++a]); if (n >= 0) sample_offset = n; }
                catch (...) {}
            }
        }
        const int total_samples = sample_offset + sample_count;
        std::cerr << "boxGPU_app: sample_count=" << sample_count
                  << " offset=" << sample_offset
                  << " (global [" << sample_offset << ", " << total_samples << "))\n";

        using HostSpace = Kokkos::HostSpace;
        Kokkos::View<TScale*,      HostSpace> mu2("mu2", total_samples);
        Kokkos::View<TMass*  [4],  HostSpace> m  ("m",   total_samples);
        Kokkos::View<TScale* [6],  HostSpace> p  ("p",   total_samples);
        Kokkos::View<TOutput*[3],  HostSpace> res("res", total_samples);

        auto fill_mu2 = [&]() {
            for (int i = 0; i < total_samples; ++i) mu2(i) = TScale(kMu2);
        };

        // Set an m / p slot to a constant (stem arg kept for verbatim recipe
        // copy from boxGPU_tracked.cpp; unused without journaling).
        auto sm = [&](int i, int slot, const char* /*stem*/, double v) {
            m(i, slot) = TMass(v);
        };
        auto sp = [&](int i, int slot, const char* /*stem*/, double v) {
            p(i, slot) = TScale(v);
        };

        // Reproducible per-integral: fresh mt19937(12345), fill [0,total) so the
        // skipped prefix advances the draw sequence identically, then dispatch
        // [offset,total) through ql::BO and print each coeff triple.
        auto run_integral = [&](const std::string& name,
                                const std::function<void(int, std::mt19937&)>& fill) {
            std::mt19937 rng(kSeedBase);
            fill_mu2();
            for (int i = 0; i < total_samples; ++i) fill(i, rng);

            std::string out;
            out.reserve(sample_count * 96);
            for (int i = sample_offset; i < total_samples; ++i) {
                ql::BO<TOutput, TMass, TScale>(res, mu2, m, p, i);
                out += "RES,";
                out += name;
                out += ',';
                out += std::to_string(i);
                for (int k = 0; k < 3; ++k) {
                    out += ',';
                    Printer::emit(out, res(i, k).real());
                    out += ',';
                    Printer::emit(out, res(i, k).imag());
                }
                out += '\n';
            }
            std::cout << out;
        };

        // ---- Massless / one-mass boxes (recipes verbatim from boxGPU_tracked) ----
        run_integral("B1", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",0.0); sp(i,3,"p4",0.0);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B2", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",0.0);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B3", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3",0.0);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B4", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B5", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B6", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",kM2); sp(i,3,"p4",kM2);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B7", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",kM2);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B8", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B9", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM2);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B10", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B11", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0); sp(i,1,"p2",kM32);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B12", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B13", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B14", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",0.0); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22); sp(i,1,"p2",kM22); sp(i,2,"p3",kM42); sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B15", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",0.0); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });
        run_integral("B16", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // ---- BIN sweep: first n_masses masses = kMassVal, rest 0; p1..p4 signed ----
        auto bin_fill = [&](int n_masses) {
            return [&, n_masses](int i, std::mt19937& rng) {
                const char* mstems[4] = {"m1","m2","m3","m4"};
                for (int s = 0; s < 4; ++s)
                    m(i, s) = TMass(s < n_masses ? kMassVal : 0.0), (void)mstems;
                sp(i,0,"p1", r_signed(rng,kLow,kUp));
                sp(i,1,"p2", r_signed(rng,kLow,kUp));
                sp(i,2,"p3", r_signed(rng,kLow,kUp));
                sp(i,3,"p4", r_signed(rng,kLow,kUp));
                sp(i,4,"p5", r_uniform(rng,kLow,kUp));
                sp(i,5,"p6", r_uniform(rng,kLow,kUp));
            };
        };
        run_integral("BIN0", bin_fill(0));
        run_integral("BIN1", bin_fill(1));
        run_integral("BIN2", bin_fill(2));
        run_integral("BIN3", bin_fill(3));
        run_integral("BIN4", bin_fill(4));
    }
    Kokkos::finalize();
    return rc;
}

}  // namespace ql_app
