// runs/qcdloop/src/boxGPU_tracked.cpp
//
// Consolidated qcdloop application driver (Stage 2 -> consolidation).
//
// A single main() that dispatches all 21 in-scope box integrals through the
// public ql::BO<TOutput, TMass, TScale>() entry point, matching how a real
// application would be instrumented with the Tracked<T> datatype. This retires
// the Stage-2 per-target scaffolding (runs/B1 .. runs/BIN4), each of which
// exercised one integral in isolation; the recipes below are copied verbatim
// from those per-target drivers (which in turn mirror qcdloop's own
// examples/boxGPU_test.cc BIN sweep + B1..B16 blocks).
//
// Design (see the consolidation spec):
//
//   * One journal.jsonl, scope-tagged. Each integral runs inside a nested RAII
//     scope pair  tracked::scope("integral=<name>") > tracked::scope("sample=<i>"),
//     so every generated op id carries the suffix "@integral=<name>/sample=<i>".
//     Downstream analysis can groupby integral=<name> in one file and still
//     filter by sample. Input track() ids stay bare (track() ignores the scope
//     stack), so per-slot provenance ("p4[5]", "m2[3]", ...) is byte-identical
//     to the Stage-2 per-target journals.
//
//   * Independent per-integral reproducibility. Each integral block re-seeds a
//     fresh std::mt19937(12345) BEFORE its input-fill loop, exactly as every
//     Stage-2 per-target driver did. This reproduces each per-target draw
//     sequence bit-for-bit, so a single integral can be re-run in isolation for
//     hotspot debugging without re-running all 21. (This is the spec's own
//     STOP-condition fallback to the srand(12345+offset) scheme: local mt19937
//     instances, which is what the whole Stage-2 sweep already used.)
//
//   * The input-fill loop runs OUTSIDE the scope pair (matching Stage-2, where
//     fill preceded the per-sample scope). Because track() emits no record and
//     ignores the scope stack, this keeps input ids bare while only the ops
//     generated inside ql::BO get the scope suffix.
//
//   * journal_meta.json is emitted alongside journal.jsonl at the end of main.
//     Build-provenance fields (shim/patch hashes, tracked version, qcdloop
//     commit, C8 site tally) are injected as compile definitions by CMake at
//     configure time; run-intrinsic fields are known here.
//
// Include order matters, and ql_tracked_interop.hpp must come FIRST: qcdloop's
// template bodies call ql::Real/Imag/Sign/kAbs/kLog on tracked types via
// *qualified* names, for which ADL does not apply, so the tracked overloads
// must be visible at each template's definition point. See the per-target
// drivers for the full rationale.

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/complex.hpp>
#include <tracked/journal.hpp>

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

// Massive-box constants (values from boxGPU_test.cc's B6..B16 blocks).
constexpr double kM2      = 10.0;         // B6..B10 : m4, and the m2-valued legs
constexpr double kM32     = 10.0;         // B11..B13, B16 : m3
constexpr double kM42     = 50.0 * 50.0;  // B11..B16 : m4
constexpr double kM22     = 4.9 * 4.9;    // B14..B16 : m2
constexpr double kMassVal = 10.0;         // BIN sweep : the first n_masses masses

constexpr int kSeedBase = 12345;

double r_uniform(std::mt19937& rng, double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return d(rng);
}

// Signed uniform, matching boxGPU_test.cc rs(low,up): draws the magnitude, then
// a second draw for the sign. Consumes two rng values (vs one for r_uniform),
// so draw order is load-bearing for bit-identical reproduction.
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

TScale make_scale(const char* stem, int i, double v) {
    std::string id = std::string(stem) + "[" + std::to_string(i) + "]";
    return tracked::track<T>(id, v);
}

// The 21 in-scope integral names, in dispatch order (B1..B16 then BIN0..BIN4).
const std::vector<std::string> kIntegrals = {
    "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10",
    "B11","B12","B13","B14","B15","B16",
    "BIN0","BIN1","BIN2","BIN3","BIN4",
};

} // namespace

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // --sample-count N  (default 256, for Stage-2 parity; 100k later).
        // --sample-offset O (default 0): dispatch global samples [O, O+N) instead
        //   of [0, N).  Inputs for samples [0, O) are still filled so the mt19937
        //   draw sequence and the per-input ids are byte-identical to a single
        //   [0, O+N) run — but those samples emit NO ops (track() is recordless),
        //   so the journal holds only this chunk.  This makes chunk [O, O+N)
        //   bit-identical to the same samples in one big run, letting a 100k
        //   characterization run in bounded-journal chunks that reduce in-process.
        int sample_count = 256;
        int sample_offset = 0;
        for (int a = 1; a < argc; ++a) {
            std::string arg = argv[a];
            if (arg == "--sample-count" && a + 1 < argc) {
                try {
                    int n = std::stoi(argv[++a]);
                    if (n > 0) sample_count = n;
                } catch (...) { /* keep default */ }
            } else if (arg == "--sample-offset" && a + 1 < argc) {
                try {
                    int n = std::stoi(argv[++a]);
                    if (n >= 0) sample_offset = n;
                } catch (...) { /* keep default */ }
            }
        }
        // Total samples to materialize (fill); only [sample_offset, total) dispatch.
        const int total_samples = sample_offset + sample_count;
        std::cout << "qcdloop consolidated driver: sample_count = "
                  << sample_count << " per integral, offset = " << sample_offset
                  << " (global samples [" << sample_offset << ", " << total_samples
                  << ")), " << kIntegrals.size() << " integrals\n";

        using HostSpace = Kokkos::HostSpace;
        Kokkos::View<TScale*,      HostSpace> mu2("mu2", total_samples);
        Kokkos::View<TMass*  [4],  HostSpace> m  ("m",   total_samples);
        Kokkos::View<TScale* [6],  HostSpace> p  ("p",   total_samples);
        Kokkos::View<TOutput*[3],  HostSpace> res("res", total_samples);

        // Fill mu2 once (constant across every integral and sample). mu2 ids are
        // bare; the value is identical to every Stage-2 driver.
        auto fill_mu2 = [&]() {
            for (int i = 0; i < total_samples; ++i)
                mu2(i) = make_scale("mu2", i, kMu2);
        };

        // Run one integral: (re)fill inputs with a fresh mt19937(12345) OUTSIDE
        // the scope pair (bare input ids), then dispatch each sample through
        // ql::BO inside integral>sample scopes.
        //
        // fill(i, rng) assigns m(i,*) and p(i,*) for sample i. Assignments must
        // execute in positional (p0..p5) order so the rng draw sequence matches
        // the Stage-2 per-target driver exactly.
        auto run_integral = [&](const std::string& name,
                                const std::function<void(int, std::mt19937&)>& fill) {
            std::mt19937 rng(kSeedBase);
            fill_mu2();
            // Fill [0, total): replays the rng draws and input ids for the skipped
            // prefix exactly (track() emits nothing), so dispatched samples match a
            // single [0, total) run bit-for-bit.
            for (int i = 0; i < total_samples; ++i) fill(i, rng);

            tracked::scope integral_scope("integral=" + name);
            bool printed = false;
            for (int i = sample_offset; i < total_samples; ++i) {
                tracked::scope sample_scope("sample=" + std::to_string(i));
                ql::BO<TOutput, TMass, TScale>(res, mu2, m, p, i);
                if (!printed) {
                    std::cout << "  " << name << "[" << i << "] coeff0 = ("
                              << res(i, 0).real().value() << ", "
                              << res(i, 0).imag().value() << ")\n";
                    printed = true;
                }
            }
        };

        // Convenience: set an m slot to a constant.
        auto sm = [&](int i, int slot, const char* stem, double v) {
            m(i, slot) = make_mass(stem, i, v);
        };
        // Convenience: set a p slot to a constant.
        auto sp = [&](int i, int slot, const char* stem, double v) {
            p(i, slot) = make_scale(stem, i, v);
        };

        // ----------------------------------------------------------------------
        // Massless / one-mass boxes (B0m/B1m/B2m families). Recipes verbatim
        // from runs/B1..B16/src/micro_driver.cpp.
        // ----------------------------------------------------------------------

        // B1: m=0000; p=[0,0,0,0, r_u, r_u]
        run_integral("B1", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",0.0); sp(i,3,"p4",0.0);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B2: m=0000; p=[0,0,0, r_s, r_u, r_u]
        run_integral("B2", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",0.0);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B3: m=0000; p=[0, r_s, 0, r_s, r_u, r_u]
        run_integral("B3", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3",0.0);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B4: m=0000; p=[0,0, r_s, r_s, r_u, r_u]
        run_integral("B4", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B5: m=0000; p=[0, r_s, r_s, r_s, r_u, r_u]
        run_integral("B5", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",0.0);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B6: m=[0,0,0,kM2]; p=[0,0,kM2,kM2, r_u, r_u]
        run_integral("B6", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",kM2); sp(i,3,"p4",kM2);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B7: m=[0,0,0,kM2]; p=[0,0,kM2, r_s, r_u, r_u]
        run_integral("B7", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0); sp(i,2,"p3",kM2);
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B8: m=[0,0,0,kM2]; p=[0,0, r_s, r_s, r_u, r_u]
        run_integral("B8", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0); sp(i,1,"p2",0.0);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B9: m=[0,0,0,kM2]; p=[0, r_s, r_s, kM2, r_u, r_u]
        run_integral("B9", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM2);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B10: m=[0,0,0,kM2]; p=[0, r_s, r_s, r_s, r_u, r_u]
        run_integral("B10", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",0.0); sm(i,3,"m4",kM2);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B11: m=[0,0,kM32,kM42]; p=[0, kM32, r_s, kM42, r_u, r_u]
        run_integral("B11", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0); sp(i,1,"p2",kM32);
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B12: m=[0,0,kM32,kM42]; p=[0, r_s, r_s, kM42, r_u, r_u]
        run_integral("B12", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B13: m=[0,0,kM32,kM42]; p=[0, r_s, r_s, r_s, r_u, r_u]  (Stage-1 lock)
        run_integral("B13", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",0.0); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",0.0);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4", r_signed(rng,kLow,kUp));
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B14: m=[0,kM22,0,kM42]; p=[kM22, kM22, kM42, kM42, r_u, r_u]
        run_integral("B14", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",0.0); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22); sp(i,1,"p2",kM22); sp(i,2,"p3",kM42); sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B15: m=[0,kM22,0,kM42]; p=[kM22, r_s, r_s, kM42, r_u, r_u]
        run_integral("B15", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",0.0); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // B16: m=[0,kM22,kM32,kM42]; p=[kM22, r_s, r_s, kM42, r_u, r_u]
        run_integral("B16", [&](int i, std::mt19937& rng) {
            sm(i,0,"m1",0.0); sm(i,1,"m2",kM22); sm(i,2,"m3",kM32); sm(i,3,"m4",kM42);
            sp(i,0,"p1",kM22);
            sp(i,1,"p2", r_signed(rng,kLow,kUp));
            sp(i,2,"p3", r_signed(rng,kLow,kUp));
            sp(i,3,"p4",kM42);
            sp(i,4,"p5", r_uniform(rng,kLow,kUp)); sp(i,5,"p6", r_uniform(rng,kLow,kUp));
        });

        // ----------------------------------------------------------------------
        // BIN sweep (boxGPU_test.cc:189 loop): the first n_masses internal
        // masses are kMassVal, the rest 0; all four invariants p1..p4 signed.
        // ----------------------------------------------------------------------

        auto bin_fill = [&](int n_masses) {
            return [&, n_masses](int i, std::mt19937& rng) {
                const char* mstems[4] = {"m1","m2","m3","m4"};
                for (int s = 0; s < 4; ++s)
                    m(i, s) = make_mass(mstems[s], i, s < n_masses ? kMassVal : 0.0);
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

        // ---- journal.jsonl -------------------------------------------------
        tracked::journal::flush("journal.jsonl");
        std::cout << "wrote journal.jsonl\n";

        // ---- journal_meta.json (build/run metadata sibling) ----------------
        // Build-provenance fields come from CMake compile definitions; the
        // run-intrinsic fields are known here. C8_SITES_JSON is a pre-formatted
        // JSON object literal.
#ifndef QCDLOOP_SHIM_SHA
#  define QCDLOOP_SHIM_SHA "unknown"
#endif
#ifndef QCDLOOP_PATCH_SHA
#  define QCDLOOP_PATCH_SHA "unknown"
#endif
#ifndef QCDLOOP_TRACKED_VERSION
#  define QCDLOOP_TRACKED_VERSION "unknown"
#endif
#ifndef QCDLOOP_COMMIT
#  define QCDLOOP_COMMIT "unknown"
#endif
        // C8 site tally is a fixed structure derived from the whole-app C8
        // patch (box/B3m.h: 3 tracked->int casts + 5 int->tracked wraps;
        // box/B4m.h: 1 tracked-comparison .value()). Emitted as a raw JSON
        // object. The build STOP-gate verifies the patch is byte-identical to
        // this 9-site patch, so this literal stays in sync with the patch.
        const char* kC8SitesJson =
            "{\"total\": 9, \"by_kind\": {\"a\": 3, \"b\": 5, \"c\": 1}, "
            "\"files\": [\"box/B3m.h\", \"box/B4m.h\"]}";
        {
            std::ofstream meta("journal_meta.json");
            meta << "{\n";
            meta << "  \"app\": \"qcdloop\",\n";
            meta << "  \"driver\": \"boxGPU_tracked\",\n";
            meta << "  \"integrals\": [";
            for (std::size_t k = 0; k < kIntegrals.size(); ++k) {
                if (k) meta << ",";
                meta << "\"" << kIntegrals[k] << "\"";
            }
            meta << "],\n";
            meta << "  \"sample_count_per_integral\": " << sample_count << ",\n";
            meta << "  \"sample_offset\": " << sample_offset << ",\n";
            meta << "  \"seed_base\": " << kSeedBase << ",\n";
            meta << "  \"backend\": \"Kokkos::Serial\",\n";
            meta << "  \"shim_source_hash\": \"" << QCDLOOP_SHIM_SHA << "\",\n";
            meta << "  \"c8_patch_sha\": \"" << QCDLOOP_PATCH_SHA << "\",\n";
            meta << "  \"c8_sites_patched\": " << kC8SitesJson << ",\n";
            meta << "  \"tracked_version\": \"" << QCDLOOP_TRACKED_VERSION << "\",\n";
            meta << "  \"qcdloop_commit\": \"" << QCDLOOP_COMMIT << "\"\n";
            meta << "}\n";
        }
        std::cout << "wrote journal_meta.json\n";
    }
    Kokkos::finalize();
    return 0;
}
