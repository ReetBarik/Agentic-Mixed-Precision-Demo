// Auto-generated micro-driver for log_sum_exp_naive
// Characterizes numerical sensitivity using the Tracked library.

#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/journal.hpp>

#include <cmath>
#include <random>
#include <iostream>

// ---------------------------------------------------------------------------
// Interop shims for std:: math on Tracked<double>.
//
// The kernel calls std::log and std::exp directly.  We inject location-
// forwarding overloads into namespace std: each takes a trailing
// tracked::SourceLocation (default {}) and threads it into tracked::log /
// tracked::exp.  The kernel passes TRACKED_HERE at the call site, so the op is
// attributed to the *kernel* function (log_sum_exp_naive), not to this shim.
//
// Namespace-injection into std is technically UB but works on all major
// compilers and is the standard interop approach for the Tracked library.
// ---------------------------------------------------------------------------
namespace std {
    inline ::tracked::Tracked<double> log(const ::tracked::Tracked<double>& x,
                                          ::tracked::SourceLocation loc = {}) {
        return ::tracked::log(x, loc);
    }
    inline ::tracked::Tracked<double> exp(const ::tracked::Tracked<double>& x,
                                          ::tracked::SourceLocation loc = {}) {
        return ::tracked::exp(x, loc);
    }
}

// Include the kernel source verbatim (it is a header-guarded .cpp with
// #pragma once and a templated definition).
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/log_sum_exp.cpp"

int main() {
    constexpr int sample_count = 512;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(0.0, 1.0);
    std::uniform_real_distribution<double> dist_b(100.0, 200.0);

    for (int i = 0; i < sample_count; ++i) {
        double a_sample = dist_a(rng);
        double b_sample = dist_b(rng);

        auto a = tracked::track("a", a_sample);
        auto b = tracked::track("b", b_sample);

        auto result = log_sum_exp_naive(a, b);

        if (i == 0) {
            std::cout << "log_sum_exp_naive(" << a_sample << ", " << b_sample
                      << ") = " << result.value() << std::endl;
        }
    }

    tracked::journal::flush("journal.jsonl");
    return 0;
}
