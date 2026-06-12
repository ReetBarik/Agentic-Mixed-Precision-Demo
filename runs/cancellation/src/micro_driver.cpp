// Auto-generated micro-driver for cancellation_check
// Characterizes catastrophic cancellation: (a + b) - a when b << a.

#include <tracked/tracked.hpp>
#include <tracked/journal.hpp>

#include <random>
#include <iostream>

// Include the kernel source directly (header-style, #pragma once guarded).
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/cancellation.cpp"

int main() {
    constexpr int sample_count = 512;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(1.0, 1.0);
    std::uniform_real_distribution<double> dist_b(1e-15, 1e-10);

    for (int i = 0; i < sample_count; ++i) {
        double a_sample = dist_a(rng);
        double b_sample = dist_b(rng);

        auto a = tracked::track("a", a_sample);
        auto b = tracked::track("b", b_sample);

        auto result = cancellation_check(a, b);

        if (i == 0) {
            std::cout << "cancellation_check(" << a_sample << ", " << b_sample
                      << ") = " << result.value() << std::endl;
        }
    }

    tracked::journal::flush("journal.jsonl");
    return 0;
}
