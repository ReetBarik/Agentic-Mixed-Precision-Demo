// Auto-generated micro-driver for kahan_sum
#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/journal.hpp>

#include <random>
#include <iostream>

// Include the kernel source verbatim (header-style with #pragma once).
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/kahan.cpp"

int main() {
    using TD = tracked::Tracked<double>;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(1.0, 1.0e6);
    std::uniform_real_distribution<double> dist_b(1e-15, 1e-8);
    std::uniform_real_distribution<double> dist_c(1e-15, 1e-8);

    constexpr int sample_count = 512;

    for (int i = 0; i < sample_count; ++i) {
        double a_sample = dist_a(rng);
        double b_sample = dist_b(rng);
        double c_sample = dist_c(rng);

        auto a = tracked::track("a", a_sample);
        auto b = tracked::track("b", b_sample);
        auto c = tracked::track("c", c_sample);

        TD result = kahan_sum<TD>(a, b, c);

        if (i == 0) {
            std::cout << "kahan_sum(" << a_sample << ", " << b_sample
                      << ", " << c_sample << ") = " << result.value() << "\n";
        }
    }

    tracked::journal::flush("journal.jsonl");
    return 0;
}
