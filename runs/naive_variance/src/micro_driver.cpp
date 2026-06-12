// Auto-generated micro-driver for `naive_variance`.
// Characterizes numerical sensitivity using the Tracked library.

#include <tracked/tracked.hpp>
#include <tracked/journal.hpp>

#include <random>
#include <iostream>

// Include the kernel source directly — do NOT rewrite the body.
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/naive_variance.cpp"

int main() {
    using T = tracked::Tracked<double>;

    // Input ranges from the kernel spec.
    constexpr double sum_x_min   = 1.0e6,  sum_x_max   = 1.0e6;
    constexpr double sum_x2_min  = 1.0e12, sum_x2_max  = 1.0e12;
    constexpr double n_min       = 1.0,    n_max       = 1.0;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_sum_x(sum_x_min, sum_x_max);
    std::uniform_real_distribution<double> dist_sum_x2(sum_x2_min, sum_x2_max);
    std::uniform_real_distribution<double> dist_n(n_min, n_max);

    constexpr int sample_count = 512;

    for (int i = 0; i < sample_count; ++i) {
        double sum_x_sample  = dist_sum_x(rng);
        double sum_x2_sample = dist_sum_x2(rng);
        double n_sample      = dist_n(rng);

        auto sum_x  = tracked::track("sum_x",  sum_x_sample);
        auto sum_x2 = tracked::track("sum_x2", sum_x2_sample);
        auto n      = tracked::track("n",      n_sample);

        T result = naive_variance<T>(sum_x, sum_x2, n);

        if (i == 0) {
            std::cout << "naive_variance(" << sum_x_sample << ", "
                      << sum_x2_sample << ", " << n_sample << ") = "
                      << result.value() << std::endl;
        }
    }

    tracked::journal::flush("journal.jsonl");
    return 0;
}
