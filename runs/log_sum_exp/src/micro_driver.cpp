// Auto-generated micro-driver for log_sum_exp_naive characterization.
#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/journal.hpp>

#include <random>
#include <iostream>

// Interop shims: inject std::log and std::exp overloads for Tracked<double>
// that delegate to tracked::log / tracked::exp.  This is technically UB
// (injecting into namespace std) but works reliably on all major compilers
// and is required because the kernel hard-codes `std::log` / `std::exp`.
namespace std {
    inline ::tracked::Tracked<double> log(const ::tracked::Tracked<double>& x) {
        return ::tracked::log(x);
    }
    inline ::tracked::Tracked<double> exp(const ::tracked::Tracked<double>& x) {
        return ::tracked::exp(x);
    }
}

// Include the original kernel source verbatim — do not rewrite it.
#include "/home/rbarik/Agentic-Mixed-Precision-Demo/tests/agents/fixtures/kernels/log_sum_exp.cpp"

int main() {
    using T = tracked::Tracked<double>;

    constexpr int sample_count = 512;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist_a(0.0, 1.0);
    std::uniform_real_distribution<double> dist_b(100.0, 200.0);

    for (int i = 0; i < sample_count; ++i) {
        double a_sample = dist_a(rng);
        double b_sample = dist_b(rng);

        auto a = tracked::track("a", a_sample);
        auto b = tracked::track("b", b_sample);

        T result = log_sum_exp_naive<T>(a, b);

        if (i == 0) {
            std::cout << "log_sum_exp_naive(" << a_sample << ", " << b_sample
                      << ") = " << result.value() << std::endl;
        }
    }

    tracked::journal::flush("journal.jsonl");
    return 0;
}
